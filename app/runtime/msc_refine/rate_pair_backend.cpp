// Fixed-scaffold Fitch step counts: read-only MSA, no inferred gene tree.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <omp.h>
namespace py=pybind11;
py::array_t<int32_t> steps(py::array_t<uint8_t,py::array::c_style> packed,
 py::array_t<int64_t,py::array::c_style> nodes,
 py::array_t<int64_t,py::array::c_style> children,int64_t length,int threads){
 auto p=packed.request(),o=nodes.request(),c=children.request();
 if(p.ndim!=2||o.ndim!=1||c.ndim!=2||c.shape[0]!=o.shape[0]||c.shape[1]!=2||
    length<1||p.shape[1]!=(length+1)/2||threads<1)throw std::invalid_argument("invalid Fitch input");
 int64_t n=p.shape[0],v=2*n-1,k=o.shape[0];
 if(k!=n-1)throw std::invalid_argument("binary virtual-root topology required");
 auto *pp=(const uint8_t*)p.ptr;auto *oo=(const int64_t*)o.ptr;auto *cc=(const int64_t*)c.ptr;
 std::vector<bool> ready(v,false);for(int i=0;i<n;i++)ready[i]=true;
 for(int64_t i=0;i<k;i++){
  int64_t u=oo[i],a=cc[2*i],b=cc[2*i+1];
  if(u<n||u>=v||ready[u]||a<0||b<0||a>=v||b>=v||!ready[a]||!ready[b]||a==b)
   throw std::invalid_argument("invalid postorder child references");ready[u]=true;
 }
 py::array_t<int32_t> result(length);auto *out=result.mutable_data();constexpr int B=128;
 {py::gil_scoped_release release;
 #pragma omp parallel num_threads(threads)
 {
  std::vector<uint8_t> scratch(v*B);std::vector<int32_t> count(B);
  #pragma omp for schedule(static)
  for(int64_t start=0;start<length;start+=B){
   int width=std::min<int64_t>(B,length-start);std::fill(count.begin(),count.end(),0);
   for(int64_t leaf=0;leaf<n;leaf++)for(int s=0;s<width;s++){
    int64_t pos=start+s;unsigned code=(pp[leaf*p.shape[1]+pos/2]>>((pos%2)*4))&15;
    scratch[leaf*B+s]=code<4?(1u<<code):15;
   }
   for(int64_t i=0;i<k;i++){
    auto *a=scratch.data()+cc[2*i]*B,*b=scratch.data()+cc[2*i+1]*B,*u=scratch.data()+oo[i]*B;
    for(int s=0;s<width;s++){uint8_t both=a[s]&b[s];u[s]=both?both:(a[s]|b[s]);count[s]+=both==0;}
   }
   std::copy(count.begin(),count.begin()+width,out+start);
  }
 }
 }
 return result;
}
PYBIND11_MODULE(_msc_rate_pair,m){m.def("steps",&steps,py::arg("packed"),py::arg("nodes"),py::arg("children"),py::arg("length"),py::arg("threads")=8);}
