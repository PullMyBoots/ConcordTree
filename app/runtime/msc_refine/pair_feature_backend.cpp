// Exact finite polynomial feature assembly;independent of existing extensions.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
namespace py=pybind11;
using F64=py::array_t<double,py::array::c_style|py::array::forcecast>;
using I64=py::array_t<int64_t,py::array::c_style|py::array::forcecast>;
using I32=py::array_t<int32_t,py::array::c_style|py::array::forcecast>;
F64 features(F64 counts,I32 linear,I32 quadratic,I64 linear_card,I64 quadratic_card){
 if(counts.ndim()!=2 || counts.shape(1)!=256 || linear.ndim()!=2 || linear.shape(0)!=3 || linear.shape(1)!=256 ||
    quadratic.ndim()!=2 || quadratic.shape(0)!=3 || quadratic.shape(1)!=65536 || linear_card.ndim()!=1 || quadratic_card.ndim()!=1)
  throw std::invalid_argument("full256 counts and valid invariant layout required");
 int64_t n=counts.shape(0),ld=linear_card.size(),qd=quadratic_card.size(),d=ld+qd;
 auto c=counts.data();auto l=linear.data();auto q=quadratic.data();auto lc=linear_card.data();auto qc=quadratic_card.data();
 for(int64_t i=0;i<3*256;++i)if(l[i]<0 || l[i]>=ld)throw std::invalid_argument("invalid linear feature index");
 for(int64_t i=0;i<3*65536;++i)if(q[i]<0 || q[i]>=qd)throw std::invalid_argument("invalid quadratic feature index");
 for(int64_t i=0;i<ld;++i)if(lc[i]<=0)throw std::invalid_argument("nonpositive linear cardinality");
 for(int64_t i=0;i<qd;++i)if(qc[i]<=0)throw std::invalid_argument("nonpositive quadratic cardinality");
 for(int64_t i=0;i<n*256;++i)if(!std::isfinite(c[i]) || c[i]<0)throw std::invalid_argument("invalid pattern count");
 F64 output(std::vector<py::ssize_t>{n,3,d});auto y=output.mutable_data();std::fill(y,y+n*3*d,0.);
 py::gil_scoped_release unlock;
 #pragma omp parallel for schedule(static)
 for(int64_t row=0;row<n;++row){
  double x[256],total=0.;for(int i=0;i<256;++i)total+=c[row*256+i];
  for(int i=0;i<256;++i)x[i]=total>0?std::sqrt(c[row*256+i]/total):0.;
  for(int cl=0;cl<3;++cl){
   auto z=y+(row*3+cl)*d;auto ll=l+cl*256;auto qq=q+cl*65536;
   for(int i=0;i<256;++i)z[ll[i]]+=x[i];
   for(int i=0;i<256;++i)for(int j=i;j<256;++j)z[ld+qq[i*256+j]]+=(i==j?1.:2.)*x[i]*x[j];
   for(int64_t k=0;k<ld;++k)z[k]*=std::sqrt(2./lc[k]);
   for(int64_t k=0;k<qd;++k)z[ld+k]/=std::sqrt(double(qc[k]));
  }
 }
 return output;
}
PYBIND11_MODULE(_pair_features,m){m.def("features",&features);}
