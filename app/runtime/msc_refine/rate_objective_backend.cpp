// Independent global-objective audit. No topology search or model fitting.
// A rooted binary DP counts every unordered quartet exactly once, at its
// highest branching node (3+1 or 2+2). Internal rooted triples retain their
// cherry identity, unlike a sum of internal-edge scores which overcounts.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <omp.h>
namespace py=pybind11;
using I=py::array_t<int64_t,py::array::c_style>;
using D=py::array_t<double,py::array::c_style>;

py::array_t<double> evaluate(py::array_t<uint8_t,py::array::c_style> packed,
 I nodes,I children,I pairs,D pq,D weight,int threads){
 auto p=packed.request(),o=nodes.request(),c=children.request(),q=pairs.request();
 auto z=pq.request(),w=weight.request();
 if(p.ndim!=2||o.ndim!=1||c.ndim!=2||c.shape[0]!=o.shape[0]||c.shape[1]!=2||
    q.ndim!=2||q.shape[1]!=2||z.ndim!=2||z.shape[0]!=q.shape[0]||z.shape[1]!=3||
    w.ndim!=1||w.shape[0]!=q.shape[0]||threads<1)
   throw std::invalid_argument("invalid global quartet objective input");
 int64_t n=p.shape[0],v=2*n-1,k=o.shape[0],m=q.shape[0];
 if(n<4||k!=n-1)throw std::invalid_argument("binary virtual root required");
 const auto *pp=(const uint8_t*)p.ptr;const auto *oo=(const int64_t*)o.ptr;
 const auto *cc=(const int64_t*)c.ptr,*qq=(const int64_t*)q.ptr;
 const auto *zz=(const double*)z.ptr,*ww=(const double*)w.ptr;
 std::vector<bool> ready(v,false);for(int64_t i=0;i<n;i++)ready[i]=true;
 for(int64_t i=0;i<k;i++){
   int64_t u=oo[i],a=cc[2*i],b=cc[2*i+1];
   if(u<n||u>=v||ready[u]||a<0||b<0||a>=v||b>=v||!ready[a]||!ready[b]||a==b)
     throw std::invalid_argument("invalid postorder");
   ready[u]=true;
 }
 for(int64_t i=0;i<2*m;i++)if(qq[i]<0||qq[i]>=2*p.shape[1])
   throw std::invalid_argument("site index outside packed MSA");
 constexpr int B=32,F=14;
 std::array<double,3> sums{0,0,0};
 {py::gil_scoped_release release;
 #pragma omp parallel num_threads(threads)
 {
   std::vector<double> state(v*F*B);
   std::array<double,3> local{0,0,0};
   #pragma omp for schedule(static)
   for(int64_t start=0;start<m;start+=B){
     int width=std::min<int64_t>(B,m-start);
     for(int r=0;r<3;r++){
       std::fill(state.begin(),state.end(),0.);
       for(int64_t leaf=0;leaf<n;leaf++){
         auto *u=state.data()+leaf*F*B;
         for(int s=0;s<width;s++){
           int64_t a=qq[2*(start+s)],b=qq[2*(start+s)+1];
           int x=(pp[leaf*p.shape[1]+a/2]>>((a%2)*4))&15;
           int y=(pp[leaf*p.shape[1]+b/2]>>((b%2)*4))&15;
           if(x>=4||y>=4)continue;
           int partner=r==0?2:(r==1?1:3);
           int bx=(x==0||x==partner),by=(y==0||y==partner);
           u[(bx+2*by)*B+s]=1.;
         }
       }
       for(int64_t i=0;i<k;i++){
         const auto *a=state.data()+cc[2*i]*F*B,*b=state.data()+cc[2*i+1]*F*B;
         auto *u=state.data()+oo[i]*F*B;
         for(int s=0;s<width;s++){
           double L1[4],L2[4],H1a[4],H1b[4],H2a[4],H2b[4];
           double na=0,nb=0,pa1=0,pb1=0,pa2=0,pb2=0;
           for(int j=0;j<4;j++){
             L1[j]=a[j*B+s];L2[j]=b[j*B+s];na+=L1[j];nb+=L2[j];
             if(j&1){pa1+=L1[j];pb1+=L2[j];}
             if(j&2){pa2+=L1[j];pb2+=L2[j];}
           }
           double c=2*zz[3*(start+s)+r],cross1=0,cross2=0,e=0;
           for(int j=0;j<4;j++){
             H1a[j]=c*na-((j&1)?na-pa1:pa1);
             H1b[j]=c*nb-((j&1)?nb-pb1:pb1);
             H2a[j]=c*na-((j&2)?na-pa2:pa2);
             H2b[j]=c*nb-((j&2)?nb-pb2:pb2);
             cross1+=L1[j]*H1b[j];cross2+=L1[j]*H2b[j];
             e+=a[(6+j)*B+s]*H2b[j]+a[(10+j)*B+s]*H1b[j]
               +b[(6+j)*B+s]*H2a[j]+b[(10+j)*B+s]*H1a[j];
           }
           double p1a=a[4*B+s],p2a=a[5*B+s],p1b=b[4*B+s],p2b=b[5*B+s];
           e+=p1a*p2b+p2a*p1b;
           local[r]+=e*ww[start+s];
           u[4*B+s]=p1a+p1b+cross1;u[5*B+s]=p2a+p2b+cross2;
           for(int j=0;j<4;j++){
             u[j*B+s]=L1[j]+L2[j];
             u[(6+j)*B+s]=a[(6+j)*B+s]+b[(6+j)*B+s]+p1a*L2[j]+p1b*L1[j];
             u[(10+j)*B+s]=a[(10+j)*B+s]+b[(10+j)*B+s]+p2a*L2[j]+p2b*L1[j];
           }
         }
       }
     }
   }
   #pragma omp critical
   {for(int r=0;r<3;r++)sums[r]+=local[r];}
 }
 }
 py::array_t<double> result(3);std::copy(sums.begin(),sums.end(),result.mutable_data());return result;
}
PYBIND11_MODULE(_msc_rate_objective,m){m.def("evaluate",&evaluate,py::arg("packed"),py::arg("nodes"),
 py::arg("children"),py::arg("pairs"),py::arg("pq"),py::arg("weight"),py::arg("threads")=8);}
