// Single pass over cached quartet classes. This extension never scores DNA.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>
namespace py=pybind11;
using A=py::array_t<int32_t,py::array::c_style|py::array::forcecast>;
using D=py::array_t<double,py::array::c_style|py::array::forcecast>;
py::tuple reduce_votes(A codes,D weights,int edge_count,bool include_margin){
 if(edge_count<0||codes.ndim()!=1||weights.ndim()!=2||weights.shape(0)!=codes.shape(0)||weights.shape(1)!=3)
  throw std::invalid_argument("invalid quartet support dimensions");
 py::array_t<int64_t> counts({edge_count,3});py::array_t<double> votes({edge_count,3});
 py::array_t<double> margin_sum(include_margin?edge_count:0);auto margin=margin_sum.mutable_unchecked<1>();
 auto c=counts.mutable_unchecked<2>();auto v=votes.mutable_unchecked<2>();
 auto code=codes.unchecked<1>();auto w=weights.unchecked<2>();
 {py::gil_scoped_release release;
  for(int e=0;e<edge_count;e++){
   for(int j=0;j<3;j++){c(e,j)=0;v(e,j)=0.;}
   if(include_margin)margin(e)=0.;
  }
  for(py::ssize_t i=0;i<codes.shape(0);i++){
   int x=code(i),e=(x>>4)-1,state=x&3,alt=(x>>2)&3;
   if(x<0||e>=edge_count||state>2)throw std::invalid_argument("invalid cached quartet topology code");
   if(e<0)continue;
   if(alt>2||alt==state)throw std::invalid_argument("invalid alternative code");
   double a=w(i,0),b=w(i,1),d=w(i,2);
   if(!std::isfinite(a)||!std::isfinite(b)||!std::isfinite(d))throw std::invalid_argument("nonfinite quartet support scores");
   double top=std::max(a,std::max(b,d));
   double tol=64*std::numeric_limits<double>::epsilon()*std::max(1.,std::max(std::abs(a),std::max(std::abs(b),std::abs(d))));
   bool win[3]={top-a<=tol,top-b<=tol,top-d<=tol};int n=int(win[0])+int(win[1])+int(win[2]);
   c(e,0)++;if(n==3)continue;c(e,1)++;if(n==2)c(e,2)++;
   if(include_margin)margin(e)+=w(i,state)-std::max(w(i,alt),w(i,3-state-alt));
   v(e,0)+=double(win[state])/n;v(e,1)+=double(win[alt])/n;v(e,2)+=double(win[3-state-alt])/n;
  }
 }
 if(include_margin)return py::make_tuple(counts,votes,margin_sum);
 return py::make_tuple(counts,votes);
}
PYBIND11_MODULE(_branch_support_reduce,m){
 m.def("reduce",[](A c,D w,int n){return reduce_votes(c,w,n,false);});
 m.def("reduce_margins",[](A c,D w,int n){return reduce_votes(c,w,n,true);});
}
