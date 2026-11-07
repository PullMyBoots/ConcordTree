// Original-order NNI gains; bounded-memory NumPy-compatible scalar sums.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace py=pybind11;
using A=py::array_t<int32_t,py::array::c_style|py::array::forcecast>;
using D=py::array_t<double,py::array::c_style|py::array::forcecast>;
using B=py::array_t<bool,py::array::c_style|py::array::forcecast>;
// NumPy's contiguous double reduction: 128-value leaves, eight accumulators,
// recursive half-splits rounded down to a multiple of eight. No reassociation.
class PairwiseStream {
 std::vector<int64_t> sizes;std::vector<double> values;
 std::array<double,8> accum{},tail{};int64_t total,buffer,index=0,offset=0;double small=-0.0;
 void plan(int64_t n){if(n<=128){if(n)sizes.push_back(n);return;}auto h=(n/2)/8*8;plan(h);plan(n-h);}
 double combine(int64_t n,size_t& at)const{if(n<=128)return values[at++];auto h=(n/2)/8*8;double left=combine(h,at),right=combine(n-h,at);return left+right;}
 public:
 explicit PairwiseStream(int64_t n,int64_t b):total(n),buffer(b){for(int64_t at=0;at<n;at+=buffer)plan(std::min<int64_t>(buffer,n-at));values.reserve(sizes.size());}
 void push(double x){
  int64_t n=sizes[index],k=offset++;
  if(n<8)small+=x;
  else if(k<8)accum[k]=x;
  else if(k<(n/8)*8)accum[k%8]+=x;
  else tail[k%8]=x;
  if(offset==n){
   double r=small;
   if(n>=8){r=((accum[0]+accum[1])+(accum[2]+accum[3]))+((accum[4]+accum[5])+(accum[6]+accum[7]));for(int k=0;k<n%8;k++)r+=tail[k];}
   values.push_back(r);index++;offset=0;small=-0.0;
  }
 }
 double result()const{if(!total)return 0.0;if(index!=int64_t(sizes.size())||offset)throw std::runtime_error("incomplete streaming sum");size_t leaf=0;double sum=0.;for(int64_t at=0;at<total;at+=buffer)sum+=combine(std::min<int64_t>(buffer,total-at),leaf);return sum;}
};
py::dict reduce_sums(A codes,D weights,B is_new,int edge_count,int buffer_size){
 if(buffer_size<16||buffer_size%16)throw std::runtime_error("invalid NumPy reduction buffer size");
 if(codes.ndim()!=1||weights.ndim()!=2||weights.shape(0)!=codes.shape(0)||weights.shape(1)!=3||is_new.ndim()!=1||is_new.shape(0)!=codes.shape(0)||edge_count<1)throw std::runtime_error("invalid dimensions");
 auto c=codes.unchecked<1>();auto w=weights.unchecked<2>();auto mask=is_new.unchecked<1>();
 int64_t count=c.shape(0),added=std::count(is_new.data(),is_new.data()+count,true);
 py::array_t<double> score({edge_count,3});auto s=score.mutable_unchecked<2>();
 for(int e=0;e<edge_count;e++)for(int j=0;j<3;j++)s(e,j)=0.;
 double objective=0.,magnitude_sum,new_sum;int64_t active=0;
 {py::gil_scoped_release release;PairwiseStream magnitudes(count,buffer_size),new_values(added,buffer_size);
  for(int64_t i=0;i<count;i++){
   int code=c(i),state=code&3,alt=(code>>2)&3,e=(code>>4)-1;
   if(code<0||state>2||e>=edge_count||(e>=0&&(alt>2||alt==state)))throw std::runtime_error("invalid integer topology code");
   double current=w(i,state);objective+=current;
   magnitudes.push(std::max(std::abs(w(i,0)),std::max(std::abs(w(i,1)),std::abs(w(i,2)))));
   if(mask(i))new_values.push(current);
   if(e<0)continue;
   s(e,1)+=w(i,alt)-current;s(e,2)+=w(i,3-state-alt)-current;active++;
  }
  magnitude_sum=magnitudes.result();new_sum=new_values.result();
 }
 py::dict result;result["scores"]=score;result["objective"]=objective;result["active_quartets"]=active;result["magnitude_sum"]=magnitude_sum;result["new_sum"]=new_sum;return result;
}
PYBIND11_MODULE(_msc_active_streamsum,m){m.def("reduce",&reduce_sums,py::arg("codes"),py::arg("weights"),py::arg("is_new"),py::arg("edge_count"),py::arg("buffer_size")=8192);}
