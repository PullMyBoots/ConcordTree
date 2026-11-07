// Bounded representative distances for sparse aggregate NJ.
// No matrix of all taxon distances is constructed or retained.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include <climits>
namespace py=pybind11;
#include "gamma_shape.cuh"
static void check(cudaError_t e,const char*m){if(e!=cudaSuccess)throw std::runtime_error(std::string(m)+": "+cudaGetErrorString(e));}
struct Memory{void*p=nullptr;size_t size;Memory(size_t n):size(n){if(n)check(cudaMalloc(&p,n),"allocate packed APNJ");}~Memory(){if(p)cudaFree(p);}Memory(const Memory&)=delete;};
constexpr uint64_t ONE=0x1111111111111111ULL;
__device__ uint32_t hash32(uint32_t x){x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;return x^(x>>16);}
template<typename T> __device__ T warp_sum(T x){for(int j=16;j;j>>=1)x+=__shfl_down_sync(0xffffffff,x,j);return x;}
__global__ void project_packed(const uint64_t*seq,int n,int length,int words,int P,uint32_t seed,double*out){
 int row=blockIdx.x,p=blockIdx.y,t=threadIdx.x;double sum=0;
 for(int s=t;s<length;s+=128){int b=(seq[int64_t(row)*words+s/16]>>((s%16)*4))&15;
  if(b<4)sum+=(hash32(uint32_t(s)*4+b+hash32(seed+p*0x9e3779b9U))&1)?1.:-1.;}
 __shared__ double v[4];sum=warp_sum(sum);if(!(t&31))v[t/32]=sum;__syncthreads();
 if(t<32){sum=t<4?v[t]:0;sum=warp_sum(sum);if(!t)out[int64_t(row)*P+p]=sum;}
}
// Exactly the same per-site random signs, prepacked once for all taxa.
__global__ void prepare_projection_masks(int length,int words,int P,uint32_t seed,uint64_t*masks){
 int word=blockIdx.x*blockDim.x+threadIdx.x,p=blockIdx.y,b=blockIdx.z;if(word>=words)return;
 uint64_t mask=0;const uint32_t salt=hash32(seed+p*0x9e3779b9U);
 for(int k=0;k<16;++k){int site=word*16+k;if(site<length && (hash32(uint32_t(site)*4+b+salt)&1))mask|=1ULL<<(k*4);}
 masks[(int64_t(p)*4+b)*words+word]=mask;
}
__global__ void project_bit_packed(const uint64_t*seq,int n,int length,int words,int P,const uint64_t*masks,double*out){
 int row=blockIdx.x,p=blockIdx.y,t=threadIdx.x;int sum=0;
 for(int w=t;w<words;w+=128){
  uint64_t u=seq[int64_t(row)*words+w],lo=u&ONE,hi=(u>>1)&ONE,ok=ONE&~((u>>2)|(u>>3));
  if(w==words-1 && (length&15))ok&=(1ULL<<((length&15)*4))-1;
  uint64_t a=ok&~lo&~hi,c=ok&lo&~hi,g=ok&~lo&hi,tt=ok&lo&hi;
  uint64_t positive=(a&masks[(int64_t(p)*4)*words+w])|(c&masks[(int64_t(p)*4+1)*words+w])|(g&masks[(int64_t(p)*4+2)*words+w])|(tt&masks[(int64_t(p)*4+3)*words+w]);
  sum+=2*__popcll(positive)-__popcll(ok);
 }
 __shared__ int value[4];sum=warp_sum(sum);if(!(t&31))value[t/32]=sum;__syncthreads();
 if(t<32){sum=t<4?value[t]:0;sum=warp_sum(sum);if(!t)out[int64_t(row)*P+p]=double(sum);}
}

// Exact observed-state evidence, independent of missingness-mask variation.
__global__ void observation_evidence(const uint64_t*seq,int n,int length,int words,unsigned long long*out){
 int w=blockIdx.x*blockDim.x+threadIdx.x;if(w>=words)return;
 uint64_t seen[4]={0,0,0,0},observed=0,joint=0;unsigned long long count=0;
 uint64_t tail=ONE;if(w==words-1 && (length&15))tail&=(1ULL<<((length&15)*4))-1;
 for(int row=0;row<n;++row){uint64_t u=seq[int64_t(row)*words+w],lo=u&ONE,hi=(u>>1)&ONE,ok=tail&~((u>>2)|(u>>3));
  seen[0]|=ok&~lo&~hi;seen[1]|=ok&lo&~hi;seen[2]|=ok&~lo&hi;seen[3]|=ok&lo&hi;
  joint|=observed&ok;observed|=ok;count+=__popcll(ok);
 }
 uint64_t variable=(seen[0]&(seen[1]|seen[2]|seen[3]))|(seen[1]&(seen[2]|seen[3]))|(seen[2]&seen[3]);
 atomicAdd(out,(unsigned long long)__popcll(variable));atomicAdd(out+1,(unsigned long long)__popcll(observed));atomicAdd(out+2,count);atomicAdd(out+3,(unsigned long long)__popcll(joint));
}

// One warp per pair of representatives; at most R^2 warps for each edge.
// Useful sites use union semantics for parity with original DIPPER, or common
// sites for an explicit observed-data distance. Tail padding never contributes.
__global__ void representative_distances(const uint64_t*seq,int length,int words,const int*reps,const double*mass,
 const int64_t*pairs,int count,int R,int model,bool common,double alpha,bool moments,bool overlap,double*out,unsigned long long*health){
 int e=blockIdx.x,t=threadIdx.x,lane=t&31,warp=t>>5;if(e>=count)return;
 int a=pairs[int64_t(e)*2],b=pairs[int64_t(e)*2+1],i=warp/R,j=warp%R;
 int columns=moments?(overlap?4:3):1;
 if(a==b){if(!t)for(int k=0;k<columns;++k)out[int64_t(e)*columns+k]=0;return;}
 double wa=mass[a*R+i],wb=mass[b*R+j],answer=0,control_p=0,control_q=0,control_c=0;
 if(wa>0 && wb>0){
  int x=reps[a*R+i],y=reps[b*R+j];int usable=0,mismatch=0,transversions=0;
  if(x!=y){
   for(int w=lane;w<words;w+=32){
    uint64_t u=seq[int64_t(x)*words+w],v=seq[int64_t(y)*words+w],z=u^v;
    uint64_t ou=ONE&~((u>>2)|(u>>3)),ov=ONE&~((v>>2)|(v>>3));
    uint64_t ok=common?(ou&ov):(ou|ov),same=~(z|(z>>1)|(z>>2)|(z>>3))&ou&ov;
    uint64_t tv=ou&ov&z&ONE;
    if(w==words-1 && (length&15)){uint64_t tail=(1ULL<<((length&15)*4))-1;ok&=tail;same&=tail;tv&=tail;}
    usable+=__popcll(ok);mismatch+=__popcll(ok&~same);transversions+=__popcll(tv);
   }
   usable=warp_sum(usable);mismatch=warp_sum(mismatch);transversions=warp_sum(transversions);
  }
  if(!lane){if(x!=y && common && !usable)atomicAdd(health,1ULL);
   double p=x==y?0.:(usable?double(mismatch)/usable:.75),q=x==y?0.:(usable?double(transversions)/usable:.5),floor=.5/(usable+1.);
   double corrected=model==0?p:(model==1?-.75*log(fmax(1.-(4./3.)*p,floor)):(model==2?.75*gamma_inv(fmax(1.-(4./3.)*p,floor),alpha):.5*gamma_inv(fmax(1-2*p+q,floor),alpha)+.25*gamma_inv(fmax(1-2*q,floor),alpha)));
   answer=corrected*wa*wb;control_p=(x==y?0.:1.-double(usable-mismatch)/length)*wa*wb;control_q=double(transversions)/length*wa*wb;control_c=(x==y?0.:double(usable)/length)*wa*wb;}
 }
 __shared__ double partial[4][16];if(!lane){partial[0][warp]=answer;partial[1][warp]=control_p;partial[2][warp]=control_q;if(overlap)partial[3][warp]=control_c;}__syncthreads();
 if(!t){for(int k=0;k<columns;++k){double total=0;for(int w=0;w<R*R;++w)total+=partial[k][w];out[int64_t(e)*columns+k]=total;}}
}
__global__ void representative_aggregate(const uint64_t*seq,int length,int words,const int*reps,const double*mass,int m,int R,int first,int sites,double*out){
 int64_t c=int64_t(blockIdx.x)*256+threadIdx.x;if(c>=int64_t(sites)*4)return;int site=first+c/4,base=c%4;double total=0;
 for(int i=blockIdx.y;i<m;i+=gridDim.y)for(int j=0;j<R;++j){double w=mass[i*R+j];if(w){int b=(seq[int64_t(reps[i*R+j])*words+site/16]>>((site%16)*4))&15;total+=w*(b==base);}}
 out[int64_t(blockIdx.y)*sites*4+c]=total;
}
__global__ void sum_aggregate(double*aggregate,int length,int tiles){int64_t c=int64_t(blockIdx.x)*256+threadIdx.x;if(c>=int64_t(length)*4)return;double sum=0;for(int j=0;j<tiles;++j)sum+=aggregate[int64_t(j)*length*4+c];aggregate[c]=sum;}
__global__ void representative_raw_rows(const uint64_t*seq,int length,int words,const int*reps,const double*mass,const double*aggregate,int m,int R,int first,int sites,bool accumulate,bool overlap,double*out){
 int i=blockIdx.x,t=threadIdx.x;double matched=0,tv=0,both=0;
 for(int64_t s=t;s<sites;s+=128){int64_t site=first+s;double f[4]={0,0,0,0};for(int j=0;j<R;++j){int b=(seq[int64_t(reps[i*R+j])*words+site/16]>>((site%16)*4))&15;if(b<4)f[b]+=mass[i*R+j];}
  for(int b=0;b<4;++b)matched+=f[b]*(aggregate[s*4+b]-f[b]);
  double pur=f[0]+f[2],pyr=f[1]+f[3],ar=aggregate[s*4]+aggregate[s*4+2],ay=aggregate[s*4+1]+aggregate[s*4+3];tv+=pur*(ay-pyr)+pyr*(ar-pur);if(overlap)both+=(pur+pyr)*(ar+ay-pur-pyr);
 }
 __shared__ double partial[3][4];matched=warp_sum(matched);tv=warp_sum(tv);if(overlap)both=warp_sum(both);if(!(t&31)){partial[0][t/32]=matched;partial[1][t/32]=tv;if(overlap)partial[2][t/32]=both;}__syncthreads();
 if(t<32){double a=t<4?partial[0][t]:0,b=t<4?partial[1][t]:0,c=overlap&&t<4?partial[2][t]:0;a=warp_sum(a);b=warp_sum(b);if(overlap)c=warp_sum(c);int stride=overlap?3:2;if(!t){if(accumulate){out[int64_t(i)*stride]+=a;out[int64_t(i)*stride+1]+=b;if(overlap)out[int64_t(i)*stride+2]+=c;}else{out[int64_t(i)*stride]=(m-1)-a/length;out[int64_t(i)*stride+1]=b/length;if(overlap)out[int64_t(i)*stride+2]=c/length;}}}
}
__global__ void normalize_raw_rows(double*out,int m,int length,bool overlap){int i=blockIdx.x*256+threadIdx.x;if(i<m){int stride=overlap?3:2;out[int64_t(i)*stride]=(m-1)-out[int64_t(i)*stride]/length;out[int64_t(i)*stride+1]/=length;if(overlap)out[int64_t(i)*stride+2]/=length;}}
class PackedState{
 int n,length,words;
 std::unique_ptr<Memory> seq,work_r,work_w,work_pairs,work_result,work_aggregate,work_rows,health;
 unsigned long long evidence_counts[4]={0,0,0,0};bool scanned=false;int gamma_triplets=-1;
 void reserve(std::unique_ptr<Memory>&p,size_t bytes){if(!p || p->size<bytes)p=std::make_unique<Memory>(bytes);}
 void upload(const py::array_t<int,py::array::c_style>& reps,const py::array_t<double,py::array::c_style>& mass){reserve(work_r,reps.size()*4);reserve(work_w,mass.size()*8);check(cudaMemcpy(work_r->p,reps.data(),reps.size()*4,cudaMemcpyHostToDevice),"upload persistent representative IDs");check(cudaMemcpy(work_w->p,mass.data(),mass.size()*8,cudaMemcpyHostToDevice),"upload persistent representative masses");}
 public:
 PackedState(py::array_t<uint8_t,py::array::c_style> packed,int sites,int workers){
  if(packed.ndim()!=2 || packed.shape(0)<4 || packed.shape(0)>INT_MAX/8 || sites<1 || sites>INT_MAX-16 || packed.shape(1)!=(sites+1)/2 || workers<1 || workers>64)
   throw std::invalid_argument("packed APNJ input layout");
  n=packed.shape(0);length=sites;words=(length+15)/16;size_t bytes=size_t(n)*words*8;
  size_t free,total;check(cudaMemGetInfo(&free,&total),"query APNJ memory");if(bytes>free-free/10)throw std::runtime_error("packed APNJ input exceeds device capacity");
  const uint8_t*input=packed.data();size_t stride=packed.shape(1);
  py::gil_scoped_release release;
  int entirely_missing=0;
  #pragma omp parallel for schedule(static) num_threads(workers) reduction(+:entirely_missing)
  for(int row=0;row<n;++row){bool observed=false;const uint8_t* data=input+size_t(row)*stride;
   for(size_t j=0;j<stride;++j)if((data[j]&15)<4 || (2*j+1<size_t(length) && (data[j]>>4)<4)){observed=true;break;}
   entirely_missing+=!observed;
  }
  if(entirely_missing)throw std::invalid_argument("AP-NJ requires at least one observed A/C/G/T base in every taxon");
  seq=std::make_unique<Memory>(bytes);
  health=std::make_unique<Memory>(sizeof(unsigned long long));check(cudaMemset(health->p,0,sizeof(unsigned long long)),"initialize APNJ diagnostics");
  if(stride==size_t(words)*8){
   check(cudaMemcpy(seq->p,input,bytes,cudaMemcpyHostToDevice),"direct upload APNJ packed sequences");
  }else{
   check(cudaMemset(seq->p,0,bytes),"initialize APNJ row padding");
   check(cudaMemcpy2D(seq->p,size_t(words)*8,input,stride,stride,size_t(n),cudaMemcpyHostToDevice),"pitched upload APNJ packed sequences");
  }
 }
 py::array_t<double> project(uint32_t seed,int P){
  if(P<1 || P>64)throw std::invalid_argument("packed APNJ projection width");
  py::array_t<double> out({n,P});Memory temp(size_t(n)*P*8);py::gil_scoped_release release;
  Memory masks(size_t(P)*4*words*8);
  prepare_projection_masks<<<dim3((words+127)/128,P,4),128>>>(length,words,P,seed,(uint64_t*)masks.p);
  project_bit_packed<<<dim3(n,P),128>>>((uint64_t*)seq->p,n,length,words,P,(uint64_t*)masks.p,(double*)temp.p);
  check(cudaMemcpy(out.mutable_data(),temp.p,size_t(n)*P*8,cudaMemcpyDeviceToHost),"download packed APNJ sketch");return out;
 }
 py::array_t<double> query(py::array_t<int,py::array::c_style> reps,py::array_t<double,py::array::c_style> mass,
   py::array_t<int64_t,py::array::c_style> pairs,int model,bool common,double alpha,bool moments,bool overlap){
  if(overlap && (!moments || !common))throw std::invalid_argument("overlap controls require moments and observed-intersection distances");
  if(reps.ndim()!=2 || mass.ndim()!=2 || mass.shape(0)!=reps.shape(0) || mass.shape(1)!=reps.shape(1) ||
   pairs.ndim()!=2 || pairs.shape(1)!=2 || (reps.shape(1)!=1 && reps.shape(1)!=2 && reps.shape(1)!=4) || model<0 || model>3 || !(alpha>0) || !std::isfinite(alpha))
   throw std::invalid_argument("APNJ representative query shape");
  if(reps.size()>INT_MAX || pairs.shape(0)>INT_MAX)throw std::invalid_argument("APNJ query exceeds native integer domain");
  int R=reps.shape(1),m=reps.shape(0);auto w=mass.unchecked<2>();
  for(py::ssize_t k=0;k<reps.size();++k)if(reps.data()[k]<0 || reps.data()[k]>=n || !std::isfinite(mass.data()[k]) || mass.data()[k]<0)throw std::invalid_argument("APNJ representative value");
  for(int i=0;i<m;++i){double total=0;for(int j=0;j<R;++j)total+=w(i,j);if(std::abs(total-1)>1e-12)throw std::invalid_argument("APNJ representative mass");}
  for(py::ssize_t k=0;k<pairs.size();++k)if(pairs.data()[k]<0 || pairs.data()[k]>=m)throw std::invalid_argument("APNJ representative pair");
  py::array_t<double> out=moments?py::array_t<double>({pairs.shape(0),py::ssize_t(overlap?4:3)}):py::array_t<double>(pairs.shape(0));if(!pairs.shape(0))return out;
  reserve(work_pairs,pairs.size()*8);reserve(work_result,out.size()*8);
  {py::gil_scoped_release release;
  upload(reps,mass);
  check(cudaMemcpy(work_pairs->p,pairs.data(),pairs.size()*8,cudaMemcpyHostToDevice),"upload representative pairs");
  representative_distances<<<pairs.shape(0),R*R*32>>>((uint64_t*)seq->p,length,words,(int*)work_r->p,(double*)work_w->p,(int64_t*)work_pairs->p,pairs.shape(0),R,model,common,alpha,moments,overlap,(double*)work_result->p,(unsigned long long*)health->p);
  check(cudaMemcpy(out.mutable_data(),work_result->p,out.size()*8,cudaMemcpyDeviceToHost),"download representative distances");}
  return out;
 }
 size_t memory(){return seq->size;}
 double gamma_shape(){return estimate_gamma((uint64_t*)seq->p,n,length,&gamma_triplets);}
 py::dict evidence(){
  if(!scanned){Memory scratch(4*sizeof(unsigned long long));{py::gil_scoped_release release;check(cudaMemset(scratch.p,0,4*sizeof(unsigned long long)),"initialize evidence scan");observation_evidence<<<(words+127)/128,128>>>((uint64_t*)seq->p,n,length,words,(unsigned long long*)scratch.p);check(cudaMemcpy(evidence_counts,scratch.p,sizeof(evidence_counts),cudaMemcpyDeviceToHost),"download evidence scan");}scanned=true;}
  py::dict out;out["variable_observed_sites"]=evidence_counts[0];out["observed_sites"]=evidence_counts[1];out["observed_bases"]=evidence_counts[2];out["jointly_observed_sites"]=evidence_counts[3];out["missing_base_fraction"]=1.-double(evidence_counts[2])/(double(n)*length);return out;
 }
 py::dict diagnostics(){py::dict out=evidence();unsigned long long zeros;check(cudaMemcpy(&zeros,health->p,sizeof(zeros),cudaMemcpyDeviceToHost),"download distance diagnostics");out["zero_overlap_pair_evaluations"]=zeros;out["gamma_usable_triplets"]=gamma_triplets;out["raw_control_scratch_limit_bytes"]=size_t(16)*131072*32;out["raw_control_scratch_bytes"]=work_aggregate?work_aggregate->size:0;return out;}
 py::array_t<double> raw_rows(py::array_t<int,py::array::c_style> reps,py::array_t<double,py::array::c_style> mass,bool overlap){
  if(reps.ndim()!=2 || mass.ndim()!=2 || reps.shape(0)!=mass.shape(0) || reps.shape(1)!=mass.shape(1) || reps.shape(0)<2 || reps.shape(1)>4 || reps.shape(1)<1)throw std::invalid_argument("raw row layout");
  for(py::ssize_t k=0;k<reps.size();++k)if(reps.data()[k]<0 || reps.data()[k]>=n || mass.data()[k]<0 || !std::isfinite(mass.data()[k]))throw std::invalid_argument("raw row representative value");
  if(reps.size()>INT_MAX)throw std::invalid_argument("APNJ raw rows exceed native integer domain");
  for(py::ssize_t i=0;i<reps.shape(0);++i){double sum=0;for(py::ssize_t j=0;j<reps.shape(1);++j)sum+=mass.data()[i*reps.shape(1)+j];
   if(std::abs(sum-1)>1e-12)throw std::invalid_argument("APNJ raw-row representative mass");}
  int m=reps.shape(0),R=reps.shape(1),tiles=std::min(16,(m+63)/64),chunk=std::min(length,131072),columns=overlap?3:2;py::array_t<double> out({m,columns});reserve(work_aggregate,size_t(tiles)*chunk*32);reserve(work_rows,size_t(m)*columns*8);
  {py::gil_scoped_release release;upload(reps,mass);
  if(length>chunk)check(cudaMemset(work_rows->p,0,size_t(m)*columns*8),"initialize streamed raw rows");
  for(int first=0;first<length;){int sites=std::min(chunk,length-first);
   representative_aggregate<<<dim3((int64_t(sites)*4+255)/256,tiles),256>>>((uint64_t*)seq->p,length,words,(int*)work_r->p,(double*)work_w->p,m,R,first,sites,(double*)work_aggregate->p);
   if(tiles>1)sum_aggregate<<<(int64_t(sites)*4+255)/256,256>>>((double*)work_aggregate->p,sites,tiles);
   representative_raw_rows<<<m,128>>>((uint64_t*)seq->p,length,words,(int*)work_r->p,(double*)work_w->p,(double*)work_aggregate->p,m,R,first,sites,length>chunk,overlap,(double*)work_rows->p);first+=sites;
  }
  if(length>chunk)normalize_raw_rows<<<(m+255)/256,256>>>((double*)work_rows->p,m,length,overlap);
  check(cudaMemcpy(out.mutable_data(),work_rows->p,size_t(m)*columns*8,cudaMemcpyDeviceToHost),"download exact raw row controls");}return out;
 }
};
PYBIND11_MODULE(_ct_apnj_packed,m){py::class_<PackedState>(m,"State",py::module_local()).def(py::init<py::array_t<uint8_t,py::array::c_style>,int,int>()).def("project",&PackedState::project).def("query",&PackedState::query,py::arg("reps"),py::arg("mass"),py::arg("pairs"),py::arg("model"),py::arg("common"),py::arg("alpha")=100.,py::arg("moments")=false,py::arg("overlap")=false).def("memory",&PackedState::memory).def("gamma_shape",&PackedState::gamma_shape).def("raw_rows",&PackedState::raw_rows,py::arg("reps"),py::arg("mass"),py::arg("overlap")=false).def("evidence",&PackedState::evidence).def("diagnostics",&PackedState::diagnostics);}
