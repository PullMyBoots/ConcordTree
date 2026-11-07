// Reused unchanged from the prior project initializer distance experiment.
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <array>
constexpr uint64_t M=0x1111111111111111ULL;
__device__ uint64_t valid_nibbles(uint64_t a,uint64_t b){return M&~((a>>2)|(a>>3)|(b>>2)|(b>>3));}
__device__ uint64_t different(uint64_t a,uint64_t b){uint64_t z=a^b;return (z|(z>>1))&M;}
__global__ void triplet_counts(const uint64_t*data,int n,int length,int*counts) {
 const int g=blockIdx.x,t=threadIdx.x,words=(length+15)/16;
 const int ia=g%n,ib=(g+1+g/8)%n,ic=(g+3+g/4)%n;
 int v=0,ab=0,ac=0,bc=0,all=0;
 for(int w=t;w<words;w+=128){uint64_t a=data[int64_t(ia)*words+w],b=data[int64_t(ib)*words+w],c=data[int64_t(ic)*words+w];
  uint64_t ok=valid_nibbles(a,b)&valid_nibbles(a,c);
  if(w==words-1 && (length&15))ok&=(1ULL<<((length&15)*4))-1;
  uint64_t qab=ok&~different(a,b),qac=ok&~different(a,c),qbc=ok&~different(b,c);
  v+=__popcll(ok);ab+=__popcll(qab);ac+=__popcll(qac);bc+=__popcll(qbc);all+=__popcll(qab&qac);
 }
 __shared__ int s[5][128];s[0][t]=v;s[1][t]=ab;s[2][t]=ac;s[3][t]=bc;s[4][t]=all;__syncthreads();
 for(int st=64;st;st>>=1){if(t<st)for(int k=0;k<5;++k)s[k][t]+=s[k][t+st];__syncthreads();}
 if(!t)for(int k=0;k<5;++k)counts[g*5+k]=s[k][0];
}
double estimate_gamma(const uint64_t*data,int n,int length,int*usable_triplets=nullptr) {
 int *device;auto e=cudaMalloc(&device,32*5*sizeof(int));if(e!=cudaSuccess)throw std::runtime_error("gamma scratch allocation");
 triplet_counts<<<32,128>>>(data,n,length,device);int c[32][5];e=cudaMemcpy(c,device,sizeof(c),cudaMemcpyDeviceToHost);cudaFree(device);
 if(e!=cudaSuccess)throw std::runtime_error("gamma count download");
 if(usable_triplets){*usable_triplets=0;for(int g=0;g<32;++g){if(c[g][0]<32)continue;bool usable=true;for(int j=0;j<3;++j){double f=(4*double(c[g][j+1])/c[g][0]-1)/3;if(f<=.001 || f>=.999999)usable=false;}*usable_triplets+=usable;}}
 auto loss=[&](double logalpha){double a=exp(logalpha),ll=0;
  for(int g=0;g<32;++g){if(c[g][0]<32)continue;double v=c[g][0];double e[3],f[3];bool usable=true;
   for(int j=0;j<3;++j){e[j]=double(c[g][j+1])/v;f[j]=(4*e[j]-1)/3;if(f[j]<=.001 || f[j]>=.999999)usable=false;}
   if(!usable)continue;
   double sum=0;for(double ff:f)sum+=expm1(-log(ff)/a);
   double fsum=exp(-a*log1p(.5*sum));double q=(1+3*(f[0]+f[1]+f[2])+6*fsum)/16;
   double prob[5]={q,e[0]-q,e[1]-q,e[2]-q,1-e[0]-e[1]-e[2]+2*q};
   int64_t observed[5]={c[g][4],c[g][1]-c[g][4],c[g][2]-c[g][4],c[g][3]-c[g][4],int64_t(c[g][0])-c[g][1]-c[g][2]-c[g][3]+2LL*c[g][4]};
   for(int j=0;j<5;++j)if(observed[j])ll-=observed[j]*log(std::max(prob[j],1e-15));
  }return ll;
 };
 double left=log(.01),right=log(100.),ratio=(sqrt(5.)-1)/2,x=right-ratio*(right-left),y=left+ratio*(right-left),fx=loss(x),fy=loss(y);
 for(int step=0;step<48;++step){if(fx<fy){right=y;y=x;fy=fx;x=right-ratio*(right-left);fx=loss(x);}else{left=x;x=y;fx=fy;y=left+ratio*(right-left);fy=loss(y);}}
 return exp((left+right)/2);
}
__device__ double gamma_inv(double f,double alpha){return alpha*expm1(fmin(30.,-log(f)/alpha));}
