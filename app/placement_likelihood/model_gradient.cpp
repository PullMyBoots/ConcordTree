#include <cmath>
#include <algorithm>
// Exact spectral Frechet pullback of the transition matrices. No pruning or
// likelihood objective changes; repeated eigenvalue limits are explicit.
extern "C" void model_gradient(int E,int K,const double* P,const double* dP,
 const double* t,const double* pi,const double* q,const double* lam,const double* U,
 const double* rates,const double* exchange,const double* dpi,const double* dw,
 const double* alpha_derivative,double pinv,double* out){
 double A[16],B[16],C[16]={},G[16]={},gr[16]={};
 for(int a=0;a<4;++a)for(int i=0;i<4;++i){A[a*4+i]=U[a*4+i]/std::sqrt(pi[a]);B[i*4+a]=U[a*4+i]*std::sqrt(pi[a]);}
 for(int e=0;e<E;++e)for(int k=0;k<K;++k){
  const double* dp=dP+((size_t)e*K+k)*16;const double* p=P+((size_t)e*K+k)*16;
  double z=t[e]*rates[k],ex[4];for(int i=0;i<4;++i)ex[i]=std::exp(z*lam[i]);
  for(int i=0;i<4;++i)for(int j=0;j<4;++j){
   double h=0.;for(int a=0;a<4;++a)for(int b=0;b<4;++b)h+=A[a*4+i]*dp[a*4+b]*B[j*4+b];
   double d=lam[i]-lam[j];double f=std::abs(d)>1e-9?(ex[i]-ex[j])/d:z*(ex[i]+ex[j])/2;
   C[i*4+j]+=h*f;
  }
  double value=0.;for(int i=0;i<4;++i)for(int j=0;j<4;++j){double qp=0.;for(int a=0;a<4;++a)qp+=q[i*4+a]*p[a*4+j];value+=dp[i*4+j]*qp;}
  gr[k]+=t[e]*value;
 }
 for(int a=0;a<4;++a)for(int b=0;b<4;++b)for(int i=0;i<4;++i)for(int j=0;j<4;++j)G[a*4+b]+=B[i*4+a]*C[i*4+j]*A[b*4+j];
 int pa[6]={0,0,0,1,1,2},pb[6]={1,2,3,2,3,3};double S[16]={},q0[16]={},mu=0.;
 for(int j=0;j<6;++j)S[pa[j]*4+pb[j]]=S[pb[j]*4+pa[j]]=exchange[j];
 for(int a=0;a<4;++a){for(int b=0;b<4;++b)if(a!=b){q0[a*4+b]=S[a*4+b]*pi[b];q0[a*4+a]-=q0[a*4+b];}mu-=pi[a]*q0[a*4+a];}
 for(int j=0;j<8;++j){double dp0[4]={},dq[16]={};
  if(j<5){int a=pa[j],b=pb[j];dq[a*4+b]=exchange[j]*pi[b];dq[b*4+a]=exchange[j]*pi[a];}
  else{int k=j-5;double rawk=pi[k]*(1+4e-8)-1e-8;
   for(int i=0;i<4;++i){double raw=pi[i]*(1+4e-8)-1e-8;dp0[i]=raw*((i==k?1.:0.)-rawk)/(1+4e-8);}
   for(int a=0;a<4;++a)for(int b=0;b<4;++b)if(a!=b)dq[a*4+b]=S[a*4+b]*dp0[b];}
  for(int a=0;a<4;++a)for(int b=0;b<4;++b)if(a!=b)dq[a*4+a]-=dq[a*4+b];
  double dmu=0.;for(int a=0;a<4;++a)dmu-=dp0[a]*q0[a*4+a]+pi[a]*dq[a*4+a];
  double val=0.;for(int a=0;a<4;++a){val+=dpi[a]*dp0[a];for(int b=0;b<4;++b)val+=G[a*4+b]*(dq[a*4+b]/mu-q0[a*4+b]*dmu/(mu*mu));}out[j]=val;
 }
 out[8]=0.;out[9]=0.;for(int k=0;k<K;++k){out[8]+=gr[k]*alpha_derivative[k];out[9]+=gr[k]*rates[k]*pinv;}
 out[9]+=dw[0]*pinv*(1-pinv);for(int k=1;k<K;++k)out[9]-=dw[k]*pinv*(1-pinv)/(K-1);
}
