#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
struct State {double ll,g[3],h[9];};
struct Scorer {
 int S,K,calls=0;const double *left,*right,*scales,*count,*pi,*weights,*lam,*U,*rates;const unsigned char* query;
 State score(const double* x){
  ++calls;double P[3][16][16],D[3][16][16],DD[3][16][16];
  for(int b=0;b<3;++b)for(int k=0;k<K;++k){
   double t=x[b]*x[b]/4;
   for(int i=0;i<4;++i)for(int j=0;j<4;++j){double p=0,d=0,dd=0;
    for(int z=0;z<4;++z){double rate=lam[z]*rates[k],v=U[i*4+z]*U[j*4+z]*std::sqrt(pi[j]/pi[i])*std::exp(t*rate);
     p+=v;d+=v*rate;dd+=v*rate*rate;}
    if(rates[k]==0){p=i==j?1.:0.;d=dd=0.;}
    P[b][k][i*4+j]=std::max(p,0.);D[b][k][i*4+j]=d*x[b]/2;
    DD[b][k][i*4+j]=dd*t+d/2;
   }
  }
  State out={0.,{0.,0.,0.},{0.,0.,0.,0.,0.,0.,0.,0.,0.}};
  for(int s=0;s<S;++s){double cat[16],cg[16][3],ch[16][9],mx=-INFINITY;
   for(int k=0;k<K;++k){double den=0.,dg[3]={},dh[9]={};
    size_t z=((size_t)k*S+s)*4;const double* a=left+z;const double* b=right+z;unsigned code=query[s];
    for(int i=0;i<4;++i){double f[3]={},df[3]={},ddf[3]={};
     for(int j=0;j<4;++j){double obs=(code==4||code==(unsigned)j)?1.:0.;double vals[3]={a[j],b[j],obs};
      for(int r=0;r<3;++r){f[r]+=P[r][k][i*4+j]*vals[r];df[r]+=D[r][k][i*4+j]*vals[r];ddf[r]+=DD[r][k][i*4+j]*vals[r];}}
     den+=pi[i]*f[0]*f[1]*f[2];
     for(int r=0;r<3;++r){int v=(r+1)%3,w=(r+2)%3;dg[r]+=pi[i]*df[r]*f[v]*f[w];dh[r*3+r]+=pi[i]*ddf[r]*f[v]*f[w];
      for(int v2=r+1;v2<3;++v2){int other=3-r-v2;dh[r*3+v2]+=pi[i]*df[r]*df[v2]*f[other];}}
    }
    den=std::max(den,1e-300);cat[k]=std::log(den)+scales[(size_t)k*S+s]+std::log(std::max(weights[k],1e-300));mx=std::max(mx,cat[k]);
    for(int r=0;r<3;++r){cg[k][r]=dg[r]/den;for(int v=r;v<3;++v)ch[k][r*3+v]=ch[k][v*3+r]=dh[r*3+v]/den;}
   }
   double mix=0.;for(int k=0;k<K;++k)mix+=std::exp(cat[k]-mx);
   out.ll+=count[s]*(mx+std::log(mix));double g[3]={},h[9]={};
   for(int k=0;k<K;++k){double post=std::exp(cat[k]-mx)/mix;
    for(int r=0;r<3;++r){g[r]+=post*cg[k][r];for(int v=0;v<3;++v)h[r*3+v]+=post*ch[k][r*3+v];}}
   for(int r=0;r<3;++r){out.g[r]+=count[s]*g[r];for(int v=0;v<3;++v)out.h[r*3+v]+=count[s]*(h[r*3+v]-g[r]*g[v]);}
  }
  return out;
 }
};

bool cholesky_solve(const double* H,const double* g,const bool* active,double damp,double* step){
 double L[9]={},y[3]={};
 for(int i=0;i<3;++i)for(int j=0;j<=i;++j){double v=(active[i]||active[j])?(i==j?1.:0.):-H[i*3+j]+(i==j?damp:0.);
  for(int k=0;k<j;++k)v-=L[i*3+k]*L[j*3+k];
  if(i==j){if(!std::isfinite(v)||v<=1e-14)return false;L[i*3+j]=std::sqrt(v);}else L[i*3+j]=v/L[j*3+j];}
 for(int i=0;i<3;++i){double v=active[i]?0.:g[i];for(int j=0;j<i;++j)v-=L[i*3+j]*y[j];y[i]=v/L[i*3+i];}
 for(int i=2;i>=0;--i){double v=y[i];for(int j=i+1;j<3;++j)v-=L[j*3+i]*step[j];step[i]=v/L[i*3+i];}
 return true;
}

extern "C" double insertion_optimize(int S,int K,const double* left,const double* right,const double* scales,
 const unsigned char* query,const double* count,const double* pi,const double* weights,const double* lam,
 const double* U,const double* rates,const double* initial,int iterations,int maxcalls,double* lengths,double* info){
 Scorer scorer={S,K,0,left,right,scales,count,pi,weights,lam,U,rates,query};
 const double lower=.002,upper=2*std::sqrt(10.);double x[3];for(int i=0;i<3;++i)x[i]=std::clamp(2*std::sqrt(initial[i]),lower,upper);
 double sites=0.;for(int s=0;s<S;++s)sites+=count[s];
 State current=scorer.score(x);int status=1,iter=0;double pg=INFINITY;
 for(;iter<iterations&&scorer.calls<maxcalls;++iter){
  bool active[3];pg=0.;
  for(int i=0;i<3;++i){active[i]=(x[i]<=lower+1e-12&&current.g[i]<0)||(x[i]>=upper-1e-12&&current.g[i]>0);
   if(!active[i])pg=std::max(pg,std::abs(current.g[i])/sites);}
  if(pg<1e-5){status=0;break;}
  double g[3],h[9],step[3]={};for(int i=0;i<3;++i){g[i]=current.g[i]/sites;for(int j=0;j<3;++j)h[i*3+j]=current.h[i*3+j]/sites;}
  double damp=0.;bool solved=false;
  for(int tries=0;tries<12;++tries){if(cholesky_solve(h,g,active,damp,step)){solved=true;break;}damp=damp==0?1e-4:damp*10;}
  if(!solved){for(int i=0;i<3;++i)step[i]=active[i]?0.:g[i];}
  // Bounded trust step prevents giant jumps near weakly identified branches.
  double maxstep=std::max({std::abs(step[0]),std::abs(step[1]),std::abs(step[2])});
  if(maxstep>1.)for(int i=0;i<3;++i)step[i]/=maxstep;
  bool accepted=false;double alpha=1.,trial[3];State next;
  for(int tries=0;tries<14&&scorer.calls<maxcalls;++tries){double dot=0.;
   for(int i=0;i<3;++i){trial[i]=std::clamp(x[i]+alpha*step[i],lower,upper);dot+=current.g[i]*(trial[i]-x[i]);}
   if(dot>0.){next=scorer.score(trial);if(std::isfinite(next.ll)&&next.ll>=current.ll+1e-4*dot){accepted=true;break;}}
   alpha*=.5;
  }
  if(!accepted){status=2;break;}
  double gain=next.ll-current.ll;current=next;for(int i=0;i<3;++i)x[i]=trial[i];
  if(gain/sites<1e-10){
   double remaining=0.;for(int i=0;i<3;++i)if(!((x[i]<=lower+1e-12&&current.g[i]<0)||(x[i]>=upper-1e-12&&current.g[i]>0)))remaining=std::max(remaining,std::abs(current.g[i])/sites);
   if(remaining<1e-5){status=0;pg=remaining;break;}
  }
 }
 for(int i=0;i<3;++i)lengths[i]=x[i]*x[i]/4;
 info[0]=status;info[1]=iter;info[2]=scorer.calls;info[3]=pg;
 return current.ll;
}

extern "C" double insertion_derivatives(int S,int K,const double* left,const double* right,const double* scales,
 const unsigned char* query,const double* count,const double* pi,const double* weights,const double* lam,
 const double* U,const double* rates,const double* x,double* grad,double* hessian){
 Scorer scorer={S,K,0,left,right,scales,count,pi,weights,lam,U,rates,query};State s=scorer.score(x);
 for(int i=0;i<3;++i)grad[i]=s.g[i];for(int i=0;i<9;++i)hessian[i]=s.h[i];return s.ll;
}
