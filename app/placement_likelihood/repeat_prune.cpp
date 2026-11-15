#include <vector>
#include <cmath>
#include <algorithm>
// Exact subtree repeat DAG; adjoints are accumulated across shared rows.
extern "C" double repeat_prune(int M,int R,int K,const int* off,const int* ce,const int* rc,
 const unsigned char* obs,const double* count,const double* P,const double* pi,const double* weight,
 double* dP,double* dpi,double* dw){
 size_t slab=(size_t)R*K*4;std::vector<double> val(slab),fac(slab*3),norm((size_t)R*K),scale((size_t)R*K),adj(slab,0.);
 std::fill(dP,dP+(size_t)(M-1)*K*16,0.);std::fill(dpi,dpi+4,0.);std::fill(dw,dw+K,0.);
 for(int u=M-1;u>=0;--u)for(int r=off[u];r<off[u+1];++r)for(int k=0;k<K;++k){
  size_t z=((size_t)r*K+k)*4;double* v=val.data()+z;double ls=0.;unsigned code=obs[r];
  for(int i=0;i<4;++i)v[i]=(code==4||code==(unsigned)i)?1.:0.;
  for(int c=0;c<3;++c){int row=rc[r*3+c];if(row<0)continue;int e=ce[u*3+c];
   const double* p=P+((size_t)e*K+k)*16;const double* x=val.data()+((size_t)row*K+k)*4;
   double* f=fac.data()+(((size_t)r*3+c)*K+k)*4;
   for(int i=0;i<4;++i){f[i]=p[i*4]*x[0]+p[i*4+1]*x[1]+p[i*4+2]*x[2]+p[i*4+3]*x[3];v[i]*=f[i];}
   ls+=scale[(size_t)row*K+k];
  }
  double n=std::max({v[0],v[1],v[2],v[3],1e-300});norm[(size_t)r*K+k]=n;
  scale[(size_t)r*K+k]=ls+std::log(n);for(int i=0;i<4;++i)v[i]/=n;
 }
 double ll=0.;std::vector<double> cat(K),den(K);
 for(int r=off[0];r<off[1];++r){double mx=-INFINITY;
  for(int k=0;k<K;++k){const double* v=val.data()+((size_t)r*K+k)*4;den[k]=std::max(1e-300,pi[0]*v[0]+pi[1]*v[1]+pi[2]*v[2]+pi[3]*v[3]);
   cat[k]=std::log(den[k])+scale[(size_t)r*K+k]+std::log(std::max(weight[k],1e-300));mx=std::max(mx,cat[k]);}
  double mix=0.;for(int k=0;k<K;++k)mix+=std::exp(cat[k]-mx);
  ll+=count[r]*(mx+std::log(mix));
  for(int k=0;k<K;++k){double post=count[r]*std::exp(cat[k]-mx)/mix;size_t z=((size_t)r*K+k)*4;
   for(int i=0;i<4;++i){adj[z+i]=post*pi[i]/den[k];dpi[i]+=post*val[z+i]/den[k];}dw[k]+=post/std::max(weight[k],1e-300);}
 }
 for(int u=0;u<M;++u)for(int r=off[u];r<off[u+1];++r)for(int k=0;k<K;++k){
  const double* a=adj.data()+((size_t)r*K+k)*4;
  for(int c=0;c<3;++c){int row=rc[r*3+c];if(row<0)continue;int e=ce[u*3+c];
   const double* x=val.data()+((size_t)row*K+k)*4;double* ax=adj.data()+((size_t)row*K+k)*4;
   const double* p=P+((size_t)e*K+k)*16;double* dp=dP+((size_t)e*K+k)*16;double b[4];
   for(int i=0;i<4;++i){b[i]=a[i]/norm[(size_t)r*K+k];
    if(obs[r]!=4&&obs[r]!=(unsigned)i)b[i]=0.;
    for(int d=0;d<3;++d)if(d!=c&&rc[r*3+d]>=0)b[i]*=fac[(((size_t)r*3+d)*K+k)*4+i];}
   for(int j=0;j<4;++j){double t=0.;for(int i=0;i<4;++i){dp[i*4+j]+=b[i]*x[j];t+=p[i*4+j]*b[i];}ax[j]+=t;}
  }
 }
 return ll;
}
