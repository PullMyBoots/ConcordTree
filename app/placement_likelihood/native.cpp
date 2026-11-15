#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
// Small bounded-context, double-precision, scaled Felsenstein pruning.
// Node IDs are preorder positions; children always have larger IDs.
extern "C" double prune(int M,int S,int K,const int* child,const int* childedge,
 const unsigned char* obs,const double* count,const double* P,const double* pi,
 const double* weight,double* dP,double* dpi,double* dw){
 const int E=M-1;
 size_t slab=(size_t)K*S*4;
 std::vector<double> val((size_t)M*slab),fac((size_t)E*slab),norm((size_t)M*K*S),
  scale((size_t)M*K*S),adj((size_t)M*slab,0.);
 std::fill(dP,dP+(size_t)E*K*16,0.);std::fill(dpi,dpi+4,0.);std::fill(dw,dw+K,0.);
 for(int u=M-1;u>=0;--u){
  for(int k=0;k<K;++k)for(int s=0;s<S;++s){
   size_t z=((size_t)k*S+s)*4, ns=((size_t)u*K+k)*S+s;
   double* v=&val[(size_t)u*slab+z];
   unsigned code=obs[(size_t)u*S+s];
   for(int i=0;i<4;++i)v[i]=(code==4||code==(unsigned)i)?1.:0.;
   double ls=0.;
   for(int c=0;c<3;++c){
    int vtx=child[u*3+c];if(vtx<0)continue;
    int e=childedge[u*3+c];const double* p=P+((size_t)e*K+k)*16;
    const double* x=&val[(size_t)vtx*slab+z];double* f=&fac[(size_t)e*slab+z];
    for(int i=0;i<4;++i){f[i]=p[i*4]*x[0]+p[i*4+1]*x[1]+p[i*4+2]*x[2]+p[i*4+3]*x[3];v[i]*=f[i];}
    ls+=scale[((size_t)vtx*K+k)*S+s];
   }
   double n=std::max(std::max(v[0],v[1]),std::max(v[2],v[3]));n=std::max(n,1e-300);
   for(int i=0;i<4;++i)v[i]/=n;
   norm[ns]=n;scale[ns]=ls+std::log(n);
  }
 }
 double ll=0.;std::vector<double> cat(K),den(K);
 for(int s=0;s<S;++s){
  double mx=-INFINITY;
  for(int k=0;k<K;++k){const double* v=&val[((size_t)k*S+s)*4];
   den[k]=std::max(1e-300,pi[0]*v[0]+pi[1]*v[1]+pi[2]*v[2]+pi[3]*v[3]);
   cat[k]=std::log(den[k])+scale[(size_t)k*S+s]+std::log(std::max(weight[k],1e-300));mx=std::max(mx,cat[k]);}
  double mix=0.;for(int k=0;k<K;++k)mix+=std::exp(cat[k]-mx);
  ll+=count[s]*(mx+std::log(mix));
  for(int k=0;k<K;++k){double post=count[s]*std::exp(cat[k]-mx)/mix;size_t z=((size_t)k*S+s)*4;
   for(int i=0;i<4;++i){adj[z+i]=post*pi[i]/den[k];dpi[i]+=post*val[z+i]/den[k];}
   dw[k]+=post/std::max(weight[k],1e-300);
  }
 }
 for(int u=0;u<M;++u)for(int k=0;k<K;++k)for(int s=0;s<S;++s){
  size_t z=((size_t)k*S+s)*4;const double* a=&adj[(size_t)u*slab+z];
  for(int c=0;c<3;++c){int vtx=child[u*3+c];if(vtx<0)continue;int e=childedge[u*3+c];
   const double* x=&val[(size_t)vtx*slab+z];double* ax=&adj[(size_t)vtx*slab+z];
   const double* p=P+((size_t)e*K+k)*16;double* dp=dP+((size_t)e*K+k)*16;
   double b[4];
   for(int i=0;i<4;++i){b[i]=a[i]/norm[((size_t)u*K+k)*S+s];
    unsigned code=obs[(size_t)u*S+s];if(code!=4&&code!=(unsigned)i)b[i]=0.;
    for(int d=0;d<3;++d)if(d!=c&&child[u*3+d]>=0)b[i]*=fac[(size_t)childedge[u*3+d]*slab+z+i];}
   for(int j=0;j<4;++j){double t=0.;for(int i=0;i<4;++i){dp[i*4+j]+=b[i]*x[j];t+=p[i*4+j]*b[i];}ax[j]+=t;}
  }
 }
 return ll;
}

extern "C" void sankoff(int M,int S,const int* child,const unsigned char* obs,
 const double* count,double* changes,double* pairs){
 std::vector<int> cost((size_t)M*4),assign(M);
 std::fill(changes,changes+M,0.);std::fill(pairs,pairs+6,0.);
 for(int s=0;s<S;++s){
  for(int u=M-1;u>=0;--u){int* v=&cost[u*4];unsigned code=obs[(size_t)u*S+s];
   for(int i=0;i<4;++i)v[i]=(code==4||code==(unsigned)i)?0:M+1;
   for(int c=0;c<3;++c){int ch=child[u*3+c];if(ch<0)continue;int* x=&cost[ch*4];
    int best=*std::min_element(x,x+4)+1;for(int i=0;i<4;++i)v[i]+=std::min(x[i],best);}}
  assign[0]=std::min_element(cost.begin(),cost.begin()+4)-cost.begin();
  for(int u=0;u<M;++u)for(int c=0;c<3;++c){int ch=child[u*3+c];if(ch<0)continue;
   int best=1000000,state=0;for(int j=0;j<4;++j){int v=cost[ch*4+j]+(j!=assign[u]);if(v<best){best=v;state=j;}}
   assign[ch]=state;if(state!=assign[u]){changes[ch]+=count[s];int a=std::min(state,assign[u]),b=std::max(state,assign[u]);
    int slot=0;for(int i=0;i<4;++i)for(int j=i+1;j<4;++j){if(i==a&&j==b)pairs[slot]+=count[s];++slot;}}
  }
 }
}

// Cavity likelihood: three incident branches are the only query parameters.
extern "C" double placement(int S,int K,const double* left,const double* right,
 const double* scales,const unsigned char* query,const double* count,
 const double* P,const double* dP,const double* pi,const double* weights,double* grad){
 std::fill(grad,grad+3,0.);double ll=0.;std::vector<double> cat(K),rat((size_t)K*3);
 for(int s=0;s<S;++s){double mx=-INFINITY;
  for(int k=0;k<K;++k){size_t z=((size_t)k*S+s)*4;
   const double* a=left+z;const double* b=right+z;unsigned code=query[s];
   double den=0.,der[3]={0.,0.,0.};
   for(int i=0;i<4;++i){double f[3]={0.,0.,0.},df[3]={0.,0.,0.};
    for(int j=0;j<4;++j){double obs=(code==4||code==(unsigned)j)?1.:0.;
     f[0]+=P[((size_t)0*K+k)*16+i*4+j]*a[j];df[0]+=dP[((size_t)0*K+k)*16+i*4+j]*a[j];
     f[1]+=P[((size_t)1*K+k)*16+i*4+j]*b[j];df[1]+=dP[((size_t)1*K+k)*16+i*4+j]*b[j];
     f[2]+=P[((size_t)2*K+k)*16+i*4+j]*obs;df[2]+=dP[((size_t)2*K+k)*16+i*4+j]*obs;}
    den+=pi[i]*f[0]*f[1]*f[2];
    der[0]+=pi[i]*df[0]*f[1]*f[2];der[1]+=pi[i]*f[0]*df[1]*f[2];der[2]+=pi[i]*f[0]*f[1]*df[2];
   }
   den=std::max(den,1e-300);cat[k]=std::log(den)+scales[(size_t)k*S+s]+std::log(std::max(weights[k],1e-300));
   mx=std::max(mx,cat[k]);for(int j=0;j<3;++j)rat[k*3+j]=der[j]/den;
  }
  double mix=0.;for(int k=0;k<K;++k)mix+=std::exp(cat[k]-mx);
  ll+=count[s]*(mx+std::log(mix));
  for(int k=0;k<K;++k){double w=count[s]*std::exp(cat[k]-mx)/mix;for(int j=0;j<3;++j)grad[j]+=w*rat[k*3+j];}
 }
 return ll;
}
