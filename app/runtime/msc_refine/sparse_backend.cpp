#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
namespace py=pybind11;
using AI=py::array_t<int64_t,py::array::c_style|py::array::forcecast>;
using AD=py::array_t<double,py::array::c_style|py::array::forcecast>;

struct Index {
 int n,V,K;std::vector<std::vector<int>> g,up;std::vector<int> depth,parent,tin,tout;
 Index(AI edges,int N):n(N),V(2*N-2),K(1),g(V),depth(V),parent(V,-1),tin(V),tout(V){
  auto e=edges.unchecked<2>();
  if(e.shape(1)!=2 || e.shape(0)!=V-1)throw std::runtime_error("invalid binary tree edges");
  for(int i=0;i<e.shape(0);i++){int a=e(i,0),b=e(i,1);if(a<0||b<0||a>=V||b>=V)throw std::runtime_error("invalid node id");g[a].push_back(b);g[b].push_back(a);}
  for(auto &a:g)std::sort(a.begin(),a.end());
  while((1<<K)<=V)K++;
  up.assign(K,std::vector<int>(V,n));
  std::vector<std::pair<int,bool>> stack{{n,false}};int tick=0;parent[n]=n;
  while(!stack.empty()){
   auto [u,done]=stack.back();stack.pop_back();
   if(done){tout[u]=tick;continue;}tin[u]=tick++;
   up[0][u]=parent[u];for(int k=1;k<K;k++)up[k][u]=up[k-1][up[k-1][u]];
   stack.push_back({u,true});
   for(auto it=g[u].rbegin();it!=g[u].rend();it++)if(*it!=parent[u]){if(parent[*it]!=-1)throw std::runtime_error("cycle");parent[*it]=u;depth[*it]=depth[u]+1;stack.push_back({*it,false});}
  }
  if(tick!=V)throw std::runtime_error("disconnected");
 }
 bool ancestor(int a,int b)const{return tin[a]<=tin[b] && tin[b]<tout[a];}
 int lca(int a,int b)const{
  if(ancestor(a,b))return a;if(ancestor(b,a))return b;
  for(int k=K-1;k>=0;k--)if(!ancestor(up[k][a],b))a=up[k][a];return parent[a];
 }
 int dist(int a,int b)const{return depth[a]+depth[b]-2*depth[lca(a,b)];}
 int median(int a,int b,int c)const{
  int u=lca(a,b),v=lca(a,c),w=lca(b,c);
  return depth[u]>=depth[v] ? (depth[u]>=depth[w]?u:w):(depth[v]>=depth[w]?v:w);
 }
 int step(int center,int leaf)const{
  if(!ancestor(center,leaf))return parent[center];
  int d=depth[leaf]-depth[center]-1;
  for(int k=0;k<K;k++)if(d&(1<<k))leaf=up[k][leaf];return leaf;
 }
};

int pairclass(int a,int b){if(a>b)std::swap(a,b);if((a==0&&b==1)||(a==2&&b==3))return 0;if((a==0&&b==2)||(a==1&&b==3))return 1;return 2;}

py::dict evaluate(AI tree,int n,AI queries,AD weights){
 Index idx(tree,n);auto q=queries.unchecked<2>();auto w=weights.unchecked<2>();
 if(q.shape(1)!=4||w.shape(0)!=q.shape(0)||w.shape(1)!=3)throw std::runtime_error("invalid quartet/weight dimensions");
 std::vector<std::array<int,2>> edges;std::vector<std::array<int,4>> branches;
 std::vector<int> edgeid(idx.V,-1);
 // Each internal edge is uniquely identified by its deeper endpoint.
 for(int u=n;u<idx.V;u++)for(int v:idx.g[u])if(v>u){
  int child=idx.parent[u]==v?u:v;edgeid[child]=edges.size();edges.push_back({u,v});
  std::array<int,4>b;int k=0;for(int z:idx.g[u])if(z!=v)b[k++]=z;for(int z:idx.g[v])if(z!=u)b[k++]=z;branches.push_back(b);
 }
 py::array_t<double> score({(int)edges.size(),3});auto out=score.mutable_unchecked<2>();
 for(int e=0;e<edges.size();e++)for(int c=0;c<3;c++)out(e,c)=0.;
 py::array_t<int64_t> klass(q.shape(0));auto cls=klass.mutable_unchecked<1>();
 double objective=0.;int active=0;
 {
 py::gil_scoped_release release;
 for(int i=0;i<q.shape(0);i++){
  int a=q(i,0),b=q(i,1),c=q(i,2),d=q(i,3);
  if(a<0||d>=n||!(a<b&&b<c&&c<d))throw std::runtime_error("quartets must be canonical unique leaf ids");
  int s0=idx.dist(a,b)+idx.dist(c,d),s1=idx.dist(a,c)+idx.dist(b,d),s2=idx.dist(a,d)+idx.dist(b,c);
  int state=(s0<s1&&s0<s2)?0:((s1<s2)?1:2);cls(i)=state;objective+=w(i,state);
  int x=a,y=b,z=c,t=d;if(state==1){y=c;z=b;}else if(state==2){y=d;z=b;t=c;}
  int u=idx.median(x,y,z),v=idx.median(z,t,x);
  if(idx.parent[u]!=v&&idx.parent[v]!=u)continue;
  if(u>v)std::swap(u,v);int child=idx.parent[u]==v?u:v;int e=edgeid[child];
  int positions[4];
  for(int j=0;j<4;j++){
   int leaf=q(i,j);int center=idx.ancestor(child,leaf)?child:idx.parent[child];
   int dir=idx.step(center,leaf);int offset=center==u?0:2;
   int group=branches[e][offset]==dir?offset:offset+1;positions[group]=j;
  }
  int k0=pairclass(positions[0],positions[1]);int k1=pairclass(positions[0],positions[2]);int k2=pairclass(positions[0],positions[3]);
  if(k0!=state)throw std::runtime_error("NNI class mapping failed");
  out(e,1)+=w(i,k1)-w(i,k0);out(e,2)+=w(i,k2)-w(i,k0);active++;
 }
 }
 py::array_t<int64_t> ee({(int)edges.size(),2});auto er=ee.mutable_unchecked<2>();
 for(int i=0;i<edges.size();i++){er(i,0)=edges[i][0];er(i,1)=edges[i][1];}
 py::dict result;result["scores"]=score;result["edges"]=ee;result["objective"]=objective;result["active_quartets"]=active;result["classes"]=klass;return result;
}

py::dict evaluate_factors(AI tree,int n,AI taxa,AI masks,AI paths){
 Index idx(tree,n);auto tax=taxa.unchecked<2>();auto mask=masks.unchecked<3>();auto path=paths.unchecked<3>();
 int P=tax.shape(0),k=tax.shape(1);if(k<4||k>60||mask.shape(0)!=P||mask.shape(1)!=k-2||mask.shape(2)!=3||path.shape(0)!=P||path.shape(1)!=k||path.shape(2)!=k)throw std::runtime_error("invalid factor dimensions");
 std::vector<std::array<int,2>> edges;std::vector<std::array<int,4>> branches;std::vector<int> edgeid(idx.V,-1);
 for(int u=n;u<idx.V;u++)for(int v:idx.g[u])if(v>u){
  int child=idx.parent[u]==v?u:v;edgeid[child]=edges.size();edges.push_back({u,v});std::array<int,4>b;int z=0;
  for(int t:idx.g[u])if(t!=v)b[z++]=t;for(int t:idx.g[v])if(t!=u)b[z++]=t;branches.push_back(b);
 }
 py::array_t<double> scores({(int)edges.size(),3});auto out=scores.mutable_unchecked<2>();
 for(int e=0;e<edges.size();e++)for(int c=0;c<3;c++)out(e,c)=0.;
 int64_t objective=0,active=0;
 {
 py::gil_scoped_release release;
 for(int p=0;p<P;p++){
  std::vector<int> nodes;nodes.reserve(2*k);std::vector<int> tips(k);std::vector<int> ds(k*k);
  for(int a=0;a<k;a++){
   int x=tax(p,a);if(x<0||x>=n)throw std::runtime_error("factor leaf out of range");tips[a]=x;nodes.push_back(x);
   for(int b=0;b<a;b++){int y=tax(p,b);int l=idx.lca(x,y);nodes.push_back(l);int d=idx.depth[x]+idx.depth[y]-2*idx.depth[l];ds[a*k+b]=ds[b*k+a]=d;}
  }
  // Objective independently enumerates only k-choose-4 local rows, never N^4.
  for(int a=0;a<k;a++)for(int b=a+1;b<k;b++)for(int c=b+1;c<k;c++)for(int d=c+1;d<k;d++){
   int s0=ds[a*k+b]+ds[c*k+d],s1=ds[a*k+c]+ds[b*k+d],s2=ds[a*k+d]+ds[b*k+c];
   int state=(s0<s1&&s0<s2)?0:((s1<s2)?1:2);
   int t0=path(p,a,b)+path(p,c,d),t1=path(p,a,c)+path(p,b,d),t2=path(p,a,d)+path(p,b,c);
   int truth=(t0<t1&&t0<t2)?0:((t1<t2)?1:2);objective+=(state==truth);
  }
  std::sort(nodes.begin(),nodes.end(),[&](int a,int b){return idx.tin[a]<idx.tin[b];});nodes.erase(std::unique(nodes.begin(),nodes.end()),nodes.end());
  std::vector<int> stack;std::vector<std::array<int,2>> ve;
  for(int u:nodes){while(!stack.empty()&&!idx.ancestor(stack.back(),u))stack.pop_back();if(!stack.empty())ve.push_back({stack.back(),u});stack.push_back(u);}
  // Suppressing the degree-two virtual root cannot introduce a new actual
  // global edge, so only original virtual edges need be considered here.
  for(auto ed:ve){
   int u=ed[0],v=ed[1];if(u<n||v<n||idx.parent[v]!=u)continue;
   if(u>v)std::swap(u,v);int child=idx.parent[u]==v?u:v;int e=edgeid[child];
   if(e<0)continue;uint64_t groups[4]={0,0,0,0};
   for(int j=0;j<k;j++){
    int leaf=tips[j];int center=idx.ancestor(child,leaf)?child:idx.parent[child];int dir=idx.step(center,leaf);int off=center==u?0:2;
    int g=branches[e][off]==dir?off:off+1;groups[g]|=uint64_t(1)<<j;
   }
   if(!groups[0]||!groups[1]||!groups[2]||!groups[3])continue;
   int64_t support[3]={0,0,0};
   for(int node=0;node<k-2;node++){
    int counts[4][3];for(int g=0;g<4;g++)for(int h=0;h<3;h++)counts[g][h]=__builtin_popcountll(groups[g]&uint64_t(mask(p,node,h)));
    for(int i=0;i<3;i++){
     int j=(i+1)%3,l=(i+2)%3;
     support[0]+=counts[0][i]*counts[1][i]*(counts[2][j]*counts[3][l]+counts[2][l]*counts[3][j]);
     support[1]+=counts[0][i]*counts[2][i]*(counts[1][j]*counts[3][l]+counts[1][l]*counts[3][j]);
     support[2]+=counts[0][i]*counts[3][i]*(counts[1][j]*counts[2][l]+counts[1][l]*counts[2][j]);
    }
   }
   out(e,1)+=double(support[1]-support[0]);out(e,2)+=double(support[2]-support[0]);active++;
  }
 }
 }
 py::array_t<int64_t> ee({(int)edges.size(),2});auto er=ee.mutable_unchecked<2>();for(int e=0;e<edges.size();e++){er(e,0)=edges[e][0];er(e,1)=edges[e][1];}
 py::dict result;result["scores"]=scores;result["edges"]=ee;result["objective"]=objective;result["active_factors"]=active;return result;
}

PYBIND11_MODULE(_msc_sparse,m){m.def("evaluate",&evaluate);m.def("evaluate_factors",&evaluate_factors);}
