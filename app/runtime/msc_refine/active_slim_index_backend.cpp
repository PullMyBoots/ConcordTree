// Euler RMQ construction without the obsolete binary-lifting table.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
namespace py=pybind11;
using A=py::array_t<int64_t,py::array::c_style|py::array::forcecast>;
py::dict export_index(A tree,int n){
 if(n<4||tree.ndim()!=2||tree.shape(1)!=2||tree.shape(0)!=2*n-3)throw std::runtime_error("invalid binary tree dimensions");
 int V=2*n-2;auto e=tree.unchecked<2>();std::vector<std::vector<int>> g(V);
 for(int i=0;i<e.shape(0);i++){int u=e(i,0),v=e(i,1);if(u<0||v<0||u>=V||v>=V||u==v)throw std::runtime_error("invalid node");g[u].push_back(v);g[v].push_back(u);}
 for(int u=0;u<V;u++){std::sort(g[u].begin(),g[u].end());if(g[u].size()!=size_t(u<n?1:3))throw std::runtime_error("binary tree required");}
 std::vector<int> parent(V,-1),tin(V),tout(V),depth(V),first(V,-1),tour;
 std::vector<std::pair<int,size_t>> stack{{n,0}};parent[n]=n;int tick=0;
 tin[n]=tick++;first[n]=0;tour.push_back(n);tour.reserve(2*V-1);
 while(!stack.empty()){
  int u=stack.back().first;size_t& next=stack.back().second;
  while(next<g[u].size()&&g[u][next]==parent[u])next++;
  if(next==g[u].size()){
   tout[u]=tick;stack.pop_back();if(!stack.empty())tour.push_back(stack.back().first);
  }else{
   int v=g[u][next++];if(parent[v]!=-1)throw std::runtime_error("cycle");
   parent[v]=u;depth[v]=depth[u]+1;tin[v]=tick++;first[v]=tour.size();tour.push_back(v);stack.push_back({v,0});
  }
 }
 if(tick!=V)throw std::runtime_error("disconnected tree");
 int W=tour.size(),K=32-__builtin_clz(unsigned(W));
 py::array_t<int32_t> nodes({8,V}),rmq({K,W});auto a=nodes.mutable_unchecked<2>();auto r=rmq.mutable_unchecked<2>();
 py::array_t<int64_t> edges({n-3,2}),branches({n-3,4});auto ee=edges.mutable_unchecked<2>();auto bb=branches.mutable_unchecked<2>();
 for(int u=0;u<V;u++){
  a(0,u)=tin[u];a(1,u)=tout[u];a(2,u)=parent[u];a(3,u)=depth[u];a(4,u)=first[u];a(5,u)=a(6,u)=a(7,u)=-1;int k=5;
  for(int v:g[u])if(v!=parent[u]&&k<7)a(k++,u)=v;
 }
 int at=0;for(int u=n;u<V;u++)for(int v:g[u])if(v>u){
  ee(at,0)=u;ee(at,1)=v;int j=0;
  for(int z:g[u])if(z!=v)bb(at,j++)=z;
  for(int z:g[v])if(z!=u)bb(at,j++)=z;
  a(7,parent[u]==v?u:v)=at++;
 }
 for(int i=0;i<W;i++)r(0,i)=tour[i];
 for(int k=1;k<K;k++)for(int i=0;i<W;i++){
  if(i+(1<<k)>W){r(k,i)=n;continue;}
  int u=r(k-1,i),v=r(k-1,i+(1<<(k-1)));r(k,i)=depth[u]<=depth[v]?u:v;
 }
 py::dict out;out["nodes"]=nodes;out["rmq"]=rmq;out["edges"]=edges;out["branches"]=branches;return out;
}
PYBIND11_MODULE(_msc_slim_index,m){m.def("export_index",&export_index);}
