// Parallel integer tree queries, then original-order FP64 reductions.
// The statistical objective and summation order match _msc_sparse.evaluate.
#include "active_index.hpp"
#include <atomic>
#include <omp.h>
using AQ=py::array_t<int32_t,py::array::c_style|py::array::forcecast>;

py::dict evaluate(AI tree,int n,AQ queries,AD weights,int workers){
 if(workers<1)throw std::runtime_error("positive execution worker count required");
 Index idx(tree,n);auto q=queries.unchecked<2>();auto w=weights.unchecked<2>();
 if(q.shape(1)!=4||w.shape(0)!=q.shape(0)||w.shape(1)!=3)throw std::runtime_error("invalid quartet/weight dimensions");
 std::vector<std::array<int,2>> edges;std::vector<std::array<int,4>> branches;
 std::vector<int> edgeid(idx.V,-1);
 for(int u=n;u<idx.V;u++)for(int v:idx.g[u])if(v>u){
  int child=idx.parent[u]==v?u:v;edgeid[child]=edges.size();edges.push_back({u,v});
  std::array<int,4>b;int k=0;for(int z:idx.g[u])if(z!=v)b[k++]=z;for(int z:idx.g[v])if(z!=u)b[k++]=z;branches.push_back(b);
 }
 const int64_t count=q.shape(0);
 py::array_t<double> score({(int)edges.size(),3});auto out=score.mutable_unchecked<2>();
 for(int e=0;e<(int)edges.size();e++)for(int c=0;c<3;c++)out(e,c)=0.;
 py::array_t<int64_t> klass(count);auto cls=klass.mutable_unchecked<1>();
 std::vector<int32_t> affected(count,-1);
 std::vector<int8_t> alternative_one(count),alternative_two(count);
 std::atomic<bool> invalid(false);double objective=0.;int active=0;
 {
  py::gil_scoped_release release;
  #pragma omp parallel for schedule(static) num_threads(workers)
  for(int64_t i=0;i<count;i++){
   int a=q(i,0),b=q(i,1),c=q(i,2),d=q(i,3);
   if(a<0||d>=n||!(a<b&&b<c&&c<d)){invalid.store(true,std::memory_order_relaxed);continue;}
   // Four DFS-ordered tips have only two noncrossing quartet splits.
   // Three consecutive LCAs identify both Steiner centers exactly.
   int order[4]={0,1,2,3};
   auto swap_if=[&](int left,int right){if(idx.tin[q(i,order[left])]>idx.tin[q(i,order[right])])std::swap(order[left],order[right]);};
   swap_if(0,1);swap_if(2,3);swap_if(0,2);swap_if(1,3);swap_if(1,2);
   int h0=idx.lca(q(i,order[0]),q(i,order[1]));
   int h1=idx.lca(q(i,order[1]),q(i,order[2]));
   int h2=idx.lca(q(i,order[2]),q(i,order[3]));
   // The shallowest of the three rooted LCAs is suppressed when degree two.
   // The remaining two (allowing a repeated shallowest root) are centers.
   int u=h0,v=h1;
   if(idx.depth[u]<idx.depth[v])std::swap(u,v);
   if(idx.depth[h2]>idx.depth[v])v=h2;
   int p0=order[0],p1=order[1],p2=order[2],p3=order[3];
   if(idx.depth[h1]>idx.depth[h0]&&idx.depth[h1]>idx.depth[h2]){
    p0=order[1];p1=order[2];p2=order[0];p3=order[3];
   }
   int state=pairclass(p0,p1);cls(i)=state;
   if(idx.parent[u]!=v&&idx.parent[v]!=u)continue;
   if(u>v)std::swap(u,v);int child=idx.parent[u]==v?u:v;int e=edgeid[child];
   if(e<0){invalid.store(true,std::memory_order_relaxed);continue;}
   int positions[4]={-1,-1,-1,-1};
   int center=idx.ancestor(child,q(i,p0))?child:idx.parent[child];
   int offset=center==u?0:2;
   int group=branches[e][offset]==idx.step(center,q(i,p0))?offset:offset+1;
   positions[group]=p0;positions[group^1]=p1;
   center=center==u?v:u;offset=center==u?0:2;
   group=branches[e][offset]==idx.step(center,q(i,p2))?offset:offset+1;
   positions[group]=p2;positions[group^1]=p3;
   if(*std::min_element(positions,positions+4)<0){invalid.store(true,std::memory_order_relaxed);continue;}
   int k0=pairclass(positions[0],positions[1]);
   if(k0!=state){invalid.store(true,std::memory_order_relaxed);continue;}
   affected[i]=e;alternative_one[i]=pairclass(positions[0],positions[2]);alternative_two[i]=pairclass(positions[0],positions[3]);
  }
  if(!invalid.load()){
   // This serial stream deliberately preserves every floating operation's
   // order. Parallelism applies only to independent exact integer queries.
   for(int64_t i=0;i<count;i++){
    const int state=cls(i);objective+=w(i,state);const int e=affected[i];
    if(e<0)continue;
    out(e,1)+=w(i,alternative_one[i])-w(i,state);
    out(e,2)+=w(i,alternative_two[i])-w(i,state);active++;
   }
  }
 }
 if(invalid.load())throw std::runtime_error("invalid quartet ids or NNI class mapping");
 py::array_t<int64_t> ee({(int)edges.size(),2});auto er=ee.mutable_unchecked<2>();
 for(int i=0;i<(int)edges.size();i++){er(i,0)=edges[i][0];er(i,1)=edges[i][1];}
 py::dict result;result["scores"]=score;result["edges"]=ee;result["objective"]=objective;
 result["active_quartets"]=active;result["classes"]=klass;return result;
}

PYBIND11_MODULE(_msc_active_fast,m){m.def("evaluate",&evaluate,py::arg("tree"),py::arg("n"),py::arg("queries"),py::arg("weights"),py::arg("workers")=8);}
