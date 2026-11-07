// Exact nearest24 query products and sorted persistent evidence operations.
// The directed nearest messages are supplied by the unchanged release backend.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <cstdint>
namespace py=pybind11;
using A=py::array_t<int32_t,py::array::c_style|py::array::forcecast>;
using D=py::array_t<double,py::array::c_style|py::array::forcecast>;
using Q=std::array<int32_t,4>;
static_assert(sizeof(Q)==16,"canonical quartet storage must be contiguous");

using Panel=std::array<int32_t,28>;
class IncrementalPlanner {
 std::unordered_map<int64_t,Panel> previous;
 public:
 py::dict propose(A tree,int n,A messages){
 if(n<24||tree.ndim()!=2||tree.shape(1)!=2||messages.ndim()!=2||messages.shape(1)!=26)
  throw std::runtime_error("binary tree with >=24 taxa and release nearest24 messages required");
 const int V=2*n-2;auto edges=tree.unchecked<2>();auto rows=messages.unchecked<2>();
 std::vector<std::vector<int>> adj(V);
 for(int i=0;i<edges.shape(0);i++){
  int u=edges(i,0),v=edges(i,1);
  if(u<0||v<0||u>=V||v>=V)throw std::runtime_error("invalid node id");
  adj[u].push_back(v);adj[v].push_back(u);
 }
 for(int u=0;u<V;u++){
  std::sort(adj[u].begin(),adj[u].end());
  if(adj[u].size()!=size_t(u<n?1:3))throw std::runtime_error("binary tree required");
 }
 std::vector<std::array<int,24>> nearest(3*V);std::vector<int> lengths(3*V,-1);
 auto slot=[&](int u,int v){
  if(u<0||u>=V)throw std::runtime_error("invalid message root");
  for(int j=0;j<(int)adj[u].size();j++)if(adj[u][j]==v)return 3*u+j;
  throw std::runtime_error("message edge absent");
 };
 for(int i=0;i<rows.shape(0);i++){
  int s=slot(rows(i,0),rows(i,1));if(lengths[s]!=-1)throw std::runtime_error("duplicate message");
  int count=0;for(int j=0;j<24;j++)if(rows(i,j+2)>=0)nearest[s][count++]=rows(i,j+2);
  lengths[s]=count;
 }
 std::vector<Q> queries;
 std::unordered_map<int64_t,Panel> current;current.reserve(n);
 int64_t proposed_count=0;int changed_edges=0;
 {
  py::gil_scoped_release release;
  for(int u=n;u<V;u++)for(int v:adj[u])if(v>u){
   int slots[4],k=0;
   for(int b:adj[u])if(b!=v)slots[k++]=slot(b,u);
   for(int b:adj[v])if(b!=u)slots[k++]=slot(b,v);
   std::array<std::array<int,24>,4> selected;int counts[4]={0,0,0,0},total=0;
   for(int rank=0;rank<24;rank++)for(int j=0;j<4;j++){
    if(lengths[slots[j]]<=0)throw std::runtime_error("missing directed message");
    if(rank<lengths[slots[j]]&&total<24){selected[j][counts[j]++]=nearest[slots[j]][rank];total++;}
   }
   if(total!=24)throw std::runtime_error("incomplete saturated panel");
   Panel panel{};int at=4;
   for(int j=0;j<4;j++){
    panel[j]=counts[j];
    for(int k=0;k<counts[j];k++)panel[at++]=selected[j][k];
   }
   proposed_count+=int64_t(counts[0])*counts[1]*counts[2]*counts[3];
   const int64_t key=(int64_t(u)<<32)|uint32_t(v);
   current.emplace(key,panel);
   auto old=previous.find(key);
   // Identical panels emitted exactly these queries last sweep; every one
   // is already retained in the caller's persistent evidence bank.
   if(old!=previous.end() && old->second==panel)continue;
   changed_edges++;

   for(int a=0;a<counts[0];a++)for(int b=0;b<counts[1];b++)
    for(int c=0;c<counts[2];c++)for(int d=0;d<counts[3];d++){
     Q q{selected[0][a],selected[1][b],selected[2][c],selected[3][d]};
     std::sort(q.begin(),q.end());
     if(q[0]<0||q[3]>=n||std::adjacent_find(q.begin(),q.end())!=q.end())throw std::runtime_error("invalid discriminating quartet");
     queries.push_back(q);
    }
  }
  // Four stable counting passes give lexicographic integer order in O(Q+N).
  std::vector<Q> scratch(queries.size());std::vector<size_t> bucket(n+1);
  for(int col=3;col>=0;col--){
   std::fill(bucket.begin(),bucket.end(),0);
   for(const auto &q:queries)bucket[q[col]+1]++;
   for(int j=1;j<=n;j++)bucket[j]+=bucket[j-1];
   for(const auto &q:queries)scratch[bucket[q[col]]++]=q;
   queries.swap(scratch);
  }
  queries.erase(std::unique(queries.begin(),queries.end()),queries.end());
 }
 A out({py::ssize_t(queries.size()),py::ssize_t(4)});
 std::copy(reinterpret_cast<const int32_t*>(queries.data()),reinterpret_cast<const int32_t*>(queries.data())+4*queries.size(),out.mutable_data());
 previous.swap(current);
 py::dict result;result["queries"]=out;result["proposed_quartets"]=proposed_count;
 result["changed_edges"]=changed_edges;return result;
 }
};
PYBIND11_MODULE(_msc_incremental_plan,m){
 py::class_<IncrementalPlanner>(m,"IncrementalPlanner")
  .def(py::init<>()).def("propose",&IncrementalPlanner::propose);
}

