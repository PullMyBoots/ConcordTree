// Exact nearest24 query products and sorted persistent evidence operations.
// The directed nearest messages are supplied by the unchanged release backend.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <vector>
#include <algorithm>
#include <stdexcept>
namespace py=pybind11;
using A=py::array_t<int32_t,py::array::c_style|py::array::forcecast>;
using D=py::array_t<double,py::array::c_style|py::array::forcecast>;
using Q=std::array<int32_t,4>;
static_assert(sizeof(Q)==16,"canonical quartet storage must be contiguous");

A nearest_queries(A tree,int n,A messages){
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
 return out;
}

A missing(A old,A proposed){
 if(old.ndim()!=2||old.shape(1)!=4||proposed.ndim()!=2||proposed.shape(1)!=4)throw std::runtime_error("canonical quartet matrices required");
 auto a=reinterpret_cast<const Q*>(old.data());auto b=reinterpret_cast<const Q*>(proposed.data());
 std::vector<Q> result;result.reserve(proposed.shape(0));
 {
  py::gil_scoped_release release;size_t i=0;
  for(size_t j=0;j<size_t(proposed.shape(0));j++){
   while(i<size_t(old.shape(0))&&a[i]<b[j])i++;
   if(i==size_t(old.shape(0))||a[i]!=b[j])result.push_back(b[j]);
  }
 }
 A out({py::ssize_t(result.size()),py::ssize_t(4)});
 std::copy(reinterpret_cast<const int32_t*>(result.data()),reinterpret_cast<const int32_t*>(result.data())+4*result.size(),out.mutable_data());
 return out;
}

py::tuple merge(A old,D weights,A added,D added_weights){
 if(old.ndim()!=2||old.shape(1)!=4||added.ndim()!=2||added.shape(1)!=4||
    weights.ndim()!=2||weights.shape(0)!=old.shape(0)||weights.shape(1)!=3||
    added_weights.ndim()!=2||added_weights.shape(0)!=added.shape(0)||added_weights.shape(1)!=3)
  throw std::runtime_error("quartet/weight dimensions disagree");
 const size_t na=old.shape(0),nb=added.shape(0),count=na+nb;
 A queries({py::ssize_t(count),py::ssize_t(4)});D scores({py::ssize_t(count),py::ssize_t(3)});py::array_t<bool> is_new(count);
 auto a=reinterpret_cast<const Q*>(old.data());auto b=reinterpret_cast<const Q*>(added.data());
 auto out=reinterpret_cast<Q*>(queries.mutable_data());auto w=scores.mutable_data();auto mask=is_new.mutable_data();
 {
  py::gil_scoped_release release;size_t i=0,j=0;
  for(size_t k=0;k<count;k++){
   if(i<na&&j<nb&&a[i]==b[j])throw std::runtime_error("new evidence overlaps existing quartet");
   const bool take_new=i==na||(j<nb&&b[j]<a[i]);mask[k]=take_new;
   if(take_new){out[k]=b[j];std::copy_n(added_weights.data()+3*j,3,w+3*k);j++;}
   else{out[k]=a[i];std::copy_n(weights.data()+3*i,3,w+3*k);i++;}
  }
 }
 return py::make_tuple(queries,scores,is_new);
}

PYBIND11_MODULE(_msc_active_plan,m){m.def("nearest_queries",&nearest_queries);m.def("missing",&missing);m.def("merge",&merge);}
