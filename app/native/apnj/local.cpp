#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <array>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <numeric>
namespace py=pybind11;
using Pair=std::pair<int,int>;
struct Message{std::array<Pair,4> leaf;int count=0;};
py::tuple plan(py::array_t<int64_t,py::array::c_style> neighbors,int n){
 if(neighbors.ndim()!=2 || neighbors.shape(1)!=3 || n<4 || neighbors.shape(0)!=2LL*n-2)throw std::invalid_argument("local plan requires contiguous binary adjacency");
 int m=neighbors.shape(0);const int64_t*adj=neighbors.data();
 auto slot=[&](int u,int v){for(int j=0;j<3;++j)if(adj[u*3+j]==v)return j;throw std::invalid_argument("asymmetric local adjacency");};
 for(int u=0;u<m;++u){int count=0;for(int j=0;j<3;++j){auto v=adj[u*3+j];if(v<0)continue;if(v>=m || v==u)throw std::invalid_argument("local plan neighbor out of range");slot(v,u);++count;}if(count!=(u<n?1:3))throw std::invalid_argument("local plan invalid degree");}
 std::vector<int> parent(m,-2),order;order.reserve(m);parent[n]=-1;order.push_back(n);
 for(size_t q=0;q<order.size();++q){int u=order[q];for(int j=0;j<3;++j){int v=adj[u*3+j];if(v<0 || v==parent[u])continue;if(parent[v]!=-2)throw std::invalid_argument("cyclic local tree");parent[v]=u;order.push_back(v);}}
 if(order.size()!=size_t(m))throw std::invalid_argument("disconnected local tree");
 std::vector<Message> messages(m*3);
 auto compute=[&](int u,int excluded){Message result;if(u<n){result.count=1;result.leaf[0]={0,u};return result;}std::array<Pair,8> candidates;int size=0;
  for(int j=0;j<3;++j){int v=adj[u*3+j];if(v==excluded)continue;const auto& source=messages[v*3+slot(v,u)];if(!source.count)throw std::runtime_error("local DP dependency missing");for(int k=0;k<source.count;++k)candidates[size++]={source.leaf[k].first+1,source.leaf[k].second};}
  std::sort(candidates.begin(),candidates.begin()+size);result.count=std::min(4,size);std::copy(candidates.begin(),candidates.begin()+result.count,result.leaf.begin());return result;
 };
 {py::gil_scoped_release release;
 for(auto it=order.rbegin();it!=order.rend();++it){int u=*it;if(parent[u]>=0)messages[u*3+slot(u,parent[u])]=compute(u,parent[u]);}
 for(int u:order)for(int j=0;j<3;++j){int v=adj[u*3+j];if(v>=0 && parent[v]==u)messages[u*3+j]=compute(u,v);}}
 const int E=n-3;py::array_t<int64_t> edges({E,2}),branches({E,4});py::array_t<int> reps({E*4,4});py::array_t<double> weights({E*4,4});int e=0;
 for(int u=n;u<m;++u)for(int j=0;j<3;++j){int v=adj[u*3+j];if(v<=u || v<n)continue;edges.mutable_data()[e*2]=u;edges.mutable_data()[e*2+1]=v;int k=0;
  for(int side=0;side<2;++side){int owner=side?v:u,other=side?u:v;std::array<int,2> ends;int count=0;for(int t=0;t<3;++t)if(adj[owner*3+t]!=other)ends[count++]=adj[owner*3+t];std::sort(ends.begin(),ends.end());
   for(int end:ends){branches.mutable_data()[e*4+k]=end;const auto& msg=messages[end*3+slot(end,owner)];for(int z=0;z<4;++z){reps.mutable_data()[(e*4+k)*4+z]=msg.leaf[z<msg.count?z:0].second;weights.mutable_data()[(e*4+k)*4+z]=z<msg.count?1./msg.count:0.;}++k;}}
  ++e;
 }
 if(e!=E)throw std::runtime_error("local plan edge count");return py::make_tuple(edges,branches,reps,weights);
}
py::array_t<int64_t> apply(py::array_t<int64_t,py::array::c_style> neighbors,py::array_t<int64_t,py::array::c_style> edges,py::array_t<int64_t,py::array::c_style> branches,py::array_t<int64_t,py::array::c_style> best,py::array_t<double,py::array::c_style> margins){
 if(neighbors.ndim()!=2 || neighbors.shape(1)!=3 || !neighbors.writeable() || edges.ndim()!=2 || edges.shape(1)!=2 || branches.ndim()!=2 || branches.shape(1)!=4 || branches.shape(0)!=edges.shape(0) || best.ndim()!=1 || best.size()!=edges.shape(0) || margins.ndim()!=1 || margins.size()!=edges.shape(0))throw std::invalid_argument("local action shape");
 const int m=neighbors.shape(0),E=edges.shape(0);auto*adj=neighbors.mutable_data();auto*es=edges.data();auto*bs=branches.data();auto*which=best.data();auto*gap=margins.data();std::vector<int64_t> order,selected;
 for(int i=0;i<E;++i){if(which[i]<0 || which[i]>2 || !std::isfinite(gap[i]))throw std::invalid_argument("local action score");for(int j=0;j<2;++j)if(es[i*2+j]<0 || es[i*2+j]>=m)throw std::invalid_argument("local action edge");for(int j=0;j<4;++j)if(bs[i*4+j]<0 || bs[i*4+j]>=m)throw std::invalid_argument("local action branch");if(which[i] && gap[i]>0)order.push_back(i);}
 {py::gil_scoped_release release;std::sort(order.begin(),order.end(),[&](int a,int b){if(gap[a]!=gap[b])return gap[a]>gap[b];if(es[a*2]!=es[b*2])return es[a*2]<es[b*2];return es[a*2+1]<es[b*2+1];});std::vector<unsigned char> blocked(m,0);
 auto replace=[&](int u,int old,int fresh){auto*start=adj+u*3;auto*at=std::find(start,start+3,old);if(at==start+3)throw std::runtime_error("stale local action");*at=fresh;std::sort(start,start+3,[](int64_t a,int64_t b){return a>=0 && (b<0 || a<b);});};
 for(int e:order){int u=es[e*2],v=es[e*2+1];if(blocked[u] || blocked[v])continue;for(int t=0;t<3;++t){if(adj[u*3+t]>=0)blocked[adj[u*3+t]]=1;if(adj[v*3+t]>=0)blocked[adj[v*3+t]]=1;}blocked[u]=blocked[v]=1;int b=bs[e*4+1],d=bs[e*4+(which[e]==1?2:3)];replace(u,b,d);replace(b,u,v);replace(v,d,b);replace(d,v,u);selected.push_back(e);}}
 py::array_t<int64_t> out(selected.size());std::copy(selected.begin(),selected.end(),out.mutable_data());return out;
}
PYBIND11_MODULE(_ct_apnj_local,m){m.def("plan",&plan);m.def("apply",&apply);}
