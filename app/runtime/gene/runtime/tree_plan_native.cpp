// Deterministic O(N) tree-to-message DAG, preserving the Python oracle order.
// No likelihood or search approximation. Only graph metadata construction.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
namespace py=pybind11;
using I=int64_t;
template<class T> py::array_t<T> array(std::vector<T> v,std::vector<py::ssize_t> shape){
 auto *p=new std::vector<T>(std::move(v));
 py::capsule own(p,[](void *x){delete static_cast<std::vector<T>*>(x);});
 return py::array_t<T>(shape,p->data(),own);
}
struct Graph{
 std::vector<I> id,order,parent,depth;
 std::vector<std::vector<I>> adj;
 std::vector<bool> leaf;
 std::unordered_map<I,I> ix;
 I root=-1,maxdepth=0;
 Graph(py::dict graph,py::dict names,bool binary=true){
  for(auto p:graph){I u=py::cast<I>(p.first);ix[u]=id.size();id.push_back(u);}
  I n=id.size();adj.resize(n);leaf.resize(n);parent.assign(n,-2);depth.assign(n,0);
  if(n==0)throw std::invalid_argument("empty tree graph");
  for(auto p:names)if(!graph.contains(p.first))throw std::invalid_argument("unknown leaf node");
  for(I a=0;a<n;++a){
   auto u=id[a];leaf[a]=names.contains(py::int_(u));
   std::vector<I> v;for(auto w:py::reinterpret_borrow<py::iterable>(graph[py::int_(u)]))v.push_back(py::cast<I>(w));std::sort(v.begin(),v.end());
   if(binary&&v.size()!=(leaf[a]?1:3))throw std::invalid_argument("binary tree required");
   for(auto w:v){if(!ix.count(w))throw std::invalid_argument("invalid neighbor");adj[a].push_back(ix.at(w));}
   if(root<0&&!leaf[a])root=a;
  }
  if(binary&&(names.size()<4||root<0))throw std::invalid_argument("at least4 taxa required");
  if(root<0)root=0;
  order.push_back(root);parent[root]=-1;
  for(size_t j=0;j<order.size();++j){auto u=order[j];
   bool parent_found=(u==root);
   for(auto v:adj[u]){if(v==parent[u]){parent_found=true;continue;}
    if(parent[v]!=-2)throw std::invalid_argument("cycle or asymmetric graph");
    parent[v]=u;depth[v]=depth[u]+1;maxdepth=std::max(maxdepth,depth[v]);order.push_back(v);
   }
   if(!parent_found)throw std::invalid_argument("asymmetric graph");
  }
  if(order.size()!=id.size())throw std::invalid_argument("disconnected tree");
 }
 I slot(I u,I v)const{for(size_t j=0;j<adj[u].size();++j)if(adj[u][j]==v)return j;throw std::invalid_argument("asymmetric graph");}
 py::tuple edge(I u,I v)const{return py::make_tuple(std::min(id[u],id[v]),std::max(id[u],id[v]));}
};
py::dict build(py::dict graph,py::dict names,py::dict lengths,py::list labels,bool nni){
 Graph g(graph,names);I n=g.id.size();
 std::unordered_map<std::string,I> lookup;
 for(I i=0;i<I(labels.size());++i)if(!lookup.emplace(py::cast<std::string>(labels[i]),i).second)throw std::invalid_argument("duplicate taxa");
 if(lookup.size()!=names.size())throw std::invalid_argument("exact taxa match required");
 std::vector<std::pair<I,I>> edges;
 std::vector<std::vector<int32_t>> row(n),eid(n);
 for(I u=0;u<n;++u){row[u].assign(g.adj[u].size(),-1);eid[u].assign(g.adj[u].size(),-1);}
 py::list pyedges,internal,branches;
 std::vector<double> lens;
 I nr=0;
 for(auto u:g.order)for(auto v:g.adj[u])if(g.id[u]<g.id[v]){
  auto key=g.edge(u,v);double t=py::cast<double>(lengths[key]);
  if(!(t>0)||!std::isfinite(t))throw std::invalid_argument("positive finite lengths required");
  I e=edges.size();edges.emplace_back(u,v);pyedges.append(key);lens.push_back(t);
  eid[u][g.slot(u,v)]=e;eid[v][g.slot(v,u)]=e;
  for(auto ab:{std::make_pair(u,v),std::make_pair(v,u)}){
   auto a=ab.first,b=ab.second;
   if((nni&&!g.leaf[b])||(!nni&&g.parent[a]==b))row[a][g.slot(a,b)]=nr++;
  }
 }
 auto R=[&](I u,I v)->int32_t{auto r=row[u][g.slot(u,v)];if(r<0)throw std::invalid_argument("missing message dependency");return r;};
 std::vector<int32_t> deps(nr*2,0),rowedge(nr),leafrows,leafindex,rootrows,scoreedges,incoming;
 for(auto uv:edges)for(auto ab:{uv,std::make_pair(uv.second,uv.first)}){
  auto u=ab.first,v=ab.second;auto r=row[u][g.slot(u,v)];if(r<0)continue;
  rowedge[r]=eid[u][g.slot(u,v)];
  if(!g.leaf[u]){I j=0;for(auto w:g.adj[u])if(w!=v)deps[r*2+j++]=R(w,u);}
 }
 std::vector<I> leaves;for(I u=0;u<n;++u)if(g.leaf[u])leaves.push_back(u);
 std::sort(leaves.begin(),leaves.end(),[&](I u,I v){return g.id[u]<g.id[v];});
 for(auto u:leaves){leafrows.push_back(R(u,g.adj[u][0]));auto name=py::cast<std::string>(names[py::int_(g.id[u])]);if(!lookup.count(name))throw std::invalid_argument("taxon mismatch");leafindex.push_back(lookup.at(name));}
 std::vector<std::vector<int32_t>> up(g.maxdepth+1),down(g.maxdepth+1);
 for(auto u:g.order)if(!g.leaf[u]){
  if(u!=g.root)up[g.depth[u]].push_back(R(u,g.parent[u]));
  if(nni)for(auto v:g.adj[u])if(g.parent[v]==u&&!g.leaf[v])down[g.depth[u]].push_back(R(u,v));
 }
 for(auto v:g.adj[g.root])rootrows.push_back(R(v,g.root));
 if(nni)for(I e=0;e<I(edges.size());++e){auto u=edges[e].first,v=edges[e].second;if(g.leaf[u]||g.leaf[v])continue;
  internal.append(g.edge(u,v));scoreedges.push_back(e);std::vector<I> b;
  for(auto w:g.adj[u])if(w!=v){b.push_back(g.id[w]);incoming.push_back(R(w,u));}
  for(auto w:g.adj[v])if(w!=u){b.push_back(g.id[w]);incoming.push_back(R(w,v));}
  branches.append(py::make_tuple(b[0],b[1],b[2],b[3]));
 }
 py::list arrays,upout,downout;
 arrays.append(array(std::move(deps),{nr,2}));arrays.append(array(std::move(rowedge),{nr}));
 arrays.append(array(std::move(leafrows),{I(leaves.size())}));arrays.append(array(std::move(leafindex),{I(leaves.size())}));
 arrays.append(array(std::move(rootrows),{3}));arrays.append(array(std::move(scoreedges),{I(internal.size())}));
 // Match the Python empty-list shape in root-only mode.
 arrays.append(nni?array(std::move(incoming),{I(internal.size()),4}):array(std::move(incoming),{0}));
 for(auto i=up.rbegin();i!=up.rend();++i)if(!i->empty()){I size=i->size();upout.append(array(std::move(*i),{size}));}
 for(auto &v:down)if(!v.empty()){I size=v.size();downout.append(array(std::move(v),{size}));}
 py::dict result;result["root"]=g.id[g.root];result["depth"]=g.maxdepth;result["edges"]=pyedges;result["lengths"]=array(std::move(lens),{I(edges.size())});
 result["rows"]=nr;result["internal_edges"]=internal;result["branches"]=branches;result["arrays"]=arrays;result["up"]=upout;result["down"]=downout;return result;
}
std::string newick(py::dict graph,py::dict names,py::dict lengths,py::dict annotations){
 Graph g(graph,names,false);std::string out;out.reserve(g.id.size()*20);
 struct Job{I u;char c;};std::vector<Job> stack{{g.root,0}};
 auto tail=[&](I u){
  if(g.leaf[u]){auto s=py::cast<std::string>(names[py::int_(g.id[u])]);
   bool quote=s.find_first_of("(),:;[]'")!=std::string::npos;
   auto value=py::reinterpret_borrow<py::str>(names[py::int_(g.id[u])]);
   for(Py_ssize_t j=0;j<PyUnicode_GetLength(value.ptr());++j)if(Py_UNICODE_ISSPACE(PyUnicode_ReadChar(value.ptr(),j)))quote=true;
   if(!quote)out+=s;
   else{out+='\'';for(char c:s){out+=c;if(c=='\'')out+='\'';}out+='\'';}
  }
  if(g.parent[u]>=0){auto e=g.edge(u,g.parent[u]);
   if(!g.leaf[u]&&annotations.contains(e))out+=py::cast<std::string>(py::str(annotations[e]));
   if(lengths.contains(e)&&!lengths[e].is_none()){char buf[64];std::snprintf(buf,sizeof(buf),"%.10g",py::cast<double>(lengths[e]));out+=':';out+=buf;}
  }
 };
 while(!stack.empty()){auto job=stack.back();stack.pop_back();if(job.c){out+=job.c;continue;}
  auto u=job.u;if(u<0){tail(-u-1);continue;}
  stack.push_back({-u-1,0});std::vector<I> children;for(auto v:g.adj[u])if(v!=g.parent[u])children.push_back(v);
  if(!children.empty()){stack.push_back({0,')'});for(I j=I(children.size())-1;j>=0;--j){stack.push_back({children[j],0});if(j>0)stack.push_back({0,','});}stack.push_back({0,'('});}
 }
 out+=';';return out;
}
PYBIND11_MODULE(_tree_plan_native,m){m.def("build",&build);m.def("newick",&newick);}
