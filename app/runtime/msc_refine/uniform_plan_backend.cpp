// Exact fixed-priority component and Cartesian-product compiler. No statistics.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace py=pybind11;
constexpr int K=24;
using I64=py::array_t<int64_t,py::array::c_style|py::array::forcecast>;
using I32=py::array_t<int32_t,py::array::c_style|py::array::forcecast>;

py::tuple components(py::dict graph,int n,I64 priority){
 const int total=2*n-2;
 if(n<4 || graph.size()!=static_cast<size_t>(total) || priority.ndim()!=1 || priority.size()!=n)
  throw std::invalid_argument("binary graph and permutation priority required");
 auto p=priority.data();std::vector<bool> seen(n,false);
 for(int i=0;i<n;++i){if(p[i]<0 || p[i]>=n || seen[p[i]])throw std::invalid_argument("priority must be a permutation");seen[p[i]]=true;}
 py::array_t<int32_t> adjacent(std::vector<py::ssize_t>{total,3});
 py::array_t<int32_t> message(std::vector<py::ssize_t>{total,3,K});
 py::array_t<int32_t> length(std::vector<py::ssize_t>{total,3});
 auto a=adjacent.mutable_data();auto m=message.mutable_data();auto l=length.mutable_data();
 std::fill(a,a+3*total,-1);std::fill(m,m+3*total*K,-1);std::fill(l,l+3*total,0);
 std::vector<int> degree(total,0);
 for(auto item:graph){int u=py::cast<int>(item.first);
  if(u<0 || u>=total)throw std::invalid_argument("node ID out of range");
  for(auto vv:py::reinterpret_borrow<py::iterable>(item.second)){
   int v=py::cast<int>(vv);if(v<0 || v>=total || degree[u]>=3)throw std::invalid_argument("adjacency out of range");
   a[3*u+degree[u]++]=v;
  }
  if(degree[u]!=(u<n?1:3))throw std::invalid_argument("fully binary tree required");
  std::sort(a+3*u,a+3*u+degree[u]);
 }
 auto slot=[&](int u,int v){for(int j=0;j<degree[u];++j)if(a[3*u+j]==v)return 3*u+j;throw std::invalid_argument("asymmetric adjacency");};
 std::vector<int> parent(total,-2),order;order.reserve(total);parent[n]=-1;order.push_back(n);
 for(size_t i=0;i<order.size();++i){int u=order[i];
  for(int j=0;j<degree[u];++j){int v=a[3*u+j];if(v==parent[u])continue;
   if(parent[v]!=-2)throw std::invalid_argument("cycle");parent[v]=u;order.push_back(v);
  }
 }
 if(order.size()!=static_cast<size_t>(total))throw std::invalid_argument("disconnected tree");
 auto compose=[&](int u,int v){int out=slot(u,v);
  if(u<n){m[out*K]=u;l[out]=1;return;}
  int input[2],used=0;
  for(int j=0;j<degree[u];++j){int w=a[3*u+j];if(w!=v)input[used++]=slot(w,u);}
  if(used!=2 || l[input[0]]==0 || l[input[1]]==0)throw std::logic_error("unready directed component");
  int i=0,j=0,count=0;
  while(count<K && (i<l[input[0]] || j<l[input[1]])){
   bool left=j==l[input[1]] || (i<l[input[0]] && p[m[input[0]*K+i]]<p[m[input[1]*K+j]]);
   m[out*K+count++]=left?m[input[0]*K+i++]:m[input[1]*K+j++];
  }
  l[out]=count;
 };
 for(auto it=order.rbegin();it!=order.rend();++it)if(parent[*it]>=0)compose(*it,parent[*it]);
 for(int u:order)for(int j=0;j<degree[u];++j){int v=a[3*u+j];if(parent[v]==u)compose(u,v);}
 return py::make_tuple(adjacent,message,length);
}

py::tuple queries(I32 adjacent,I32 message,I32 length,I64 edges,I64 branches,I64 rows){
 if(adjacent.ndim()!=2 || adjacent.shape(1)!=3 || message.ndim()!=3 || message.shape(1)!=3 || message.shape(2)!=K ||
    length.ndim()!=2 || length.shape(1)!=3 || message.shape(0)!=adjacent.shape(0) || length.shape(0)!=adjacent.shape(0) ||
    edges.ndim()!=2 || edges.shape(1)!=2 || branches.ndim()!=2 || branches.shape(0)!=edges.shape(0) || branches.shape(1)!=4 || rows.ndim()!=1)
  throw std::invalid_argument("invalid component/plan shapes");
 const int n=(adjacent.shape(0)+2)/2;auto a=adjacent.data();auto m=message.data();auto l=length.data();
 auto ep=edges.data();auto bp=branches.data();auto rp=rows.data();size_t batch=rows.size();
 std::vector<std::array<std::array<int,K>,4>> chosen(batch);std::vector<std::array<int,4>> counts(batch);
 py::array_t<int64_t> offsets(batch+1);auto off=offsets.mutable_data();off[0]=0;
 for(size_t i=0;i<batch;++i){int row=rp[i];if(row<0 || row>=edges.shape(0))throw std::invalid_argument("edge row out of range");
  int slots[4];counts[i].fill(0);
  for(int j=0;j<4;++j){int u=bp[4*row+j],v=ep[2*row+(j<2?0:1)];slots[j]=-1;
   if(u<0 || u>=adjacent.shape(0))throw std::invalid_argument("component root out of range");
   for(int k=0;k<3;++k)if(a[3*u+k]==v)slots[j]=3*u+k;
   if(slots[j]<0 || l[slots[j]]<=0)throw std::invalid_argument("missing directed component");
  }
  int total=0;for(int rank=0;rank<K;++rank)for(int j=0;j<4;++j)
   if(rank<l[slots[j]] && total<std::min(n,K)){chosen[i][j][counts[i][j]++]=m[slots[j]*K+rank];++total;}
  int product=1;for(int j=0;j<4;++j){if(counts[i][j]==0)throw std::logic_error("empty sampled component");product*=counts[i][j];}
  off[i+1]=off[i]+product;
 }
 py::array_t<int64_t> quartet(std::vector<py::ssize_t>{off[batch],4});
 py::array_t<int64_t> classes(std::vector<py::ssize_t>{off[batch],3});
 auto q=quartet.mutable_data();auto c=classes.mutable_data();
 constexpr int cls[4][4]={{-1,0,1,2},{0,-1,2,1},{1,2,-1,0},{2,1,0,-1}};
 for(size_t i=0;i<batch;++i){int64_t row=off[i];auto &x=chosen[i];auto &ct=counts[i];
  for(int aa=0;aa<ct[0];++aa)for(int bb=0;bb<ct[1];++bb)for(int cc=0;cc<ct[2];++cc)for(int dd=0;dd<ct[3];++dd){
   std::array<int,4> value={x[0][aa],x[1][bb],x[2][cc],x[3][dd]},rank{};
   for(int j=0;j<4;++j)for(int k=0;k<4;++k)rank[j]+=value[j]>value[k];
   for(int j=0;j<3;++j)c[3*row+j]=cls[rank[0]][rank[j+1]];
   std::sort(value.begin(),value.end());for(int j=0;j<4;++j)q[4*row+j]=value[j];++row;
  }
 }
 return py::make_tuple(quartet,classes,offsets);
}

PYBIND11_MODULE(_uniform_plan,m){m.def("components",&components);m.def("queries",&queries);}
