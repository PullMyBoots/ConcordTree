#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <array>
#include <vector>
#include <stdexcept>
#include <limits>
namespace py=pybind11;
using Array=py::array_t<int64_t,py::array::c_style>;
void shape(const Array& a){if(a.ndim()!=2 || a.shape(1)!=3 || !a.writeable())throw std::invalid_argument("writable contiguous adjacency required");}
void add(int64_t* a,int m,int u,int v){if(u<0||v<0||u>=m||v>=m||u==v)throw std::invalid_argument("invalid topology endpoint");auto* r=a+u*3;for(int j=0;j<3;++j)if(r[j]==v)throw std::invalid_argument("duplicate topology edge");auto pos=std::find(r,r+3,-1);if(pos==r+3)throw std::invalid_argument("topology degree overflow");*pos=v;std::sort(r,r+3,[](int64_t a,int64_t b){return a>=0&&(b<0||a<b);});}
void attach(Array a,Array triples){shape(a);if(triples.ndim()!=2 || triples.shape(1)!=3)throw std::invalid_argument("merge triple shape");auto* g=a.mutable_data();int m=a.shape(0);const auto* t=triples.data();py::gil_scoped_release release;for(int i=0;i<triples.shape(0);++i){int u=t[i*3],v=t[i*3+1],w=t[i*3+2];add(g,m,w,u);add(g,m,w,v);add(g,m,u,w);add(g,m,v,w);}}
void center(Array a,Array active,int w){shape(a);if(active.ndim()!=1||active.size()!=3)throw std::invalid_argument("terminal three expected");auto* g=a.mutable_data();int m=a.shape(0);py::gil_scoped_release release;for(int j=0;j<3;++j){int v=active.data()[j];add(g,m,w,v);add(g,m,v,w);}}
py::dict export_graph(Array a,int n,bool canonical){shape(a);int m=a.shape(0);if(n<3||m!=2*n-2)throw std::invalid_argument("binary topology dimensions");const auto*g=a.data();std::vector<int> parent(m,-2),order,minimum(m,std::numeric_limits<int>::max()),label(m,-1);order.reserve(m);parent[0]=-1;order.push_back(0);
 {py::gil_scoped_release release;
 for(int u=0;u<m;++u){int count=0;for(int j=0;j<3;++j){int v=g[u*3+j];if(v<0)continue;if(v>=m||v==u||std::count(g+v*3,g+v*3+3,u)!=1)throw std::invalid_argument("asymmetric topology");++count;}if(count!=(u<n?1:3))throw std::invalid_argument("binary degree mismatch");}
 for(size_t k=0;k<order.size();++k){int u=order[k];for(int j=0;j<3;++j){int v=g[u*3+j];if(v<0||v==parent[u])continue;if(parent[v]!=-2)throw std::invalid_argument("cyclic topology");parent[v]=u;order.push_back(v);}}
 if(order.size()!=size_t(m))throw std::invalid_argument("disconnected topology");
 for(auto it=order.rbegin();it!=order.rend();++it){int u=*it;if(u<n)minimum[u]=u;else for(int j=0;j<3;++j){int v=g[u*3+j];if(v!=parent[u])minimum[u]=std::min(minimum[u],minimum[v]);}}
 if(canonical){std::vector<int> stack{0};int next=n;while(!stack.empty()){int u=stack.back();stack.pop_back();label[u]=u<n?u:next++;std::array<int,3> children;int count=0;for(int j=0;j<3;++j){int v=g[u*3+j];if(v>=0&&v!=parent[u])children[count++]=v;}std::sort(children.begin(),children.begin()+count,[&](int a,int b){return minimum[a]>minimum[b];});for(int j=0;j<count;++j)stack.push_back(children[j]);}if(next!=m)throw std::runtime_error("canonical internal count");}else for(int u=0;u<m;++u)label[u]=u;
 }
 py::dict result;for(int u=0;u<m;++u){py::set adjacent;for(int j=0;j<3;++j){int v=g[u*3+j];if(v>=0)adjacent.add(py::int_(label[v]));}result[py::int_(label[u])]=adjacent;}return result;
}
PYBIND11_MODULE(_ct_apnj_topology,m){m.def("attach",&attach);m.def("connect_center",&center);m.def("export_graph",&export_graph);}
