// Integer RMQ index export and original lexicographic FP64 reduction stream.
#include "active_rmq_index.hpp"
#include <cmath>
using A32=py::array_t<int32_t,py::array::c_style|py::array::forcecast>;
py::dict export_index(AI tree,int n){
 RmqIndex idx(tree,n);int V=idx.V,W=idx.rmq[0].size();
 py::array_t<int32_t> nodes({8,V}),rmq({int(idx.rmq.size()),W});
 auto a=nodes.mutable_unchecked<2>();auto r=rmq.mutable_unchecked<2>();
 py::array_t<int64_t> edges({n-3,2}),branches({n-3,4});auto ee=edges.mutable_unchecked<2>();auto bb=branches.mutable_unchecked<2>();
 for(int u=0;u<V;u++){
  a(0,u)=idx.tin[u];a(1,u)=idx.tout[u];a(2,u)=idx.parent[u];a(3,u)=idx.depth[u];a(4,u)=idx.first[u];
  a(5,u)=a(6,u)=-1;a(7,u)=-1;int k=5;
  for(int v:idx.g[u])if(v!=idx.parent[u]){if(k<7)a(k++,u)=v;}
 }
 int e=0;for(int u=n;u<V;u++)for(int v:idx.g[u])if(v>u){
  ee(e,0)=u;ee(e,1)=v;int j=0;
  for(int z:idx.g[u])if(z!=v)bb(e,j++)=z;
  for(int z:idx.g[v])if(z!=u)bb(e,j++)=z;
  a(7,idx.parent[u]==v?u:v)=e++;
 }
 for(int k=0;k<int(idx.rmq.size());k++)for(int i=0;i<W;i++)r(k,i)=i<int(idx.rmq[k].size())?idx.rmq[k][i]:n;
 py::dict out;out["nodes"]=nodes;out["rmq"]=rmq;out["edges"]=edges;out["branches"]=branches;return out;
}
py::dict reduce(A32 codes,AD weights,int edge_count){
 auto c=codes.unchecked<1>();auto w=weights.unchecked<2>();
 if(w.shape(0)!=c.shape(0)||w.shape(1)!=3||edge_count<1)throw std::runtime_error("invalid dimensions");
 py::array_t<double> score({edge_count,3});auto s=score.mutable_unchecked<2>();
 py::array_t<int64_t> classes(c.shape(0));auto cls=classes.mutable_unchecked<1>();
 for(int e=0;e<edge_count;e++)for(int j=0;j<3;j++)s(e,j)=0;
 double objective=0.;int active=0;
 {py::gil_scoped_release release;
  for(int64_t i=0;i<c.shape(0);i++){
   int code=c(i),state=code&3,alt=(code>>2)&3,e=(code>>4)-1;
   if(code<0||state>2||e>=edge_count||(e>=0&&(alt>2||alt==state)))throw std::runtime_error("invalid integer topology code");
   cls(i)=state;objective+=w(i,state);
   if(e<0)continue;
   s(e,1)+=w(i,alt)-w(i,state);s(e,2)+=w(i,3-state-alt)-w(i,state);active++;
  }
 }
 py::dict result;result["scores"]=score;result["objective"]=objective;result["active_quartets"]=active;result["classes"]=classes;return result;
}
PYBIND11_MODULE(_msc_active_gpu,m){m.def("export_index",&export_index);m.def("reduce",&reduce);}
