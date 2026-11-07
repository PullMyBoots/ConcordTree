#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <functional>
namespace py=pybind11;
using AI=py::array_t<int64_t,py::array::c_style|py::array::forcecast>;
using AD=py::array_t<double,py::array::c_style|py::array::forcecast>;

struct Index {
 int n,V,K;std::vector<std::vector<int>> g,up;std::vector<int> depth,parent,tin,tout,order;
 Index(AI edges,int N):n(N),V(2*N-2),K(1),g(V),depth(V),parent(V,-1),tin(V),tout(V){
  auto e=edges.unchecked<2>();
  if(e.shape(1)!=2 || e.shape(0)!=V-1)throw std::runtime_error("invalid binary tree edges");
  for(int i=0;i<e.shape(0);i++){int a=e(i,0),b=e(i,1);if(a<0||b<0||a>=V||b>=V||a==b)throw std::runtime_error("invalid node id");g[a].push_back(b);g[b].push_back(a);}
  for(int i=0;i<V;i++){auto &a=g[i];std::sort(a.begin(),a.end());if(a.size()!=(i<n?1:3)||std::adjacent_find(a.begin(),a.end())!=a.end())throw std::runtime_error("invalid binary degree");}
  while((1<<K)<=V)K++;
  up.assign(K,std::vector<int>(V,n));
  std::vector<std::pair<int,bool>> stack{{n,false}};int tick=0;parent[n]=n;
  while(!stack.empty()){
   auto [u,done]=stack.back();stack.pop_back();
   if(done){tout[u]=tick;continue;}tin[u]=tick++;order.push_back(u);
   up[0][u]=parent[u];for(int k=1;k<K;k++)up[k][u]=up[k-1][up[k-1][u]];
   stack.push_back({u,true});
   for(auto it=g[u].rbegin();it!=g[u].rend();it++)if(*it!=parent[u]){if(parent[*it]!=-1)throw std::runtime_error("cycle");parent[*it]=u;depth[*it]=depth[u]+1;stack.push_back({*it,false});}
  }
  if(tick!=V)throw std::runtime_error("disconnected");
 }
 bool ancestor(int a,int b)const{return tin[a]<=tin[b]&&tin[b]<tout[a];}
 int lca(int a,int b)const{
  if(ancestor(a,b))return a;if(ancestor(b,a))return b;
  for(int k=K-1;k>=0;k--)if(!ancestor(up[k][a],b))a=up[k][a];return parent[a];
 }
 int dist(int a,int b)const{return depth[a]+depth[b]-2*depth[lca(a,b)];}
 int median(int a,int b,int c)const{
  int u=lca(a,b),v=lca(a,c),w=lca(b,c);
  return depth[u]>=depth[v]?(depth[u]>=depth[w]?u:w):(depth[v]>=depth[w]?v:w);
 }
 int step(int center,int leaf)const{
  if(!ancestor(center,leaf))return parent[center];
  int d=depth[leaf]-depth[center]-1;
  for(int k=0;k<K;k++)if(d&(1<<k))leaf=up[k][leaf];return leaf;
 }
};

int pairclass(int a,int b){if(a>b)std::swap(a,b);if((a==0&&b==1)||(a==2&&b==3))return 0;if((a==0&&b==2)||(a==1&&b==3))return 1;return 2;}

py::dict proposals(AI tree,int n,AI queries,AD weights){
 Index idx(tree,n);auto q=queries.unchecked<2>();auto w=weights.unchecked<2>();
 if(q.shape(1)!=4||w.shape(0)!=q.shape(0)||w.shape(1)!=3)throw std::runtime_error("invalid quartet/weight dimensions");
 std::vector<std::vector<int>> incident(n);
 std::vector<int> klass(q.shape(0));double objective=0.;
 for(int i=0;i<q.shape(0);i++){
  int a=q(i,0),b=q(i,1),c=q(i,2),d=q(i,3);
  if(a<0||d>=n||!(a<b&&b<c&&c<d))throw std::runtime_error("canonical four distinct leaves required");
  for(int s=0;s<3;s++)if(!std::isfinite(w(i,s)))throw std::runtime_error("finite weights required");
  int s0=idx.dist(a,b)+idx.dist(c,d),s1=idx.dist(a,c)+idx.dist(b,d),s2=idx.dist(a,d)+idx.dist(b,c);
  klass[i]=(s0<s1&&s0<s2)?0:((s1<s2)?1:2);objective+=w(i,klass[i]);
  for(int k=0;k<4;k++)incident[q(i,k)].push_back(i);
 }
 py::array_t<int64_t> edits({n,4});auto out=edits.mutable_unchecked<2>();
 py::array_t<double> gains(n);auto gain=gains.mutable_unchecked<1>();
 for(int x=0;x<n;x++){for(int j=0;j<4;j++)out(x,j)=-1;gain(x)=0.;}
 {
 py::gil_scoped_release release;
 for(int x=0;x<n;x++){
  if(incident[x].empty())continue;
  std::vector<std::pair<int,double>> events;events.reserve(incident[x].size()*10+2);
  auto add=[&](int a,int b,double value){if(a<b&&value!=0.){events.emplace_back(a,value);events.emplace_back(b,-value);}};
  double before=0.;
  for(int qi:incident[x]){
   int pos=0,other[3],op[3],k=0;
   for(int j=0;j<4;j++)if(q(qi,j)==x)pos=j;else {other[k]=q(qi,j);op[k++]=j;}
   int median=idx.median(other[0],other[1],other[2]);before+=w(qi,klass[qi]);
   for(int j=0;j<3;j++){
    int dir=idx.step(median,other[j]);double value=w(qi,pairclass(pos,op[j]));
    if(idx.parent[dir]==median)add(idx.tin[dir],idx.tout[dir],value);
    else {add(0,idx.V,value);add(idx.tin[median],idx.tout[median],-value);add(idx.tin[median],idx.tin[median]+1,value);}
   }
  }
  events.emplace_back(0,0.);events.emplace_back(idx.V,0.);
  std::sort(events.begin(),events.end(),[](auto a,auto b){return a.first<b.first;});
  double running=0.,best=before;int target=-1;
  for(size_t i=0;i<events.size();){
   int at=events[i].first;do{running+=events[i++].second;}while(i<events.size()&&events[i].first==at);
   int end=i<events.size()?events[i].first:idx.V;
   int p=at;while(p<end&&(idx.order[p]==idx.n||idx.order[p]==x))p++;
   if(p<end&&running>best){best=running;target=idx.order[p];}
  }
  if(target<0)continue;
  int u=idx.g[x][0],a=-1,b=-1;
  for(int v:idx.g[u])if(v!=x){if(a<0)a=v;else b=v;}
  int c=target,d=idx.parent[target];
  // Both old edges through u represent the same suppressed current edge.
  if(c==u||d==u){c=a;d=b;}
  if(c==d||c==x||d==x||c==u||d==u)throw std::runtime_error("invalid suppressed target");
  if((c==a&&d==b)||(c==b&&d==a))continue;
  out(x,0)=x;out(x,1)=u;out(x,2)=std::min(c,d);out(x,3)=std::max(c,d);gain(x)=best-before;
 }
 }
 py::dict r;r["edits"]=edits;r["gains"]=gains;r["objective"]=objective;return r;
}

struct HeavyIndex:Index {
 std::vector<int> size,heavy,head,pos,end,at;
 HeavyIndex(AI tree,int n):Index(tree,n),size(V,1),heavy(V,-1),head(V),pos(V),end(V),at(V){
  for(auto it=order.rbegin();it!=order.rend();it++){
   int u=*it;for(int v:g[u])if(parent[v]==u&&v!=u){size[u]+=size[v];if(heavy[u]<0||size[v]>size[heavy[u]])heavy[u]=v;}
  }
  int tick=0;std::vector<std::pair<int,int>> todo{{n,n}};
  while(!todo.empty()){
   auto [u,h]=todo.back();todo.pop_back();head[u]=h;pos[u]=tick;at[tick++]=u;
   for(auto it=g[u].rbegin();it!=g[u].rend();it++)if(parent[*it]==u&&*it!=u&&*it!=heavy[u])todo.emplace_back(*it,*it);
   if(heavy[u]>=0)todo.emplace_back(heavy[u],h);
  }
  if(tick!=V)throw std::runtime_error("HLD traversal incomplete");
  for(int u=0;u<V;u++)end[u]=pos[u]+size[u];
 }
 template<class F> void path(int a,int b,F f,bool exclude_last=false)const{
  while(head[a]!=head[b]){
   if(depth[head[a]]<depth[head[b]])std::swap(a,b);
   f(pos[head[a]],pos[a]+1);a=parent[head[a]];
  }
  if(depth[a]<depth[b])std::swap(a,b);
  int first=pos[b]+(exclude_last?1:0);if(first<=pos[a])f(first,pos[a]+1);
 }
};

struct Maximum {
 int n;std::vector<double> value,lazy;std::vector<int> arg;
 Maximum(int N):n(N),value(4*N,0.),lazy(4*N,0.),arg(4*N){build(1,0,n);}
 void build(int v,int l,int r){arg[v]=l;if(r-l>1){int m=(l+r)/2;build(2*v,l,m);build(2*v+1,m,r);}}
 void push(int v){if(lazy[v]!=0.){for(int c:{v*2,v*2+1}){value[c]+=lazy[v];lazy[c]+=lazy[v];}lazy[v]=0.;}}
 void pull(int v){int a=2*v,b=a+1;int k=value[a]>=value[b]?a:b;value[v]=value[k];arg[v]=arg[k];}
 void add(int l,int r,double x){if(l<r&&x!=0.)add(1,0,n,l,r,x);}
 void add(int v,int a,int b,int l,int r,double x){
  if(l<=a&&b<=r){value[v]+=x;lazy[v]+=x;return;}
  push(v);int m=(a+b)/2;if(l<m)add(v*2,a,m,l,r,x);if(r>m)add(v*2+1,m,b,l,r,x);pull(v);
 }
 std::pair<double,int> query(int l,int r){if(l>=r)return {-INFINITY,-1};return query(1,0,n,l,r);}
 std::pair<double,int> query(int v,int a,int b,int l,int r){
  if(l<=a&&b<=r)return {value[v],arg[v]};push(v);int m=(a+b)/2;
  auto x=l<m?query(v*2,a,m,l,r):std::pair<double,int>{-INFINITY,-1};
  auto y=r>m?query(v*2+1,m,b,l,r):std::pair<double,int>{-INFINITY,-1};
  return x.first>=y.first?x:y;
 }
};

struct Event {int l,r;double value;};

py::dict rooted_proposals(AI tree,int n,AI queries,AD weights){
 HeavyIndex idx(tree,n);auto q=queries.unchecked<2>();auto w=weights.unchecked<2>();
 if(q.shape(1)!=4||w.shape(0)!=q.shape(0)||w.shape(1)!=3)throw std::runtime_error("invalid quartet/weight dimensions");
 py::array_t<int64_t> edits({idx.V,4});auto out=edits.mutable_unchecked<2>();
 py::array_t<double> gains(idx.V);auto gain=gains.mutable_unchecked<1>();
 for(int x=0;x<idx.V;x++){for(int j=0;j<4;j++)out(x,j)=-1;gain(x)=0.;}
 std::vector<std::vector<Event>> events(idx.V+1);int64_t eventcount=0;double objective=0.;
 std::vector<int64_t> selected;
 {
 py::gil_scoped_release release;
 for(int qi=0;qi<q.shape(0);qi++){
  int a=q(qi,0),b=q(qi,1),c=q(qi,2),d=q(qi,3);
  if(a<0||d>=n||!(a<b&&b<c&&c<d))throw std::runtime_error("canonical four distinct leaves required");
  for(int s=0;s<3;s++)if(!std::isfinite(w(qi,s)))throw std::runtime_error("finite weights required");
  int s0=idx.dist(a,b)+idx.dist(c,d),s1=idx.dist(a,c)+idx.dist(b,d),s2=idx.dist(a,d)+idx.dist(b,c);
  int state=(s0<s1&&s0<s2)?0:((s1<s2)?1:2);objective+=w(qi,state);
  for(int k=0;k<4;k++){
   int leaf=q(qi,k),other[3],op[3],j=0,stop=idx.n;
   for(int h=0;h<4;h++)if(h!=k){other[j]=q(qi,h);op[j++]=h;int l=idx.lca(leaf,q(qi,h));if(idx.depth[l]>idx.depth[stop])stop=l;}
   int median=idx.median(other[0],other[1],other[2]);
   std::vector<Event> targets;
   for(int h=0;h<3;h++){
    double delta=w(qi,pairclass(k,op[h]))-w(qi,state);if(delta==0.)continue;
    int dir=idx.step(median,other[h]);
    if(idx.parent[dir]==median)targets.push_back({idx.pos[dir],idx.end[dir],delta});
    else {
     if(idx.pos[median]+1>1)targets.push_back({1,idx.pos[median]+1,delta});
     if(idx.end[median]<idx.V)targets.push_back({idx.end[median],idx.V,delta});
    }
   }
   idx.path(leaf,stop,[&](int l,int r){
    for(auto e:targets){events[l].push_back(e);events[r].push_back({e.l,e.r,-e.value});eventcount+=2;}
   },true);
  }
 }
 Maximum score(idx.V);
 for(int position=0;position<idx.V;position++){
  for(auto e:events[position])score.add(e.l,e.r,e.value);
  int x=idx.at[position];if(x==idx.n)continue;
  auto left=score.query(1,idx.pos[x]),right=score.query(idx.end[x],idx.V);
  auto best=left.first>=right.first?left:right;if(!(best.first>0.)||best.second<0)continue;
  int p=idx.parent[x],a=-1,b=-1;for(int v:idx.g[p])if(v!=x){if(a<0)a=v;else b=v;}
  int c=idx.at[best.second],d=idx.parent[c];
  if(c==p||d==p){c=a;d=b;}
  if((c==a&&d==b)||(c==b&&d==a))continue;
  if(c==x||d==x||c==p||d==p||idx.ancestor(x,c)||idx.ancestor(x,d))throw std::runtime_error("invalid rooted SPR target");
  out(x,0)=x;out(x,1)=p;out(x,2)=std::min(c,d);out(x,3)=std::max(c,d);gain(x)=best.first;
 }
 // Vertex-disjoint affected paths guarantee structural legality of a batch.
 std::vector<int> ranking;for(int x=0;x<idx.V;x++)if(gain(x)>0.)ranking.push_back(x);
 std::sort(ranking.begin(),ranking.end(),[&](int a,int b){return gain(a)!=gain(b)?gain(a)>gain(b):a<b;});
 Maximum occupied(idx.V);
 for(int x:ranking){
  int p=out(x,1),c=out(x,2),d=out(x,3);std::vector<std::pair<int,int>> ranges;
  idx.path(p,c,[&](int l,int r){ranges.emplace_back(l,r);});
  for(int v:{x,d})ranges.emplace_back(idx.pos[v],idx.pos[v]+1);
  for(int v:idx.g[p])ranges.emplace_back(idx.pos[v],idx.pos[v]+1);
  bool conflict=false;for(auto [l,r]:ranges)if(occupied.query(l,r).first>0.){conflict=true;break;}
  if(conflict)continue;selected.push_back(x);for(auto [l,r]:ranges)occupied.add(l,r,1.);
 }
 }
 py::array_t<int64_t> batch(selected.size());std::copy(selected.begin(),selected.end(),batch.mutable_data());
 py::dict r;r["edits"]=edits;r["gains"]=gains;r["objective"]=objective;r["batch"]=batch;r["range_events"]=eventcount;return r;
}

PYBIND11_MODULE(_msc_spr,m){m.def("proposals",&proposals);m.def("rooted_proposals",&rooted_proposals);}
