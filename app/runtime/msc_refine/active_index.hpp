// Exact index implementation copied from project sparse_backend.cpp.
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
namespace py=pybind11;
using AI=py::array_t<int64_t,py::array::c_style|py::array::forcecast>;
using AD=py::array_t<double,py::array::c_style|py::array::forcecast>;

struct Index {
 int n,V,K;std::vector<std::vector<int>> g,up;std::vector<int> depth,parent,tin,tout;
 Index(AI edges,int N):n(N),V(2*N-2),K(1),g(V),depth(V),parent(V,-1),tin(V),tout(V){
  auto e=edges.unchecked<2>();
  if(e.shape(1)!=2 || e.shape(0)!=V-1)throw std::runtime_error("invalid binary tree edges");
  for(int i=0;i<e.shape(0);i++){int a=e(i,0),b=e(i,1);if(a<0||b<0||a>=V||b>=V)throw std::runtime_error("invalid node id");g[a].push_back(b);g[b].push_back(a);}
  for(auto &a:g)std::sort(a.begin(),a.end());
  while((1<<K)<=V)K++;
  up.assign(K,std::vector<int>(V,n));
  std::vector<std::pair<int,bool>> stack{{n,false}};int tick=0;parent[n]=n;
  while(!stack.empty()){
   auto [u,done]=stack.back();stack.pop_back();
   if(done){tout[u]=tick;continue;}tin[u]=tick++;
   up[0][u]=parent[u];for(int k=1;k<K;k++)up[k][u]=up[k-1][up[k-1][u]];
   stack.push_back({u,true});
   for(auto it=g[u].rbegin();it!=g[u].rend();it++)if(*it!=parent[u]){if(parent[*it]!=-1)throw std::runtime_error("cycle");parent[*it]=u;depth[*it]=depth[u]+1;stack.push_back({*it,false});}
  }
  if(tick!=V)throw std::runtime_error("disconnected");
 }
 bool ancestor(int a,int b)const{return tin[a]<=tin[b] && tin[b]<tout[a];}
 int lca(int a,int b)const{
  if(ancestor(a,b))return a;if(ancestor(b,a))return b;
  for(int k=K-1;k>=0;k--)if(!ancestor(up[k][a],b))a=up[k][a];return parent[a];
 }
 int dist(int a,int b)const{return depth[a]+depth[b]-2*depth[lca(a,b)];}
 int median(int a,int b,int c)const{
  int u=lca(a,b),v=lca(a,c),w=lca(b,c);
  return depth[u]>=depth[v] ? (depth[u]>=depth[w]?u:w):(depth[v]>=depth[w]?v:w);
 }
 int step(int center,int leaf)const{
  if(!ancestor(center,leaf))return parent[center];
  int d=depth[leaf]-depth[center]-1;
  for(int k=0;k<K;k++)if(d&(1<<k))leaf=up[k][leaf];return leaf;
 }
};

int pairclass(int a,int b){if(a>b)std::swap(a,b);if((a==0&&b==1)||(a==2&&b==3))return 0;if((a==0&&b==2)||(a==1&&b==3))return 1;return 2;}

