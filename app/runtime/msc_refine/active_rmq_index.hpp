// Integer-only acceleration of the unchanged quartet topology queries.
#pragma once
#include "active_index.hpp"

struct RmqIndex : Index {
 std::vector<int> first;
 std::vector<std::vector<int>> rmq;

 RmqIndex(AI edges, int n) : Index(edges,n), first(V,-1) {
  // Iterative Euler tour is safe for maximally unbalanced trees.
  std::vector<int> tour; tour.reserve(2*V-1);
  std::vector<std::pair<int,size_t>> stack{{n,0}};
  first[n]=0; tour.push_back(n);
  while(!stack.empty()) {
   int u=stack.back().first;
   size_t &next=stack.back().second;
   while(next<g[u].size() && g[u][next]==parent[u]) next++;
   if(next==g[u].size()) {
    stack.pop_back();
    if(!stack.empty()) tour.push_back(stack.back().first);
   } else {
    int v=g[u][next++];
    first[v]=tour.size(); tour.push_back(v); stack.push_back({v,0});
   }
  }
  rmq.push_back(std::move(tour));
  for(size_t width=2;width<=rmq[0].size();width*=2) {
   size_t half=width/2, count=rmq[0].size()-width+1;
   std::vector<int> row(count);
   const auto &previous=rmq.back();
   for(size_t i=0;i<count;i++) {
    int a=previous[i],b=previous[i+half];
    row[i]=depth[a]<=depth[b]?a:b;
   }
   rmq.push_back(std::move(row));
  }
 }

 int lca(int a,int b) const {
  unsigned left=first[a],right=first[b];
  if(left>right) std::swap(left,right);
  unsigned length=right-left+1;
  unsigned level=31-__builtin_clz(length);
  int u=rmq[level][left],v=rmq[level][right-(1u<<level)+1];
  return depth[u]<=depth[v]?u:v;
 }

 int step(int center,int leaf) const {
  if(!ancestor(center,leaf)) return parent[center];
  // A binary tree has at most three incident directions. Test their DFS
  // intervals directly rather than lifting the leaf through log(N) levels.
  for(int v:g[center]) if(v!=parent[center] && ancestor(v,leaf)) return v;
  throw std::runtime_error("no direction from center to distinct leaf");
 }
};
