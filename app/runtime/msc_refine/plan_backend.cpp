// Exact native counterpart of core.compile_plan. No statistical decisions.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
namespace py=pybind11;

py::dict compile(py::dict graph,int n){
    if(n<4 || n>(std::numeric_limits<int>::max()-2)/2)
        throw std::invalid_argument("invalid binary tree taxon count");
    const int total=2*n-2,ecount=n-3;
    if(static_cast<int>(graph.size())!=total)
        throw std::invalid_argument("contiguous binary tree IDs required");
    std::vector<std::array<int,3>> adj(total);
    std::vector<int> degree(total,0);
    for(auto item:graph){
        int u=py::cast<int>(item.first);
        if(u<0 || u>=total || degree[u])throw std::invalid_argument("invalid node ID");
        for(auto vv:py::reinterpret_borrow<py::iterable>(item.second)){
            int v=py::cast<int>(vv);
            if(v<0 || v>=total || v==u || degree[u]>=3)
                throw std::invalid_argument("invalid adjacency");
            adj[u][degree[u]++]=v;
        }
        if(degree[u]!=(u<n?1:3))throw std::invalid_argument("fully binary tree required");
        std::sort(adj[u].begin(),adj[u].begin()+degree[u]);
        for(int i=1;i<degree[u];++i)if(adj[u][i]==adj[u][i-1])throw std::invalid_argument("duplicate neighbor");
    }
    for(int u=0;u<total;++u){
        if(degree[u]!=(u<n?1:3))throw std::invalid_argument("missing node");
        for(int i=0;i<degree[u];++i){int v=adj[u][i];
            if(!std::binary_search(adj[v].begin(),adj[v].begin()+degree[v],u))
                throw std::invalid_argument("asymmetric adjacency");
        }
    }
    std::vector<int> parent(total,-2),left(total),right(total),order;
    std::vector<std::pair<int,bool>> stack;stack.reserve(2*total);order.reserve(n);
    parent[n]=-1;stack.emplace_back(n,false);int visited=0;
    while(!stack.empty()){
        auto [u,done]=stack.back();stack.pop_back();
        if(done){right[u]=static_cast<int>(order.size());continue;}
        ++visited;left[u]=static_cast<int>(order.size());
        if(u<n){order.push_back(u);right[u]=static_cast<int>(order.size());continue;}
        stack.emplace_back(u,true);
        for(int i=degree[u]-1;i>=0;--i){int v=adj[u][i];if(v==parent[u])continue;
            if(parent[v]!=-2)throw std::invalid_argument("cycle");
            parent[v]=u;stack.emplace_back(v,false);
        }
    }
    if(visited!=total || static_cast<int>(order.size())!=n)throw std::invalid_argument("disconnected tree");
    py::array_t<int64_t> leaves(n);
    py::array_t<int64_t> edges(std::vector<py::ssize_t>{ecount,2});
    py::array_t<int64_t> branches(std::vector<py::ssize_t>{ecount,4});
    py::array_t<int64_t> intervals(std::vector<py::ssize_t>{ecount,4,2});
    py::array_t<bool> complement(std::vector<py::ssize_t>{ecount,4});
    py::array_t<int64_t> sizes(std::vector<py::ssize_t>{ecount,4});
    auto lp=leaves.mutable_data();auto ep=edges.mutable_data();auto bp=branches.mutable_data();
    auto ip=intervals.mutable_data();auto cp=complement.mutable_data();auto sp=sizes.mutable_data();
    for(int i=0;i<n;++i)lp[i]=order[i];
    int edge=0;
    for(int u=n;u<total;++u)for(int jj=0;jj<degree[u];++jj){int v=adj[u][jj];
        if(v<=u || v<n)continue;
        if(edge>=ecount)throw std::invalid_argument("too many internal edges");
        ep[2*edge]=u;ep[2*edge+1]=v;int parts[4],k=0;
        for(int i=0;i<3;++i)if(adj[u][i]!=v)parts[k++]=adj[u][i];
        for(int i=0;i<3;++i)if(adj[v][i]!=u)parts[k++]=adj[v][i];
        for(int g=0;g<4;++g){int center=g<2?u:v,neighbor=parts[g];
            bool rev=parent[center]==neighbor;int node=rev?center:neighbor;
            bp[4*edge+g]=neighbor;cp[4*edge+g]=rev;
            ip[8*edge+2*g]=left[node];ip[8*edge+2*g+1]=right[node];
            int size=right[node]-left[node];sp[4*edge+g]=rev?n-size:size;
        }
        ++edge;
    }
    if(edge!=ecount)throw std::invalid_argument("incorrect internal edge count");
    py::dict result;
    result["leaf_order"]=leaves;result["edges"]=edges;result["branches"]=branches;
    result["intervals"]=intervals;result["complement"]=complement;result["sizes"]=sizes;
    return result;
}

py::array_t<int64_t> byte_histogram(py::array_t<uint8_t,0> input){
    auto info=input.request();
    if(info.ndim!=2)throw std::invalid_argument("two-dimensional uint8 packed input required");
    py::array_t<int64_t> result(256);auto out=result.mutable_data();
    std::fill(out,out+256,0);
    const auto* data=static_cast<const uint8_t*>(info.ptr);
    {
        py::gil_scoped_release release;
        for(py::ssize_t r=0;r<info.shape[0];++r){
            const auto* row=data+r*info.strides[0];
            for(py::ssize_t c=0;c<info.shape[1];++c)++out[row[c*info.strides[1]]];
        }
    }
    return result;
}

PYBIND11_MODULE(_msc_plan,m){
    m.def("compile",&compile,"Exact linear-state Euler NNI plan");
    m.def("byte_histogram",&byte_histogram,py::arg("input").noconvert(),
          "Exact byte counts without an int64 input copy; supports strided views");
}
