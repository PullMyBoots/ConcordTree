#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
namespace py=pybind11;
py::tuple solve(py::array_t<double,py::array::c_style> matrix,
                py::array_t<int64_t,py::array::c_style> node_ids,
                py::array_t<int64_t,py::array::c_style> tie_ranks,int64_t next_node){
    if(matrix.ndim()!=2 || matrix.shape(0)!=matrix.shape(1) || matrix.shape(0)<4 || matrix.shape(0)>512 ||
       node_ids.ndim()!=1 || tie_ranks.ndim()!=1 || node_ids.size()!=matrix.shape(0) || tie_ranks.size()!=node_ids.size() ||
       next_node<0 || next_node>std::numeric_limits<int64_t>::max()-512)
        throw std::invalid_argument("terminal NJ requires a bounded4..512 square matrix and matching IDs/ranks");
    const int m=node_ids.size();
    std::vector<double> d(matrix.data(),matrix.data()+matrix.size()),row(m);
    std::vector<int64_t> ids(node_ids.data(),node_ids.data()+m),ranks(tie_ranks.data(),tie_ranks.data()+m);
    std::vector<int> active(m);for(int i=0;i<m;++i)active[i]=i;
    for(int i=0;i<m;++i){
        if(ids[i]<0 || ids[i]>=next_node || d[size_t(i)*m+i]!=0)throw std::invalid_argument("terminal NJ ID/diagonal");
        for(int j=0;j<m;++j){double x=d[size_t(i)*m+j];
            if(!std::isfinite(x) || x!=d[size_t(j)*m+i])throw std::invalid_argument("terminal NJ matrix must be finite symmetric");
            row[i]+=x;
        }
    }
    py::array_t<int64_t> triples({py::ssize_t(m-3),py::ssize_t(3)}),tail(3);
    auto* out=triples.mutable_data();auto* remaining=tail.mutable_data();
    {py::gil_scoped_release release;
    int step=0;
    while(active.size()>3){
        const int size=active.size();double best=std::numeric_limits<double>::infinity();int a=-1,b=-1;
        std::array<int64_t,4> best_tie{};
        for(int x=0;x<size;++x)for(int y=x+1;y<size;++y){int i=active[x],j=active[y];
            double q=double(size-2)*d[size_t(i)*m+j];q-=row[i];q-=row[j];
            if(a>=0 && q>best)continue;
            int left=i,right=j;if(ranks[left]>ranks[right] || (ranks[left]==ranks[right] && ids[left]>ids[right]))std::swap(left,right);
            const std::array<int64_t,4> tie{ranks[left],ranks[right],ids[left],ids[right]};
            if(a<0 || q<best || (q==best && tie<best_tie)){best=q;a=left;b=right;best_tie=tie;}
        }
        out[step*3]=ids[a];out[step*3+1]=ids[b];out[step*3+2]=next_node++;
        const double joined=d[size_t(a)*m+b];double sum=0;
        for(int t:active){if(t==a || t==b)continue;
            const double old_a=d[size_t(a)*m+t],old_b=d[size_t(b)*m+t];
            const double fresh=.5*(old_a+old_b-joined);row[t]+=fresh-old_a-old_b;
            d[size_t(a)*m+t]=d[size_t(t)*m+a]=fresh;sum+=fresh;
        }
        row[a]=sum;d[size_t(a)*m+a]=0;ids[a]=out[step*3+2];ranks[a]=std::min(ranks[a],ranks[b]);
        active.erase(std::find(active.begin(),active.end(),b));++step;
    }
    for(int i=0;i<3;++i)remaining[i]=ids[active[i]];
    }
    return py::make_tuple(triples,tail);
}
PYBIND11_MODULE(_ct_apnj_terminal,m){m.def("solve",&solve);}
