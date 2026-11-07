#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

template<class Key> static void radix_sort_keys(std::vector<Key>&keys){
    if(keys.size()<4096){std::sort(keys.begin(),keys.end());return;}
    uint64_t upper=0;for(auto key:keys){if(key<0)throw std::invalid_argument("negative candidate key");upper=std::max(upper,uint64_t(key));}
    if(!upper)return;
    constexpr size_t width=2048,mask=width-1;
    std::vector<Key> scratch(keys.size());
    const int workers=keys.size()>=65536?std::min(16,omp_get_max_threads()):1;
    std::vector<std::array<size_t,width>> counts(workers),offsets(workers);
    for(unsigned shift=0;shift<64 && (upper>>shift);shift+=11){
        #pragma omp parallel num_threads(workers)
        {
            const int worker=omp_get_thread_num(),threads=omp_get_num_threads();
            const size_t first=keys.size()*worker/threads,last=keys.size()*(worker+1)/threads;
            auto& count=counts[worker];count.fill(0);
            for(size_t i=first;i<last;++i)++count[(uint64_t(keys[i])>>shift)&mask];
            #pragma omp barrier
            #pragma omp single
            {
                size_t position=0;
                for(size_t bucket=0;bucket<width;++bucket)for(int w=0;w<threads;++w){offsets[w][bucket]=position;position+=counts[w][bucket];}
            }
            auto& offset=offsets[worker];
            for(size_t i=first;i<last;++i){Key key=keys[i];scratch[offset[(uint64_t(key)>>shift)&mask]++]=key;}
        }
        keys.swap(scratch);
    }
}




template<class Key> py::array_t<int64_t> projection_pairs_typed(
    py::array_t<double, py::array::c_style | py::array::forcecast> projected,
    int window,
    int workers
) {
    if (projected.ndim() != 2 || projected.shape(0) < 2 || window <= 0 || workers <= 0) {
        throw std::runtime_error("invalid projected array, window, or worker count");
    }
    const int64_t n = projected.shape(0);
    const int64_t columns = projected.shape(1);
    const int64_t width = std::min<int64_t>(window, n - 1);
    const auto values = projected.unchecked<2>();
    for(py::ssize_t i=0;i<projected.size();++i)if(!std::isfinite(projected.data()[i]))throw std::invalid_argument("projected observations must be finite");
    std::vector<std::vector<Key>> column_keys(static_cast<size_t>(columns));
#ifdef _OPENMP
    omp_set_num_threads(workers);
#pragma omp parallel for schedule(static)
#endif
    for (int64_t column = 0; column < columns; ++column) {
        std::vector<int64_t> order(static_cast<size_t>(n));
        std::iota(order.begin(), order.end(), int64_t{0});
        std::sort(order.begin(), order.end(), [&](int64_t left, int64_t right) {
            if(values(left,column)<values(right,column))return true;
            if(values(right,column)<values(left,column))return false;
            return left<right;
        });
        auto& keys = column_keys[static_cast<size_t>(column)];
        keys.reserve(static_cast<size_t>(width * n - width * (width + 1) / 2));
        for (int64_t offset = 1; offset <= width; ++offset) {
            for (int64_t rank = 0; rank + offset < n; ++rank) {
                const int64_t one = order[static_cast<size_t>(rank)];
                const int64_t two = order[static_cast<size_t>(rank + offset)];
                keys.push_back(std::min(one, two) * n + std::max(one, two));
            }
        }
    }
    size_t total = 0;
    for (const auto& keys : column_keys) {
        total += keys.size();
    }
    std::vector<Key> keys;
    keys.reserve(total);
    for (auto& one_column : column_keys) {
        keys.insert(keys.end(), one_column.begin(), one_column.end());
    }
    radix_sort_keys(keys);
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    py::array_t<int64_t> output({static_cast<py::ssize_t>(keys.size()), py::ssize_t{2}});
    auto pairs = output.mutable_unchecked<2>();
    for (size_t row = 0; row < keys.size(); ++row) {
        pairs(static_cast<py::ssize_t>(row), 0) = keys[row] / n;
        pairs(static_cast<py::ssize_t>(row), 1) = keys[row] % n;
    }
    return output;
}



struct IncidenceRow {
    int64_t* first;
    int64_t* last;
    int64_t* begin() const {return first;}
    int64_t* end() const {return last;}
    size_t size() const {return last-first;}
    int64_t& operator[](size_t j) const {return first[j];}
};
struct CsrIncidence {
    std::vector<int64_t> offsets, entries;
    CsrIncidence(int64_t n,int64_t count,const int64_t* pairs):offsets(n+1,0),entries(size_t(count)*2){
        for(int64_t e=0;e<count;++e){int64_t a=pairs[e*2],b=pairs[e*2+1];
            if(a<0 || b<=a || b>=n)throw std::runtime_error("pair rows must be canonical and in range");
            ++offsets[a+1];++offsets[b+1];}
        for(int64_t i=0;i<n;++i)offsets[i+1]+=offsets[i];
        auto next=offsets;
        for(int64_t e=0;e<count;++e){entries[next[pairs[e*2]]++]=e;entries[next[pairs[e*2+1]]++]=e;}
    }
    IncidenceRow operator[](size_t i){return {entries.data()+offsets[i],entries.data()+offsets[i+1]};}
};


py::array_t<int64_t> select_directed_pairs(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> pair_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> distances,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tie_rank,
    int cap,
    int workers
) {
    if (pair_array.ndim() != 2 || pair_array.shape(1) != 2 ||
        distances.ndim() != 1 || distances.shape(0) != pair_array.shape(0) ||
        tie_rank.ndim() != 1 || cap <= 0 || workers <= 0) {
        throw std::runtime_error("invalid pair, distance, tie-rank, cap, or worker input");
    }
    const int64_t n = tie_rank.shape(0);
    const auto pairs = pair_array.unchecked<2>();
    const auto edge_distance = distances.unchecked<1>();
    const auto ranks = tie_rank.unchecked<1>();
    CsrIncidence incident(n,pair_array.shape(0),pair_array.data());
    for(py::ssize_t i=0;i<distances.size();++i)if(!std::isfinite(distances.data()[i]))throw std::invalid_argument("candidate scores must be finite");
#ifdef _OPENMP
    omp_set_num_threads(workers);
#pragma omp parallel for schedule(dynamic, 32)
#endif
    for (int64_t owner = 0; owner < n; ++owner) {
        auto rows = incident[static_cast<size_t>(owner)];
        std::partial_sort(rows.begin(), rows.begin()+std::min<size_t>(cap,rows.size()), rows.end(), [&](int64_t first, int64_t second) {
            if (edge_distance(first) != edge_distance(second)) {
                return edge_distance(first) < edge_distance(second);
            }
            const int64_t first_other =
                pairs(first, 0) == owner ? pairs(first, 1) : pairs(first, 0);
            const int64_t second_other =
                pairs(second, 0) == owner ? pairs(second, 1) : pairs(second, 0);
            return ranks(first_other) < ranks(second_other);
        });
    }
    std::vector<int64_t> starts(static_cast<size_t>(n + 1), 0);
    for (int64_t owner = 0; owner < n; ++owner) {
        starts[static_cast<size_t>(owner + 1)] = starts[static_cast<size_t>(owner)] +
            std::min<int64_t>(cap, incident[static_cast<size_t>(owner)].size());
    }
    py::array_t<int64_t> output(
        {static_cast<py::ssize_t>(starts.back()), py::ssize_t{2}}
    );
    auto selected = output.mutable_unchecked<2>();
    for (int64_t owner = 0; owner < n; ++owner) {
        const auto rows = incident[static_cast<size_t>(owner)];
        const int64_t count = std::min<int64_t>(cap, rows.size());
        for (int64_t local = 0; local < count; ++local) {
            const int64_t pair_row = rows[static_cast<size_t>(local)];
            const int64_t other = pairs(pair_row, 0) == owner
                ? pairs(pair_row, 1) : pairs(pair_row, 0);
            const int64_t output_row = starts[static_cast<size_t>(owner)] + local;
            selected(output_row, 0) = owner;
            selected(output_row, 1) = other;
        }
    }
    return output;
}


py::tuple select_bounded_nj_merges(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> pair_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> distances,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tie_rank,
    py::array_t<double, py::array::c_style | py::array::forcecast> row_sums,
    py::array_t<double, py::array::c_style | py::array::forcecast> offsets,
    int cap,
    int maximum_merges,
    int workers,
    py::array_t<int64_t,py::array::c_style> path_order,
    bool bounded,
    int minimum_divisor,
    bool q_cap
) {
    if (pair_array.ndim() != 2 || pair_array.shape(1) != 2 ||
        distances.ndim() != 1 || distances.shape(0) != pair_array.shape(0) ||
        tie_rank.ndim() != 1 || row_sums.ndim() != 1 || offsets.ndim() != 1 ||
        row_sums.shape(0) != tie_rank.shape(0) ||
        offsets.shape(0) != tie_rank.shape(0) ||
        cap <= 0 || maximum_merges <= 0 || workers <= 0) {
        throw std::runtime_error(
            "invalid pair, distance, rank, row-sum, offset, cap, merge, or worker input"
        );
    }
    const int64_t n = tie_rank.shape(0);
    if (n < 4) {
        throw std::runtime_error("ordered NJ fold requires at least four active nodes");
    }
    if(path_order.ndim()!=1 || path_order.size()!=n)throw std::runtime_error("path order shape mismatch");
    auto po=path_order.unchecked<1>();
    std::vector<int64_t> path_position(n,-1);
    for(int64_t i=0;i<n;++i){if(po(i)<0 || po(i)>=n || path_position[po(i)]!=-1)throw std::runtime_error("path is not a permutation");path_position[po(i)]=i;}
    const auto pairs = pair_array.unchecked<2>();
    const auto edge_distance = distances.unchecked<1>();
    const auto ranks = tie_rank.unchecked<1>();
    const auto rows = row_sums.unchecked<1>();
    const auto node_offsets = offsets.unchecked<1>();
    for(int64_t i=0;i<n;++i)if(!std::isfinite(rows(i)) || !std::isfinite(node_offsets(i)))throw std::invalid_argument("NJ row sums and offsets must be finite");
    for(py::ssize_t i=0;i<distances.size();++i)if(!std::isfinite(distances.data()[i]))throw std::invalid_argument("NJ distances must be finite");

    struct RankedEdge {
        int64_t pool_row;
        int64_t left;
        int64_t right;
        double q;
    };
    std::vector<std::array<int64_t, 3>> accepted;
    int64_t candidate_count = 0;
    int64_t maximum_degree = 0;
    int64_t reciprocal_count = 0;
    if(minimum_divisor<4)throw std::runtime_error("minimum divisor must be >=4");
    {
        py::gil_scoped_release release;
        CsrIncidence incident(n,pair_array.shape(0),pair_array.data());
#ifdef _OPENMP
        omp_set_num_threads(workers);
#pragma omp parallel for schedule(dynamic, 32)
#endif
        for (int64_t owner = 0; owner < n; ++owner) {
            auto owned_rows = incident[static_cast<size_t>(owner)];
            std::partial_sort(owned_rows.begin(), owned_rows.begin()+std::min<size_t>(cap,owned_rows.size()), owned_rows.end(), [&](int64_t first, int64_t second) {
                const int64_t other_first=pairs(first,0)==owner?pairs(first,1):pairs(first,0);
                const int64_t other_second=pairs(second,0)==owner?pairs(second,1):pairs(second,0);
                const double first_score=edge_distance(first)+(q_cap?node_offsets(other_first)-rows(other_first)/double(n-2):0.);
                const double second_score=edge_distance(second)+(q_cap?node_offsets(other_second)-rows(other_second)/double(n-2):0.);
                if(first_score!=second_score)return first_score<second_score;
                const int64_t first_other =
                    pairs(first, 0) == owner ? pairs(first, 1) : pairs(first, 0);
                const int64_t second_other =
                    pairs(second, 0) == owner ? pairs(second, 1) : pairs(second, 0);
                if (ranks(first_other) != ranks(second_other)) {
                    return ranks(first_other) < ranks(second_other);
                }
                return first_other < second_other;
            });
        }

        std::vector<uint8_t> retained(static_cast<size_t>(pair_array.shape(0)), 0);
        for (int64_t owner = 0; owner < n; ++owner) {
            const auto owned_rows = incident[static_cast<size_t>(owner)];
            const int64_t count = std::min<int64_t>(cap, owned_rows.size());
            maximum_degree = std::max(maximum_degree, count);
            for (int64_t local = 0; local < count; ++local) {
                retained[static_cast<size_t>(owned_rows[static_cast<size_t>(local)])] = 1;
            }
        }

        if(bounded){
            int found=0;
            for(int64_t i=0;i<pair_array.shape(0);++i){if(std::abs(path_position[pairs(i,0)]-path_position[pairs(i,1)])==1){retained[i]=1;++found;}}
            if(found!=n-1)throw std::runtime_error("candidate pool lost projection path");
        }
        std::vector<RankedEdge> candidates;
        candidates.reserve(static_cast<size_t>(pair_array.shape(0)));
        for (int64_t pool_row = 0; pool_row < pair_array.shape(0); ++pool_row) {
            if (!retained[static_cast<size_t>(pool_row)]) {
                continue;
            }
            int64_t left = pairs(pool_row, 0);
            int64_t right = pairs(pool_row, 1);
            if (ranks(right) < ranks(left)) {
                std::swap(left, right);
            }
            double adjusted = edge_distance(pool_row);
            adjusted = adjusted + node_offsets(left);
            adjusted = adjusted + node_offsets(right);
            double q = static_cast<double>(n - 2) * adjusted;
            q = q - rows(left);
            q = q - rows(right);
            candidates.push_back({pool_row, left, right, q});
        }
        candidate_count = static_cast<int64_t>(candidates.size());
        if (candidates.empty()) {
            throw std::runtime_error("sparse candidate graph has no edges");
        }

        // Only minimum values are needed for set-valued reciprocity.
        // Store them directly: no rank-tied argmin or candidate indirection.
        std::vector<double> best_q(static_cast<size_t>(n), std::numeric_limits<double>::infinity());
        for (const auto& edge : candidates) {
            best_q[edge.left] = std::min(best_q[edge.left], edge.q);
            best_q[edge.right] = std::min(best_q[edge.right], edge.q);
        }

        std::vector<int64_t> order(static_cast<size_t>(candidate_count));
        std::iota(order.begin(), order.end(), int64_t{0});
        std::sort(order.begin(), order.end(), [&](int64_t first, int64_t second) {
            const auto& one = candidates[static_cast<size_t>(first)];
            const auto& two = candidates[static_cast<size_t>(second)];
            if (one.q != two.q) {
                return one.q < two.q;
            }
            if (ranks(one.left) != ranks(two.left)) {
                return ranks(one.left) < ranks(two.left);
            }
            if (ranks(one.right) != ranks(two.right)) {
                return ranks(one.right) < ranks(two.right);
            }
            if (one.left != two.left) {
                return one.left < two.left;
            }
            return one.right < two.right;
        });

        std::vector<uint8_t> consumed(static_cast<size_t>(n), 0);
        accepted.reserve(static_cast<size_t>(std::min<int64_t>(maximum_merges, n / 2)));
        for (int64_t candidate : order) {
            if (static_cast<int>(accepted.size()) >= maximum_merges) {
                break;
            }
            const auto& edge = candidates[static_cast<size_t>(candidate)];
            if (consumed[static_cast<size_t>(edge.left)] ||
                consumed[static_cast<size_t>(edge.right)]) {
                continue;
            }
            // Match any edge that is exactly minimal at BOTH endpoints.
            // A unique argmin tie-break creates a star under symmetric evidence.
            // The sorted greedy pass gives a deterministic maximal disjoint
            // matching of mutual minima, preserving every strict preference.
            if (edge.q == best_q[edge.left] &&
                edge.q == best_q[edge.right]) {
                accepted.push_back({edge.left, edge.right, edge.pool_row});
                consumed[static_cast<size_t>(edge.left)] = 1;
                consumed[static_cast<size_t>(edge.right)] = 1;
            }
        }
        reciprocal_count=accepted.size();
        if(bounded){
            const int64_t target=std::min<int64_t>(maximum_merges,minimum_divisor==4?(n+2)/4:(n+minimum_divisor-1)/minimum_divisor);
            for(int64_t candidate:order){
                if(int64_t(accepted.size())>=target)break;
                const auto& edge=candidates[candidate];
                if(consumed[edge.left] || consumed[edge.right])continue;
                accepted.push_back({edge.left,edge.right,edge.pool_row});consumed[edge.left]=consumed[edge.right]=1;
            }
            if(int64_t(accepted.size())<target)throw std::runtime_error("contraction proof invariant violated");
        }
        if (accepted.empty()) {
            const auto& edge = candidates[static_cast<size_t>(order.front())];
            accepted.push_back({edge.left, edge.right, edge.pool_row});
        }
    }

    py::array_t<int64_t> output(
        {static_cast<py::ssize_t>(accepted.size()), py::ssize_t{3}}
    );
    auto selected = output.mutable_unchecked<2>();
    for (size_t row = 0; row < accepted.size(); ++row) {
        selected(static_cast<py::ssize_t>(row), 0) = accepted[row][0];
        selected(static_cast<py::ssize_t>(row), 1) = accepted[row][1];
        selected(static_cast<py::ssize_t>(row), 2) = accepted[row][2];
    }
    return py::make_tuple(output, candidate_count, maximum_degree, reciprocal_count);
}


template<class Key> py::array_t<int64_t> expand_directed_pairs_typed(
    py::array_t<int64_t,py::array::c_style> directed,
    py::array_t<int64_t,py::array::c_style> path_order,bool expand) {
    if(directed.ndim()!=2 || directed.shape(1)!=2 || path_order.ndim()!=1 || path_order.size()<2)
        throw std::runtime_error("invalid directed expansion layout");
    const int64_t n=path_order.size();auto p=directed.unchecked<2>();auto path=path_order.unchecked<1>();
    if(n>INT32_MAX)throw std::runtime_error("expansion owner stamp exceeds integer domain");
    std::vector<int64_t> offset(n+1,0),adj(directed.shape(0));std::vector<unsigned char> seen(n,0);
    for(int64_t i=0;i<n;++i)if(path(i)<0 || path(i)>=n || seen[path(i)]++)throw std::runtime_error("invalid expansion path");
    for(py::ssize_t e=0;e<directed.shape(0);++e){
        if(p(e,0)<0 || p(e,0)>=n || p(e,1)<0 || p(e,1)>=n || p(e,0)==p(e,1))throw std::runtime_error("invalid directed edge");
        ++offset[p(e,0)+1];
    }
    for(int64_t i=0;i<n;++i)offset[i+1]+=offset[i];
    auto next=offset;for(py::ssize_t e=0;e<directed.shape(0);++e)adj[next[p(e,0)]++]=p(e,1);
    std::vector<int64_t> starts(n+1,0);std::vector<Key> keys;
    int64_t maximum_degree=0;for(int64_t i=0;i<n;++i)maximum_degree=std::max(maximum_degree,offset[i+1]-offset[i]);
    const int workers=size_t(adj.size())*size_t(maximum_degree+1)>=65536?std::min(16,omp_get_max_threads()):1;
    {py::gil_scoped_release release;
    std::vector<int32_t> stamps(size_t(workers)*size_t(n),-1);
    #pragma omp parallel num_threads(workers)
    {
        int32_t* stamp=stamps.data()+size_t(omp_get_thread_num())*size_t(n);
        #pragma omp for schedule(static)
        for(int64_t i=0;i<n;++i){
            int64_t count=0;
            auto visit=[&](int64_t k){if(k!=i && stamp[k]!=int32_t(i)){stamp[k]=int32_t(i);++count;}};
            for(int64_t at=offset[i];at<offset[i+1];++at){int64_t j=adj[at];visit(j);
                if(expand)for(int64_t at2=offset[j];at2<offset[j+1];++at2)visit(adj[at2]);
            }
            starts[i+1]=count;
        }
        #pragma omp single
        {
            for(int64_t i=0;i<n;++i)starts[i+1]+=starts[i];
            keys.resize(size_t(starts.back())+n-1);
        }
        std::fill(stamp,stamp+n,-1);
        #pragma omp for schedule(static)
        for(int64_t i=0;i<n;++i){
            int64_t out=starts[i];
            auto visit=[&](int64_t k){if(k!=i && stamp[k]!=int32_t(i)){
                stamp[k]=int32_t(i);keys[out++]=Key(std::min(i,k)*n+std::max(i,k));}};
            for(int64_t at=offset[i];at<offset[i+1];++at){int64_t j=adj[at];visit(j);
                if(expand)for(int64_t at2=offset[j];at2<offset[j+1];++at2)visit(adj[at2]);
            }
        }
    }
    for(int64_t i=1;i<n;++i){int64_t a=path(i-1),b=path(i);keys[size_t(starts.back())+i-1]=Key(std::min(a,b)*n+std::max(a,b));}
    radix_sort_keys(keys);keys.erase(std::unique(keys.begin(),keys.end()),keys.end());}
    py::array_t<int64_t> out({py::ssize_t(keys.size()),py::ssize_t(2)});auto result=out.mutable_unchecked<2>();
    for(size_t i=0;i<keys.size();++i){result(i,0)=keys[i]/n;result(i,1)=keys[i]%n;}return out;
}



py::array_t<int64_t> projection_pairs(
    py::array_t<double,py::array::c_style|py::array::forcecast> projected,
    int window,int workers) {
    if(projected.ndim()==2 && projected.shape(0)<=65536)
        return projection_pairs_typed<uint32_t>(projected,window,workers);
    return projection_pairs_typed<int64_t>(projected,window,workers);
}
py::array_t<int64_t> expand_directed_pairs(
    py::array_t<int64_t,py::array::c_style> directed,
    py::array_t<int64_t,py::array::c_style> path_order,bool expand) {
    if(path_order.ndim()==1 && path_order.size()<=65536)
        return expand_directed_pairs_typed<uint32_t>(directed,path_order,expand);
    return expand_directed_pairs_typed<int64_t>(directed,path_order,expand);
}

PYBIND11_MODULE(_ct_apnj_graph,m){
 m.def("projection_pairs",&projection_pairs);
 m.def("select_directed_pairs",&select_directed_pairs);
 m.def("select_bounded_nj_merges",&select_bounded_nj_merges);
 m.def("expand_directed_pairs",&expand_directed_pairs);
}
