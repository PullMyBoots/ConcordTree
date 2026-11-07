// Exact radix child: postorder live-ID frontier and zero-copy owning exports.
// All first-site pattern IDs, array layouts and CSR order preserve radix oracle.
// No hashing or ordered-map lookup. IDs retain FIRST original-site appearance.
// Project implementation of exact site-repeat IDs and reverse pattern CSR.
// Site repeats: Kobert et al. (2017), doi:10.1093/sysbio/syw075.
// No libpll/RAxML source copied. Narrow uint64 keys or wide tuples are sorted exactly;
// stable byte counting passes and first-site rank scans never merge unequal keys.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <numeric>
#include <vector>
#include <atomic>
#include "repeat_tuple_keys.h"

namespace py=pybind11;
template<class T> py::array_t<T> arr(std::vector<T>& v, std::vector<py::ssize_t> shape) {
    auto* owned = new std::vector<T>(std::move(v));
    py::capsule owner(owned, [](void* p){delete static_cast<std::vector<T>*>(p);});
    return py::array_t<T>(shape, owned->data(), owner);
}
template<class Index> py::dict build(py::array_t<uint8_t,py::array::c_style> seq,
               py::array_t<int64_t,py::array::c_style> children,
               py::array_t<int64_t,py::array::c_style> depths,int threads) {
    if(seq.ndim()!=2||children.ndim()!=2||children.shape(1)!=3||depths.ndim()!=1)
        throw std::invalid_argument("invalid native builder shapes");
    const int64_t ni=depths.shape(0),m=children.shape(0)-1,n=seq.shape(0),L=seq.shape(1);
    if(ni<1||m!=ni+n||L<1||threads<1||threads>256)
        throw std::invalid_argument("invalid native builder controls");
    if(L>INT32_MAX)throw std::overflow_error("site-repeat local IDs require sites <= INT32_MAX");
    auto C=children.data();auto dep=depths.data();auto X=seq.data();
    std::vector<int64_t> parent(m,-1),slot(m,-1),depth(m),offset(m+1),sizes(m,5);
    int64_t maxdepth=0;
    for(int64_t u=0;u<ni;++u){
        depth[u]=dep[u];maxdepth=std::max(maxdepth,dep[u]);
        int degree=0;
        for(int j=0;j<3;++j){
            auto v=C[u*3+j];
            if(v==m)continue;
            if(v<=u||v>=m||parent[v]!=-1)throw std::invalid_argument("invalid ordered tree");
            parent[v]=u;slot[v]=j;++degree;
            if(v<ni&&dep[v]!=dep[u]+1)throw std::invalid_argument("invalid levels");
        }
        if(degree<2||(u>0&&degree!=2))throw std::invalid_argument("binary tree required");
    }
    if(dep[0]!=0)throw std::invalid_argument("root must be first");
    for(int64_t u=1;u<m;++u){
        if(parent[u]<0)throw std::invalid_argument("disconnected tree");
        if(u>=ni)depth[u]=depth[parent[u]]+1;
    }
    std::vector<std::vector<int64_t>> levels(maxdepth+1);
    for(int64_t u=0;u<ni;++u)levels[dep[u]].push_back(u);
    std::vector<std::vector<std::array<int32_t,3>>> maps(ni);
    std::vector<uint8_t> wide_tuple_nodes(ni,0);
    std::vector<Index> pattern_nodes,links,starts,degrees,chunks,up_levels;
    std::vector<int64_t> mapping_offsets,root_frequency;
    std::vector<int32_t> mapping;
    std::vector<uint8_t> root_invariant_masks;
    int powers=1;for(uint64_t width=1;width<uint64_t(L);width*=2)++powers;
    const int64_t bucket_depths=maxdepth+2;
    std::vector<std::vector<Index>> buckets(size_t(bucket_depths)*powers);
    int64_t peak_live_rows=0,job_count=0,register_bound=0;
    {
        py::gil_scoped_release release;
        std::vector<std::vector<int32_t>> site_index(ni);
        std::vector<std::vector<uint8_t>> invariant_masks(ni);
        std::vector<int64_t> subtree(m,1),registers(m,0);
        std::vector<uint8_t> done(ni,0);
        int bad=0;
        #pragma omp parallel for num_threads(threads) reduction(|:bad) schedule(static)
        for(int64_t i=0;i<n*L;++i)bad|=(X[i]>4);
        if(bad)throw std::invalid_argument("invalid nucleotide code");
        auto inner_children=[&](int64_t u){
            std::vector<int64_t> v;
            for(int j=0;j<3;++j)if(C[u*3+j]<ni)v.push_back(C[u*3+j]);
            return v;
        };
        for(int64_t u=ni-1;u>=0;--u){
            subtree[u]=0;auto cc=inner_children(u);std::vector<int64_t> r;
            for(int j=0;j<3;++j)if(C[u*3+j]<m)subtree[u]+=subtree[C[u*3+j]];
            for(auto v:cc)r.push_back(registers[v]);
            std::sort(r.rbegin(),r.rend());int64_t need=int64_t(r.size())+1;
            for(int64_t j=0;j<int64_t(r.size());++j)need=std::max(need,r[j]+j);
            registers[u]=need;
        }
        register_bound=registers[0];
        std::vector<int64_t> jobs{0};
        while(jobs.size()<size_t(threads*4)){
            int64_t best=-1,cost=-1;
            for(int64_t j=0;j<int64_t(jobs.size());++j)
                if(!inner_children(jobs[j]).empty()&&subtree[jobs[j]]>cost){best=j;cost=subtree[jobs[j]];}
            if(best<0)break;
            auto cc=inner_children(jobs[best]);jobs.erase(jobs.begin()+best);
            jobs.insert(jobs.end(),cc.begin(),cc.end());
        }
        job_count=jobs.size();std::atomic<int64_t> live{0},peak{0};
        auto compute=[&](int64_t u){
            site_index[u].resize(L);
            auto current=live.fetch_add(1)+1;auto prior=peak.load();
            while(prior<current&&!peak.compare_exchange_weak(prior,current)){}
            auto get=[&](int64_t v,int64_t s)->int32_t{
                return v>=ni?int32_t(X[(v-ni)*L+s]):site_index[v][s];
            };
                auto a=C[u*3],b=C[u*3+1],c=C[u*3+2];
                uint64_t bc=sizes[b],cc=c<m?sizes[c]:1;
                uint64_t space=0;
                const bool narrow=repeat_tuple_space(uint64_t(sizes[a]),bc,cc,space);
                wide_tuple_nodes[u]=!narrow;
                auto& mm=maps[u];mm.reserve(narrow?std::min<uint64_t>(L,space):uint64_t(L));
                if(narrow&&space<=uint64_t(L)){
                    // Direct table initialization/work is bounded by raw L.
                    std::vector<int32_t> table(space,-1);
                    for(int64_t s=0;s<L;++s){
                        int32_t pa=get(a,s),pb=get(b,s),pc=c<m?get(c,s):0;
                        uint64_t key=(uint64_t(pa)*bc+pb)*cc+pc;
                        int32_t id=table[key];
                        if(id<0){id=mm.size();table[key]=id;mm.push_back({pa,pb,pc});}
                        site_index[u][s]=id;
                    }
                }else{
                    std::vector<uint64_t> keys(narrow?L:0);
                    std::vector<int32_t> order(L),temp(L),rank(L,-1);
                    uint64_t maximum=0;
                    if(narrow){
                    for(int64_t s=0;s<L;++s){
                        int32_t pa=get(a,s),pb=get(b,s),pc=c<m?get(c,s):0;
                        keys[s]=(uint64_t(pa)*bc+pb)*cc+pc;
                        maximum=std::max(maximum,keys[s]);order[s]=int32_t(s);
                    }
                    // Stable LSD byte radix: <=8 passes for the guarded uint64 keys.
                    for(int shift=0;shift<64&&(maximum>>shift);shift+=8){
                        std::array<int64_t,256> count{};
                        for(auto pos:order)++count[(keys[pos]>>shift)&255];
                        int64_t cursor=0;
                        for(auto& n:count){int64_t old=n;n=cursor;cursor+=old;}
                        for(auto pos:order)temp[count[(keys[pos]>>shift)&255]++]=pos;
                        order.swap(temp);
                    }
                    }else{
                        repeat_tuple_order(order,temp,L,[&](int64_t s,int j){
                            return C[u*3+j]<m?get(C[u*3+j],s):0;
                        });
                    }
                    // Within a key group stable sorting preserves original site
                    // order, so the first index is its minimum original site.
                    int32_t representative=-1;
                    for(int64_t j=0;j<L;++j){
                        auto pos=order[j];
                        bool different=j==0;
                        if(j>0){
                            auto prior=order[j-1];
                            different=narrow?keys[pos]!=keys[prior]:
                                (get(a,pos)!=get(a,prior)||get(b,pos)!=get(b,prior)||
                                 (c<m&&get(c,pos)!=get(c,prior)));
                        }
                        if(different){
                            representative=pos;rank[pos]=0;
                        }
                        site_index[u][pos]=representative;
                    }
                    for(int64_t s=0;s<L;++s)if(rank[s]>=0){
                        rank[s]=mm.size();
                        mm.push_back({get(a,s),get(b,s),c<m?get(c,s):0});
                    }
                    for(int64_t s=0;s<L;++s)site_index[u][s]=rank[site_index[u][s]];
                }
                sizes[u]=mm.size();
            // Exact zero-rate compatibility, on the SAME site-repeat DAG.
            // Each pattern's compatible ancestral states are the intersection
            // of child masks. Unknown leaves allow all four states.
            auto& masks=invariant_masks[u];masks.assign(mm.size(),uint8_t(15));
            for(int64_t p=0;p<int64_t(mm.size());++p){
                for(int j=0;j<3;++j){
                    auto child=C[u*3+j];if(child>=m)continue;
                    auto local=mm[p][j];
                    uint8_t mask=child<ni?invariant_masks[child][local]:
                        (local==4?uint8_t(15):uint8_t(1u<<local));
                    masks[p]&=mask;
                }
            }
            done[u]=1;
            for(auto v:inner_children(u)){
                std::vector<int32_t>().swap(site_index[v]);
                std::vector<uint8_t>().swap(invariant_masks[v]);live.fetch_sub(1);
            }
        };
        #pragma omp parallel for num_threads(threads) schedule(dynamic,1)
        for(int64_t j=0;j<int64_t(jobs.size());++j){
            std::vector<std::pair<int64_t,bool>> stack{{jobs[j],false}};
            while(!stack.empty()){
                auto item=stack.back();stack.pop_back();auto u=item.first;
                if(item.second){compute(u);continue;}
                stack.push_back({u,true});auto cc=inner_children(u);
                std::sort(cc.begin(),cc.end(),[&](int64_t a,int64_t b){
                    if(registers[a]!=registers[b])return registers[a]>registers[b];
                    if(subtree[a]!=subtree[b])return subtree[a]>subtree[b];return a<b;
                });
                for(auto it=cc.rbegin();it!=cc.rend();++it)stack.push_back({*it,false});
            }
        }
        // Remaining nodes are ancestors of the disjoint jobs. Ordered IDs ensure
        // children are ready; every retained row is consumed by one parent only.
        for(int64_t u=ni-1;u>=0;--u)if(!done[u])compute(u);
        peak_live_rows=peak.load();
        for(int64_t u=0;u<m;++u)offset[u+1]=offset[u]+sizes[u];
        const int64_t R=offset[ni],A=offset[m];
        pattern_nodes.resize(R);mapping.resize(R*3);
        #pragma omp parallel for num_threads(threads) schedule(static)
        for(int64_t u=0;u<ni;++u)for(int64_t p=0;p<sizes[u];++p){
            pattern_nodes[offset[u]+p]=u;
            for(int j=0;j<3;++j)mapping[(offset[u]+p)*3+j]=maps[u][p][j];
        }
        root_invariant_masks.swap(invariant_masks[0]);
        root_frequency.assign(sizes[0],0);
        for(int64_t s=0;s<L;++s)++root_frequency[site_index[0][s]];
        // Site IDs are no longer needed. Release O(NL) temporary before CSR.
        std::vector<std::vector<int32_t>>().swap(site_index);
        starts.resize(A);degrees.assign(A,0);
        mapping_offsets.assign(m+1,0);
        for(int64_t u=1;u<m;++u)mapping_offsets[u+1]=mapping_offsets[u]+sizes[parent[u]];
        links.resize(mapping_offsets[m]);
        #pragma omp parallel for num_threads(threads) schedule(dynamic,1)
        for(int64_t u=1;u<m;++u){
            auto pa=parent[u];auto& mm=maps[pa];auto j=slot[u];auto off=offset[u];
            for(auto& pattern:mm)++degrees[off+pattern[j]];
            auto next=mapping_offsets[u];
            for(int64_t p=0;p<sizes[u];++p){starts[off+p]=next;next+=degrees[off+p];}
            std::vector<Index> cursor(starts.begin()+off,starts.begin()+off+sizes[u]);
            for(int64_t q=0;q<int64_t(mm.size());++q)links[cursor[mm[q][j]]++]=offset[pa]+q;
        }
        // Reserve disjoint deterministic (node,power) intervals first.
        // A bucket keeps exactly the old node order and first-site pattern order.
        std::vector<int64_t> node_counts(size_t(m)*powers,0),
            destination(size_t(m)*powers,0),bucket_counts(buckets.size(),0);
        #pragma omp parallel for num_threads(threads) schedule(dynamic,1)
        for(int64_t u=1;u<m;++u){
            for(int64_t p=0;p<sizes[u];++p){
                auto deg=degrees[offset[u]+p];if(!deg)continue;
                int power=0;int64_t width=1;while(width<deg){width*=2;++power;}
                ++node_counts[size_t(u)*powers+power];
            }
        }
        for(int64_t u=1;u<m;++u)for(int power=0;power<powers;++power){
            size_t np=size_t(u)*powers+power,bp=size_t(depth[u])*powers+power;
            destination[np]=bucket_counts[bp];bucket_counts[bp]+=node_counts[np];
        }
        #pragma omp parallel for num_threads(threads) schedule(dynamic,1)
        for(int64_t bp=0;bp<int64_t(buckets.size());++bp)
            buckets[bp].resize(size_t(bucket_counts[bp])*5);
        #pragma omp parallel for num_threads(threads) schedule(dynamic,1)
        for(int64_t u=1;u<m;++u){
            for(int64_t p=0;p<sizes[u];++p){
                const int64_t row=offset[u]+p,deg=degrees[row];if(!deg)continue;
                int power=0;int64_t width=1;while(width<deg){width*=2;++power;}
                auto& rec=buckets[size_t(depth[u])*powers+power];
                auto out=5*destination[size_t(u)*powers+power]++;
                rec[out]=u;rec[out+1]=row;rec[out+2]=p;
                rec[out+3]=starts[row];rec[out+4]=deg;
            }
        }
        for(int64_t u=1;u<m;++u)
            for(int64_t p=0;p<sizes[u];p+=128)
                chunks.insert(chunks.end(),{Index(u),Index(offset[u]+p),Index(std::min<int64_t>(128,sizes[u]-p))});
        for(int64_t d=maxdepth;d>=0;--d)if(!levels[d].empty()){
            auto& v=levels[d];up_levels.insert(up_levels.end(),{Index(offset[v.front()]),Index(offset[v.back()+1])});
        }
        std::vector<std::vector<std::array<int32_t,3>>>().swap(maps);
    }
    py::dict out;
    out["offsets"]=arr(offset,{m+1});out["sizes"]=arr(sizes,{m});
    out["pattern_nodes"]=arr(pattern_nodes,{int64_t(pattern_nodes.size())});
    out["mapping"]=arr(mapping,{int64_t(mapping.size()/3),3});
    out["links"]=arr(links,{int64_t(links.size())});
    out["chunks"]=arr(chunks,{int64_t(chunks.size()/3),3});
    out["up_levels"]=arr(up_levels,{int64_t(up_levels.size()/2),2});
    out["root_frequency"]=arr(root_frequency,{int64_t(root_frequency.size())});
    out["root_invariant_masks"]=arr(root_invariant_masks,{int64_t(root_invariant_masks.size())});
    py::list bs;
    for(int64_t d=0;d<bucket_depths;++d)for(int power=0;power<powers;++power){
        auto& b=buckets[size_t(d)*powers+power];
        if(!b.empty())bs.append(py::make_tuple(d,power,arr(b,{int64_t(b.size()/5),5})));
    }
    out["buckets"]=bs;
    py::dict audit;
    audit["wide_tuple_nodes"]=std::count(wide_tuple_nodes.begin(),wide_tuple_nodes.end(),uint8_t(1));
    audit["peak_live_site_id_rows"]=peak_live_rows;
    audit["peak_live_site_id_bytes"]=peak_live_rows*L*int64_t(sizeof(int32_t));
    audit["old_site_id_bytes"]=m*L*int64_t(sizeof(int32_t));
    audit["jobs"]=job_count;audit["register_bound"]=register_bound;
    audit["owning_zero_copy_arrays"]=true;out["frontier_audit"]=audit;
    return out;
}

// Same exact repeat metadata, with terminal scheduling/count descriptors lowered
// into native loops. Preserve all original records and stable reduction order.
template<class Index> py::dict build_scheduled(py::array_t<uint8_t,py::array::c_style> seq,
               py::array_t<int64_t,py::array::c_style> children,
               py::array_t<int64_t,py::array::c_style> depths,int threads) {
    py::dict out=build<Index>(seq,children,depths,threads);
    const int64_t ni=depths.shape(0),node_count=children.shape(0);
    using Records=py::array_t<Index>;
    struct Bucket {int64_t depth,power;Records records;};
    std::vector<Bucket> scheduled;
    std::vector<std::vector<Records>> leaves;
    int64_t leaf_rows=0,last_depth=0;
    py::list original=out["buckets"].cast<py::list>();
    for(auto item:original){
        auto tuple=py::reinterpret_borrow<py::tuple>(item);
        int64_t depth=tuple[0].cast<int64_t>(),power=tuple[1].cast<int64_t>();
        auto array=tuple[2].cast<Records>();
        const Index* records=array.data();const int64_t size=array.shape(0);
        if(power>=int64_t(leaves.size()))leaves.resize(power+1);
        int64_t cut=0;while(cut<size&&records[cut*5]<ni)++cut;
        if(cut==size)scheduled.push_back({depth,power,array});
        else {
            if(cut)scheduled.push_back({depth,power,Records(
                std::vector<py::ssize_t>{cut,5},std::vector<py::ssize_t>{5*sizeof(Index),sizeof(Index)},records,array)});
            leaves[power].push_back(Records(std::vector<py::ssize_t>{size-cut,5},
                std::vector<py::ssize_t>{5*sizeof(Index),sizeof(Index)},records+cut*5,array));
            leaf_rows+=size-cut;
        }
        last_depth=std::max(last_depth,depth);
    }
    for(int64_t power=0;power<int64_t(leaves.size());++power){
        if(leaves[power].empty())continue;
        if(leaves[power].size()==1)scheduled.push_back({last_depth+1,power,leaves[power][0]});
        else {
            int64_t total=0;for(auto& array:leaves[power])total+=array.size();
            std::vector<Index> joined(total);int64_t offset=0;
            for(auto& array:leaves[power]){
                std::memcpy(joined.data()+offset,array.data(),array.size()*sizeof(Index));
                offset+=array.size();
            }
            scheduled.push_back({last_depth+1,power,arr(joined,{total/5,5})});
        }
    }
    std::vector<std::vector<Index>> descriptors(scheduled.size());
    std::vector<int64_t> sizes(node_count,0),ranges(node_count*2,0),cursor(node_count,0);
    int64_t total=0;
    for(size_t bucket=0;bucket<scheduled.size();++bucket){
        const auto* records=scheduled[bucket].records.data();
        const int64_t count=scheduled[bucket].records.shape(0);
        const int64_t tile=std::max<int64_t>(1,128/(int64_t(1)<<scheduled[bucket].power));
        auto& desc=descriptors[bucket];
        for(int64_t start=0;start<count;){
            int64_t end=start+1,node=records[start*5];
            while(end<count&&records[end*5]==node)++end;
            for(int64_t first=start;first<end;first+=tile){
                desc.insert(desc.end(),{Index(first),Index(std::min(tile,end-first)),Index(node),0});
                ++sizes[node];++total;
            }
            start=end;
        }
    }
    int64_t sum=0,maximum=1;
    for(int64_t node=0;node<node_count;++node){
        ranges[node*2]=sum;ranges[node*2+1]=sizes[node];cursor[node]=sum;
        sum+=sizes[node];maximum=std::max(maximum,sizes[node]);
    }
    for(auto& desc:descriptors)
        for(size_t row=0;row<desc.size()/4;++row)desc[row*4+3]=cursor[desc[row*4+2]]++;
    int64_t capacity=1;while(capacity<maximum)capacity*=2;
    py::list bs,blocks;
    for(size_t index=0;index<scheduled.size();++index){
        auto& b=scheduled[index];auto& d=descriptors[index];
        bs.append(py::make_tuple(b.depth,b.power,b.records));
        blocks.append(arr(d,{int64_t(d.size()/4),4}));
    }
    out["scheduled_buckets"]=bs;out["endpoint_blocks"]=blocks;
    out["endpoint_ranges"]=arr(ranges,{node_count,2});
    out["endpoint_capacity"]=capacity;out["endpoint_block_count"]=total;
    py::dict audit;
    audit["terminal_leaf_schedule"]=true;audit["terminal_leaf_records"]=leaf_rows;
    audit["reverse_buckets_before"]=py::len(original);audit["reverse_buckets_after"]=py::len(bs);
    out["terminal_schedule_audit"]=audit;
    return out;
}

py::dict compact_build(py::array_t<uint8_t,py::array::c_style> seq,
               py::array_t<int64_t,py::array::c_style> children,
               py::array_t<int64_t,py::array::c_style> depths,int threads) {
    if(seq.ndim()!=2 || children.ndim()!=2 || depths.ndim()!=1)
        throw std::invalid_argument("invalid native builder shapes");
    // Conservative integer-capacity proof: R <= ni*L, A <= R+5*n,
    // links <= 3*R. Every stored start/degree/row/block is bounded by max(A,3R).
    // Wide layouts remain available automatically beyond this representation.
    const uint64_t ni=depths.shape(0), n=seq.shape(0), L=seq.shape(1);
    const uint64_t limit=uint64_t(INT32_MAX);
    bool small=ni<=limit && n<=limit/5 && L<=limit/3
        && ni<=limit/(3*std::max<uint64_t>(1,L))
        && ni*L<=limit-5*n;
    auto out=small ? build_scheduled<int32_t>(seq,children,depths,threads)
                   : build_scheduled<int64_t>(seq,children,depths,threads);
    out["index_bits"]=small ? 32 : 64;
    return out;
}
PYBIND11_MODULE(_repeat_builder_schedule,m){m.def("build",&compact_build);}


