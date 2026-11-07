// Exact radix child: postorder live-ID frontier and zero-copy owning exports.
// All first-site pattern IDs, array layouts and CSR order preserve radix oracle.
// No hashing or ordered-map lookup. IDs retain FIRST original-site appearance.
// Project implementation of exact site-repeat IDs and reverse pattern CSR.
// Site repeats: Kobert et al. (2017), doi:10.1093/sysbio/syw075.
// No libpll/RAxML source copied. uint64 tuple keys are sorted exactly;
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

namespace py=pybind11;
template<class T> py::array_t<T> arr(std::vector<T>& v, std::vector<py::ssize_t> shape) {
    auto* owned = new std::vector<T>(std::move(v));
    py::capsule owner(owned, [](void* p){delete static_cast<std::vector<T>*>(p);});
    return py::array_t<T>(shape, owned->data(), owner);
}
py::dict build(py::array_t<uint8_t,py::array::c_style> seq,
               py::array_t<int64_t,py::array::c_style> children,
               py::array_t<int64_t,py::array::c_style> depths,int threads) {
    if(seq.ndim()!=2||children.ndim()!=2||children.shape(1)!=3||depths.ndim()!=1)
        throw std::invalid_argument("invalid native builder shapes");
    const int64_t ni=depths.shape(0),m=children.shape(0)-1,n=seq.shape(0),L=seq.shape(1);
    if(ni<1||m!=ni+n||L<1||threads<1||threads>64||L>2000000)
        throw std::invalid_argument("invalid native builder controls");
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
    std::vector<int64_t> pattern_nodes,mapping_offsets,links,starts,degrees,chunks,up_levels,root_frequency;
    std::vector<int32_t> mapping;
    std::vector<uint8_t> root_invariant_masks;
    int powers=1;for(uint64_t width=1;width<uint64_t(L);width*=2)++powers;
    const int64_t bucket_depths=maxdepth+2;
    std::vector<std::vector<int64_t>> buckets(size_t(bucket_depths)*powers);
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
                uint64_t space=uint64_t(sizes[a])*bc*cc;
                auto& mm=maps[u];mm.reserve(std::min<uint64_t>(L,space));
                if(space<=uint64_t(L)){
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
                    std::vector<uint64_t> keys(L);
                    std::vector<int32_t> order(L),temp(L),rank(L,-1);
                    uint64_t maximum=0;
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
                    // Within a key group stable sorting preserves original site
                    // order, so the first index is its minimum original site.
                    int32_t representative=-1;
                    for(int64_t j=0;j<L;++j){
                        auto pos=order[j];
                        if(j==0||keys[pos]!=keys[order[j-1]]){
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
            std::vector<int64_t> cursor(starts.begin()+off,starts.begin()+off+sizes[u]);
            for(int64_t q=0;q<int64_t(mm.size());++q)links[cursor[mm[q][j]]++]=offset[pa]+q;
        }
        for(int64_t u=1;u<m;++u){
            for(int64_t p=0;p<sizes[u];++p){
                auto row=offset[u]+p,deg=degrees[row];if(!deg)continue;
                int power=0;int64_t width=1;while(width<deg){width*=2;++power;}
                auto& rec=buckets[size_t(depth[u])*powers+power];
                rec.insert(rec.end(),{u,row,p,starts[row],deg});
            }
            for(int64_t p=0;p<sizes[u];p+=128)
                chunks.insert(chunks.end(),{u,offset[u]+p,std::min<int64_t>(128,sizes[u]-p)});
        }
        for(int64_t d=maxdepth;d>=0;--d)if(!levels[d].empty()){
            auto& v=levels[d];up_levels.insert(up_levels.end(),{offset[v.front()],offset[v.back()+1]});
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
    audit["peak_live_site_id_rows"]=peak_live_rows;
    audit["peak_live_site_id_bytes"]=peak_live_rows*L*int64_t(sizeof(int32_t));
    audit["old_site_id_bytes"]=m*L*int64_t(sizeof(int32_t));
    audit["jobs"]=job_count;audit["register_bound"]=register_bound;
    audit["owning_zero_copy_arrays"]=true;out["frontier_audit"]=audit;
    return out;
}
PYBIND11_MODULE(_repeat_builder_invariant,m){m.def("build",&build);}
