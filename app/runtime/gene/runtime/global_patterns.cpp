// Global exact equality folding of derived chunks. Input MSA is never modified.
// Hashes only index candidates; every equal-hash group is compared exactly.
// In-place MSD radix avoids a second O(patterns) record buffer.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <omp.h>
namespace py=pybind11;
struct Record{uint64_t hash,id;};
struct Chunk{const uint8_t* x;int64_t* w;int64_t n,p;};
void partition(Record* a,size_t n,int shift,std::array<size_t,257>& start){
 std::array<size_t,256> count{};
 for(size_t i=0;i<n;++i)++count[(a[i].hash>>shift)&255];
 start[0]=0;for(int b=0;b<256;++b)start[b+1]=start[b]+count[b];
 std::array<size_t,256> next;std::copy(start.begin(),start.begin()+256,next.begin());
 for(int b=0;b<256;++b)while(next[b]<start[b+1]){
  auto k=(a[next[b]].hash>>shift)&255;
  if(k==uint64_t(b))++next[b];else std::swap(a[next[b]],a[next[k]++]);
 }
}
void radix(Record* a,size_t n,int shift){
 if(n<256||shift<0){std::sort(a,a+n,[](auto x,auto y){return x.hash<y.hash;});return;}
 std::array<size_t,257> start;partition(a,n,shift,start);
 for(int b=0;b<256;++b)if(start[b+1]-start[b]>1)radix(a+start[b],start[b+1]-start[b],shift-8);
}
py::dict fold(py::list arrays,py::list weights,int threads,bool collisions){
 if(arrays.empty()||arrays.size()!=weights.size()||threads<1||threads>64)throw std::invalid_argument("invalid chunks/threads");
 std::vector<Chunk> chunks;int64_t n=-1;uint64_t total=0,sites=0;
 for(size_t c=0;c<arrays.size();++c){
  if(!py::array_t<uint8_t>::check_(arrays[c])||!py::array_t<int64_t>::check_(weights[c]))throw std::invalid_argument("native uint8/int64 arrays required");
  auto x=py::reinterpret_borrow<py::array_t<uint8_t>>(arrays[c]);
  auto w=py::reinterpret_borrow<py::array_t<int64_t>>(weights[c]);
  if(!(x.flags()&py::array::c_style)||!(w.flags()&py::array::c_style))throw std::invalid_argument("contiguous derived arrays required");
  if(x.ndim()!=2||w.ndim()!=1||x.shape(1)!=w.shape(0)||!w.writeable()||x.shape(0)<1||x.shape(1)<1||uint64_t(x.shape(1))>=uint64_t(1)<<32)throw std::invalid_argument("invalid array shape/type/writeability");
  if(n<0)n=x.shape(0);if(x.shape(0)!=n)throw std::invalid_argument("taxa mismatch");
  chunks.push_back({x.data(),w.mutable_data(),x.shape(0),x.shape(1)});total+=x.shape(1);
  for(int64_t j=0;j<w.shape(0);++j){if(w.data()[j]<=0)throw std::invalid_argument("positive weights required");sites+=w.data()[j];}
 }
 if(arrays.size()>uint64_t(1)<<32||sites>uint64_t(std::numeric_limits<int64_t>::max()))throw std::overflow_error("pattern count encoding overflow");
 uint64_t classes=0,hash_collisions=0,removed=0;
 {
  py::gil_scoped_release release;
  std::vector<Record> records(total);size_t offset=0;
  for(size_t c=0;c<chunks.size();++c){auto &ch=chunks[c];
   #pragma omp parallel for num_threads(threads) schedule(static)
   for(int64_t b=0;b<ch.p;b+=512){auto end=std::min(b+512,ch.p);
    for(int64_t j=b;j<end;++j)records[offset+j]={1469598103934665603ULL,(uint64_t(c)<<32)|uint64_t(j)};
    for(int64_t i=0;i<n;++i)for(int64_t j=b;j<end;++j){
     auto code=ch.x[i*ch.p+j];records[offset+j].hash=(records[offset+j].hash^uint64_t(code+1))*1099511628211ULL;
    }
    if(collisions)for(int64_t j=b;j<end;++j)records[offset+j].hash=0;
   }
   offset+=ch.p;
  }
  std::array<size_t,257> start;partition(records.data(),total,56,start);
  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  for(int b=0;b<256;++b)if(start[b+1]-start[b]>1)radix(records.data()+start[b],start[b+1]-start[b],48);
  auto equal=[&](uint64_t a,uint64_t b){auto &aa=chunks[a>>32];auto &bb=chunks[b>>32];uint64_t j=uint32_t(a),k=uint32_t(b);
   for(int64_t i=0;i<n;++i)if(aa.x[i*aa.p+j]!=bb.x[i*bb.p+k])return false;return true;
  };
  auto less=[&](const Record&a,const Record&b){auto &aa=chunks[a.id>>32];auto &bb=chunks[b.id>>32];uint64_t j=uint32_t(a.id),k=uint32_t(b.id);
   for(int64_t i=0;i<n;++i){auto x=aa.x[i*aa.p+j],y=bb.x[i*bb.p+k];if(x!=y)return x<y;}return a.id<b.id;
  };
  #pragma omp parallel num_threads(threads) reduction(+:classes,hash_collisions,removed)
  {
   size_t begin=total*omp_get_thread_num()/omp_get_num_threads(),end=total*(omp_get_thread_num()+1)/omp_get_num_threads();
   if(begin)while(begin<total&&records[begin].hash==records[begin-1].hash)++begin;
   for(size_t pos=begin;pos<end;){size_t last=pos+1;while(last<total&&records[last].hash==records[pos].hash)++last;
    bool identical=true;for(size_t j=pos+1;j<last;++j)if(!equal(records[pos].id,records[j].id)){identical=false;break;}
    if(!identical){++hash_collisions;std::sort(records.begin()+pos,records.begin()+last,less);}
    size_t lo=pos;
    while(lo<last){size_t hi=lo+1;while(hi<last&&equal(records[lo].id,records[hi].id))++hi;
     uint64_t first=records[lo].id;int64_t count=0;
     for(size_t j=lo;j<hi;++j){first=std::min(first,records[j].id);auto id=records[j].id;auto &ch=chunks[id>>32];count+=ch.w[uint32_t(id)];ch.w[uint32_t(id)]=0;}
     chunks[first>>32].w[uint32_t(first)]=count;++classes;removed+=hi-lo-1;lo=hi;
    }
    pos=last;
   }
  }
 }
 uint64_t after_sites=0,after_classes=0;for(auto ch:chunks)for(int64_t j=0;j<ch.p;++j){after_sites+=ch.w[j];after_classes+=ch.w[j]>0;}
 if(after_sites!=sites||after_classes!=classes||classes+removed!=total)throw std::runtime_error("global fold count mismatch");
 py::dict out;out["input_patterns"]=total;out["global_patterns"]=classes;out["removed_duplicate_patterns"]=removed;out["hash_collision_groups"]=hash_collisions;out["original_sites"]=sites;out["scratch_bytes"]=total*sizeof(Record);out["exact_equality_verified"]=true;return out;
}
PYBIND11_MODULE(_global_patterns,m){m.def("fold",[](py::list x,py::list w,int t){return fold(x,w,t,false);});m.def("_fold_all_hashes_collide",[](py::list x,py::list w,int t){return fold(x,w,t,true);});}
