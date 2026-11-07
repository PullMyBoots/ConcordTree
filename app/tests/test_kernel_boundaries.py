"""Execution-boundary regressions with independent integer/FP64 references.

Run with the tested scientific Python; all temporary build files stay in output.
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from runtime_config import configure_runtime,PROJECT
configure_runtime(gpu=0,cpu_threads=16)
import torch
import triton
from _repeat_builder_schedule import build
from repeats_factorized_cuda import _sum_partials
from msc_refine.barycenter_tile_features import tiled_barycenter
from msc_refine.barycenter_streamed import streamed_barycenter
from msc_refine.bit_orbits import numpy_encode,numpy_counts,_events


def cpu_features(counts):
    x=np.asarray(counts,dtype=np.float64).copy();x[:,:,0]=0
    x=np.sqrt(x/np.maximum(x.sum(2,keepdims=True),1))
    x=x.sum(1)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-300)


class NativeBoundaries(unittest.TestCase):
    def test_wide_tuple_builder_reconstructs_original_columns(self):
        # Three 12-taxon children create a real >uint64 root tuple space.
        n=36;ni=n-2;m=ni+n;length=3000001
        children=np.full((m+1,3),m,dtype=np.int64);depths=np.zeros(ni,dtype=np.int64)
        next_id=1
        def subtree(leaves,depth):
            nonlocal next_id
            if len(leaves)==1:return ni+leaves[0]
            node=next_id;next_id+=1;depths[node]=depth;half=len(leaves)//2
            children[node,:2]=[subtree(leaves[:half],depth+1),subtree(leaves[half:],depth+1)]
            return node
        children[0]=[subtree(list(range(i,i+12)),1) for i in (0,12,24)]
        seq=np.random.default_rng(914532).integers(0,4,(n,length),dtype=np.uint8)
        meta=build(seq,children,depths,16)
        self.assertGreater(np.prod([int(meta['sizes'][c]) for c in children[0]],dtype=object),2**64-1)
        self.assertEqual(meta['frontier_audit']['wide_tuple_nodes'],1)
        np.testing.assert_array_equal(meta['root_frequency'],np.ones(length,dtype=np.int64))
        def decode(node,pattern,out):
            if node>=ni:out[node-ni]=pattern;return
            p=meta['mapping'][int(meta['offsets'][node])+pattern]
            for j,c in enumerate(children[node]):
                if c<m:decode(int(c),int(p[j]),out)
        sites=np.r_[0,1,17,65535,length-1,np.random.default_rng(19).integers(0,length,123)]
        for site in sites:
            recovered=np.empty(n,dtype=np.uint8);decode(0,int(site),recovered)
            np.testing.assert_array_equal(recovered,seq[:,site])

    def test_long_alignment_preserves_exact_pattern_frequencies(self):
        # Four taxa need <=625 observed patterns, regardless of raw L.
        length=2097153
        cycle=np.random.default_rng(972).integers(0,5,(4,257),dtype=np.uint8)
        seq=np.tile(cycle,(1,(length+256)//257))[:,:length].copy()
        children=np.full((7,3),6,dtype=np.int64)
        children[0]=[1,4,5];children[1]=[2,3,6]
        out=build(seq,children,np.array([0,1],dtype=np.int64),16)
        expected={}
        for site in range(cycle.shape[1]):
            key=tuple(cycle[:,site]);expected[key]=expected.get(key,0)+(length+256-site)//257
        maps=out['mapping'];offsets=out['offsets']
        actual={}
        for i,frequency in enumerate(out['root_frequency']):
            child=maps[i];sub=maps[int(offsets[1])+child[0]]
            actual[(int(sub[0]),int(sub[1]),int(child[1]),int(child[2]))]=int(frequency)
        self.assertEqual(expected,actual)
        self.assertEqual(sum(actual.values()),length)

    def test_overflow_tuple_order_is_exact_and_stable(self):
        # Distinct tuples whose naive uint64 mixed-radix keys collide.
        source=r'''
        #include "repeat_tuple_keys.h"
        #include <cassert>
        int main(){
            uint64_t space=0;
            assert(repeat_tuple_space(5,5,5,space)&&space==125);
            assert(!repeat_tuple_space(1ull<<22,1ull<<22,1ull<<22,space));
            std::vector<std::array<int32_t,3>> a{{1048576,0,0},{0,0,0},
                {1048576,0,0},{0,0,1},{0,0,0},{2147483646,2147483646,2147483646}};
            std::vector<int32_t> order(a.size()),temp(a.size()),expected{1,4,3,0,2,5};
            repeat_tuple_order(order,temp,a.size(),[&](int64_t s,int j){return a[s][j];});
            assert(order==expected);
        }
        '''
        with tempfile.TemporaryDirectory(dir=PROJECT/'output',prefix='tuple-boundary-') as directory:
            path=Path(directory);(path/'check.cpp').write_text(source)
            subprocess.run(['c++','-O2','-std=c++17','-I'+str(HERE/'runtime/gene/runtime'),
                            str(path/'check.cpp'),'-o',str(path/'check')],check=True)
            subprocess.run([str(path/'check')],check=True)


@unittest.skipUnless(torch.cuda.is_available(),'requires CUDA')
class CudaBoundaries(unittest.TestCase):
    def test_endpoint_reduction_crosses_old_tensor_limit(self):
        generator=torch.Generator(device='cuda').manual_seed(1814)
        for k in (1,4):
            for size in (1,4096,4097,65537,131073):
                with self.subTest(categories=k,size=size):
                    partial=torch.randn((size+4,k,16),generator=generator,device='cuda',dtype=torch.float64)
                    ranges=torch.tensor([[2,size],[0,0],[1,1],[3,17 if size>17 else size]],device='cuda',dtype=torch.int64)
                    out=torch.empty((4,k,16),device='cuda',dtype=torch.float64)
                    expected=torch.stack([partial[int(a):int(a+b)].sum(0) for a,b in ranges.cpu().tolist()])
                    _sum_partials[(4,k)](partial,ranges,out,k,triton.next_power_of_2(size))
                    torch.testing.assert_close(out,expected,rtol=2e-12,atol=2e-10)

    def test_long_single_gene_integer_sum_and_empty_queries(self):
        for tiles in (1,255,257,65537,1048577):
            with self.subTest(tiles=tiles):
                # Int64 totals exceed int32 at the largest boundary.
                counts=torch.full((1,1,tiles,15),4096,device='cuda',dtype=torch.int32)
                counts[0,0,0,1]=3
                expected=cpu_features(counts.sum(2,dtype=torch.int64).cpu().numpy())
                np.testing.assert_allclose(tiled_barycenter(counts).cpu().numpy(),expected,rtol=3e-15,atol=3e-15)
        self.assertEqual(tiled_barycenter(torch.empty((0,1,1,15),device='cuda',dtype=torch.int32)).shape,(0,15))

    def test_streamed_loci_and_word_tiles_match_independent_cpu(self):
        rng=np.random.default_rng(761)
        lengths=np.r_[32779,rng.integers(1,73,129)]
        ends=lengths.cumsum();regions=np.column_stack([np.r_[0,ends[:-1]],ends]).astype(np.int64)
        states=rng.integers(0,5,(4,int(ends[-1])),dtype=np.uint8)
        query=np.array([[0,1,2,3],[3,1,0,2],[1,0,3,2]],dtype=np.int64)
        planes=numpy_encode(states)
        expected=cpu_features(numpy_counts(planes,query,regions))
        p=torch.tensor(planes,device='cuda');q=torch.tensor(query,device='cuda');r=torch.tensor(regions,device='cuda')
        actual=streamed_barycenter(p,q,regions,r,budget=128<<10).cpu().numpy()
        np.testing.assert_allclose(actual,expected,rtol=3e-15,atol=3e-15)
        tiles=triton.cdiv(int(((regions[:,1]+31)//32-regions[:,0]//32).max()),128)
        full=torch.empty((len(q),len(r),tiles,15),device='cuda',dtype=torch.int32)
        _events[(len(q),len(r),tiles)](p,q,r,full,p.shape[2],len(r),tiles,128,num_warps=1)
        self.assertTrue(torch.equal(tiled_barycenter(full),torch.tensor(actual,device='cuda')))

    def test_more_than_65535_loci_and_flat_grid(self):
        loci=65537
        states=np.random.default_rng(783).integers(0,4,(4,loci),dtype=np.uint8)
        regions=np.column_stack([np.arange(loci),np.arange(1,loci+1)]).astype(np.int64)
        planes=numpy_encode(states);queries=np.array([[0,1,2,3]],dtype=np.int64)
        p=torch.tensor(planes,device='cuda');q=torch.tensor(queries,device='cuda');r=torch.tensor(regions,device='cuda')
        out=torch.empty((1,loci,1,15),device='cuda',dtype=torch.int32)
        _events[(loci,)](p,q,r,out,p.shape[2],loci,1,128,FLAT=True,num_warps=1)
        self.assertTrue(torch.equal(out.sum(3),torch.ones((1,loci,1),device='cuda',dtype=torch.int64)))
        take=np.array([0,31,32,64,4095,65534,65535,65536])
        np.testing.assert_array_equal(out[0,take,0].cpu().numpy(),numpy_counts(planes,queries,regions[take])[0])
        actual=streamed_barycenter(p,q,regions,r)
        self.assertTrue(torch.equal(actual,tiled_barycenter(out)))


if __name__=='__main__':unittest.main(verbosity=2)
