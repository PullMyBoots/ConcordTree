"""Bounded-memory exact counter and original2048 direct kernel execution.

Integer counting uses one warp/program. For a single word tile the integer
output feeds the FP64 barycenter directly, avoiding an int64 copy. Gaussian
sums explicitly preserve the original32-lane then4-group reduction tree.
"""
import numpy as np
import torch
from .barycenter_state import BarycenterState
from .kernel_transposed import TransposedBarycenterKernel
from .bit_orbits import _events
from .barycenter_tile_features import tiled_barycenter

class TiledBarycenterScorer(BarycenterState):
    def __init__(self,packed,length,regions,asset,*,device='cuda:0',batch_size=8192):
        super().__init__(packed,length,regions,asset,device=device)
        self.batch_size=batch_size
        if self.model.expanded_coefficients.dtype==torch.float32:
            self.model=TransposedBarycenterKernel(asset,device=device,query_tile=16,warps=2,reduction_group=32)
        live_bytes=len(self.regions)*self.count_tiles*15*4+15*8+4*8
        self.count_rows=max(1,min(8192,(16<<20)//live_bytes))
        self.metadata.update(count_query_rows=self.count_rows,counter_live_bytes_per_query=live_bytes,scratch_layout_budget_bytes=16<<20)
        self.metadata.update(counter_warps=1,direct_all_tile_features=True,gaussian_reference_layout='transposed',gaussian_reduction_group=32)
        self.stream_counts=(live_bytes>(16<<20) or len(self.regions)>65535 or self.count_tiles>65535)
        if self.stream_counts:
            self.count_rows=min(self.count_rows,(16<<20)//(64*15*20+16*8+15*8))
        self.metadata['streamed_count_layout']=self.stream_counts

    @torch.inference_mode()
    def predict(self,q):
        q=np.asarray(q)
        if q.ndim!=2 or q.shape[1]!=4 or not np.issubdtype(q.dtype,np.integer):
            raise ValueError('an integer Qx4 CPU query matrix is required')
        if len(q):
            if q.min()<0 or q.max()>=len(self.planes):
                raise ValueError('four distinct in-range query taxa required')
            if not np.all(q[:,1:]>q[:,:-1]) and np.any(np.diff(np.sort(q,axis=1),axis=1)==0):
                raise ValueError('four distinct in-range query taxa required')
        values=np.empty((len(q),3),dtype=np.float64)
        for start in range(0,len(q),self.batch_size):
            query=torch.as_tensor(np.ascontiguousarray(q[start:start+self.batch_size]),device=self.device,dtype=torch.int64)
            features=[]
            for first in range(0,len(query),self.count_rows):
                if self.check is not None:self.check()
                block=query[first:first+self.count_rows]
                if self.stream_counts:
                    from .barycenter_streamed import streamed_barycenter
                    features.append(streamed_barycenter(self.planes,block,self.regions,self.count_regions,check=self.check))
                    continue
                out=torch.empty((len(block),len(self.regions),self.count_tiles,15),device=self.device,dtype=torch.int32)
                _events[(len(block),len(self.regions),self.count_tiles)](self.planes,block,self.count_regions,out,self.planes.shape[2],len(self.regions),self.count_tiles,128,num_warps=1)
                features.append(tiled_barycenter(out))
                del out
            values[start:start+len(query)]=self.model.predict_features(torch.cat(features)).cpu().numpy()
        return values
