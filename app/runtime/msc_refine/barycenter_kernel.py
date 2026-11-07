"""Numerically audited direct15 Gaussian evaluator on barycenter features."""
import torch
import triton
from .kernel_direct_dynamic import DynamicDirectMSCKernel,_direct_runtime


class BarycenterKernel(DynamicDirectMSCKernel):
    @torch.inference_mode()
    def predict_features(self,features):
        x=torch.as_tensor(features,device=self.device,dtype=torch.float32).contiguous()
        if x.ndim!=2 or x.shape[1]!=15:raise ValueError('quartet x15 barycenter features required')
        tiles=triton.cdiv(len(self.expanded),128)
        partial=torch.empty((len(x),3,tiles),device=self.device,dtype=torch.float32)
        _direct_runtime[(triton.cdiv(len(x),16),3,tiles)](x,self.perms,self.expanded,self.expanded_coefficients,partial,
            len(x),len(self.expanded),tiles,1/(2*self.bandwidth**2),16,128,num_warps=4)
        return partial.sum(-1,dtype=torch.float64).float()
