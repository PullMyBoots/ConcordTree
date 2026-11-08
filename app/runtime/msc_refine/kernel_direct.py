"""Direct coordinate-difference Gaussian expansion; same mathematical score."""
import torch
from .msc_kernel import MSCKernel,group_permutations


class DirectMSCKernel(MSCKernel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        p=torch.as_tensor(group_permutations(),device=self.device)
        self.expanded=self.landmarks[:,p].reshape(-1,15).contiguous()
        self.expanded_coefficients=(self.coefficients/8).repeat_interleave(8).contiguous()

