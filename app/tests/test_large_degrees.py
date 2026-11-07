"""GPU regression for a CSR row wider than Triton's live tensor limit."""
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from runtime_config import configure_runtime
configure_runtime(gpu=0, cpu_threads=16)
import torch
from repeats_fused_counts import _down


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class LargeDegrees(unittest.TestCase):
    def test_all_parent_links_contribute_to_each_nucleotide(self):
        torch.manual_seed(12041)
        for dtype in [torch.float32, torch.float64]:
            for degree in [129, 311148, 1048579]:
                with self.subTest(dtype=dtype, degree=degree):
                    k = 4
                    theta = torch.rand((degree+2,k,4), device='cuda', dtype=dtype)
                    expected_sum = theta[2:].double().sum(0).to(dtype)
                    up = torch.rand((1,k,4), device='cuda', dtype=torch.float64)
                    matrix = torch.rand((1,k,4,4), device='cuda', dtype=torch.float64)
                    matrix /= matrix.sum(-1, keepdim=True)
                    offsets = torch.zeros(2, device='cuda', dtype=torch.int64)
                    links = torch.arange(2, degree+2, device='cuda', dtype=torch.int32)
                    records = torch.tensor([[0,0,0,0,degree]], device='cuda', dtype=torch.int32)
                    blocks = torch.tensor([[0,1,0,0]], device='cuda', dtype=torch.int64)
                    partial = torch.empty((1,k,4,4), device='cuda', dtype=torch.float64)
                    u, p = up[0].to(dtype), matrix[0].to(dtype)
                    denominator = (p*u[:,None,:]).sum(-1)
                    joint = (expected_sum/denominator)[:,:,None]*p*u[:,None,:]
                    _down[(1,k)](up,theta,partial,matrix,offsets,links,records,blocks,k,1,256,1,1e-30 if dtype==torch.float32 else 1e-300)
                    tolerance = 2e-6 if dtype==torch.float32 else 2e-12
                    torch.testing.assert_close(partial[0],joint.double(),rtol=tolerance,atol=tolerance)
                    torch.testing.assert_close(theta[0],joint.sum(1),rtol=tolerance,atol=tolerance)


if __name__ == '__main__':
    unittest.main(verbosity=2)
