"""Same exact scorer with resident counting inputs and one CPU input check."""
import numpy as np
import torch
import triton
from .barycenter_tree_batched import BatchedFeatureBarycenterScorer
from .bit_orbits import _events
from .barycenter_features_fused import fused_barycenter


class ResidentFeatureBarycenterScorer(BatchedFeatureBarycenterScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The parent has already checked this complete observed partition.
        self.count_regions = torch.as_tensor(self.regions, device=self.device)
        self.count_tiles = int(triton.cdiv(int(((self.regions[:, 1]+31)//32-self.regions[:, 0]//32).max()), 128))

    @torch.inference_mode()
    def predict(self, q):
        q = np.asarray(q)
        if q.ndim != 2 or q.shape[1] != 4 or not np.issubdtype(q.dtype, np.integer):
            raise ValueError('an integer Qx4 CPU query matrix is required')
        if len(q):
            if q.min() < 0 or q.max() >= len(self.planes):
                raise ValueError('four distinct in-range query taxa required')
            canonical = np.all(q[:, 1:] > q[:, :-1])
            if not canonical and np.any(np.diff(np.sort(q, axis=1), axis=1) == 0):
                raise ValueError('four distinct in-range query taxa required')
        values = np.empty((len(q), 3), dtype=np.float64)
        for start in range(0, len(q), 8192):
            query = torch.as_tensor(np.ascontiguousarray(q[start:start+8192]),
                                    device=self.device, dtype=torch.int64)
            features = []
            for first in range(0, len(query), self.count_rows):
                if self.check is not None:
                    self.check()
                block = query[first:first+self.count_rows]
                out = torch.empty((len(block), len(self.regions), self.count_tiles, 15),
                                  device=self.device, dtype=torch.int32)
                _events[(len(block), len(self.regions), self.count_tiles)](
                    self.planes, block, self.count_regions, out, self.planes.shape[2],
                    len(self.regions), self.count_tiles, 128, num_warps=4)
                c = out.sum(2, dtype=torch.int64)
                del out
                features.append(fused_barycenter(c))
                del c
            values[start:start+len(query)] = self.model.predict_features(torch.cat(features)).cpu().numpy()
        return values
