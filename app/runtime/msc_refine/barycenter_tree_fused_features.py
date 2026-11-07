"""Same scorer with the independently audited FP64 feature execution."""
import numpy as np
import torch
from .barycenter_tree import BarycenterScorer
from .bit_orbits import counts
from .barycenter_features_fused import fused_barycenter

class FastFeatureBarycenterScorer(BarycenterScorer):
    @torch.inference_mode()
    def predict(self,q):
        values=np.empty((len(q),3),dtype=np.float64)
        for start in range(0,len(q),8192):
            query=q[start:start+8192];features=[]
            for first in range(0,len(query),self.count_rows):
                if self.check is not None:self.check()
                c=counts(self.planes,query[first:first+self.count_rows],self.regions)
                features.append(fused_barycenter(c));del c
            values[start:start+len(query)]=self.model.predict_features(torch.cat(features)).cpu().numpy()
        return values
