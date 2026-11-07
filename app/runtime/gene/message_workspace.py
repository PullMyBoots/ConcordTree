"""One serial scratch arena for all conditional likelihood scores.

Root checks and NNI scans never overlap; their logical tensor shapes differ,
but their storage does not need to. Returned views keep the incumbent layout
and arithmetic. No live message is reused across changes to tree/parameters.
The arena is relinquished at the model-fit boundary.
"""
import torch
from collections import OrderedDict


class MessageWorkspace:
    def __init__(self, device):
        self.device=torch.device(device)
        self.capacity=0
        self.graph_cache=OrderedDict()
        self.values=self.scales=None

    def clear(self):
        self.graph_cache.clear()
        self.values=self.scales=None
        self.capacity=0

    def borrow(self, rows, categories, sites, *, scale_dtype=torch.float64):
        size=rows*categories*sites
        if size>self.capacity or (self.scales is not None and self.scales.dtype!=scale_dtype):
            self.clear()
            self.values=torch.empty(size*4,device=self.device,dtype=torch.float32)
            self.scales=torch.empty(size,device=self.device,dtype=scale_dtype)
            self.capacity=size
        return (self.values[:size*4].view(rows,categories,sites,4),
                self.scales[:size].view(rows,categories,sites))
