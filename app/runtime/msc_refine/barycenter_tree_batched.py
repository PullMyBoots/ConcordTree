"""Use the fused counter's live storage to size its unchanged query batches."""
import triton
from .barycenter_tree_fused_features import FastFeatureBarycenterScorer


class BatchedFeatureBarycenterScorer(FastFeatureBarycenterScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rr = self.regions
        tiles = triton.cdiv(int(((rr[:, 1] + 31)//32 - rr[:, 0]//32).max()), 128)
        # int32 tile counts + int64 reduced counts, one FP64 feature row and
        # four int64 query indices. The old budget counted unfused torch
        # intermediates that no longer exist. No site/query/locus is dropped.
        live_bytes = len(rr)*15*(4*tiles + 8) + 15*8 + 4*8
        old = self.count_rows
        self.count_rows = max(1, min(8192, (64 << 20)//live_bytes))
        self.metadata.update(count_query_rows=self.count_rows,
                             original_count_query_rows=old,
                             counter_live_bytes_per_query=live_bytes)
