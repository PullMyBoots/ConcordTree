"""Independent frontier-row packing only; exact same conditional likelihood."""
import triton
from flat_messages import _flat_leaf,_flat_send
from analytic_invariant_likelihood import AnalyticInvariantLikelihood,_invariant_root,_invariant_nni,_invariant_values
from uniformized_likelihood import UniformizedLikelihood,_em_nni,_em_leaf

class FlatUniformizedLikelihood(UniformizedLikelihood):
    def _kernels(self,slab,weights):
        if self.width>=128:
            return super()._kernels(slab,weights)
        k,w,parts=self.k,self.width,self.parts
        _invariant_values[(triton.cdiv(w,256),)](self.codes,self.pi,self.iv,self.invariant_weight,w,256)
        _flat_leaf[(triton.cdiv(len(self.leafrows)*w,128),k)](self.u,self.logs,self.p,slab,self.leafrows,self.leafindex,self.rowedge,len(self.leafrows),k,w,128,1e-30,enable_fp_fusion=False)
        for level in self.levels:
            _flat_send[(triton.cdiv(len(level)*w,128),k)](self.u,self.logs,self.p,level,self.deps,self.rowedge,len(level),k,w,128,1e-30,enable_fp_fusion=False)
        _invariant_root[(parts,)](self.u,self.logs,self.pi,self.logweights,self.rootrows,self.rootpartial,weights,self.iv,k,w,w,128,enable_fp_fusion=False)
        _em_nni[(len(self.internal_edges),3,parts)](self.u,self.logs,self.p,self.dp,self.pi,self.logweights,self.rates,self.incoming,self.score_edges,self.partial,weights,self.iv,k,w,128,parts,enable_fp_fusion=False)
        _em_leaf[(len(self.leafrows),parts)](self.u,self.logs,self.p,self.dp,self.pi,self.logweights,self.rates,slab,self.leafrows,self.leafindex,self.rowedge,self.leafincoming,self.leafpartial,weights,self.iv,k,w,128,parts,enable_fp_fusion=False)


class FlatInvariantLikelihood(AnalyticInvariantLikelihood):
    def _kernels(self,slab,weights):
        if self.width>=128:
            return super()._kernels(slab,weights)
        k,w,parts=self.k,self.width,self.parts
        _invariant_values[(triton.cdiv(w,256),)](self.codes,self.pi,self.iv,self.invariant_weight,w,256)
        _flat_leaf[(triton.cdiv(len(self.leafrows)*w,128),k)](self.u,self.logs,self.p,slab,self.leafrows,self.leafindex,self.rowedge,len(self.leafrows),k,w,128,1e-30,enable_fp_fusion=False)
        for level in self.levels:
            _flat_send[(triton.cdiv(len(level)*w,128),k)](self.u,self.logs,self.p,level,self.deps,self.rowedge,len(level),k,w,128,1e-30,enable_fp_fusion=False)
        _invariant_root[(parts,)](self.u,self.logs,self.pi,self.logweights,self.rootrows,self.rootpartial,weights,self.iv,k,w,w,128,enable_fp_fusion=False)
        if self.nni:
            _invariant_nni[(len(self.internal_edges),parts)](self.u,self.logs,self.p,self.pi,self.logweights,self.score_edges,self.incoming,self.partial,weights,self.iv,k,w,w,128,parts,enable_fp_fusion=False)

