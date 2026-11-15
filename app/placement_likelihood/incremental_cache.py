"""Reuse unaffected directed messages after insertion; update each changed side once."""
import numpy as np
from .newton import NewtonCache
from .fast_likelihood import transitions
from taxon_placement import edge

class IncrementalCache(NewtonCache):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        edges=sorted(self.tree.lengths)
        self.pm=dict(zip(edges,transitions(self.model,[self.tree.lengths[e] for e in edges])))

    def update(self,new_tree,selected,leaf,query_states):
        a,b=selected;center=leaf+1;self.tree=new_tree
        self.messages[a,center]=self.messages.pop((a,b));self.scales[a,center]=self.scales.pop((a,b))
        self.messages[b,center]=self.messages.pop((b,a));self.scales[b,center]=self.scales.pop((b,a))
        self.pm.pop(selected)
        new_edges=[edge(a,center),edge(b,center),edge(leaf,center)]
        self.pm.update(zip(new_edges,transitions(self.model,[new_tree.lengths[e] for e in new_edges])))
        values=np.where(query_states[:,None]==4,1.,np.eye(4)[np.minimum(query_states,3)])
        self.messages[leaf,center]=np.ascontiguousarray(np.broadcast_to(values,(self.K,self.S,4)))
        self.scales[leaf,center]=np.zeros((self.K,self.S))
        def message(u,v):
            val=np.ones((self.K,self.S,4));ls=np.zeros((self.K,self.S))
            for w in sorted(new_tree.graph[u]):
                if w==v:continue
                val*=np.einsum('kij,ksj->ksi',self.pm[edge(u,w)],self.messages[w,u],optimize=False)
                ls+=self.scales[w,u]
            n=np.maximum(val.max(2),1e-300);self.messages[u,v]=np.ascontiguousarray(val/n[:,:,None]);self.scales[u,v]=ls+np.log(n)
        parent={center:None};order=[center]
        for u in order:
            for v in sorted(new_tree.graph[u]):
                if v==parent[u]:continue
                parent[v]=u;order.append(v);message(u,v)
        if len(order)!=len(new_tree.graph):raise AssertionError('cache update disconnected')
