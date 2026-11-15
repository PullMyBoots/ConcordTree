"""Exact constraint checks against exhaustive small-tree NNI neighborhoods."""
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE/'runtime/gene/src'))
from backbone import Backbone, treeio, restricted_graph
from concordtree.treeio import Tree, parse_newick, nrf


def insert(tree, name, e):
    g={u:set(v) for u,v in tree.graph.items()};labels=dict(tree.names)
    leaf=max(g)+1;center=leaf+1;a,b=e
    g[a].remove(b);g[b].remove(a)
    g[center]={a,b,leaf};g[leaf]={center};g[a].add(center);g[b].add(center)
    labels[leaf]=name
    return Tree(g,labels,{})


def moves(tree):
    for u,vs in tree.graph.items():
        if u in tree.names:continue
        for v in sorted(vs):
            if v<=u or v in tree.names:continue
            a,b=sorted(vs-{v});c,d=sorted(tree.graph[v]-{u})
            for x in (c,d):
                out=tree.copy()
                out.graph[u].remove(b);out.graph[b].remove(u)
                out.graph[v].remove(x);out.graph[x].remove(v)
                out.graph[u].add(x);out.graph[x].add(u)
                out.graph[v].add(b);out.graph[b].add(v)
                yield (u,v),out


class BackboneTests(unittest.TestCase):
    def test_all_six_taxon_nni_neighborhoods(self):
        trees=[parse_newick('(a,b,c);')]
        for name in ('d','x','y'):
            trees=[insert(tree,name,(u,v)) for tree in trees for u in tree.graph for v in tree.graph[u] if u<v]
        self.assertEqual(len(trees),105)
        constraint=Backbone('((a,b),(c,d));',list('abcdxy'))
        checked=0
        for tree in trees:
            g,n=restricted_graph(tree.graph,tree.names,constraint.taxa)
            if nrf(Tree(g,n,{}),constraint.tree):continue
            constraint.assert_tree(tree)
            candidates=list(moves(tree));edges=[e for e,_ in candidates]
            allowed=constraint.allowed_edges(tree.graph,tree.names,edges)
            for ok,(e,out) in zip(allowed,candidates):
                g,n=restricted_graph(out.graph,out.names,constraint.taxa)
                self.assertEqual(ok,nrf(Tree(g,n,{}),constraint.tree)==0,(e,ok))
                if ok:constraint.assert_tree(out)
                checked+=1
        self.assertEqual(checked,210)

    def test_completion_ignores_conflicting_guide_backbone(self):
        guide=parse_newick('(((a,c),(x,y)),((b,d),z));')
        names=list('abcdxyz')
        constraint=Backbone('((a,b),(c,d));',names)
        completed=constraint.complete(guide,names)
        constraint.assert_tree(completed)
        self.assertEqual(completed.names,dict(enumerate(names)))
        for e,out in moves(completed):
            if constraint.allowed_edges(completed.graph,completed.names,[e])[0]:
                constraint.assert_tree(out)

    def test_complete_backbone_and_quoted_names(self):
        text="(('a:b','x''y'),(c,d));"
        guide=parse_newick(text);names=list(guide.names.values())
        c=Backbone(text,names);c.assert_tree(c.complete(guide,names))
        self.assertFalse(any(c.allowed_edges(guide.graph,guide.names,[e for e,_ in moves(guide)])))
        # Rooting and branch length changes do not change the unrooted constraint.
        c=Backbone('((a:1,b:2):3,(c:4,d:5):6);',list('abcd'))
        c.assert_tree(parse_newick('(a,b,(c,d));'))

    def test_invalid_backbones(self):
        for text in ('', '(a,b,c,d);','((a,b),(c,z));','((a,a),(c,d));',
                     "((a,b),(c,d));'",'((a,b),(c,d));[unterminated',
                     '((a,b),(c,d));((a,b),(c,d));'):
            with self.subTest(text=text),self.assertRaises(ValueError):Backbone(text,list('abcd'))

    def test_ladder_is_iterative(self):
        text='(a,b,(c,d));';constraint=Backbone(text,list('abcd')+[f'x{i}' for i in range(2000)])
        tree=parse_newick(text)
        e=(next(u for u,n in tree.names.items() if n=='a'),)
        anchor=e[0]
        for i in range(2000):tree=insert(tree,f'x{i}',(anchor,next(iter(tree.graph[anchor]))))
        constraint.assert_tree(tree)
        edge_list=[(u,v) for u in tree.graph for v in tree.graph[u] if u<v and u not in tree.names and v not in tree.names]
        allowed=constraint.allowed_edges(tree.graph,tree.names,edge_list)
        self.assertEqual(sum(not x for x in allowed),1)


if __name__=='__main__':unittest.main(verbosity=2)
