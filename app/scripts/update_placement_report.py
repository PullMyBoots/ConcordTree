"""Rebuild placement accuracy from retained, paired development results; no inference."""
from pathlib import Path
import csv,json,hashlib,html,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RELEASE=Path(__file__).resolve().parents[1];PROJECT=RELEASE.parents[1];OUT=PROJECT/'output'
REPORT=RELEASE/'technical-report';DATA=REPORT/'data';FIG=REPORT/'assets/benchmark'
ORIGINAL_SOURCES={
 'gene_baselines':OUT/'raxmlgrove-signal500-placement-20261113/records.csv',
 'gene_current':OUT/'placement-likelihood-fast-20261113/records.csv',
 'species_external':OUT/'species-placement-external-20261114/records.csv',
 'species_estimated':OUT/'species-placement-iqfast-instral-20261114/records.csv'}
SOURCES={key:DATA/'placement-source'/(key+'.csv') for key in ORIGINAL_SOURCES}
LABELS=dict(concordtree='ConcordTree',iqtree='IQ-TREE constrained',tipars='TIPars',epa_ng='EPA-ng · sequential',pplacer='pplacer · sequential',apples2='APPLES-2 · sequential',
 instral_iqfast='IQ-TREE fast + INSTRAL',apples_species='APPLES-2 · independent')

def read(path):return list(csv.DictReader(path.open()))
def boolean(s):return str(s).lower()=='true'
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def main():
 rows=[];failures=[];oracle=[]
 def add(task,r,method,missing,nrf,exact,status='complete',protocol='sequential'):
  if status!='complete':failures.append(dict(task=task,method=method,source_id=r['source_id'],missing=int(missing),status=status));return
  rows.append(dict(task=task,source_id=r['source_id'],taxa=int(r['taxa']),missing=int(missing),method=method,nrf=float(nrf),exact_batch=boolean(exact),protocol=protocol))
 for r in read(SOURCES['gene_baselines']):
  if r['method']!='concordtree':add('gene',r,r['method'],r['deleted'],r['nrf'],r['exact_topology'],r['status'],'constrained search' if r['method']=='iqtree' else 'sequential')
 for r in read(SOURCES['gene_current']):add('gene',r,'concordtree',r['missing'],r['candidate_nrf'],r['candidate_exact_tree'])
 for r in read(SOURCES['species_external']):
  if r['method']=='instral_oracle':oracle.append(r);continue
  method='concordtree' if r['method']=='concordtree_kernel' else 'apples_species'
  add('species',r,method,r['deleted'],r['nrf'],r['exact_topology'],r['status'],'sequential' if method=='concordtree' else 'independent; coincident queries unresolved')
 for r in read(SOURCES['species_estimated']):add('species',r,'instral_iqfast',r['deleted'],r['nrf'],r['exact_topology'],r['status'],'independent; coincident queries unresolved')
 expected=dict(gene=['concordtree','iqtree','tipars','epa_ng','pplacer','apples2'],species=['concordtree','instral_iqfast','apples_species'])
 summaries=[];exports=[];rng=np.random.default_rng(20261114)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
 for task in ('gene','species'):
  for m in (1,4,8):
   selected=[r for r in rows if r['task']==task and r['missing']==m];methods=expected[task]
   lookup={}
   for r in selected:
    key=(r['source_id'],r['method']);assert key not in lookup,key;lookup[key]=r
   ids=sorted(set.intersection(*[{r['source_id'] for r in selected if r['method']==method} for method in methods]))
   assert ids
   losses=np.array([[lookup[s,method]['nrf'] for method in methods] for s in ids]);assert np.isfinite(losses).all()
   # Same input and taxa: nRF has the same scientific interpretation. No averaging of raw nRF across sizes.
   # A tie tolerance below any attainable RF step in this corpus.
   delta=losses[:,:,None]-losses[:,None,:]
   pair=(delta < -1e-12).astype(float)+.5*(np.abs(delta)<=1e-12)
   scores=(pair.sum(axis=2)-.5)/(len(methods)-1)
   assert np.allclose(scores.sum(axis=1),len(methods)/2)
   bootstrap=rng.integers(0,len(ids),(4000,len(ids)))
   entries=[]
   for j,method in enumerate(methods):
    ci=np.quantile(scores[bootstrap,j].mean(axis=1),[.025,.975]);exact=np.mean([lookup[s,method]['exact_batch'] for s in ids])
    e=dict(method=method,label=LABELS[method],rank_score=float(scores[:,j].mean()),rank_score_ci95=ci.tolist(),exact_batch_rate=float(exact),shared_records=len(ids),available_records=sum(r['method']==method for r in selected))
    entries.append(e)
    for i,s in enumerate(ids):exports.append(dict(**lookup[s,method],relative_rank_score=float(scores[i,j])))
   entries.sort(key=lambda e:(-e['rank_score'],e['label']))
   summaries.append(dict(task=task,missing=m,methods=entries,shared_ids=ids))
   fig,ax=plt.subplots(figsize=(7.2,3.35 if task=='gene' else 2.6));fig.subplots_adjust(left=.30,right=.78,top=.82,bottom=.22)
   y=np.arange(len(entries));colors=['#39766a' if e['method']=='concordtree' else '#b3bebd' for e in entries]
   ax.barh(y,[e['rank_score'] for e in entries],height=.43,color=colors,zorder=3)
   for i,e in enumerate(entries):
    lo,hi=e['rank_score_ci95'];ax.plot([lo,hi],[i,i],color='#263e39',lw=1.2,zorder=4)
    ax.text(e['rank_score']+.028,i,f"{e['rank_score']:.3f}",va='center',fontsize=11,color='#263e39')
    ax.text(1.34,i,f"{100*e['exact_batch_rate']:.1f}%",va='center',ha='right',fontsize=11,color='#263e39',transform=ax.get_yaxis_transform(),clip_on=False)
   ax.set_yticks(y,[e['label'] for e in entries],fontsize=10.5);ax.invert_yaxis();ax.set_xlim(0,1.12);ax.set_xticks([0,.25,.5,.75,1],['0','.25','.50','.75','1']);ax.set_xlabel('Relative rank score',labelpad=8)
   ax.text(1.34,1.10,'Exact batch',ha='right',transform=ax.transAxes,color='#67746e',fontsize=10)
   ax.grid(axis='x',alpha=.12,zorder=0);ax.spines[['left','bottom']].set_visible(False);ax.tick_params(axis='both',length=0,pad=8)
   stem=f'placement-{task}-{m}';fig.savefig(FIG/(stem+'.svg'),facecolor='white');fig.savefig(FIG/(stem+'.png'),dpi=180,facecolor='white');plt.close(fig)
 summary=dict(metric='Mean within-input pairwise superiority (tie-aware normalized rank)',formula='For each input: mean over competitors of 1 if lower whole-tree nRF, 0.5 if tied, 0 otherwise; then average inputs equally.',
  direction='higher is better; 0 to 1; not probability of correct placement',ties='absolute nRF difference <= 1e-12',
  exact_batch='All completed unrooted nontrivial splits exactly equal simulated truth, including relationships among added taxa.',
  pairing='Within each task and missing count, intersect successful source IDs across every displayed method. No failed run is given a score.',
  interval='95% percentile bootstrap over whole alignments; 4,000 paired resamples; seed 20261114.',
  limitations=['Exposed development datasets, not an independent final holdout.',
    'Scores depend on the displayed comparator set; do not compare absolute values between gene and species panels.',
    'Gene baselines use sequential adapters except native TIPars and IQ-TREE constrained search.',
    'Species APPLES-2 and INSTRAL place independently on the original backbone; coincident queries remain unresolved. ConcordTree inserts jointly/sequentially. Whole-output topology evaluates these actual outputs.',
    'IQ-TREE fast + INSTRAL uses estimated observed-locus gene trees (one individual per species). True-gene-tree INSTRAL is privileged and excluded from rankings.'],
  panels=summaries,failures=failures,oracle_reference=[dict(missing=m,method='INSTRAL with true gene trees (oracle)',exact_batch_rate=float(np.mean([boolean(r['exact_topology']) for r in oracle if int(r['deleted'])==m])),records=sum(int(r['deleted'])==m for r in oracle)) for m in (1,4,8)],
  sources={key:dict(path=str(p.relative_to(RELEASE)),original_source=str(ORIGINAL_SOURCES[key].relative_to(PROJECT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for key,p in SOURCES.items()})
 save(DATA/'PLACEMENT_SUMMARY.json',summary)
 with (DATA/'PLACEMENT_RECORDS.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(exports[0]));writer.writeheader();writer.writerows(exports)
 blocks=[]
 for task,label in [('gene','Gene-tree completion'),('species','Species-tree completion')]:
  tabs=''.join(f'<button type="button" role="tab" id="{task}-tab-{m}" aria-controls="{task}-placement-{m}" aria-selected="{str(m==1).lower()}" data-missing="{m}">+{m}</button>' for m in (1,4,8))
  panels=''
  for m in (1,4,8):
   stem=f'placement-{task}-{m}'
   entry=next(panel for panel in summaries if panel['task']==task and panel['missing']==m)
   compact='<div class="placement-mobile"><div class="placement-mobile-labels"><span>Method</span><span>Rank</span><span>Exact</span></div>'+''.join(f'<div class="placement-mobile-row" data-method="{e["method"]}"><span>{html.escape(e["label"])}<meter min="0" max="1" value="{e["rank_score"]:.6f}" aria-hidden="true"></meter></span><b>{e["rank_score"]:.3f}</b><span>{100*e["exact_batch_rate"]:.1f}%</span></div>' for e in entry['methods'])+'</div>'
   panels+=f'<div class="placement-view" id="{task}-placement-{m}" role="tabpanel" aria-labelledby="{task}-tab-{m}"><figure class="figure"><a href="assets/benchmark/{stem}.svg" aria-label="Open {label}, {m} added taxa"><img src="assets/benchmark/{stem}.svg" width="1296" height="{603 if task=="gene" else 468}" alt="{label}, {m} additions: methods ranked by paired relative topology rank score; exact batch recovery shown at right." loading="lazy"></a></figure>{compact}</div>'
  blocks.append(f'<div class="placement-task" data-placement-task="{task}"><header><h3>{label}</h3><div class="placement-tabs" role="tablist" aria-label="Number of added taxa">{tabs}</div></header>{panels}</div>')
 section='<section class="report-section" id="completion"><div class="section-title"><span>06</span><h2>Taxon completion</h2></div><div class="section-body"><p>Paired topology ranking across all taxon sizes. Higher is better. Exact batch: the entire completed topology is correct.</p>'+''.join(blocks)+'''<details class="placement-definition"><summary>Metric &amp; comparison protocol</summary><p>On each input, a method earns 1 for each competitor it beats in whole-tree nRF, ½ for a tie, and 0 for a loss. We average across competitors, then equally across shared completed inputs. The score measures relative rank, not probability of correctness. Bars show 95% alignment-bootstrap intervals.</p><p>+1, +4 and +8 use separate paired panels. Gene methods use sequential placement, native TIPars or IQ-TREE constrained search. Species APPLES-2 and INSTRAL place independently; coincident queries remain unresolved, while ConcordTree can resolve relationships among new taxa. Exact batch recovery tests the complete output, including those relationships. INSTRAL uses IQ-TREE fast inferred gene trees; the true-gene-tree oracle is excluded. Results are descriptive development comparisons.</p></details><div class="placement-downloads"><a href="data/PLACEMENT_RECORDS.csv" download>Measurements</a> · <a href="data/PLACEMENT_SUMMARY.json" download>Summary &amp; coverage</a></div></div></section>'''
 p=REPORT/'accuracy.html';text=p.read_text();text=re.sub(r'<section class="report-section" id="completion">.*?</section>','',text,flags=re.S);text=text.replace('</main>',section+'</main>')
 if 'assets/placement.js' not in text:text=text.replace('</body>','<script src="assets/placement.js" defer></script></body>')
 p.write_text(text)
 print(json.dumps([{k:s[k] for k in ('task','missing','methods')} for s in summaries],indent=2))
if __name__=='__main__':main()
