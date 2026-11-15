"""Render size/family topology comparisons and exact paired runtime ranges.

Inputs are retained measurements shipped in technical-report/data. This command
never runs inference and never writes into the immutable source data directory.
The effective ledger includes current 2048–4096 full-pipeline measurements.
Archived source ledgers remain available separately for provenance.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator, NullLocator
import numpy as np

RELEASE = Path(__file__).resolve().parents[1]
REPORT = RELEASE / 'technical-report'
DATA = REPORT / 'data'
FIGURES = REPORT / 'assets/benchmark'
SEED, BOOTSTRAPS = 20261105, 1000
COLORS = {'ConcordTree': '#39766a', 'DIPPER GPU': '#9a7f82', 'FastTree': '#b38a62',
          'CASTER-site': '#708c9f', 'VeryFastTree': '#8a9568', 'RapidNJ': '#aaa7a0',
          'FastME': '#958f89', 'IQ-TREE --fast': '#667fa0'}
SMALL_METHODS = ['ConcordTree', 'DIPPER GPU', 'FastTree', 'CASTER-site']
LARGE_METHODS = ['ConcordTree', 'FastME', 'RapidNJ', 'VeryFastTree', 'DIPPER GPU']


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_figure(fig, name):
    paths = [FIGURES / (name + '.' + suffix) for suffix in ('svg', 'png')]
    for path in paths:
        fig.savefig(path, dpi=240, facecolor='white')
    plt.close(fig)
    return [str(p.relative_to(REPORT)) for p in paths]


def read_records(path):
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        r['taxa'] = int(r['taxa'])
        for field in ('nrf', 'seconds'):
            r[field] = float(r[field]) if r.get(field) else None
    assert len({(r['task'], r['method'], r['id']) for r in rows}) == len(rows)
    return rows


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    assert len(values) and np.isfinite(values).all()
    rng = np.random.default_rng(SEED)
    means = values[rng.integers(0, len(values), (BOOTSTRAPS, len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), ci95=np.quantile(means, [.025, .975]).tolist())


def topology_summary(rows):
    groups = []
    for size, methods, taxa in [('small', SMALL_METHODS, [32, 64, 128, 192, 256]),
                                ('large', LARGE_METHODS, [512, 1024, 2048, 3072, 4096])]:
        for family, label in [('ILS', 'Simple'), ('complex', 'Complex')]:
            selected = [r for r in rows if r['taxa'] in taxa and r['family'] == family]
            expected = {r['id'] for r in selected if r['method'] == 'ConcordTree'}
            valid = {m: {r['id']: r for r in selected if r['method'] == m and
                         r['status'] == 'complete' and r['nrf'] is not None} for m in methods}
            # Small comparisons share the same successful source alignments.
            # Large keeps the 4096-taxon observations for the four complete methods;
            # FastME's missing 4096 results remain an explicitly partial bar.
            shared = set.intersection(*(set(v) for v in valid.values()))
            metrics = []
            for method in methods:
                ids = sorted(shared if size == 'small' else valid[method])
                stats = mean_ci([valid[method][rid]['nrf'] for rid in ids])
                metrics.append(dict(method=method, **stats, complete=len(ids), expected=len(expected),
                                    available_complete=len(valid[method]), record_ids=ids,
                                    taxa=sorted({valid[method][rid]['taxa'] for rid in ids}),
                                    partial_coverage=size == 'large' and set(ids) != expected))
            metrics.sort(key=lambda r: (r['mean'], r['method']))
            groups.append(dict(id=f'{size}-{label.lower()}', size=size, family=family, label=label,
                               taxa=taxa, expected=len(expected), shared_complete=len(shared), metrics=metrics))
    return dict(seed=SEED, bootstrap=BOOTSTRAPS, bootstrap_unit='whole alignment',
                statistic='arithmetic mean of per-alignment unrooted normalized RF; lower is better',
                small_policy='same successfully completed alignments for all four methods within each family',
                large_policy='all available alignments; the four full-coverage methods share identical IDs; '
                             'FastME excludes 4096 timeouts and is explicitly marked partial',
                concordtree_provenance='Current V1.0.2 full-pipeline reruns at 2048–4096, 2026-11-14. '
                                      'Report snapshot at 32–1024 independently spot-checked: 14/14 exact topologies match.',
                limitations=['Historical development data; not independent final holdout.',
                             'FastME partial-bar mean is not a matched full-Large ranking.',
                             'Small timeout exclusions are nonrandom and remain in the exported record ledger.'],
                groups=groups)


def draw_bars(ax, group, xmax):
    metrics = group['metrics']
    for y, r in enumerate(metrics):
        value = r['mean']
        lo, hi = r['ci95']
        method = r['method']
        ax.barh(y, value, height=.57, color=COLORS[method], edgecolor='white', linewidth=.6,
                hatch='//' if method == 'ConcordTree' else '..' if r['partial_coverage'] else None,
                zorder=3)
        ax.errorbar(value, y, xerr=[[max(0, value-lo)], [max(0, hi-value)]], fmt='none',
                    ecolor='#4d5350', elinewidth=.85, capsize=2.3, zorder=4)
        ax.text(1.035, y, f'{value:.4f}', transform=ax.get_yaxis_transform(), va='center',
                fontsize=10.5, color=COLORS['ConcordTree'] if method == 'ConcordTree' else '#4d5350',
                weight='bold' if method == 'ConcordTree' else 'normal')
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([r['method'] + (' *' if r['partial_coverage'] else '') for r in metrics], fontsize=10)
    for text, metric in zip(ax.get_yticklabels(), metrics):
        if metric['method'] == 'ConcordTree':
            text.set_fontweight('bold')
    ax.set(xlim=(0, xmax), ylim=(len(metrics)-.4, -.65), xlabel='Mean nRF (lower is better)')
    ax.tick_params(axis='y', length=0, pad=8)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.grid(False)
    ax.xaxis.grid(True, color='#e8eae7', linewidth=.6)
    ax.spines[['top', 'right', 'left']].set_visible(False)


def topology_figures(summary):
    xmax = max(r['ci95'][1] for g in summary['groups'] for r in g['metrics']) * 1.07
    exports = []
    for group in summary['groups']:
        fig, ax = plt.subplots(figsize=(5.3, 3.35))
        draw_bars(ax, group, xmax)
        fig.subplots_adjust(left=.31, right=.86, bottom=.18, top=.97)
        exports += save_figure(fig, 'species-topology-' + group['id'])
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.5))
    for ax, group in zip(axes.flat, summary['groups']):
        draw_bars(ax, group, xmax)
        lo, hi = min(group['taxa']), max(group['taxa'])
        ax.set_title(f'{group["size"].capitalize()} · {group["label"]}  |  {lo:,}–{hi:,} taxa',
                     loc='left', fontsize=12, pad=15)
    fig.subplots_adjust(left=.15, right=.93, top=.94, bottom=.09, wspace=.73, hspace=.5)
    exports += save_figure(fig, 'species-topology-size-family')
    return exports


def paired_ratios(records):
    tasks = {}
    for task in ('gene', 'species'):
        good = [r for r in records if r['task'] == task and r['status'] == 'complete'
                and r['seconds'] is not None and math.isfinite(r['seconds']) and r['seconds'] > 0
                and r['method'] != 'IQ-TREE' and not (task == 'gene' and r['method'] == 'CASTER-site')]
        if task == 'species':
            good = [r for r in good if r['method'] in
                    (SMALL_METHODS if r['taxa'] <= 256 else LARGE_METHODS)]
        ours = {r['id']: r for r in good if r['method'] == 'ConcordTree'}
        speedups = []
        for method in sorted({r['method'] for r in good} - {'ConcordTree'}):
            pairs = []
            for r in sorted(good, key=lambda x: x['id']):
                if r['method'] != method or r['id'] not in ours:
                    continue
                a = ours[r['id']]
                assert r['taxa'] == a['taxa'] and r['family'] == a['family']
                pairs.append(dict(id=r['id'], taxa=r['taxa'], family=r['family'],
                                  comparator_seconds=r['seconds'], concordtree_seconds=a['seconds'],
                                  ratio=r['seconds']/a['seconds']))
            if not pairs:
                continue
            points = []
            for taxa in sorted({r['taxa'] for r in pairs}):
                group = [r for r in pairs if r['taxa'] == taxa]
                values = [r['ratio'] for r in group]
                points.append(dict(taxa=taxa, min_ratio=min(values), max_ratio=max(values),
                                   pairs=len(group), record_ids=[r['id'] for r in group]))
            values = [r['ratio'] for r in pairs]
            speedups.append(dict(method=method, min_ratio=min(values), max_ratio=max(values),
                                 points=points, pairs=pairs))
        tasks[task] = dict(speedups=speedups)
    return dict(definition='For each identical successful input ID, comparator seconds / ConcordTree seconds. '
                          'At each taxon count show the exact minimum and maximum of these individual ratios.',
                statistic='observed min–max range, not mean, median, quantile interval, or confidence interval',
                missing_policy='Only shared successful records with positive finite full-inference timings. '
                               'Failed, missing, censored and refinement-only times are not ratios.',
                limitations='Historical hardware, concurrency and timing boundaries differ across campaigns. '
                            'Observed ratios are not hardware-normalized speedups. '
                            'ConcordTree 32–1024 timings use a retained engine; 2048–4096 use engine '
                            'elapsed time in fresh CLI processes, including task-internal first-use loading. '
                            'DIPPER 2048–4096 and other baselines retain archived timing scopes.', tasks=tasks)


def fmt_ratio(value):
    if value < .01:
        return f'{value:.2g}'
    if value >= 1000:
        return f'{value:,.0f}'
    if value >= 10:
        return f'{value:.1f}'
    return f'{value:.2f}'


def ratio_tone(low, high):
    return 'faster' if low > 1 else 'slower' if high < 1 else 'mixed'


RATIO_COLORS = {'faster': ('#e4eee8', '#325e4c'), 'slower': ('#f3e7dd', '#835a3c'),
                'mixed': ('#eef0f1', '#58656c'), 'missing': ('#fafaf8', '#a2a9aa')}


def ordered_ratio_methods(data):
    order = ['DIPPER GPU', 'FastTree', 'CASTER-site', 'VeryFastTree', 'FastME', 'RapidNJ', 'IQ-TREE --fast']
    return sorted(data['speedups'], key=lambda x: order.index(x['method']))


def ratio_figures(summary):
    """One annotated interval matrix per task; no average encoded in color."""
    from matplotlib.patches import FancyBboxPatch
    exports = []
    for task, data in summary['tasks'].items():
        methods = ordered_ratio_methods(data)
        taxa = sorted({p['taxa'] for item in methods for p in item['points']})
        fig, ax = plt.subplots(figsize=(12.2 if task == 'species' else 9.6, 4.7))
        for row, item in enumerate(methods):
            points = {p['taxa']: p for p in item['points']}
            for col, size in enumerate(taxa):
                p = points.get(size)
                tone = ratio_tone(p['min_ratio'], p['max_ratio']) if p else 'missing'
                bg, fg = RATIO_COLORS[tone]
                ax.add_patch(FancyBboxPatch((col-.46, row-.39), .92, .78,
                    boxstyle='round,pad=0,rounding_size=0.06', facecolor=bg, edgecolor='none'))
                label = f'{fmt_ratio(p["min_ratio"])}–\n{fmt_ratio(p["max_ratio"])}×' if p else '—'
                ax.text(col, row, label, ha='center', va='center', fontsize=9.1, color=fg, linespacing=1.35)
        ax.set_xlim(-.55, len(taxa)-.45); ax.set_ylim(len(methods)-.5, -.65)
        ax.set_xticks(range(len(taxa)), [f'{n:,}' for n in taxa], fontsize=10)
        ax.set_yticks(range(len(methods)), [m['method'] for m in methods], fontsize=10)
        ax.xaxis.tick_top(); ax.tick_params(length=0, pad=10)
        ax.grid(False); ax.spines[:].set_visible(False)
        fig.text(.02, .94, 'Paired runtime ratios', fontsize=13, weight='bold', color='#263e35')
        fig.text(.02, .035, 'Green: ConcordTree faster throughout  ·  Sand: slower throughout  ·  Gray: crosses 1×  ·  —: unavailable', fontsize=8.7, color='#65716c')
        fig.subplots_adjust(left=.15 if task == 'species' else .20, right=.99, top=.82, bottom=.12)
        exports += save_figure(fig, task+'-speedup')
    return exports


def runtime_summary(records):
    tasks = {}
    for task in ('gene', 'species'):
        good = [r for r in records if r['task']==task and r['status']=='complete' and
                r['seconds'] is not None and math.isfinite(r['seconds']) and r['seconds']>0
                and r['method']!='IQ-TREE' and not (task=='gene' and r['method']=='CASTER-site')]
        if task=='species':
            good=[r for r in good if r['method'] in (SMALL_METHODS if r['taxa']<=256 else LARGE_METHODS)]
        curves=[]
        for method in sorted({r['method'] for r in good},key=lambda x:(x!='ConcordTree',x)):
            rr=[r for r in good if r['method']==method]
            points=[]
            for size in sorted({r['taxa'] for r in rr}):
                selected=[r for r in rr if r['taxa']==size]
                points.append(dict(taxa=size,median_seconds=float(np.median([r['seconds'] for r in selected])),
                    record_ids=sorted(r['id'] for r in selected)))
            curves.append(dict(method=method,points=points))
        tasks[task]=dict(runtime_curves=curves)
    return dict(source='CURRENT_BENCHMARK_RECORDS.csv',runtime_statistic='median recorded inference seconds at each taxon count',
                limitations='Recorded execution conditions differ; see timing scope on the Runtime page.',tasks=tasks)


def runtime_figures(summary):
    exports=[]
    for task,data in summary['tasks'].items():
        curves=data['runtime_curves'];taxa=sorted({p['taxa'] for item in curves for p in item['points']})
        fig,ax=plt.subplots(figsize=(10.6,5.1))
        for item in curves:
            method=item['method'];lookup={p['taxa']:p['median_seconds'] for p in item['points']}
            values=np.array([lookup.get(size,np.nan) for size in taxa]); x=np.arange(len(taxa))
            style=dict(color=COLORS[method],marker='o',markersize=4.5 if method=='ConcordTree' else 3.7,
                       lw=2.15 if method=='ConcordTree' else 1.25,zorder=5 if method=='ConcordTree' else 3)
            ax.plot(x,values,label=method,**style)
            if method=='ConcordTree':
                for i,value in enumerate(values):
                    if np.isfinite(value):ax.annotate(f'{value:.2g}s',(i,value),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8,color=COLORS[method])
        ax.set_yscale('log',base=10);ax.set_xticks(range(len(taxa)),[f'{n:,}' for n in taxa])
        ax.set_xlabel('Number of taxa');ax.set_ylabel('Median inference time (seconds, log₁₀)')
        ax.yaxis.set_minor_locator(NullLocator());ax.grid(False);ax.yaxis.grid(True,color='#e5e9e5',lw=.6)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='lower left',bbox_to_anchor=(-.01,1.025),ncol=4,frameon=False,fontsize=9,columnspacing=1.6)
        ax.margins(x=.045,y=.23)
        if task=='species':
            boundary=taxa.index(2048)-.5
            ax.axvline(boundary,color='#c4ccc7',lw=.8,ls=':')
        fig.subplots_adjust(left=.10,right=.98,bottom=.14,top=.82)
        exports+=save_figure(fig,task+'-runtime')
    return exports


def ratio_html(task, data):
    import html
    methods=ordered_ratio_methods(data)
    taxa=sorted({p['taxa'] for item in methods for p in item['points']})
    out=['<div class="ratio-matrix-wrap" tabindex="0" aria-label="Scrollable paired runtime ratios"><table class="ratio-matrix">',
         '<caption class="sr-only">Observed minimum to maximum of individual paired comparator / ConcordTree runtime ratios by number of taxa.</caption>',
         '<thead><tr><th scope="col">Method <span>/ taxa</span></th>']
    for n in taxa:out.append(f'<th scope="col">{n:,}</th>')
    out.append('</tr></thead><tbody>')
    for item in methods:
        out.append(f'<tr><th scope="row">{html.escape(item["method"])}</th>')
        points={p['taxa']:p for p in item['points']}
        for n in taxa:
            p=points.get(n)
            if p:
                tone=ratio_tone(p['min_ratio'],p['max_ratio'])
                title=f'{p["pairs"]} paired completed inputs; observed minimum {p["min_ratio"]:.6g}×, maximum {p["max_ratio"]:.6g}×'
                out.append(f'<td class="ratio-{tone}" title="{title}"><span>{fmt_ratio(p["min_ratio"])}–</span><span>{fmt_ratio(p["max_ratio"])}<small>×</small></span></td>')
            else:
                title='FastME: all 4096-taxon runs timed out at 12 hours.' if item['method']=='FastME' and n==4096 else 'No paired successful measurement in this comparison scope.'
                out.append(f'<td class="ratio-missing" title="{title}"><span aria-label="Unavailable">—</span></td>')
        out.append('</tr>')
    out.append('</tbody></table></div>')
    out.append('<div class="ratio-legend"><span class="ratio-key faster">ConcordTree faster throughout</span><span class="ratio-key slower">Slower throughout</span><span class="ratio-key mixed">Crosses 1×</span><span>— Unavailable</span></div>')
    out.append(f'<div class="ratio-downloads"><a href="assets/benchmark/{task}-speedup.svg" download>SVG</a><a href="assets/benchmark/{task}-speedup.png" download>PNG</a><a href="data/PAIRED_RUNTIME_RANGES.json" download>Paired data</a></div>')
    return ''.join(out)


def update_html(ratios):
    path=REPORT/'accuracy.html';text=path.read_text()
    text,count=re.subn(r'<section class="report-section" id="distance-comparison">.*?</section>','',text,flags=re.S);assert count==1 or 'JC to K2P-gamma' not in text
    text=re.sub(r'<p class="scope-note">.*?</p>',
        '<p class="scope-note">Current V1.0.2 results: 2,048–4,096 taxa rerun in full; 32–1,024 taxa retained after matching current-version spot checks. Comparator accuracy is reused. Development data.</p>',text,count=1,flags=re.S)
    text=re.sub(r'<p class="species-comparison-note">.*?</p>',
        '<p class="species-comparison-note">Small uses shared completed alignments. Large includes all available scales; FastME * excludes 4,096-taxon timeouts. <a href="data/SPECIES_SIZE_FAMILY_SUMMARY.json" download>Data &amp; coverage</a></p>',text,flags=re.S)
    # The support ROC retains its original evaluation panel; topology and time updates do not enlarge it.
    start=text.index('<section class="report-section" id="support">')
    text=text[:start]+text[start:].replace('<tr><td>Species</td>', '<tr><td>Species · 32–1,024 taxa</td>',1)
    path.write_text(text)
    path=REPORT/'performance.html';text=path.read_text()
    text,count=re.subn(r'<section class="report-section" id="distance-timing">.*?</section>','',text,flags=re.S);assert count==1 or 'JC to K2P-gamma' not in text
    text=re.sub(r'<p class="scope-note">.*?</p>','<p class="scope-note">Recorded inference times, with current V1.0.2 results through 4,096 taxa. Process startup and nRF evaluation are excluded. <a href="#timing-scope">Timing scope</a></p>',text,count=1,flags=re.S)
    # Rebuild only the two task sections; unrelated report sections remain intact.
    for task,title,num in [('gene','Gene trees','01'),('species','Species trees','02')]:
        timing_alt='Median recorded inference time by taxon count; log base 10 seconds.'
        section=(f'<section class="report-section" id="{task}"><div class="section-title"><span>{num}</span><h2>{title}</h2></div><div class="section-body">'
          f'<figure class="figure chart-figure"><a href="assets/benchmark/{task}-runtime.svg" aria-label="Open vector figure"><img src="assets/benchmark/{task}-runtime.svg" width="2544" height="1224" alt="{timing_alt}" loading="lazy"></a></figure>'
          '<div class="ratio-heading"><h3>Observed runtime ratios</h3><span>Min–max · paired inputs</span></div>'
          '<p class="paired-ratio-note">Comparator / ConcordTree time. Above 1× favors ConcordTree.</p>'
          +ratio_html(task,ratios['tasks'][task])+'</div></section>')
        text,count=re.subn(r'<section class="report-section" id="'+task+r'">.*?</section>',section,text,flags=re.S);assert count==1
    # Preserve the provenance links in the expandable note, keeping the chart page compact.
    text=re.sub(r'<section class="report-section"><h2>DIPPER timing audit</h2>.*?</section>','',text,flags=re.S)
    details=('<details class="timing-details" id="timing-scope"><summary>Timing scope &amp; data</summary>'
       '<p>ConcordTree at 32–1,024 taxa uses the retained-engine report measurements. At 2,048–4,096 taxa, the 14 November full-pipeline runs use engine elapsed time in fresh CLI processes: decompression, parsing, initialization, refinement, support and tree export are included, as is task-internal first-use loading. Process startup and nRF evaluation are excluded. The dotted divider marks the change in measurement conditions; connecting lines guide the eye.</p>'
       '<p>Gene timings are retained. DIPPER at 32–1,024 species taxa and on the gene panel uses the remeasured timing contract; DIPPER at 2,048–4,096 taxa and the other methods retain archived measurement conditions. Hardware, threads, concurrency and timing boundaries differ; these are observed runtime ratios, not controlled algorithmic speedups.</p>'
       '<p>Each cell reports the exact observed minimum and maximum of individual ratios on identical successful input IDs. Missing, failed and timed-out runs are not assigned ratios. FastME at 4,096 taxa timed out after 12 hours.</p>'
       '<p><a href="data/CURRENT_BENCHMARK_RECORDS.csv" download>Records</a> · <a href="data/SPECIES_CURRENT_LARGE_TIMING.json" download>Large-input timing audit</a> · <a href="data/DIPPER_TIMING_AUDIT.json" download>DIPPER timing contract</a> · <a href="data/dipper-timing/TIMING_SUMMARY.json" download>DIPPER measurements</a> · <a href="data/dipper-timing/RERUN_ACCURACY.json" download>DIPPER topology check</a></p></details>')
    text=re.sub(r'<details class="timing-details" id="timing-scope">.*?</details>','',text,flags=re.S)
    text=text.replace('</main>',details+'</main>');path.write_text(text)
    css=REPORT/'assets/site.css';text=css.read_text()
    text=re.sub(r'/\* Runtime ratio matrix \*/.*?/\* End runtime ratio matrix \*/','',text,flags=re.S)
    text+='''
/* Runtime ratio matrix */
.ratio-heading{display:flex;align-items:baseline;justify-content:space-between;gap:16px;margin:32px 0 8px}.ratio-heading h3{margin:0}.ratio-heading>span{font-size:11px;color:var(--muted);white-space:nowrap;letter-spacing:.03em}
.ratio-matrix-wrap{max-width:100%;overflow-x:auto;margin:18px 0 14px;scrollbar-width:thin;overscroll-behavior-x:contain}
.ratio-matrix{border-collapse:separate;border-spacing:5px;width:100%;font-size:12px;font-variant-numeric:tabular-nums}
.ratio-matrix caption.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}
.ratio-matrix th{border:0;text-align:center;color:#53635c;padding:11px 5px;font-weight:600;white-space:nowrap}
.ratio-matrix th:first-child{text-align:left;min-width:114px;position:sticky;left:0;background:var(--paper);z-index:1;padding-left:0}
.ratio-matrix th span{font-weight:400;color:var(--muted);font-size:10px}.ratio-matrix td{border:0;border-radius:5px;min-width:67px;padding:12px 5px;text-align:center;line-height:1.5}
.ratio-matrix td span{display:block;white-space:nowrap}.ratio-matrix small{font-size:10px;margin-left:1px}.ratio-faster{background:#e4eee8;color:#325e4c}.ratio-slower{background:#f3e7dd;color:#835a3c}.ratio-mixed{background:#eef0f1;color:#58656c}.ratio-missing{background:#f2f3f0;color:#9aa39d}
.ratio-legend{display:flex;flex-wrap:wrap;gap:9px 20px;font-size:10px;color:var(--muted);align-items:center;margin:12px 0}.ratio-key::before{content:"";display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px}.ratio-key.faster::before{background:#b2cbbb}.ratio-key.slower::before{background:#d8bca5}.ratio-key.mixed::before{background:#c8ced2}
.ratio-downloads{display:flex;gap:16px;margin-top:15px;font-size:11px}.timing-details{border-top:1px solid var(--line);margin:42px 0 0;padding:20px 0 5px;color:var(--muted);font-size:12px;line-height:1.8}.timing-details summary{cursor:pointer;width:fit-content;color:var(--accent);font-size:13px}.timing-details p{max-width:860px}
@media(max-width:650px){.ratio-heading{align-items:flex-start;flex-direction:column;gap:4px}.ratio-matrix{font-size:11px;border-spacing:4px;min-width:600px}.ratio-matrix th:first-child{min-width:100px}.ratio-matrix td{min-width:63px;padding:10px 4px}.ratio-legend{font-size:10px;gap:8px 14px}}
/* End runtime ratio matrix */
'''
    css.write_text(text)

def main():
    plt.style.use(REPORT / 'assets/academic.mplstyle')
    plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelsize': 10, 'savefig.bbox': None, 'axes.titlelocation': 'left'})
    species = read_records(DATA / 'SPECIES_SIZE_RECORDS.csv')
    summary = topology_summary(species)
    save_json(DATA / 'SPECIES_SIZE_FAMILY_SUMMARY.json', summary)
    fields = ['panel', 'size', 'family', 'method', 'mean_nrf', 'ci95_low', 'ci95_high', 'complete', 'expected', 'partial_coverage']
    with (DATA / 'species-size-family.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for group in summary['groups']:
            for r in group['metrics']:
                writer.writerow(dict(panel=group['id'], size=group['size'], family=group['family'],
                                     method=r['method'], mean_nrf=r['mean'], ci95_low=r['ci95'][0],
                                     ci95_high=r['ci95'][1], complete=r['complete'], expected=r['expected'],
                                     partial_coverage=r['partial_coverage']))
    topology_exports = topology_figures(summary)
    # Preserve the original ledger; consume the effective current merged records.
    records = read_records(DATA / 'CURRENT_BENCHMARK_RECORDS.csv')
    ratios = paired_ratios(records)
    save_json(DATA / 'PAIRED_RUNTIME_RANGES.json', ratios)
    runtime_exports = ratio_figures(ratios)
    current_runtime = runtime_summary(records)
    runtime_exports += runtime_figures(current_runtime)
    with (DATA / 'paired-runtime-ranges.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task', 'method', 'taxa', 'min_ratio', 'max_ratio', 'pairs'])
        writer.writeheader()
        for task, value in ratios['tasks'].items():
            for item in value['speedups']:
                for p in item['points']:
                    writer.writerow(dict(task=task, method=item['method'], **{k: p[k] for k in ['taxa', 'min_ratio', 'max_ratio', 'pairs']}))
    # Reconcile the pre-existing runtime metadata with the new rendered statistic.
    path = DATA / 'RUNTIME_COMPARISON.json'
    original = current_runtime
    original['speedup_statistic'] = ratios['statistic']
    original['speedup_definition'] = ratios['definition']
    original['paired_range_source'] = 'PAIRED_RUNTIME_RANGES.json'
    for task in ratios['tasks']:
        original['tasks'][task]['speedups'] = ratios['tasks'][task]['speedups']
    save_json(path, original)
    update_html(ratios)
    catalog = dict(surface='appendix', script=str(Path(__file__).relative_to(RELEASE)),
                   topology_source='data/SPECIES_SIZE_RECORDS.csv', runtime_source='data/CURRENT_BENCHMARK_RECORDS.csv',
                   exports=topology_exports + runtime_exports,
                   review='pending render and browser inspection; verify before accepting',
                   record_sha256=sha(DATA / 'records.csv'))
    save_json(DATA / 'SIZE_REPORT_FIGURE_REVIEW.json', catalog)
    print(json.dumps({g['id']: [(r['method'], round(r['mean'], 6), r['complete']) for r in g['metrics']]
                      for g in summary['groups']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
