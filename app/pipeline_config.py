"""JSON-only configuration contract shared by the UI, CLI and scientific engine."""
import copy
import math

VERSION = 1

def field(label, default, kind='number', minimum=0, maximum=None, **extra):
    return dict(label=label, default=default, type=kind, min=minimum, max=maximum, **extra)

LOOP = {
    'max_rounds': field('Maximum rounds', None, 'integer', 1, 1000000, nullable=True),
    'stop_ratio': field('Stop below move ratio', .005, minimum=0, maximum=.999999, step=.001),
}
APNJ = {
    'projections': field('Projection dimensions', 16, 'integer', 1, 128),
    'window': field('Projection neighbor window', 8, 'integer', 1, 64),
    'cap': field('Merge candidate count', 32, 'integer', 2, 128),
    'representatives': field('Representatives per cluster', 2, 'integer', choices=[1, 2, 4]),
    'landmarks': field('Row-sum landmarks', 32, 'integer', 2, 512),
    'ann_k': field('ANN neighbors', 16, 'integer', 1, 64),
    'ann_rounds': field('ANN expansion rounds', 2, 'integer', 0, 8),
    'terminal_limit': field('Exact NJ terminal size', 512, 'integer', 4, 4096),
    'minimum_divisor': field('Contraction divisor', 514, 'integer', 4, 4096),
    'seed': field('Random seed', 20260934, 'integer', 0, 4294967295),
    'distance_model': field('Distance model', 3, 'integer', choices=[1, 2, 3], choice_labels=['JC', 'JC-gamma', 'K2P-gamma']),
}
MODULES = {
    'distance': dict(label='Distance NNI', tasks=['gene', 'species'], basic={
        'max_rounds': {**LOOP['max_rounds'], 'default': 4},
        'stop_ratio': {**LOOP['stop_ratio'], 'default': 0.},
    }, advanced={
        'representatives': field('Representatives per component', 4, 'integer', 1, 4),
        'min_gain': field('Minimum distance improvement', 0., minimum=0, maximum=1000000, step=.001),
    }),
    'mlp': dict(label='MLP NNI', tasks=['gene', 'species'], basic=LOOP, advanced={
        'representatives': field('Representatives per component', 4, 'integer', 1, 4),
        'min_margin': field('Minimum probability improvement', .10, minimum=0, maximum=1, step=.01),
        'batch_size': field('Quartet prediction batch', 65536, 'integer', 1, 65536),
    }),
    'likelihood': dict(label='Maximum likelihood', tasks=['gene'], basic=LOOP, advanced={
        'model': dict(label='Evolutionary model',default='GTR+FO+I+G4',type='string',choices=['GTR+FO+I+G4','GTR+I+G4','GTR+G4','HKY+G4','GTR','SYM','TN93','HKY','K80','F81','JC']),
        'min_gain': field('Minimum likelihood improvement', .001, minimum=0, maximum=1000000, step=.001),
        'optimizer_iterations': field('Parameter optimizer iterations', 32, 'integer', 1, 1024),
    }),
    'kernel': dict(label='Statistical kernel', tasks=['species'], basic=LOOP, advanced={
        'min_gain': field('Minimum objective improvement', 0., minimum=0, maximum=1000000, step=.001),
        'batch_size': field('Quartet prediction batch', 8192, 'integer', 1, 65536),
    }),
}

def defaults(task):
    if task not in ('gene', 'species'):raise ValueError('Unknown tree task.')
    ids=['distance', 'mlp', 'likelihood' if task=='gene' else 'kernel']
    return dict(version=VERSION, task=task,
        apnj={k:copy.deepcopy(v['default']) for k,v in APNJ.items()},
        stages=[dict(id=id, enabled=True,
            **{k:copy.deepcopy(v['default']) for k,v in MODULES[id]['basic'].items()},
            advanced={k:copy.deepcopy(v['default']) for k,v in MODULES[id]['advanced'].items()}) for id in ids])

def _fields(value, definitions, name):
    if not isinstance(value, dict):raise ValueError(name+' must be an object.')
    unknown=set(value)-set(definitions)
    if unknown:raise ValueError(name+': unknown parameter '+sorted(unknown)[0])
    out={}
    for k,spec in definitions.items():
        v=value.get(k, spec['default'])
        if v is None and spec.get('nullable'):out[k]=None;continue
        if spec['type']=='string':
            if not isinstance(v,str) or v not in spec['choices']:raise ValueError(name+'.'+k+' has an unsupported value.')
            out[k]=v;continue
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):raise ValueError(name+'.'+k+' must be a finite number.')
        if spec['type']=='integer' and not isinstance(v,int):raise ValueError(name+'.'+k+' must be an integer.')
        if 'choices' in spec and v not in spec['choices']:raise ValueError(name+'.'+k+' has an unsupported value.')
        if v<spec.get('min',0) or (spec.get('max') is not None and v>spec['max']):raise ValueError(name+'.'+k+' is outside its supported range.')
        out[k]=v
    return out

def normalize(value, task):
    if value is None:return defaults(task)
    if not isinstance(value,dict):raise ValueError('Pipeline settings must be an object.')
    if set(value)-{'version','task','apnj','stages'}:raise ValueError('Unknown pipeline setting.')
    if value.get('version')!=VERSION or value.get('task')!=task:raise ValueError('Pipeline version or tree task does not match.')
    base=defaults(task);stages=value.get('stages')
    if not isinstance(stages,list) or len(stages)!=len(base['stages']):raise ValueError('Include each compatible module exactly once; disable unused modules.')
    allowed={s['id'] for s in base['stages']};seen=set();out=[]
    for stage in stages:
        if not isinstance(stage,dict) or set(stage)-{'id','enabled','max_rounds','stop_ratio','advanced'}:raise ValueError('Invalid refinement module.')
        id=stage.get('id')
        if id not in allowed or id in seen:raise ValueError('Duplicate or incompatible refinement module.')
        seen.add(id)
        if not isinstance(stage.get('enabled'),bool):raise ValueError('Module enabled must be true or false.')
        spec=MODULES[id]
        basic=_fields({k:stage[k] for k in spec['basic'] if k in stage},spec['basic'],id)
        out.append(dict(id=id,enabled=stage['enabled'],**basic,advanced=_fields(stage.get('advanced',{}),spec['advanced'],id)))
    return dict(version=VERSION,task=task,apnj=_fields(value.get('apnj',{}),APNJ,'AP-NJ'),stages=out)

def schema():
    return dict(version=VERSION,apnj=APNJ,modules=MODULES,defaults={t:defaults(t) for t in ('gene','species')})
