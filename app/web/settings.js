'use strict';
// Settings are edited as a draft. Each submitted job receives its own JSON copy.
window.pipelineSettings = (() => {
  const el=id=>document.getElementById(id), clone=value=>JSON.parse(JSON.stringify(value));
  const storageKey='concordtree-pipelines-v1',computeKey='concordtree-compute-v1';
  let schema, saved, draft, computeSchema, savedResources, draftResources, devices=[], task='gene', loading, dragged;
  async function load() {
    if (schema) return;
    if (loading) return loading;
    loading=(async()=>{
      const response=await fetch('/api/config');
      if (!response.ok) throw new Error('Could not load pipeline settings.');
      const payload=await response.json();schema=payload.pipeline_settings;computeSchema=payload.compute_settings;
      if(!computeSchema)throw new Error('Restart ConcordTree to load compute settings.');
      savedResources=clone(computeSchema.defaults);
      try {
        const stored=JSON.parse(localStorage.getItem(computeKey));
        if(Number.isInteger(stored?.cpu_threads)&&stored.cpu_threads>=1)savedResources.cpu_threads=Math.min(stored.cpu_threads,computeSchema.max_cpu_threads);
        if(Number.isInteger(stored?.gpu)&&stored.gpu>=0&&stored.gpu<=65535)savedResources.gpu=stored.gpu;
      } catch {}
      if (!schema) throw new Error('Restart ConcordTree to load pipeline settings.');
      saved=clone(schema.defaults);
      try {
        const stored=JSON.parse(localStorage.getItem(storageKey));
        for (const type of ['gene','species']) {
          const value=stored?.[type], defaults=schema.defaults[type];
          if (value?.version===schema.version && value.task===type && Array.isArray(value.stages) &&
              value.stages.length===defaults.stages.length && new Set(value.stages.map(s=>s.id)).size===defaults.stages.length &&
              value.stages.every(s=>defaults.stages.some(d=>d.id===s.id))) saved[type]=value;
        }
      } catch {}
    })().catch(e=>{schema=null;throw e;}).finally(()=>{loading=null;});
    return loading;
  }
  function resourceName() {
    const gpu=devices.find(d=>String(d.index)===String(draftResources.gpu));
    el('gpuChoiceName').textContent=gpu?gpu.name:`GPU ${draftResources.gpu}`;
  }
  function renderResources() {
    el('computeThreads').max=computeSchema.max_cpu_threads;
    el('computeThreads').value=draftResources.cpu_threads;el('computeGpu').value=draftResources.gpu;
    el('computeHint').textContent=`Up to ${computeSchema.max_cpu_threads} threads`;
    el('gpuChoices').replaceChildren(...devices.map(d=>{
      const option=document.createElement('option');option.value=d.index;option.label=d.name;return option;
    }));resourceName();
  }
  async function refreshDevices() {
    try {
      const response=await fetch('/api/host',{signal:AbortSignal.timeout(4000)});
      if(!response.ok)return;const host=await response.json();
      if(host.gpus?.length)devices=host.gpus;
      if(draftResources)renderResources();
    } catch {} // A missed telemetry sample does not change the selected GPU.
  }
  function node(tag,cls,text) {
    const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;
  }
  function button(label,fn,cls='module-action') {
    const b=node('button',cls,label);b.type='button';b.addEventListener('click',fn);return b;
  }
  function dirty() {el('settingsMessage').textContent='Unsaved changes';el('settingsError').hidden=true;preview();}
  function field(key,spec,values,scope) {
    const label=node('label','parameter-field'), text=node('span','',spec.label);
    const input=document.createElement(spec.choices?'select':'input');input.id=`parameter-${scope}-${key}`;
    input.name=`${scope}.${key}`;
    if (spec.choices) {
      spec.choices.forEach((value,i)=>{
        const option=node('option','',spec.choice_labels?.[i]||String(value));option.value=value;input.append(option);
      });
    } else {
      input.type='number';input.min=spec.min;if(spec.max!==null)input.max=spec.max;
      input.step=spec.type==='integer'?1:(spec.step||'any');
      input.required=!spec.nullable;
      if(spec.nullable)input.placeholder='Unlimited';
    }
    input.value=values[key]??'';
    input.addEventListener('input',()=>{
      const value=spec.type==='string'?input.value:input.value===''&&spec.nullable?null:Number(input.value);
      values[key]=value;
      input.setCustomValidity(spec.type==='integer'&&value!==null&&!Number.isInteger(value)?'Enter a whole number.':'');
      dirty();
    });
    label.append(text,input);return label;
  }
  function preview() {
    const box=el('pipelinePreview');box.replaceChildren();
    [node('span','pipeline-chip','AP-NJ'),...draft[task].stages.filter(s=>s.enabled).map(s=>node('span','pipeline-chip',schema.modules[s.id].label))].forEach((n,i)=>{
      if(i)box.append(node('span','pipeline-arrow','→'));box.append(n);
    });
  }
  function move(id,offset) {
    const stages=draft[task].stages,index=stages.findIndex(s=>s.id===id),next=index+offset;
    if(next<0||next>=stages.length)return;
    [stages[index],stages[next]]=[stages[next],stages[index]];dirty();render();
    el(`module-${id}`).querySelector('[data-direction]')?.focus();
  }
  function render() {
    const config=draft[task],box=el('pipelineBlocks');box.replaceChildren();preview();
    document.querySelectorAll('[data-pipeline-task]').forEach(b=>{
      const selected=b.dataset.pipelineTask===task;b.classList.toggle('active',selected);b.setAttribute('aria-pressed',String(selected));
    });
    const apnj=node('details','algorithm-block apnj-block');apnj.id='module-apnj';
    const heading=node('summary','module-heading');heading.append(node('span','module-symbol','01'),node('strong','','AP-NJ'),node('span','module-tag','Starting tree'),node('span','expand-mark','⌄'));apnj.append(heading);
    const fields=node('div','advanced-grid');
    Object.entries(schema.apnj).forEach(([key,spec])=>fields.append(field(key,spec,config.apnj,'apnj')));
    apnj.append(fields,button('Reset AP-NJ parameters',()=>{config.apnj=clone(schema.defaults[task].apnj);dirty();render();el('module-apnj').open=true;},'module-reset'));box.append(apnj);
    config.stages.forEach((stage,index)=>{
      const spec=schema.modules[stage.id],card=node('article','algorithm-block refinement-block');card.id=`module-${stage.id}`;card.dataset.module=stage.id;card.classList.toggle('is-disabled',!stage.enabled);
      const header=node('div','module-heading');
      const handle=button('⠿',()=>{},'drag-handle');handle.draggable=true;handle.title='Drag to reorder';handle.setAttribute('aria-label',`Drag ${spec.label} to reorder`);
      handle.addEventListener('dragstart',e=>{dragged=stage.id;e.dataTransfer.setData('application/x-concordtree-stage',stage.id);e.dataTransfer.effectAllowed='move';card.classList.add('is-dragging');});
      handle.addEventListener('dragend',()=>{dragged=null;document.querySelectorAll('.drop-target,.is-dragging').forEach(n=>n.classList.remove('drop-target','is-dragging'));});
      const toggle=node('label','module-toggle');const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.checked=stage.enabled;checkbox.setAttribute('aria-label',`Enable ${spec.label}`);
      checkbox.addEventListener('change',()=>{stage.enabled=checkbox.checked;card.classList.toggle('is-disabled',!stage.enabled);dirty();});toggle.append(checkbox,node('span','toggle-track'));
      const up=button('↑',()=>move(stage.id,-1)),down=button('↓',()=>move(stage.id,1));
      up.dataset.direction='up';down.dataset.direction='down';up.disabled=index===0;down.disabled=index===config.stages.length-1;up.setAttribute('aria-label',`Move ${spec.label} up`);down.setAttribute('aria-label',`Move ${spec.label} down`);
      header.append(handle,node('strong','',spec.label),up,down,toggle);card.append(header);
      const basic=node('div','basic-grid');Object.entries(spec.basic).forEach(([key,value])=>basic.append(field(key,value,stage,stage.id)));card.append(basic);
      const advanced=node('details','module-advanced');advanced.append(node('summary','','Advanced parameters'));
      const grid=node('div','advanced-grid');Object.entries(spec.advanced).forEach(([key,value])=>grid.append(field(key,value,stage.advanced,stage.id+'-advanced')));advanced.append(grid);card.append(advanced);
      card.append(button('Reset parameters',()=>{
        const defaults=clone(schema.defaults[task].stages.find(s=>s.id===stage.id));Object.assign(stage,{max_rounds:defaults.max_rounds,stop_ratio:defaults.stop_ratio,advanced:defaults.advanced});dirty();render();
      },'module-reset'));
      card.addEventListener('dragover',e=>{if(dragged&&dragged!==stage.id){e.preventDefault();e.dataTransfer.dropEffect='move';card.classList.add('drop-target');}});
      card.addEventListener('dragleave',()=>card.classList.remove('drop-target'));
      card.addEventListener('drop',e=>{
        if(!dragged||dragged===stage.id)return;e.preventDefault();e.stopPropagation();
        const from=config.stages.findIndex(s=>s.id===dragged),to=config.stages.findIndex(s=>s.id===stage.id);
        config.stages.splice(to,0,config.stages.splice(from,1)[0]);dragged=null;dirty();render();
      });box.append(card);
    });
  }
  function validate() {
    for(const input of el('runtimeSettingsFields').querySelectorAll('input,select')) {
      if(!input.checkValidity()) {
        const parent=input.closest('details');if(parent)parent.open=true;input.reportValidity();return false;
      }
    }
    return true;
  }
  async function open(type) {
    el('settingsError').hidden=true;
    try {await load();task=type;draft=clone(saved);draftResources=clone(savedResources);el('settingsMessage').textContent='';render();renderResources();void refreshDevices();}
    catch(e){el('settingsError').textContent=e.message;el('settingsError').hidden=false;}
  }
  document.addEventListener('DOMContentLoaded',()=>{
    for(const [id,key] of [['computeThreads','cpu_threads'],['computeGpu','gpu']])el(id).addEventListener('input',()=>{
      if(!draftResources)return;draftResources[key]=Number(el(id).value);resourceName();dirty();
    });
    document.querySelectorAll('[data-pipeline-task]').forEach(b=>b.addEventListener('click',()=>{
      if(!validate())return;task=b.dataset.pipelineTask;el('settingsMessage').textContent='';render();
    }));
    el('resetWorkflow').addEventListener('click',()=>{
      const stages=draft[task].stages;
      draft[task].stages=schema.defaults[task].stages.map(s=>({...stages.find(old=>old.id===s.id),enabled:true}));dirty();render();
    });
    el('resetAllSettings').addEventListener('click',()=>{draft=clone(schema.defaults);draftResources=clone(computeSchema.defaults);dirty();render();renderResources();});
    el('discardSettings').addEventListener('click',()=>el('settingsDialog').close());
    el('saveSettings').addEventListener('click',()=>{
      if(!draft||!validate())return;
      try {localStorage.setItem(storageKey,JSON.stringify(draft));localStorage.setItem(computeKey,JSON.stringify(draftResources));saved=clone(draft);savedResources=clone(draftResources);el('settingsMessage').textContent='Saved';el('settingsDialog').close();}
      catch {el('settingsError').textContent='Settings could not be saved in this browser.';el('settingsError').hidden=false;}
    });
  });
  return {open,selectedGpu(){if(savedResources)return savedResources.gpu;try{return JSON.parse(localStorage.getItem(computeKey))?.gpu;}catch{return undefined;}},async resources(){await load();return clone(savedResources);},async configuration(type){await load();return clone(saved[type]);}};
})();
