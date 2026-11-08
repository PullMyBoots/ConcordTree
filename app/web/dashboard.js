'use strict';
window.runDashboard = (() => {
  const el=id=>document.getElementById(id);
  const names={apnj:'AP-NJ',distance:'Distance NNI',mlp:'MLP NNI',likelihood:'Likelihood',kernel:'Statistical kernel'};
  let signature='',selectedStage=null,lastLiveStage=null,lastJob=null,lastEvents=[];
  function plan(job) {
    return ['apnj',...(job.configuration?.stages || [
      {id:'distance',enabled:true},{id:'mlp',enabled:true},{id:job.task==='species'?'kernel':'likelihood',enabled:true}
    ]).filter(s=>s.enabled).map(s=>s.id)];
  }
  function moduleOf(event) {
    return event.module || ({input_ready:'apnj',apnj_running:'apnj',apnj:'distance',
      gene_refinement:'likelihood',species_refinement:'kernel'})[event.stage] || event.stage;
  }
  function position(job) {
    const ids=plan(job),event=job.progress || {},index=ids.indexOf(moduleOf(event));
    let completed=0,active=0;
    if (['succeeded'].includes(job.status) || ['branch_support','complete'].includes(event.stage)) completed=ids.length;
    else if (index>=0) {
      completed=index+(event.stage==='module_complete' || event.stage==='apnj'?1:0);
      active=Math.min(completed,ids.length-1);
    } else {
      const at=ids.indexOf(job.checkpoint?.stage);
      if(at>=0){completed=at+(job.checkpoint.stage==='apnj'?1:0);active=Math.min(completed,ids.length-1);}
    }
    return {ids,completed,active};
  }
  function current(job) {
    const p=position(job);return p.ids[p.completed>=p.ids.length?p.ids.length-1:p.active];
  }
  function stages(job) {
    const {ids,completed,active}=position(job),box=el('pipelineProgress'),live=current(job);
    if(lastJob?.id!==job.id||lastLiveStage!==live)selectedStage=null;
    lastLiveStage=live;
    if(box.dataset.plan!==ids.join(',')) {
      box.replaceChildren(...ids.map(id=>{
        const li=document.createElement('li');li.dataset.module=id;
        const button=document.createElement('button');button.type='button';button.textContent=names[id];
        button.addEventListener('click',()=>{selectedStage=id;if(lastJob){stages(lastJob);history(lastJob,lastEvents);}});
        li.append(button);return li;
      }));box.dataset.plan=ids.join(',');
    }
    [...box.children].forEach((li,i)=>{
      const status=i<completed?'complete':i===active&&job.status!=='queued'&&completed<ids.length?'active':'pending';
      li.dataset.status=status;li.dataset.viewed=String(ids[i]===(selectedStage||live));
      const button=li.firstElementChild;button.disabled=status==='pending';
      if(status==='active')button.setAttribute('aria-current','step');else button.removeAttribute('aria-current');
      button.setAttribute('aria-pressed',li.dataset.viewed);
      button.title=names[ids[i]];
    });
  }
  function number(value,digits=2) {
    if(typeof value!=='number'||!Number.isFinite(value))return '—';
    if(Math.abs(value)>=1e6 || (value!==0 && Math.abs(value)<1e-3))return value.toExponential(2);
    return value.toFixed(digits);
  }
  function score(info) {
    for(const [key,title,delta] of [
      ['after_ll','Whole-tree log likelihood',false],['log_likelihood','Whole-tree log likelihood',false],
      ['summed_objective_gain','Total score gain for moved edges in this round',true],
      ['maximum_proposal_gain','Largest proposed distance improvement in this round',true],
      ['mean_probability_gain','Mean probability improvement for moved edges in this round',true]
    ])if(typeof info[key]==='number'&&Number.isFinite(info[key])) {
      const value=info[key],text=number(value,4);
      return {text:(delta&&value>0?'+':'')+text,title:`${title}: ${value}`};
    }
    return {text:'—',title:'This step does not report a single score'};
  }
  function history(job,events) {
    const id=selectedStage||current(job),rows=[],roundRows=new Map();
    let latest=moduleOf(job.progress||{})===id?job.progress?.elapsed_seconds:undefined;
    for(const event of events) {
      if(moduleOf(event)!==id)continue;
      const info={...event,...event.detail};latest=event.elapsed_seconds??latest;
      const round=info.round??info.sweep??info.pass;
      if(event.stage==='module_complete' && rows.length)continue;
      if(!Number.isSafeInteger(round)&&!(event.stage==='module_complete'||event.stage==='apnj'||typeof info.log_likelihood==='number'))continue;
      const row={round:Number.isSafeInteger(round)?round:null,seconds:event.elapsed_seconds,score:score(info),moves:info.moves};
      const key=row.round??'summary';
      if(roundRows.has(key))rows[roundRows.get(key)]=row;
      else {roundRows.set(key,rows.length);rows.push(row);}
    }
    const refinement=rows.some(r=>r.round!==null);
    // Only show fields meaningful for this stage and phase. Before NNI begins,
    // construction/model fitting needs time alone; its rounds do not exist yet.
    const metric={distance:'Distance gain',mlp:'Probability gain',likelihood:'Log likelihood',kernel:'Score gain'}[id];
    const columns=refinement?[
      {label:'Round',width:15,value:r=>number(r.round,0)},
      {label:'Time (s)',width:24,value:r=>number(r.seconds),title:'Seconds since inference started, including pauses'},
      {label:metric,width:39,value:r=>r.score.text,title:r=>r.score.title},
      {label:'Moves',width:22,value:r=>number(r.moves,0)}
    ]:[{label:'Time (s)',width:100,value:r=>number(r.seconds),title:'Seconds since inference started, including pauses'}];
    const visible=refinement?rows.filter(r=>r.round!==null):rows.length?[rows.at(-1)]:typeof latest==='number'?[{seconds:latest}]:[];
    const key=JSON.stringify([job.id,job.status,id,columns.map(c=>c.label),visible]);if(signature===key)return;signature=key;
    const panel=el('logTail'),follow=panel.scrollTop+panel.clientHeight>=panel.scrollHeight-30;
    el('logColumns').replaceChildren(...columns.map(c=>{const col=document.createElement('col');col.style.width=c.width+'%';return col;}));
    const head=document.createElement('tr');
    columns.forEach(c=>{const th=document.createElement('th');th.scope='col';th.textContent=c.label;head.append(th);});
    el('logHead').replaceChildren(head);
    el('logRows').replaceChildren(...visible.map(row=>{
      const tr=document.createElement('tr');tr.dataset.module=id;
      columns.forEach(c=>{const cell=document.createElement('td');cell.textContent=c.value(row);if(c.title)cell.title=typeof c.title==='function'?c.title(row):c.title;tr.append(cell);});return tr;
    }));
    el('logEmpty').hidden=visible.length>0;
    el('logEmpty').textContent=job.status==='queued'?'Waiting to start':job.status==='cancelled'?'Cancelled':job.status==='failed'?'Stopped':job.status==='paused'?'Paused':'Preparing…';
    if(follow)requestAnimationFrame(()=>{panel.scrollTop=panel.scrollHeight;});
  }
  return {plan,position,stages,render(job,events){stages(job);lastJob=job;lastEvents=events;history(job,events);},clear(){signature='';selectedStage=null;lastLiveStage=null;lastJob=null;lastEvents=[];el('logRows').replaceChildren();el('logHead').replaceChildren();el('logColumns').replaceChildren();el('pipelineProgress').replaceChildren();delete el('pipelineProgress').dataset.plan;el('logEmpty').hidden=false;el('logEmpty').textContent='Waiting to start';}};
})();
