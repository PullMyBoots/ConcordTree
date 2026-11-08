'use strict';
const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const PREVIEW_LIMIT = 400;
const labels = {gene: 'Gene Tree', species: 'Species Tree'};
const state = {task: 'gene', file: null, compute: false, busy: false, submitting: false, tree: null, supportMethod: null, job: null, poll: null, fileSequence: 0, queue: null, queueStatus: 'loading'};

function error(message = '') {
  $('formError').textContent = message;
  $('formError').hidden = !message;
}
function refreshControls() {
  $('taskForm').querySelectorAll('button,input,select').forEach(el => { el.disabled = state.busy; });
  $('taskForm').setAttribute('aria-busy', String(state.submitting));
  $('backButton').disabled = state.busy;
  $('cancelButton').hidden = !state.busy || state.submitting;
  const status=state.job?.status;
  $('pauseButton').hidden=!['running','pausing','paused'].includes(status);
  $('pauseButton').disabled=status==='pausing';
  $('pauseButton').textContent=status==='paused'?'Resume':status==='pausing'?'Pausing…':'Pause';
  $('checkpointDownload').hidden=status!=='paused'||!state.job?.checkpoint?.available;
  if(status==='paused')$('checkpointDownload').href=`/api/jobs/${state.job.id}/checkpoint`;
}
function selectTask(task, restored = false) {
  if ((state.busy && !restored) || !Object.hasOwn(labels, task)) return;
  if (state.task !== task) clearFile();
  state.task = task;
  $('alignmentFile').accept = task === 'gene' ? '.phy,.phylip,.txt' : '.phy,.phylip,.paml,.txt';
  document.querySelectorAll('[data-task]').forEach(button => {
    const active = button.dataset.task === task;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
  error();
}
function clearFile() {
  state.fileSequence++;
  state.file = null;
  $('alignmentFile').value = '';
  $('fileCard').hidden = true;
  $('uploadTrigger').hidden = false;
  error();
  refreshControls();
}
async function firstAlignmentLine(file) {
  const reader = file.stream().getReader(), decoder = new TextDecoder();
  let pending = '';
  try {
    while (true) {
      const {value, done} = await reader.read();
      pending += decoder.decode(value, {stream: !done});
      let end;
      while ((end = pending.search(/[\r\n]/)) !== -1) {
        const line = pending.slice(0, end).trim();
        pending = pending.slice(end + 1);
        if (line) return line;
      }
      if (done) return pending.trim();
    }
  } finally {
    await reader.cancel();
    reader.releaseLock();
  }
}
async function useFile(file) {
  if (!file || state.busy) return;
  clearFile();
  const sequence = state.fileSequence;
  if (!file.size) return error('The file is empty.');
  try {
    const firstLine = await firstAlignmentLine(file);
    if (sequence !== state.fileSequence) return;
    const header = firstLine.match(/^(\d+)\s+(\d+)(?:\s+[IS])?$/i);
    if (!header) throw new Error(state.task === 'gene' ? 'Upload a PHYLIP alignment, not FASTA.' : 'Upload PHYLIP/PAML locus blocks, each starting with sequence and site counts.');
    const taxa = Number(header[1]), sites = Number(header[2]);
    if (!Number.isSafeInteger(taxa) || !Number.isSafeInteger(sites) || taxa < (state.task === 'gene' ? 4 : 2) || sites < 1) throw new Error('Invalid PHYLIP sequence or site counts.');
    state.file = file;
    $('fileName').textContent = file.name;
    $('fileMeta').textContent = state.task === 'gene' ? `${taxa.toLocaleString()} taxa · ${sites.toLocaleString()} sites` : 'Checking locus alignments…';
    $('fileCard').hidden = false;
    $('uploadTrigger').hidden = true;
    refreshControls();
    await submitAlignment();
  } catch (e) { if (sequence === state.fileSequence) error(e.message || 'Could not read the file.'); }
}
// Iterative parser/layout: avoid recursion depth failures on ladder-like trees.
function parseTree(text) {
  const tokens = text.match(/\[[^\]]*\]|'(?:[^']|'')*'|[(),:;]|[^\s(),:;\[\]]+/g) || [];
  let root, current, expectLength = false;
  const stack = [];
  for (const token of tokens) {
    if (token.startsWith('[')) continue;
    if (token === '(') {
      const node = {children: [], name: '', length: 0};
      if (stack.length) stack.at(-1).children.push(node); else if (root) throw new Error('Only one tree can be previewed.'); else root = node;
      stack.push(node); current = node;
    } else if (token === ',') current = null;
    else if (token === ')') { current = stack.pop(); if (!current) throw new Error('Unmatched Newick parentheses.'); }
    else if (token === ':') { if (!current) throw new Error('Invalid Newick branch length.'); expectLength = true; }
    else if (token === ';') break;
    else if (expectLength) { const length = Number(token); if (!Number.isFinite(length) || length < 0) throw new Error('Invalid Newick branch length.'); current.length = length; expectLength = false; }
    else if (current && current.children.length) current.support = token;
    else {
      if (!stack.length) throw new Error('Invalid Newick structure.');
      current = {children: [], name: token.startsWith("'") ? token.slice(1, -1).replace(/''/g, "'") : token, length: 0};
      stack.at(-1).children.push(current);
    }
  }
  if (!root || stack.length || expectLength || !text.trim().endsWith(';')) throw new Error('Incomplete Newick tree.');
  return root;
}
function svgNode(name, attributes = {}, text) {
  const el = document.createElementNS(NS, name);
  Object.entries(attributes).forEach(([key, value]) => el.setAttribute(key, value));
  if (text !== undefined) el.textContent = text;
  return el;
}
const viewport = {x: 0, y: 0, scale: 1, width: 760, height: 500};
function transformTree() {
  const group = $('treeDrawing');
  const cx = viewport.width / 2, cy = viewport.height / 2;
  if (group) group.setAttribute('transform', `translate(${viewport.x} ${viewport.y}) translate(${cx} ${cy}) scale(${viewport.scale}) translate(${-cx} ${-cy})`);
}
function resetView() { viewport.x = 0; viewport.y = 0; viewport.scale = 1; $('treeScroll').scrollTo(0, 0); transformTree(); }
function drawTree() {
  const svg = $('treeSvg');
  svg.replaceChildren();
  $('canvasEmpty').hidden = true;
  $('resetView').hidden = true;
  svg.hidden = false;
  resetView();
  if (!state.tree) return;
  try {
    const root = parseTree(state.tree), order = [root], leaves = [];
    root.depth = 0; root.distance = 0;
    for (let i = 0; i < order.length; i++) {
      const node = order[i];
      for (const child of node.children) {
        child.parent = node; child.depth = node.depth + 1; child.distance = node.distance + child.length; order.push(child);
      }
    }
    // A subtree's tips must form a contiguous interval. Breadth-first tip order
    // interleaves unrelated clades and makes binary branches appear to cross.
    const pending = [root];
    while (pending.length) {
      const node = pending.pop();
      if (node.children.length) pending.push(...[...node.children].reverse());
      else leaves.push(node);
    }
    if (leaves.length > PREVIEW_LIMIT) throw new Error(`${leaves.length.toLocaleString()} taxa. Preview supports up to ${PREVIEW_LIMIT} taxa. Download the full tree to view it.`);
    const positions = new Map(leaves.map((leaf, i) => [leaf, i]));
    const maxDistance = Math.max(...leaves.map(node => node.distance));
    const maxDepth = Math.max(...leaves.map(node => node.depth));
    const measure = document.createElement('canvas').getContext('2d');
    const family = getComputedStyle(svg).fontFamily;
    measure.font = `13px ${family}`;
    const nameWidth = Math.max(...leaves.map(node => measure.measureText(node.name.replace(/_/g, ' ')).width));
    const treeWidth = Math.max(540, maxDepth * 52);
    viewport.width = Math.ceil(50 + treeWidth + nameWidth + 40);
    viewport.height = Math.max(300, 80 + (leaves.length - 1) * 28);
    svg.setAttribute('viewBox', `0 0 ${viewport.width} ${viewport.height}`);
    const available = $('treeScroll').clientWidth || 894;
    const displayWidth = Math.max(760, Math.min(viewport.width, Math.max(available, viewport.width * .8)));
    svg.style.width = `${displayWidth}px`;
    svg.style.height = `${displayWidth * viewport.height / viewport.width}px`;
    for (const node of [...order].reverse()) {
      node.position = node.children.length ? (node.children[0].position + node.children.at(-1).position) / 2 : positions.get(node);
      node.x = 50 + treeWidth * (maxDistance > 0 ? node.distance / maxDistance : node.depth / (maxDepth || 1));
      node.y = 40 + node.position * 28;
    }
    svg.setAttribute('aria-label', `${leaves.length}-taxon inferred tree`);
    const group = svgNode('g', {id: 'treeDrawing'});
    const supports = [], occupied = [];
    for (const node of order.slice(1)) {
      group.append(svgNode('path', {d: `M${node.parent.x},${node.parent.y}V${node.y}H${node.x}`, stroke: '#52734a', fill: 'none'}));
      if (node.children.length && node.support !== undefined && Number.isFinite(Number(node.support))) {
        const value = Number(node.support);
        const label = svgNode('text', {'text-anchor': 'end', class: 'branch-support'}, value.toFixed(1));
        label.append(svgNode('title', {}, `${state.supportMethod || 'Branch support'}: ${node.support}`));
        supports.push({node, label, value: value.toFixed(1)});
      }
      if (!node.children.length) {
        const label = svgNode('text', {x: node.x + 10, y: node.y + 4, class: 'tip-label'}, node.name.replace(/_/g, ' '));
        label.append(svgNode('title', {}, node.name));
        group.append(label);
        occupied.push({x: node.x + 10, y: node.y - 10, width: measure.measureText(node.name.replace(/_/g, ' ')).width, height: 16});
      }
    }
    measure.font = `9px ${family}`;
    const overlaps = (a, b) => a.x < b.x + b.width + 3 && a.x + a.width + 3 > b.x && a.y < b.y + b.height + 2 && a.y + a.height + 2 > b.y;
    for (const {node, label, value} of supports.sort((a, b) => b.node.x - a.node.x)) {
      const width = measure.measureText(value).width;
      let spot;
      search: for (const dy of [-6, 14, -21, 29, -36, 44, -51, 59]) {
        for (const dx of [-5, -20, 10, -35, 25, -50, 40]) {
          const box = {x: node.x + dx - width, y: node.y + dy - 9, width, height: 11};
          if (box.x < 4 || box.y < 4 || box.x + width > viewport.width - 4 || box.y + box.height > viewport.height - 4) continue;
          if (!occupied.some(other => overlaps(box, other))) { spot = box; break search; }
        }
      }
      spot ||= {x: node.x - width - 5, y: node.y - 15, width, height: 11};
      label.setAttribute('x', spot.x + width); label.setAttribute('y', spot.y + 9);
      occupied.push(spot);
      if (Math.abs(spot.y + 9 - (node.y - 6)) > 1 || Math.abs(spot.x + width - (node.x - 5)) > 1) {
        group.append(svgNode('path', {class: 'support-leader', d: `M${node.x - 2},${node.y}L${spot.x + width / 2},${spot.y + 10}`, stroke: '#a8b59f', fill: 'none'}));
      }
      group.append(label);
    }
    svg.append(group);
    $('resetView').hidden = false;
  } catch (e) {
    svg.hidden = true;
    $('canvasEmpty').textContent = e.message;
    $('canvasEmpty').hidden = false;
  }
}
async function api(path, options = {}) {
  const response = await fetch(path, {...options, signal: Object.hasOwn(options, 'signal') ? options.signal : AbortSignal.timeout(20000)});
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const failure = new Error(typeof body.detail === 'string' ? body.detail : `Request failed (${response.status})`);
    failure.status = response.status;
    throw failure;
  }
  return body;
}
function renderQueue(queue, status = 'offline') {
  if (queue) status = !queue.compute_enabled ? 'paused' : !queue.accepting_jobs ? 'full' : queue.running || queue.queued ? 'busy' : 'idle';
  state.queueStatus = status;
  if (!queue && status === 'offline') renderHost(null);
}
function percent(value) { return typeof value === 'number' && Number.isFinite(value) ? `${Math.round(value)}%` : '—'; }
function gib(value) { return typeof value === 'number' ? (value / 1024**3).toFixed(1) : '—'; }
let lastHost=null;
function renderHost(host) {
  if(host)lastHost=host;else if(lastHost)host={...lastHost,connection_stale:true};
  $('hostLoad').dataset.status = host ? 'live' : 'offline';
  const stale = host?.connection_stale || (host?.sampled_at && Date.now() - Date.parse(host.sampled_at) > 15000);
  $('hostStatus').textContent = !host ? 'Reconnecting…' : stale ? 'Last sample · reconnecting' : !host.sampled_at ? 'Sampling…' : 'Live · 2 s';
  const cpu = host?.cpu?.utilization_percent, memory = host?.memory;
  const selectedIndex=state.busy?state.job?.resources?.gpu:window.pipelineSettings?.selectedGpu();
  const selected = host?.gpus?.find(gpu => String(gpu.index)===String(selectedIndex)) || host?.gpus?.find(gpu=>gpu.selected);
  $('hostSummary').textContent = host ? `CPU ${percent(cpu)} · RAM ${percent(memory?.utilization_percent)} · GPU ${percent(selected?.utilization_percent)}` : 'Reconnecting…';
  $('cpuUsage').textContent = percent(cpu); $('cpuMeter').value = cpu ?? 0;
  $('cpuMeter').hidden = typeof cpu !== 'number';
  $('cpuDetail').textContent = host?.cpu?.logical_cores ? `${host.cpu.logical_cores} logical cores` : 'Unavailable';
  $('memoryUsage').textContent = percent(memory?.utilization_percent);
  $('memoryMeter').value = memory?.utilization_percent ?? 0; $('memoryMeter').hidden = !memory;
  $('memoryDetail').textContent = memory ? `${gib(memory.used_bytes)} / ${gib(memory.total_bytes)} GiB` : 'Unavailable';
  const cards = [];
  for (const gpu of host?.gpus || []) {
    const card = document.createElement('div'); card.className = 'gpu-card';
    const heading = document.createElement('div'); heading.className = 'gpu-heading';
    const name = document.createElement('span'); name.textContent = `GPU ${gpu.index} · ${gpu.name}`;
    heading.append(name);
    if (String(gpu.index)===String(selected?.index)) { const badge = document.createElement('small'); badge.textContent = 'Inference'; heading.append(badge); }
    if(host.gpu_status==='stale'||stale){const note=document.createElement('small');note.textContent='Last reading · retrying';heading.append(note);}
    card.append(heading);
    const metrics = document.createElement('div'); metrics.className = 'gpu-readings';
    for (const [label, usage, detail] of [
      ['Utilization', gpu.utilization_percent, ''],
      ['VRAM', gpu.memory_used_bytes !== null && gpu.memory_total_bytes > 0 ? gpu.memory_used_bytes / gpu.memory_total_bytes * 100 : null,
        gpu.memory_used_bytes !== null && gpu.memory_total_bytes > 0 ? `${gib(gpu.memory_used_bytes)} / ${gib(gpu.memory_total_bytes)} GiB` : 'Unavailable']
    ]) {
      const field = document.createElement('div'); field.className = 'host-metric';
      const row = document.createElement('div'), title = document.createElement('span'), value = document.createElement('strong');
      title.textContent = label; value.textContent = percent(usage); row.append(title, value); field.append(row);
      if (typeof usage === 'number') {
        const meter = document.createElement('meter'); meter.min = 0; meter.max = 100; meter.value = usage;
        meter.setAttribute('aria-label', `GPU ${gpu.index} ${label}`); field.append(meter);
      }
      if (detail) { const text = document.createElement('small'); text.textContent = detail; field.append(text); }
      metrics.append(field);
    }
    card.append(metrics); cards.push(card);
  }
  if (!cards.length) {
    const empty = document.createElement('p'); empty.className = 'host-unavailable';
    empty.textContent = !host ? 'GPU telemetry unavailable' : host.gpu_status === 'sampling' ? 'Checking NVIDIA GPUs…' : 'NVIDIA GPU telemetry unavailable'; cards.push(empty);
  }
  $('gpuMetrics').replaceChildren(...cards);
}
let queueTimer = null, queueRequest = null;
function refreshQueue(force = false) {
  clearTimeout(queueTimer);
  if (queueRequest) return queueRequest;
  if (document.hidden && !force) return Promise.resolve(null);
  queueRequest = (async () => {
    try {
      const queue = await api('/api/queue', {signal: AbortSignal.timeout(6000)});
      if (![queue.running, queue.queued, queue.capacity].every(Number.isSafeInteger) ||
          queue.capacity < 1 || queue.capacity > 128 || queue.running < 0 || queue.queued < 0 ||
          queue.running + queue.queued > queue.capacity ||
          typeof queue.compute_enabled !== 'boolean' || typeof queue.accepting_jobs !== 'boolean') throw new Error('Invalid queue status');
      state.compute = queue.compute_enabled; state.queue = queue;
      renderQueue(queue);
      try { renderHost(await api('/api/host', {signal: AbortSignal.timeout(4000)})); }
      catch { renderHost(null); }
      return queue;
    } catch {
      state.compute = false; renderQueue(null);
      return null;
    } finally {
      queueRequest = null;
      if (!document.hidden) queueTimer = setTimeout(() => refreshQueue(), 2000);
    }
  })();
  return queueRequest;
}
function setPage(result) {
  document.body.classList.toggle('viewing-result', result);
  $('submitPage').hidden = result;
  $('resultPage').hidden = !result;
  const container = result ? $('resultPage') : $('taskForm');
  if ($('hostLoad').parentElement !== container) container.append($('hostLoad'));
  syncGrowthMotion();
}
function setStatus(title, detail = '', failed = false) {
  $('resultTitle').textContent = title;
  $('jobStatus').textContent = detail;
  $('jobStatus').hidden = !detail;
  $('jobStatus').classList.toggle('error', failed);
}
const progressState = {step: 0, total: 6};
// Visual growth eases between observed stages; only the backend marks completion.
// Keep the stage value in ARIA; the decorative tree is not an elapsed-time estimate.
const growth = {parts: [], leaves: [], value: 0, target: 0, frame: 0, last: 0,
  stageStarted: 0, mode: 'idle', motion: matchMedia('(prefers-reduced-motion: reduce)')};
function prepareGrowth() {
  if (growth.parts.length) return;
  let serial = 0;
  function branch(x, y, length, angle, depth) {
    const id = serial++, bend = Math.sin(id*2.37)*.08;
    const nx = x + Math.sin(angle)*length, ny = y - Math.cos(angle)*length;
    const d = `M${x} ${y} C${x+Math.sin(angle-bend)*length*.35} ${y-Math.cos(angle)*length*.35} ${nx-Math.sin(angle+bend)*length*.2} ${ny+Math.cos(angle+bend)*length*.2} ${nx} ${ny}`;
    const path = svgNode('path', {d, class: 'growth-branch', 'stroke-width': Math.max(.8, 5.5-depth*.9), pathLength: 1});
    $('growthBranches').append(path);
    const end = .16+depth*.14;
    const part = {path, start: .02+depth*.14, end};
    growth.parts.push(part);
    if (depth < 3) {
      const sap = svgNode('circle', {r: depth ? 1.1 : 1.6, class: 'growth-sap'});
      sap.append(svgNode('animateMotion', {path: d, dur: `${4+depth*1.4}s`, begin: `${-id*1.17}s`, repeatCount: 'indefinite'}));
      $('growthSap').append(sap); part.sap = sap;
    }
    if (depth < 5) {
      branch(nx, ny, length*(.75+Math.sin(id*1.9)*.025), angle-.43+bend, depth+1);
      branch(nx, ny, length*(.78+Math.cos(id*1.3)*.025), angle+.45+bend, depth+1);
    }
    if (depth >= 3) {
      const terminal = depth === 5;
      const holder = svgNode('g');
      const leaf = svgNode('path', {d: terminal ? 'M0 2C-9-3-7-13 0-16C7-10 7-2 0 2Z' : 'M0 1C-5-2-5-9 0-11C5-7 5-2 0 1Z', class: 'growth-leaf'});
      leaf.style.setProperty('--leaf-delay', `${-id*.63}s`);
      holder.append(leaf); $('growthLeaves').append(holder);
      growth.leaves.push({holder, x: nx, y: ny, angle: angle*180/Math.PI+(id%2 ? 22 : -22), start: end, end: Math.min(.99, end+.10+(id%4)*.012)});
    }
  }
  branch(200, 274, 69, -.025, 0);
  for (const side of [-1, 1]) {
    const path = svgNode('path', {d: `M200 277 Q${200+side*9} 289 ${200+side*32} 294`, class: 'growth-root', pathLength: 1});
    $('growthRoots').append(path); growth.parts.push({path, start: 0, end: .15});
  }
  for (let i = 0; i < 9; i++) {
    const mote = svgNode('circle', {cx: 145+(i*37)%115, cy: 265-(i*23)%90, r: i%3 ? .9 : 1.4, class: 'growth-mote'});
    mote.style.setProperty('--mote-delay', `${-i*1.37}s`);
    mote.style.setProperty('--mote-drift', `${(i%2 ? 1 : -1)*(8+i)}px`);
    $('growthMotes').append(mote);
  }
}
function drawGrowth(value) {
  prepareGrowth(); growth.value = value;
  for (const part of growth.parts) {
    const fraction = Math.max(0, Math.min(1, (value-part.start)/(part.end-part.start)));
    part.path.style.strokeDashoffset = String(1-fraction);
    part.path.style.opacity = fraction > 0 ? '1' : '0';
    if (part.sap) part.sap.style.opacity = fraction >= 1 ? '.75' : '0';
  }
  for (const leaf of growth.leaves) {
    const scale = Math.max(0, Math.min(1, (value-leaf.start)/(leaf.end-leaf.start)));
    leaf.holder.setAttribute('transform', `translate(${leaf.x} ${leaf.y}) rotate(${leaf.angle}) scale(${scale})`);
  }
  $('growthSeed').style.opacity = String(Math.max(.15, 1-value*2));
}
function growthVisible() {
  return !document.hidden && !$('resultPage').hidden && !$('inferenceProgress').hidden;
}
function syncGrowthMotion() {
  cancelAnimationFrame(growth.frame); growth.frame = 0; growth.last = 0;
  const active = growthVisible() && !growth.motion.matches;
  $('progressBar').classList.toggle('growth-active', active && ['running', 'waiting'].includes(growth.mode));
  const svg = $('growthTree');
  if (active && growth.mode === 'running') svg.unpauseAnimations(); else svg.pauseAnimations();
  if (!active) {
    if (growth.motion.matches) drawGrowth(growth.target);
    return;
  }
  function frame(now) {
    if (!growth.last) growth.last = now;
    const dt = Math.min(100, now-growth.last);
    if (dt < 32) { growth.frame = requestAnimationFrame(frame); return; }
    growth.last = now;
    let target = growth.target;
    if (growth.mode === 'running') {
      // Slow, bounded unfurling within this stage; never finish before success.
      const withinStage = .8/progressState.total*(1-Math.exp(-(now-growth.stageStarted)/90000));
      target = Math.min(.96, target+withinStage);
    }
    target = Math.max(growth.value, target);
    const next = growth.value+(target-growth.value)*(1-Math.exp(-dt/(growth.mode === 'complete' ? 900 : 2600)));
    if (Math.abs(next-growth.value) > .000001) drawGrowth(next);
    if (growth.mode === 'complete' && 1-next < .0002) { drawGrowth(1); growth.frame = 0; return; }
    if (['stopped', 'idle', 'waiting', 'paused'].includes(growth.mode)) { growth.frame = 0; return; }
    growth.frame = requestAnimationFrame(frame);
  }
  growth.frame = requestAnimationFrame(frame);
}
growth.motion.addEventListener('change', syncGrowthMotion);
document.addEventListener('visibilitychange', syncGrowthMotion);
function showProgress(message, detail = '', {step = progressState.step, total = progressState.total, mode = 'running'} = {}) {
  progressState.step = Math.max(progressState.step, step); progressState.total = total;
  const wasHidden = $('inferenceProgress').hidden;
  $('inferenceProgress').hidden = false; $('inferenceProgress').dataset.mode = mode;
  if (wasHidden) requestAnimationFrame(() => { $('logTail').scrollTop = $('logTail').scrollHeight; });
  const value = Math.min(1, progressState.step/total), percentage = Math.round(value*100);
  if (growth.target !== value || growth.mode !== mode) growth.stageStarted = performance.now();
  growth.target = value; growth.mode = mode;
  prepareGrowth(); syncGrowthMotion();
  $('progressBar').setAttribute('aria-valuenow', String(percentage));
  $('progressBar').setAttribute('aria-valuetext', `${percentage}% of stages. ${message}. ${detail}`);
  $('progressBar').setAttribute('aria-busy', String(mode === 'running'));
}
let logRequest = null;
async function refreshLog(job) {
  if (logRequest === job.id) return;
  logRequest = job.id;
  try {
    const data = await api(`/api/jobs/${job.id}/log`);
    if (state.job?.id !== job.id) return;
    window.runDashboard.render(job,data.events || []);

  } catch { /* Keep the last verified stage and rows during reconnection. */ }
  finally { if (logRequest === job.id) logRequest = null; }
}
function jobProgressPlan(job) {
  const ids=window.runDashboard.plan(job);return {ids,total:ids.length};
}
function updateProgress(job) {
  window.runDashboard.stages(job);
  const {ids,completed,active}=window.runDashboard.position(job);
  const waiting=job.status==='queued';
  const step=waiting?0:Math.min(ids.length-.05,completed+(completed<ids.length ? .35 : 0));
  showProgress(waiting?'Waiting':'Inferring','',{step,total:ids.length,mode:waiting?'waiting':'running'});
}
function clearResult() {
  $('resultTiming').hidden = true;
  $('pauseButton').hidden=true;$('checkpointDownload').hidden=true;$('checkpointDownload').removeAttribute('href');
  state.tree = null; state.supportMethod = null;
  $('supportLegend').hidden = true;
  $('resultActions').hidden = true;
  $('treeCanvas').hidden = true;
  $('downloadTree').removeAttribute('href');
  $('previewButton').hidden = true;
  $('previewButton').textContent = 'Preview';
  $('previewButton').setAttribute('aria-expanded', 'false');
  $('treeSvg').replaceChildren();
  $('inferenceProgress').hidden = true;
  window.runDashboard.clear();
  progressState.step = 0; progressState.total = 6;
  cancelAnimationFrame(growth.frame); growth.target = 0; growth.mode = 'idle'; drawGrowth(0); syncGrowthMotion();
}
function formatElapsed(seconds) {
  if (Math.round(seconds * 100) < 6000) return `${seconds.toFixed(2)} s`;
  const total = Math.round(seconds), hours = Math.floor(total / 3600);
  const minutes = Math.floor(total % 3600 / 60), remainder = String(total % 60).padStart(2, '0');
  return hours ? `${hours}h ${String(minutes).padStart(2, '0')}m ${remainder}s` : `${minutes}m ${remainder}s`;
}
async function showJob(job) {
  if (state.job?.id !== job.id) return;
  state.job = job;
  window.runDashboard.stages(job);
  void refreshLog(job);
  state.busy = ['queued', 'running', 'pausing', 'paused'].includes(job.status);
  if (state.task !== job.task) selectTask(job.task, true);
  if (typeof job.support === 'boolean') $('showSupport').checked = job.support;
  clearTimeout(state.poll);
  refreshControls();
  renderQueue(state.queueStatus === 'offline' ? null : state.queue, state.queueStatus);
  const meta = job.metadata;
  const detail = meta ? ` · ${meta.taxa.toLocaleString()} taxa · ${meta.sites.toLocaleString()} sites${job.task === 'species' ? ` · ${meta.loci} ${meta.loci === 1 ? 'locus' : 'loci'}` : ''}` : '';
  $('resultDetail').textContent = `${labels[job.task] || 'Inference'} · ${job.filename}${detail}`;
  const timed = job.status === 'succeeded' && typeof job.elapsed_seconds === 'number' && Number.isFinite(job.elapsed_seconds) && job.elapsed_seconds >= 0;
  $('resultTiming').hidden = !timed;
  if (timed) {
    $('resultTiming').textContent = `Inference time · ${formatElapsed(job.elapsed_seconds)}`;
    $('resultTiming').title = `${job.elapsed_seconds.toFixed(2)} seconds; excludes upload, queueing and process startup.`;
  }
  if (job.status === 'succeeded') {
    setStatus('Loading result');
    const data = await api(`/api/jobs/${job.id}/tree`);
    if (state.job?.id !== job.id) return;
    state.tree = data.newick;
    const support = data.branch_support || job.branch_support;
    state.supportMethod = support?.enabled ? support.label : null;
    $('supportLegend').hidden = true;
    $('supportLegend').textContent = state.supportMethod ? `${state.supportMethod} · 0–100 · not a correctness probability` : '';
    $('downloadTree').href = `/api/jobs/${job.id}/download`;
    $('resultActions').hidden = false;
    $('treeCanvas').hidden = true;
    const taxa = meta?.taxa ?? (() => {
      const pending = [parseTree(state.tree)]; let count = 0;
      while (pending.length) { const node = pending.pop(); if (node.children.length) pending.push(...node.children); else count++; }
      return count;
    })();
    $('previewButton').hidden = taxa > PREVIEW_LIMIT;
    $('previewButton').textContent = 'Preview';
    $('previewButton').setAttribute('aria-expanded', 'false');
    const {total}=jobProgressPlan(job);
    showProgress('Inference complete', '', {step: total, total, mode: 'complete'});
    const guarded = job.quality?.guard_interventions?.length > 0;
    const uninformative = job.quality?.guard_interventions?.some(item => item.reason === 'no_variable_sites');
    const warning = guarded ? (uninformative
      ? 'The alignment has no variable sites. Its branching order cannot be determined from these sequences.'
      : job.quality.search_converged === false || job.quality.guard_interventions.some(item => item.stage === 'parameter_fit')
        ? 'Search stopped before convergence was confirmed. The best verified tree is available for download.'
        : 'An initialization search was stopped to avoid repeated states. Final refinement completed.') : '';
    setStatus(guarded ? 'Complete · search notice' : 'Complete', warning);
  } else if (job.status === 'failed') {
    const message = job.error?.includes('exceeds triton maximum tensor numel') ? 'This run hit a GPU kernel limit for long alignments. The backend has been updated; submit again to retry.' : job.error;
    setStatus('Inference failed', message || 'Check your alignment and try again.', true);
    if (!$('inferenceProgress').hidden) showProgress('Inference stopped', 'See the error above', {mode: 'stopped'});
  }
  else if (job.status === 'cancelled') {
    setStatus('Cancelled');
    if (!$('inferenceProgress').hidden) showProgress('Cancelled', '', {mode: 'stopped'});
  }
  else {
    if(job.status==='paused') {
      setStatus('Paused');updateProgress(job);showProgress('Paused','',{mode:'paused'});drawGrowth(Math.max(growth.value,growth.target));
    } else {
      setStatus(job.status==='pausing'?'Pausing at next complete tree':job.status==='queued'?'Queued':'Inferring');
      updateProgress(job);
    }
    pollJob(job.id);
  }
}
function pollJob(id) {
  clearTimeout(state.poll);
  state.poll = setTimeout(async () => {
    if (state.job?.id !== id) return;
    try { await showJob(await api(`/api/jobs/${id}`)); }
    catch {
      if (state.job?.id === id) { setStatus('Reconnecting'); showProgress('Connection lost', 'Reconnecting to your task…', {mode: 'waiting', record: false}); pollJob(id); }
    }
  }, 1500);
}
async function route() {
  if (state.submitting) return;
  const match = location.hash.match(/^#job=([0-9a-f]{32})$/);
  setPage(Boolean(match));
  if (!match) return;
  if (state.job?.id === match[1]) return;
  clearResult(); clearTimeout(state.poll);
  state.job = {id: match[1]};
  setStatus('Loading task');
  $('resultDetail').textContent = '';
  try { await showJob(await api(`/api/jobs/${match[1]}`)); }
  catch (e) {
    if (state.job?.id === match[1]) { state.busy = false; refreshControls(); setStatus('Could not load task', e.message, true); }
  }
}
async function submitAlignment(serverPath = null) {
  if (!state.file || state.busy) return;
  error(); clearResult(); state.busy = true; state.submitting = true; state.job = null;
  refreshControls(); setPage(true); setStatus('Connecting');
  showProgress('Checking host', '');
  $('resultDetail').textContent = `${labels[state.task]} · ${state.file.name}`;
  const queue = await refreshQueue(true);
  if (!queue || !queue.compute_enabled || !queue.accepting_jobs) {
    state.busy = false; state.submitting = false; refreshControls(); setPage(false);
    return error(!queue ? 'Cannot connect to ConcordTree. Your file has not been uploaded.' : !queue.compute_enabled ? 'Inference is paused. Your file has not been submitted.' : 'Queue full. Please try again later.');
  }
  setStatus(serverPath ? 'Reading server file' : 'Uploading');
  showProgress(serverPath ? 'Reading alignment' : 'Uploading alignment', 'Preparing your input');
  const form = new FormData();
  if (serverPath) form.append('path', serverPath); else form.append('file', state.file);
  form.append('task', state.task);
  form.append('support', String($('showSupport').checked));
  try {
    if (window.pipelineSettings) {
      form.append('configuration',JSON.stringify(await window.pipelineSettings.configuration(state.task)));
      form.append('resources',JSON.stringify(await window.pipelineSettings.resources()));
    }
    const job = await api(serverPath ? '/api/jobs/from-server' : '/api/jobs', {method: 'POST', body: form, signal: null});
    state.submitting = false; state.job = job;
    location.hash = `job=${job.id}`;
  } catch (e) {
    state.busy = false; state.submitting = false; refreshControls();
    setPage(false);
    error(e.status ? e.message : 'Upload could not be confirmed. Check your connection before submitting again.');
    refreshQueue(true);
    return;
  }
  const activeJob = state.job;
  try { await showJob(activeJob); }
  catch {
    if (state.job?.id === activeJob.id) {
      setStatus('Reconnecting'); showProgress('Connection lost', 'Reconnecting to your task…', {mode: 'waiting', record: false}); pollJob(activeJob.id);
    }
  }
  refreshQueue(true);
  if (state.job?.id === activeJob.id) $('resultTitle').focus();
}
// Module and resource drafts are managed in settings.js.
(() => {
  const dialog=$('settingsDialog');
  $('settingsButton').addEventListener('click',async()=>{dialog.showModal();await window.pipelineSettings.open(state.task);});
  $('closeSettings').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('click',event=>{
    const box=dialog.getBoundingClientRect();
    if(event.target===dialog&&(event.clientX<box.left||event.clientX>box.right||event.clientY<box.top||event.clientY>box.bottom))dialog.close();
  });
})();
$('taskForm').addEventListener('submit', event => event.preventDefault());
$('backButton').addEventListener('click', () => { if (!state.busy) { clearTimeout(state.poll); state.job = null; clearFile(); location.hash = 'submit'; setPage(false); } });
$('pauseButton').addEventListener('click', async () => {
  if(!state.job)return;
  $('pauseButton').disabled=true;
  try {await showJob(await api(`/api/jobs/${state.job.id}/${state.job.status==='paused'?'resume':'pause'}`,{method:'POST'}));}
  catch(e){$('jobStatus').hidden=false;$('jobStatus').textContent=e.message;refreshControls();}
});
$('cancelButton').addEventListener('click', async () => {
  if (!state.job) return;
  try { await showJob(await api(`/api/jobs/${state.job.id}/cancel`, {method: 'POST'})); refreshQueue(); }
  catch (e) { $('jobStatus').hidden = false; $('jobStatus').textContent = e.message; }
});
$('previewButton').addEventListener('click', () => {
  if ($('previewButton').hidden || !state.tree) return;
  $('treeCanvas').hidden = !$('treeCanvas').hidden;
  if (!$('treeCanvas').hidden && !$('treeDrawing')) drawTree();
  $('previewButton').textContent = $('treeCanvas').hidden ? 'Preview' : 'Hide Preview';
  $('previewButton').setAttribute('aria-expanded', String(!$('treeCanvas').hidden));
  $('supportLegend').hidden = $('treeCanvas').hidden || !state.supportMethod;
});
document.querySelectorAll('[data-task]').forEach(button => button.addEventListener('click', () => selectTask(button.dataset.task)));
$('uploadTrigger').addEventListener('click', () => $('alignmentFile').click());
$('alignmentFile').addEventListener('change', event => useFile(event.target.files[0]));
$('removeFile').addEventListener('click', () => { clearFile(); $('alignmentFile').click(); });
for (const name of ['dragenter', 'dragover']) $('uploadTrigger').addEventListener(name, event => { event.preventDefault(); event.stopPropagation(); if (!state.busy) { event.dataTransfer.dropEffect = 'copy'; $('uploadTrigger').classList.add('dragging'); } });
for (const name of ['dragleave', 'drop']) $('uploadTrigger').addEventListener(name, event => { event.preventDefault(); $('uploadTrigger').classList.remove('dragging'); });
function serverDropPaths(transfer) {
  const types = Array.from(transfer.types);
  const get = name => transfer.getData(types.find(type => type.toLowerCase() === name.toLowerCase()) || name);
  // Prefer complete VS Code lists; text/uri-list may contain only the first resource.
  for (const type of ['ResourceURLs', 'CodeFiles', 'CodeEditors', 'application/vnd.code.uri-list', 'text/uri-list', 'text/plain']) {
    const raw = get(type);
    if (!raw) continue;
    let values;
    if (['ResourceURLs', 'CodeFiles', 'CodeEditors'].includes(type)) {
      values = JSON.parse(raw);
      if (!Array.isArray(values)) throw new Error('Could not read the editor file list.');
      if (type === 'CodeEditors') values = values.map(item => typeof item.resource === 'string' ? item.resource : item.resource?.path);
    } else values = raw.split(/\r?\n/).filter(line => line.trim() && !line.startsWith('#'));
    if (values.length && values.every(value => typeof value === 'string' && /^(\/|file:|vscode-remote:)/i.test(value))) return values;
  }
  return [];
}
async function useServerFile(path) {
  if (state.busy) return;
  clearFile();
  let name = path.split('/').pop();
  try { name = decodeURIComponent(name); } catch {}
  state.file = {name};
  $('fileName').textContent = name;
  $('fileMeta').textContent = 'Server file';
  $('fileCard').hidden = false;
  $('uploadTrigger').hidden = true;
  await submitAlignment(path);
}
$('uploadTrigger').addEventListener('drop', event => {
  event.preventDefault(); event.stopPropagation();
  if (state.busy) return;
  if (event.dataTransfer.files.length > 1) return error('Multiple files received. Upload one file at a time.');
  if (event.dataTransfer.files.length === 1) return useFile(event.dataTransfer.files[0]);
  try {
    const paths = serverDropPaths(event.dataTransfer);
    if (paths.length > 1) return error('Multiple files received. Upload one file at a time.');
    if (paths.length === 1) return useServerFile(paths[0]);
    error('No file or server path was received. Choose a file or use Try example.');
  } catch (e) { error(e.message || 'Could not read the editor file path.'); }
});
$('exampleButton').addEventListener('click', async () => {
  if (state.busy) return;
  const sequence = state.fileSequence;
  const name = state.task === 'gene' ? 'gene.phy' : 'species.paml';
  $('exampleButton').disabled = true;
  try {
    const response = await fetch(`/examples/${name}`);
    if (!response.ok) throw new Error('Could not load the example. Please try again.');
    const blob = await response.blob();
    if (state.busy || sequence !== state.fileSequence) return;
    await useFile(new File([blob], name, {type: 'text/plain'}));
  } catch (e) { error(e.message || 'Could not load the example.'); }
  finally { refreshControls(); }
});
window.addEventListener('dragover', event => event.preventDefault());
window.addEventListener('drop', event => event.preventDefault());
window.addEventListener('hashchange', route);
$('resetView').addEventListener('click', resetView);
let drag = null;
$('treeSvg').addEventListener('pointerdown', event => {
  if (!state.tree || event.pointerType !== 'mouse') return;
  drag = {x: event.clientX, y: event.clientY, ox: viewport.x, oy: viewport.y};
  $('treeSvg').setPointerCapture(event.pointerId);
});
$('treeSvg').addEventListener('pointermove', event => {
  if (!drag) return;
  const ratio = viewport.width / $('treeSvg').getBoundingClientRect().width;
  viewport.x = drag.ox + (event.clientX - drag.x) * ratio;
  viewport.y = drag.oy + (event.clientY - drag.y) * ratio;
  transformTree();
});
for (const name of ['pointerup', 'pointercancel']) $('treeSvg').addEventListener(name, () => { drag = null; });
$('treeSvg').addEventListener('wheel', event => {
  if (!state.tree || (!event.ctrlKey && !event.metaKey)) return;
  event.preventDefault();
  viewport.scale = Math.max(.5, Math.min(4, viewport.scale * Math.exp(-event.deltaY * .004)));
  transformTree();
}, {passive: false});
document.addEventListener('visibilitychange', () => {
  if (document.hidden) clearTimeout(queueTimer);
  else refreshQueue(true);
});
window.addEventListener('offline', () => { state.compute = false; renderQueue(null); });
window.addEventListener('online', () => refreshQueue(true));
renderQueue(null, 'loading');
route(); refreshQueue();

// The illustrated landscape follows one ten-minute seasonal cycle; photos stay separate.
(() => {
  const canvas = $('evolutionCanvas');
  const landscapes={geometry:window.growingLandscape(canvas),seasons:window.seasonLandscape(canvas)};
  const scenes = [...document.querySelectorAll('.scene')];
  const buttons = [...document.querySelectorAll('button[data-theme]')];
  const toggle = $('backgroundToggle'), menu = $('themeMenu');
  const seasonSlider=$('seasonProgress'),seasonNames=['spring','summer','autumn','winter'];
  let seasonChoice='auto',seasonOffset=0,seasonPosition=0;
  try {
    const saved=JSON.parse(localStorage.getItem('concordtree-season-control-v2'));
    if(saved&&['auto','manual'].includes(saved.mode)&&Number.isFinite(saved.position)&&saved.position>=0&&saved.position<=1) {
      seasonChoice=saved.mode;seasonPosition=saved.position;
    } else {
      const legacy=localStorage.getItem('concordtree-season-choice');
      if(seasonNames.includes(legacy)){seasonChoice='manual';seasonPosition=seasonNames.indexOf(legacy)/4;}
    }
  } catch {}
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let paused = false, request = 0, theme = 'geometry', frame = 0;
  let backgrounds, previousPhoto = -1, currentScene = 0;
  let elapsed = 0, origin = performance.now(), stopped = true, lastPaint = 0;
  function currentTime() { return stopped?elapsed:elapsed+performance.now()-origin; }
  function saveSeasonControl() {
    try {localStorage.setItem('concordtree-season-control-v2',JSON.stringify({mode:seasonChoice,position:seasonPosition}));} catch {}
  }
  function renderSeasonControl(phase) {
    const index=Math.floor(phase)%4,next=(index+1)%4,part=phase-Math.floor(phase);
    const title=name=>name[0].toUpperCase()+name.slice(1);
    if(seasonChoice==='auto')seasonPosition=phase/4;
    seasonSlider.value=Math.round(seasonPosition*1000);
    $('seasonLabel').textContent=title(seasonNames[Math.round(phase)%4]);
    seasonSlider.setAttribute('aria-valuetext',part<.01?title(seasonNames[index]):`${title(seasonNames[index])} → ${title(seasonNames[next])}`);
  }
  function paint(time=elapsed,force=false) {
    if(theme==='seasons') {
      const phase=seasonChoice==='auto'?(((time+seasonOffset)%600000+600000)%600000)/150000:(seasonPosition*4)%4;
      landscapes.seasons.paint(time,phase,force);renderSeasonControl(phase);
    } else if(theme==='geometry')landscapes.geometry.paint(time+2800);
  }
  function syncBackgroundButton() {
    const held=paused || (theme==='seasons'&&seasonChoice==='manual');
    document.body.classList.toggle('background-paused',held || Boolean(navigator.connection?.saveData));
    toggle.setAttribute('aria-pressed',String(held));
    const label=theme==='seasons'?(held?'Play seasonal cycle':'Pause seasonal cycle'):(held?'Play background':'Pause background');
    toggle.setAttribute('aria-label',label);toggle.title=label;
  }
  seasonSlider.addEventListener('input',()=>{
    seasonChoice='manual';seasonPosition=Number(seasonSlider.value)/1000;
    paint(currentTime(),true);syncBackgroundButton();
  });
  seasonSlider.addEventListener('change',saveSeasonControl);
  function animate(now) {
    if (now-lastPaint >= 50) { paint(elapsed+now-origin); lastPaint=now; }
    frame=requestAnimationFrame(animate);
  }
  function updateMotion() {
    const now=performance.now();
    if(!stopped)elapsed+=now-origin;
    origin=now;cancelAnimationFrame(frame);
    stopped=paused || reducedMotion.matches || Boolean(navigator.connection?.saveData) || theme==='nature';
    syncBackgroundButton();
    document.body.classList.toggle('background-hidden',document.hidden);
    if(theme!=='nature') {paint(elapsed,true);if(!stopped&&!document.hidden)frame=requestAnimationFrame(animate);}
  }
  function resize() { Object.values(landscapes).forEach(scene=>scene.resize());paint(stopped?elapsed:elapsed+performance.now()-origin); }
  function closeMenu() { menu.hidden = true; $('themeToggle').setAttribute('aria-expanded','false'); }
  async function selectTheme(name) {
    const ticket = ++request;
    if (name === 'nature') {
      $('themeStatus').textContent = 'Loading background';
      try {
        if (!backgrounds) {
          const response = await fetch('/backgrounds.json');
          if (!response.ok) throw new Error('Background library unavailable');
          backgrounds = await response.json();
        }
        if (ticket !== request) return;
        if (!backgrounds.length) throw new Error('No backgrounds');
        const choices = backgrounds.map((_,i)=>i).filter(i=>i!==previousPhoto || backgrounds.length===1);
        const index = choices[Math.floor(Math.random()*choices.length)], photo = backgrounds[index];
        const nextScene = 1-currentScene;
        const scene = scenes[nextScene], image = scene.querySelector('img'), source = scene.querySelector('source');
        source.srcset = photo.mobile;
        image.src = photo.desktop;
        await image.decode();
        if (ticket !== request) return;
        scenes.forEach((item,i)=>item.classList.toggle('is-active',i===nextScene));
        currentScene=nextScene;previousPhoto=index;
        document.body.dataset.photo=photo.id;
        $('photoInfo').textContent=photo.author;
        $('photoInfo').href='/credits.html#'+photo.id;
        $('photoInfo').title=photo.title+' — '+photo.kind+' — image details and credits';
        $('photoLicense').textContent=photo.license;
        $('photoLicense').href=photo.license_url;
      } catch {
        if(ticket === request) $('themeStatus').textContent='Background could not load. Please try again.';
        return;
      }
    } else scenes.forEach(scene=>scene.classList.remove('is-active'));
    $('photoCredit').hidden=name!=='nature';
    $('nextBackground').hidden=name!=='nature';
    $('seasonControls').hidden=name!=='seasons';
    theme = name;
    document.body.dataset.theme = name;
    buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.theme === name)));
    $('themeName').textContent = ({geometry:'Geometry',seasons:'Seasons',nature:'Nature'})[name];
    $('themeStatus').textContent = $('themeName').textContent + ' background selected';
    try { localStorage.setItem('phylogeny-theme', name); } catch {}
    updateMotion();
  }
  buttons.forEach(button => button.addEventListener('click', () => {
    selectTheme(button.dataset.theme); closeMenu(); $('themeToggle').focus();
  }));
  $('themeToggle').addEventListener('click', () => {
    menu.hidden = !menu.hidden; $('themeToggle').setAttribute('aria-expanded', String(!menu.hidden));
  });
  document.addEventListener('click', event => { if (!event.target.closest('.background-controls')) closeMenu(); });
  document.addEventListener('keydown', event => { if (event.key === 'Escape' && !menu.hidden) { closeMenu(); $('themeToggle').focus(); } });
  $('nextBackground').addEventListener('click',()=>selectTheme('nature'));
  toggle.addEventListener('click', () => {
    if(theme==='seasons'&&seasonChoice==='manual') {
      seasonOffset=seasonPosition*600000-currentTime();seasonChoice='auto';paused=false;saveSeasonControl();
    } else paused=!paused;
    updateMotion();
  });
  reducedMotion.addEventListener('change', updateMotion);
  document.addEventListener('visibilitychange', updateMotion);
  window.addEventListener('resize', resize);
  resize();
  let saved = 'geometry';
  try { saved = localStorage.getItem('phylogeny-theme') || saved; } catch {}
  selectTheme(['geometry','seasons','nature'].includes(saved)?saved:'geometry');
})();
