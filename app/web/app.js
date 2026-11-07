'use strict';
const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const MAX_BYTES = 100 * 1024 * 1024;
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
}
function selectTask(task, restored = false) {
  if ((state.busy && !restored) || !Object.hasOwn(labels, task)) return;
  if (state.task !== task) clearFile();
  state.task = task;
  $('modelOption').hidden = task === 'species';
  $('alignmentFile').accept = task === 'gene' ? '.phy,.phylip,.txt' : '.phy,.phylip,.paml,.txt';
  $('formatHint').textContent = task === 'gene' ? 'PHYLIP · one alignment' : 'PHYLIP / PAML · one block per locus';
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
async function useFile(file) {
  if (!file || state.busy) return;
  clearFile();
  const sequence = state.fileSequence;
  if (!file.size) return error('The file is empty.');
  if (file.size > MAX_BYTES) return error('Maximum file size: 100 MB.');
  try {
    const text = await file.text();
    if (sequence !== state.fileSequence) return;
    const header = text.replace(/^\uFEFF/, '').trimStart().split(/\r?\n/, 1)[0].trim().match(/^(\d+)\s+(\d+)(?:\s+[IS])?$/i);
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
  const response = await fetch(path, {...options, signal: options.signal || AbortSignal.timeout(20000)});
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
  const titles = {loading: 'Connecting', offline: 'Server offline', paused: 'Server paused', full: 'Queue full', busy: 'Server busy', idle: 'Server ready'};
  const hints = {loading: 'Checking server', offline: 'Reconnecting…', paused: 'Not accepting jobs', full: queue ? `${queue.queued} queued` : '', busy: queue ? `${queue.running} running · ${queue.queued} queued` : '', idle: 'No active tasks'};
  const text = (id, value) => { if ($(id).textContent !== value) $(id).textContent = value; };
  $('serverQueue').dataset.status = status;
  text('queueState', titles[status]); text('queueHint', hints[status]);
  text('runningCount', queue ? queue.running.toLocaleString() : '—');
  text('queuedCount', queue ? queue.queued.toLocaleString() : '—');
  const capacity = queue?.capacity || state.queue?.capacity || 16;
  function slots(id, total, filled, mine) {
    const container = $(id);
    if (container.children.length !== total) container.replaceChildren(...Array.from({length: total}, () => {
      const block = document.createElement('i'); block.className = 'job-slot'; return block;
    }));
    [...container.children].forEach((block, index) => {
      block.classList.toggle('is-filled', index < filled);
      block.classList.toggle('is-mine', index < filled && index === mine);
    });
  }
  const ownRunning = state.job?.status === 'running' ? 0 : -1;
  const ownQueued = state.job?.status === 'queued' ? state.job.queue_position - 1 : -1;
  slots('runningSlots', Math.max(1, queue?.running || 0), queue?.running || 0, ownRunning);
  slots('queuedSlots', capacity, queue?.queued || 0, ownQueued);
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
      return queue;
    } catch {
      state.compute = false; renderQueue(null);
      return null;
    } finally {
      queueRequest = null;
      if (!document.hidden) queueTimer = setTimeout(() => refreshQueue(), 5000);
    }
  })();
  return queueRequest;
}
function setPage(result) {
  document.body.classList.toggle('viewing-result', result);
  $('submitPage').hidden = result;
  $('resultPage').hidden = !result;
  const container = result ? $('resultPage') : $('taskForm');
  if ($('serverQueue').parentElement !== container) container.append($('serverQueue'));
}
function setStatus(title, detail = '', failed = false) {
  $('resultTitle').textContent = title;
  $('jobStatus').textContent = detail;
  $('jobStatus').hidden = !detail;
  $('jobStatus').classList.toggle('error', failed);
}
const progressState = {step: 0, total: 6, lastEntry: ''};
function showProgress(message, detail = '', {step = progressState.step, total = progressState.total, mode = 'running', record = true} = {}) {
  progressState.step = Math.max(progressState.step, step);
  progressState.total = total;
  $('inferenceProgress').hidden = false;
  $('inferenceProgress').dataset.mode = mode;
  // Milestones describe completed stages, never an estimated time percentage.
  $('progressFill').style.width = `${progressState.step / total * 100}%`;
  $('progressMessage').textContent = message;
  $('progressDetail').textContent = detail;
  $('progressStage').textContent = progressState.step ? `Stage ${Math.min(progressState.step + 1, total)} / ${total}` : '';
  $('progressBar').setAttribute('aria-valuetext', [message, detail, $('progressStage').textContent].filter(Boolean).join('. '));
  $('progressBar').setAttribute('aria-busy', String(mode === 'running'));
  const entry = [message, detail].filter(Boolean).join(' · ');
  if (record && entry !== progressState.lastEntry) {
    const row = document.createElement('li'); row.textContent = entry;
    $('progressLog').append(row);
    if ($('progressLog').children.length > 50) $('progressLog').firstElementChild.remove();
    progressState.lastEntry = entry;
    $('progressActivity').hidden = false;
  }
}
function updateProgress(job) {
  const total = job.support === false ? 5 : 6;
  if (job.status === 'queued') {
    showProgress('Waiting in queue', job.queue_position > 0 ? `Position ${job.queue_position}` : 'Waiting for a worker', {total, mode: 'waiting'});
    return;
  }
  const event = job.progress || {}, info = event.detail || event;
  const stages = {
    input_ready: ['Preparing initial tree', 1], apnj_running: ['Building initial tree', 1], apnj: ['Refining initial tree', 2],
    dipper_running: ['Building initial tree', 1], dipper: ['Refining initial tree', 2],
    mlp: ['Refining initial tree', 2], gene_refinement: ['Refining gene tree', 3],
    species_refinement: ['Refining species tree', 3], branch_support: ['Summarizing branch support', 4],
    complete: ['Preparing result', total - 1]
  };
  let [message, step] = stages[event.stage] || ['Starting inference', 0];
  if (info.phase === 'parameter_fit') message = 'Fitting model and branch lengths';
  let detail = '';
  if (['apnj_running', 'dipper_running'].includes(event.stage) && Number.isSafeInteger(event.sites)) detail = `${event.sites.toLocaleString()} sites`;
  const round = info.round ?? info.sweep ?? info.pass;
  if (Number.isSafeInteger(round)) detail = `Round ${round}`;
  if (Number.isSafeInteger(info.moves)) detail += `${detail ? ' · ' : ''}${info.moves.toLocaleString()} branch updates`;
  if (info.phase === 'parameter_fit' && Number.isSafeInteger(info.sites)) detail = `${info.sites.toLocaleString()} sites`;
  showProgress(message, detail, {step, total});
}
function clearResult() {
  $('resultTiming').hidden = true;
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
  $('progressActivity').hidden = true;
  $('progressActivity').open = false;
  $('progressLog').replaceChildren();
  progressState.step = 0; progressState.total = 6; progressState.lastEntry = '';
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
  state.busy = ['queued', 'running'].includes(job.status);
  if (state.task !== job.task) selectTask(job.task, true);
  if (job.task === 'gene' && job.model) $('evolutionModel').value = job.model;
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
    $('inferenceProgress').hidden = true;
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
    setStatus(job.status === 'queued' ? 'Queued' : 'Inferring');
    updateProgress(job);
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
  showProgress('Checking server', '', {record: false});
  $('resultDetail').textContent = `${labels[state.task]} · ${state.file.name}`;
  const queue = await refreshQueue(true);
  if (!queue || !queue.compute_enabled || !queue.accepting_jobs) {
    state.busy = false; state.submitting = false; refreshControls(); setPage(false);
    return error(!queue ? 'Server offline. Your file has not been uploaded.' : !queue.compute_enabled ? 'Server paused. Your file has not been queued.' : 'Queue full. Please try again later.');
  }
  setStatus(serverPath ? 'Reading server file' : 'Uploading');
  showProgress(serverPath ? 'Reading alignment' : 'Uploading alignment', 'Your task will join the queue when ready');
  const form = new FormData();
  if (serverPath) form.append('path', serverPath); else form.append('file', state.file);
  form.append('task', state.task);
  form.append('support', String($('showSupport').checked));
  if (state.task === 'gene') form.append('model', $('evolutionModel').value);
  try {
    const job = await api(serverPath ? '/api/jobs/from-server' : '/api/jobs', {method: 'POST', body: form, signal: AbortSignal.timeout(120000)});
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
$('taskForm').addEventListener('submit', event => event.preventDefault());
$('backButton').addEventListener('click', () => { if (!state.busy) { clearTimeout(state.poll); state.job = null; clearFile(); location.hash = 'submit'; setPage(false); } });
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
    error('No file or server path was received. Click Upload or use Try example.');
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

// Two backgrounds: growing geometric trees, or a randomly selected photograph.
(() => {
  const canvas = $('evolutionCanvas'), ctx = canvas.getContext('2d');
  const scenes = [...document.querySelectorAll('.scene')];
  const buttons = [...document.querySelectorAll('button[data-theme]')];
  const toggle = $('backgroundToggle'), menu = $('themeMenu');
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let paused = false, request = 0, theme = 'geometry', frame = 0;
  let backgrounds, previousPhoto = -1, currentScene = 0;
  let width = 0, height = 0, branches = [], elapsed = 2800, previous = null, lastPaint = 0;
  const duration = 23000;
  // Small original line illustrations share one 32-unit visual language.
  const species = {
    bird: ['M3 23c5 0 8-3 10-8L8 7c5 0 9 3 11 6l4-3 4 1 3 3-5 1c-1 8-7 12-15 10L3 28l3-5', 'M24 12h.1M14 18l6 2'],
    butterfly: ['M16 14C7-3 0 6 5 16c-5 8 4 14 11 3 7 11 16 5 11-3 5-10-2-19-11-2Z', 'M16 11v14M16 11l-3-5m3 5 3-5'],
    fish: ['M5 16c5-10 15-10 22 0-7 10-17 10-22 0Zm0 0-4-6v12l4-6', 'M20 10c-3 3-3 9 0 12M23 15h.1M12 9l3-4 4 3M12 23l4 4 3-4'],
    leaf: ['M7 26C-1 10 16 2 28 4c0 14-5 26-18 20', 'M5 29 23 9M11 22l1-9m5 3 8-1'],
    fern: ['M12 30 20 3M17 14C3 15 4 5 17 10M19 9c1-7 9-7 8-2l-8 5M15 21C1 22 1 12 15 17M17 18c9-12 17-7 8-1l-8 5M13 27C2 30 0 22 13 23M15 25c11-11 17-4 7 0l-8 4'],
    flower: ['M16 14c-8-14 9-17 5-3 14-4 13 12 0 9 7 13-9 18-9 3C-1 31-5 15 9 16-4 6 10-3 13 11Z', 'M16 13a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z'],
    mushroom: ['M3 18C3 2 29 2 29 18H3Z', 'M13 18l-2 11h10l-2-11M9 12h.1M17 9h.1M23 13h.1'],
    turtle: ['M7 19c-1-14 19-17 19-1L7 19ZM26 16c8-5 8 5 0 4M9 19l-4 5 5-1 3-4M21 19l4 5-5-1-3-4M7 17l-4 1', 'M11 10l5 3 7-3M16 13v6'],
    deer: ['M10 29V19l-5-5 4-1 4 3h7l4-3 4 1-6 5v10M13 23l3 4 3-4', 'M11 16 8 8 4 7 3 3M8 8l3-4M21 16l3-8 4-1 1-4M24 8l-3-4M12 20h.1M20 20h.1'],
    dragonfly: ['M16 7v23M14 5a2 2 0 1 0 4 0 2 2 0 0 0-4 0M16 13C-2-1-3 16 16 16c19 0 18-17 0-3ZM16 17C0 12 0 26 16 21c16 5 16-9 0-4Z'],
    jellyfish: ['M5 17C3-2 29-2 27 17c-3-3-4 3-7 0-3-3-5 3-8 0-3-3-4 3-7 0ZM9 18c-6 8 7 6 1 13M16 18c-5 8 5 7 0 13M23 18c-6 8 7 6 1 13'],
    snail: ['M13 23C-2 23 1 4 13 5c14 1 11 19 0 17-7-1-6-11 0-11 7 0 5 8 0 6', 'M4 23v4h22c4-2 4-8 0-9l-5 5H4M26 18l-2-5M28 18l3-4']
  };
  const icons = Object.fromEntries(Object.entries(species).map(([name,paths]) => [name, paths.map(path=>new Path2D(path))]));
  function buildTrees() {
    branches = [];
    const mobile = width < 700;
    const layouts = mobile ? [
      [[.11,.09,'bird'],[.37,.14,'butterfly'],[.65,.075,'leaf'],[.88,.15,'fish']]
    ] : [
      [[.15,.13,'bird'],[.052,.235,'butterfly'],[.25,.29,'dragonfly'],[.105,.39,'leaf'],[.265,.48,'fern'],[.045,.57,'mushroom'],[.20,.66,'flower'],[.095,.75,'snail']],
      [[.825,.15,'butterfly'],[.95,.245,'bird'],[.745,.32,'deer'],[.89,.415,'turtle'],[.755,.52,'fish'],[.955,.60,'jellyfish'],[.805,.68,'leaf'],[.905,.77,'fern']]
    ];
    for (let side=0;side<layouts.length;side++) {
      const leaves = layouts[side].map(([x,y,icon])=>({x:x*width,y:y*height,icon}));
      function node(items) {
        if(items.length===1)return items[0];
        const mid=Math.floor(items.length/2), children=[node(items.slice(0,mid)),node(items.slice(mid))];
        return {x:(children[0].x+children[1].x)/2+(side ? -1 : 1)*width*.009,
          y:Math.max(...children.map(child=>child.y))+height*(mobile ? .065 : .057),children};
      }
      let crown;
      if(mobile) crown=node(leaves);
      else {
        crown=leaves[0];
        for(let i=1;i<leaves.length;i++) {
          crown={x:width*((side ? .84 : .16)+Math.sin(i*1.7+side)*.012),
            y:leaves[i].y+height*.055,children:[crown,leaves[i]]};
        }
      }
      const root={x:crown.x+(side ? -.035 : .035)*width,y:mobile ? height*.47 : height*1.03};
      function connect(start,end,depth,order) {
        const c1={x:start.x+(end.x-start.x)*.12,y:start.y+(end.y-start.y)*.58};
        const c2={x:end.x-(end.x-start.x)*.22,y:end.y-(end.y-start.y)*.12};
        const points=[];
        for(let i=0;i<=32;i++){
          const t=i/32,u=1-t;
          points.push({x:u*u*u*start.x+3*u*u*t*c1.x+3*u*t*t*c2.x+t*t*t*end.x,
            y:u*u*u*start.y+3*u*u*t*c1.y+3*u*t*t*c2.y+t*t*t*end.y});
        }
        branches.push({points,end,depth,side,delay:depth*1150+order*100+side*430});
        end.children?.forEach((child,index)=>connect(end,child,depth+1,order+index));
      }
      connect(root,crown,0,0);
    }
  }
  function paint() {
    if (!ctx) return;
    ctx.clearRect(0,0,width,height);
    const still=reducedMotion.matches || Boolean(navigator.connection?.saveData);
    const time=still ? 16000 : elapsed % duration;
    const fade=time>19500 ? Math.max(0,1-(time-19500)/2800) : 1;
    ctx.lineCap='round';ctx.lineJoin='round';
    function stroke(points,amount,sway) {
      const step=amount*(points.length-1),last=Math.floor(step);
      ctx.beginPath();ctx.moveTo(points[0].x,points[0].y);
      for(let i=1;i<=last;i++)ctx.lineTo(points[i].x+sway*i/32,points[i].y);
      if(last<points.length-1){const a=points[last],b=points[last+1],part=step-last;
        ctx.lineTo(a.x+(b.x-a.x)*part+sway*step/32,a.y+(b.y-a.y)*part);}
      ctx.stroke();
    }
    for(const branch of branches) {
      const {points,end,depth,side,delay}=branch;
      const raw=Math.max(0,Math.min(1,(time-delay)/1100));
      const progress=raw*raw*(3-2*raw);
      const sway=still ? 0 : Math.sin(time/3600+side+depth*.2)*1.4;
      const rgb=side ? '211,202,164' : '165,205,180';
      ctx.strokeStyle=`rgba(${rgb},.042)`;ctx.lineWidth=.8;stroke(points,1,sway);
      if(!progress || !fade)continue;
      ctx.strokeStyle=`rgba(${rgb},${Math.max(.18,.38-depth*.025)*fade})`;
      ctx.lineWidth=Math.max(.6,1.3-depth*.09);stroke(points,progress,sway);
      if(progress<1){
        const tip=points[Math.floor(progress*32)];
        ctx.fillStyle=`rgba(${rgb},${.5*fade})`;ctx.shadowColor=`rgba(${rgb},.4)`;ctx.shadowBlur=9;
        ctx.beginPath();ctx.arc(tip.x+sway*progress,tip.y,1.5,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;
      }
      if(end.icon){
        const reveal=Math.max(0,Math.min(1,(time-delay-850)/1000));
        if(!reveal)continue;
        const size=(width<700 ? 27 : Math.max(28,Math.min(36,width*.025)));
        ctx.save();ctx.translate(end.x+sway,end.y);
        // A quiet ink-colored disc keeps the branch out of the illustration.
        ctx.globalAlpha=reveal*fade;ctx.fillStyle='#11221a';
        ctx.beginPath();ctx.arc(0,0,size*.64,0,Math.PI*2);ctx.fill();
        ctx.scale(size/32*(.85+.15*reveal),size/32*(.85+.15*reveal));ctx.translate(-16,-16);
        ctx.strokeStyle=side ? 'rgba(220,211,175,.75)' : 'rgba(183,216,193,.8)';ctx.lineWidth=1.05;
        for(const path of icons[end.icon])ctx.stroke(path);
        ctx.restore();
      } else if(progress===1){
        ctx.fillStyle=`rgba(${rgb},${.45*fade})`;
        ctx.beginPath();ctx.arc(end.x+sway,end.y,1.8,0,Math.PI*2);ctx.fill();
      }
    }
  }
  function animate(now) {
    if (previous !== null) elapsed += Math.min(now - previous, 100);
    previous = now;
    if (now - lastPaint >= 32) { paint(); lastPaint = now; }
    frame = requestAnimationFrame(animate);
  }
  function updateMotion() {
    cancelAnimationFrame(frame); previous = null;
    const stopped = paused || reducedMotion.matches || document.hidden || Boolean(navigator.connection?.saveData);
    document.body.classList.toggle('background-paused', paused || Boolean(navigator.connection?.saveData));
    document.body.classList.toggle('background-hidden', document.hidden);
    toggle.setAttribute('aria-pressed', String(paused));
    toggle.setAttribute('aria-label', paused ? 'Play background' : 'Pause background');
    toggle.title = paused ? 'Play background' : 'Pause background';
    if (theme === 'geometry') { paint(); if (!stopped && ctx) frame = requestAnimationFrame(animate); }
  }
  function resize() {
    width = innerWidth; height = innerHeight;
    const ratio = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    if (ctx) ctx.setTransform(ratio,0,0,ratio,0,0);
    buildTrees(); paint();
  }
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
    theme = name;
    document.body.dataset.theme = name;
    buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.theme === name)));
    $('themeName').textContent = name === 'geometry' ? 'Geometry' : 'Nature';
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
  toggle.addEventListener('click', () => { paused = !paused; updateMotion(); });
  reducedMotion.addEventListener('change', updateMotion);
  document.addEventListener('visibilitychange', updateMotion);
  window.addEventListener('resize', resize);
  resize();
  let saved = 'geometry';
  try { saved = localStorage.getItem('phylogeny-theme') || saved; } catch {}
  selectTheme(saved === 'nature' ? 'nature' : 'geometry');
})();
