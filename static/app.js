// TTS Studio — vanilla JS frontend logic. Talks to the FastAPI backend in server.py.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// ── Toast ──────────────────────────────────────────────────────────────
let toastTimer = null;
function showToast(msg, isError = false) {
  const el = $('#toast');
  el.textContent = msg;
  el.classList.toggle('error', isError);
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 3000);
}

// ── Sidebar navigation ─────────────────────────────────────────────────
function initNav() {
  $$('.nav-item').forEach((btn) => {
    btn.addEventListener('click', () => {
      $$('.nav-item').forEach((b) => b.classList.remove('active'));
      $$('.panel').forEach((p) => p.classList.remove('active'));
      btn.classList.add('active');
      $(`#panel-${btn.dataset.panel}`).classList.add('active');
      if (btn.dataset.panel === 'history') loadHistory();
    });
  });
}

// ── Slider value labels ────────────────────────────────────────────────
function initSliders() {
  [
    ['emotion-temp', 'emotion-temp-val'], ['emotion-topp', 'emotion-topp-val'], ['emotion-reppen', 'emotion-reppen-val'],
    ['clone-temp', 'clone-temp-val'], ['clone-topp', 'clone-topp-val'], ['clone-reppen', 'clone-reppen-val'],
    ['fish-temp', 'fish-temp-val'], ['fish-topp', 'fish-topp-val'], ['fish-reppen', 'fish-reppen-val'],
  ].forEach(([inputId, labelId]) => {
    const input = $(`#${inputId}`);
    const label = $(`#${labelId}`);
    input.addEventListener('input', () => { label.textContent = input.value; });
  });
}

// ── Config load: populate dropdowns + chip rows ────────────────────────
async function loadConfig() {
  const res = await fetch('/api/config');
  const cfg = await res.json();

  const speakerSel = $('#emotion-speaker');
  cfg.speakers.forEach((s) => {
    const opt = document.createElement('option');
    opt.value = s.id; opt.textContent = s.label;
    speakerSel.appendChild(opt);
  });

  ['emotion-language', 'clone-language'].forEach((id) => {
    const sel = $(`#${id}`);
    cfg.languages.forEach((lang) => {
      const opt = document.createElement('option');
      opt.value = lang; opt.textContent = lang;
      sel.appendChild(opt);
    });
  });
  $('#clone-language').value = 'English';

  const emoRow = $('#emotion-chip-row');
  cfg.emotion_tags.forEach(({ label, value }) => {
    const chip = document.createElement('button');
    chip.className = 'chip'; chip.textContent = label;
    chip.addEventListener('click', () => appendTag('#emotion-text', value));
    emoRow.appendChild(chip);
  });

  const fishRow = $('#fish-chip-row');
  cfg.fish_markers.forEach(({ label, value }) => {
    const chip = document.createElement('button');
    chip.className = 'chip'; chip.textContent = label;
    chip.addEventListener('click', () => appendTag('#fish-text', value));
    fishRow.appendChild(chip);
  });

  renderDeviceInfo(cfg.device);
  return cfg;
}

function appendTag(textareaSel, tag) {
  const el = $(textareaSel);
  const current = (el.value || '').replace(/\s+$/, '');
  el.value = current ? `${current} ${tag} ` : `${tag} `;
  el.focus();
}

function renderDeviceInfo(info) {
  const el = $('#device-info');
  if (info.cuda_available) {
    el.innerHTML = `🟢 GPU: <b>${info.device_name}</b> · VRAM: ${info.vram_total_gb} GB total, ${info.vram_free_gb} GB free`;
  } else {
    el.innerHTML = '🟡 CPU mode (no CUDA GPU detected)';
  }
}

// ── Status banner ──────────────────────────────────────────────────────
function setStatusBanner(text, kind) {
  const el = $('#status-banner');
  el.textContent = text;
  el.className = 'status-banner' + (kind ? ` ${kind}` : '');
}

// ── Reference-audio trimmer ────────────────────────────────────────────
// Decodes the clip in the browser, lets the user pick a section on the
// waveform, and uploads only that section (as mono 16-bit WAV).
let sharedAudioCtx = null;
const audioCtx = () => (sharedAudioCtx ||= new (window.AudioContext || window.webkitAudioContext)());
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function encodeWav(buffer, start, end) {
  const sr = buffer.sampleRate;
  const s0 = Math.floor(start * sr);
  const n = Math.max(0, Math.floor(end * sr) - s0);
  const mono = new Float32Array(n);
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const d = buffer.getChannelData(c);
    for (let i = 0; i < n; i++) mono[i] += d[s0 + i] / buffer.numberOfChannels;
  }
  const view = new DataView(new ArrayBuffer(44 + n * 2));
  const str = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); view.setUint32(4, 36 + n * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sr, true); view.setUint32(28, sr * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  str(36, 'data'); view.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const v = Math.max(-1, Math.min(1, mono[i]));
    view.setInt16(44 + i * 2, v < 0 ? v * 0x8000 : v * 0x7fff, true);
  }
  return new Blob([view], { type: 'audio/wav' });
}

function createTrimmer(root, range) {
  let { recMin, recMax, quick } = range;
  root.innerHTML = `
    <div class="trim-head"><span class="trim-title">✂️ Trim reference</span><span class="trim-info"></span></div>
    <canvas class="trim-wave"></canvas>
    <div class="trim-controls">
      <button type="button" class="btn btn-secondary trim-play">▶ Play selection</button>
      <span class="trim-field">Start <input type="number" class="trim-start" min="0" step="0.1"> s</span>
      <span class="trim-field">End <input type="number" class="trim-end" min="0" step="0.1"> s</span>
      <button type="button" class="btn btn-secondary trim-quick">First ${quick}s</button>
      <button type="button" class="btn btn-secondary trim-full">Full clip</button>
    </div>
    <div class="trim-hint">Drag across the waveform to choose the part to use. The reference transcript must match only the selected part.</div>`;

  const canvas = root.querySelector('.trim-wave');
  const info = root.querySelector('.trim-info');
  const playBtn = root.querySelector('.trim-play');
  const startIn = root.querySelector('.trim-start');
  const endIn = root.querySelector('.trim-end');

  let file = null, buffer = null, start = 0, end = 0;
  let peaks = null, peaksWidth = 0;
  let source = null, playhead = null, raf = 0, dragFrom = null, prevSel = null;

  const duration = () => (buffer ? buffer.duration : 0);
  const timeToX = (t) => (t / duration()) * canvas.clientWidth;
  const xToTime = (x) => Math.max(0, Math.min(duration(), (x / canvas.clientWidth) * duration()));

  function computePeaks(width) {
    const data = buffer.getChannelData(0);
    const step = Math.max(1, Math.floor(data.length / width));
    peaks = new Float32Array(width);
    for (let x = 0; x < width; x++) {
      let peak = 0;
      for (let i = x * step, stop = Math.min(data.length, i + step); i < stop; i++) {
        const v = Math.abs(data[i]);
        if (v > peak) peak = v;
      }
      peaks[x] = peak;
    }
    peaksWidth = width;
  }

  function draw() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!buffer || !w) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const g = canvas.getContext('2d');
    g.scale(dpr, dpr);
    if (peaksWidth !== w) computePeaks(w);
    const x0 = timeToX(start), x1 = timeToX(end);
    g.fillStyle = 'rgba(91,124,250,0.12)';
    g.fillRect(x0, 0, x1 - x0, h);
    const on = cssVar('--accent'), off = cssVar('--border-color');
    for (let x = 0; x < w; x++) {
      const bh = Math.max(1, peaks[x] * h * 0.9);
      g.fillStyle = x >= x0 && x <= x1 ? on : off;
      g.fillRect(x, (h - bh) / 2, 1, bh);
    }
    g.fillStyle = cssVar('--text-primary');
    g.fillRect(x0 - 1, 0, 2, h);
    g.fillRect(x1 - 1, 0, 2, h);
    if (playhead !== null) {
      g.fillStyle = cssVar('--success');
      g.fillRect(timeToX(playhead) - 1, 0, 2, h);
    }
  }

  function setSelection(s, e) {
    start = Math.max(0, Math.min(s, duration()));
    end = Math.max(start, Math.min(e, duration()));
    startIn.value = start.toFixed(1);
    endIn.value = end.toFixed(1);
    const len = end - start;
    const outside = len < recMin || len > recMax;
    info.textContent = `Selection ${len.toFixed(1)}s of ${duration().toFixed(1)}s` +
      (outside ? ` · recommended ${recMin}–${recMax}s` : '');
    info.classList.toggle('warn', outside);
    draw();
  }

  function stopPlayback() {
    if (source) { source.onended = null; try { source.stop(); } catch {} source = null; }
    cancelAnimationFrame(raf);
    playhead = null;
    playBtn.textContent = '▶ Play selection';
    draw();
  }

  async function togglePlayback() {
    if (source) return stopPlayback();
    const ctx = audioCtx();
    if (ctx.state === 'suspended') await ctx.resume();
    source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    const t0 = ctx.currentTime;
    source.start(0, start, end - start);
    source.onended = stopPlayback;
    playBtn.textContent = '■ Stop';
    const tick = () => {
      playhead = start + (ctx.currentTime - t0);
      draw();
      raf = requestAnimationFrame(tick);
    };
    tick();
  }

  playBtn.addEventListener('click', () => buffer && togglePlayback());
  const quickBtn = root.querySelector('.trim-quick');
  quickBtn.addEventListener('click', () => { stopPlayback(); setSelection(0, quick); });
  root.querySelector('.trim-full').addEventListener('click', () => { stopPlayback(); setSelection(0, duration()); });
  startIn.addEventListener('change', () => { stopPlayback(); setSelection(Math.min(+startIn.value || 0, end - 0.1), end); });
  endIn.addEventListener('change', () => { stopPlayback(); setSelection(start, Math.max(+endIn.value || 0, start + 0.1)); });

  canvas.addEventListener('pointerdown', (e) => {
    if (!buffer) return;
    stopPlayback();
    canvas.setPointerCapture(e.pointerId);
    dragFrom = xToTime(e.offsetX);
    prevSel = [start, end];
  });
  canvas.addEventListener('pointermove', (e) => {
    if (dragFrom === null) return;
    const t = xToTime(e.offsetX);
    setSelection(Math.min(dragFrom, t), Math.max(dragFrom, t));
  });
  canvas.addEventListener('pointerup', () => {
    if (dragFrom === null) return;
    if (end - start < 0.3) setSelection(...prevSel); // a plain click shouldn't wipe the selection
    dragFrom = null;
  });

  // Redraw when the panel becomes visible or the window resizes.
  new ResizeObserver(() => draw()).observe(canvas);

  return {
    setRange(r) {
      ({ recMin, recMax, quick } = r);
      quickBtn.textContent = `First ${quick}s`;
      if (buffer) setSelection(start, end);
    },
    async load(f) {
      stopPlayback();
      file = f; buffer = null; peaksWidth = 0;
      root.hidden = false;
      try {
        buffer = await audioCtx().decodeAudioData(await f.arrayBuffer());
        setSelection(0, buffer.duration);
      } catch {
        info.textContent = "Couldn't decode this file in the browser — it will be uploaded untrimmed.";
        info.classList.add('warn');
      }
    },
    // The file to upload: the original if untrimmed/undecodable, else the selected section as WAV.
    async getFile() {
      if (!file || !buffer) return file;
      if (start <= 0.05 && end >= buffer.duration - 0.05) return file;
      const base = file.name.replace(/\.[^.]+$/, '');
      const name = `${base}_${start.toFixed(1)}-${end.toFixed(1)}s.wav`;
      return new File([encodeWav(buffer, start, end)], name, { type: 'audio/wav' });
    },
  };
}

// ── Dropzones ──────────────────────────────────────────────────────────
function initDropzone(zoneId, fileInputId, filenameId, onFile) {
  const zone = $(`#${zoneId}`);
  const input = $(`#${fileInputId}`);
  const filenameEl = $(`#${filenameId}`);

  const setFile = (file) => {
    if (!file) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    filenameEl.textContent = file.name;
    onFile(file);
  };

  zone.addEventListener('click', () => input.click());
  input.addEventListener('change', () => setFile(input.files[0]));
  ['dragenter', 'dragover'].forEach((evt) =>
    zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.add('dragover'); })
  );
  ['dragleave', 'drop'].forEach((evt) =>
    zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.remove('dragover'); })
  );
  zone.addEventListener('drop', (e) => setFile(e.dataTransfer.files[0]));
}

// ── Generic button-busy helper ─────────────────────────────────────────
function setBusy(btn, busy, busyLabel) {
  if (busy) {
    btn.dataset.originalLabel = btn.innerHTML;
    btn.innerHTML = `<span class="spinner"></span> ${busyLabel || 'Working…'}`;
    btn.disabled = true;
  } else {
    btn.innerHTML = btn.dataset.originalLabel || btn.innerHTML;
    btn.disabled = false;
  }
}

function renderOutput(containerSel, data) {
  const el = $(containerSel);
  const cacheNote = data.cache_hit === true ? ' · Voice prompt: <b>reused (fast)</b>'
                   : data.cache_hit === false ? ' · Voice prompt: <b>computed</b>' : '';
  el.innerHTML = `
    <audio controls src="${data.audio_url}"></audio>
    <div class="stats-line">✅ Duration: <b>${data.duration_s}s</b>${cacheNote}</div>
  `;
}

function renderError(containerSel, message) {
  $(containerSel).innerHTML = `<div class="error-banner">⚠️ ${message}</div>`;
}

// ── Emotion Studio ─────────────────────────────────────────────────────
function initEmotionTab() {
  $('#btn-ollama-rephrase').addEventListener('click', async () => {
    const btn = $('#btn-ollama-rephrase');
    const text = $('#emotion-text').value;
    if (!text.trim()) { showToast('Enter some text first', true); return; }
    setBusy(btn, true, 'Rephrasing…');
    try {
      const res = await fetch('/api/ollama/rephrase', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, url: $('#ollama-url').value, model: $('#ollama-model').value }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Rephrase failed');
      $('#emotion-text').value = data.text;
      $('#ollama-status').textContent = `✅ Rephrased with ${$('#ollama-model').value}`;
    } catch (e) {
      $('#ollama-status').textContent = `🔴 ${e.message}`;
    } finally {
      setBusy(btn, false);
    }
  });

  $('#btn-generate-emotion').addEventListener('click', async () => {
    const btn = $('#btn-generate-emotion');
    const text = $('#emotion-text').value;
    if (!text.trim()) { showToast('Enter some text to synthesize', true); return; }
    setBusy(btn, true, 'Generating…');
    const form = new FormData();
    form.append('text', text);
    form.append('speaker', $('#emotion-speaker').value);
    form.append('language', $('#emotion-language').value);
    form.append('temperature', $('#emotion-temp').value);
    form.append('top_p', $('#emotion-topp').value);
    form.append('repetition_penalty', $('#emotion-reppen').value);
    try {
      const res = await fetch('/api/generate/emotion', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Generation failed');
      renderOutput('#emotion-output', data);
      showToast('🔊 Speech generated!');
    } catch (e) {
      renderError('#emotion-output', e.message);
      showToast(e.message, true);
    } finally {
      setBusy(btn, false);
    }
  });
}

// ── Voice Clone ─────────────────────────────────────────────────────────
function initCloneTab() {
  const trimmer = createTrimmer($('#clone-trimmer'), { recMin: 5, recMax: 15, quick: 15 });
  initDropzone('clone-dropzone', 'clone-file', 'clone-filename', (f) => trimmer.load(f));

  $('#clone-text').addEventListener('input', () => {
    $('#clone-char-count').textContent = `${$('#clone-text').value.length} characters`;
  });

  $('#btn-generate-clone').addEventListener('click', async () => {
    const btn = $('#btn-generate-clone');
    const text = $('#clone-text').value;
    const refFile = await trimmer.getFile();
    const refText = $('#clone-ref-text').value;
    if (!text.trim()) { showToast('Enter some text to synthesize', true); return; }
    if (!refFile) { showToast('Upload a reference audio file', true); return; }
    if (!refText.trim()) { showToast('Provide the reference transcript', true); return; }

    setBusy(btn, true, 'Cloning voice…');
    const form = new FormData();
    form.append('text', text);
    form.append('language', $('#clone-language').value);
    form.append('ref_text', refText);
    form.append('temperature', $('#clone-temp').value);
    form.append('top_p', $('#clone-topp').value);
    form.append('repetition_penalty', $('#clone-reppen').value);
    form.append('ref_audio', refFile);
    try {
      const res = await fetch('/api/generate/clone', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Generation failed');
      renderOutput('#clone-output', data);
      showToast('🔊 Speech generated!');
    } catch (e) {
      renderError('#clone-output', e.message);
      showToast(e.message, true);
    } finally {
      setBusy(btn, false);
    }
  });
}

// ── Fish-Speech ─────────────────────────────────────────────────────────
function initFishTab() {
  const trimmer = createTrimmer($('#fish-trimmer'), { recMin: 10, recMax: 30, quick: 20 });
  initDropzone('fish-dropzone', 'fish-file', 'fish-filename', (f) => trimmer.load(f));

  $('#btn-load-fish').addEventListener('click', async () => {
    const btn = $('#btn-load-fish');
    setBusy(btn, true, 'Loading…');
    $('#fish-load-log').textContent = 'Starting load… (first run also downloads the checkpoint)';
    try {
      await fetch('/api/models/fish/load', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: $('#fish-model-choice').value, compile: $('#fish-compile').checked }),
      });
      await pollStatus('/api/models/fish/status', '#fish-load-log', () => {
        setBusy(btn, false);
      });
    } catch (e) {
      $('#fish-load-log').textContent = `❌ ${e.message}`;
      setBusy(btn, false);
    }
  });

  $('#btn-generate-fish').addEventListener('click', async () => {
    const btn = $('#btn-generate-fish');
    const text = $('#fish-text').value;
    const refFile = await trimmer.getFile();
    const refText = $('#fish-ref-text').value;
    if (!text.trim()) { showToast('Enter some text to synthesize', true); return; }
    if (!refFile) { showToast('Upload a reference audio file', true); return; }
    if (!refText.trim()) { showToast('Provide the reference transcript', true); return; }

    setBusy(btn, true, 'Generating…');
    const form = new FormData();
    form.append('text', text);
    form.append('ref_text', refText);
    form.append('temperature', $('#fish-temp').value);
    form.append('top_p', $('#fish-topp').value);
    form.append('repetition_penalty', $('#fish-reppen').value);
    form.append('ref_audio', refFile);
    try {
      const res = await fetch('/api/generate/fish', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Generation failed');
      renderOutput('#fish-output', data);
      showToast('🔊 Speech generated!');
    } catch (e) {
      renderError('#fish-output', e.message);
      showToast(e.message, true);
    } finally {
      setBusy(btn, false);
    }
  });
}

// ── Setup & Models ──────────────────────────────────────────────────────
async function pollStatus(url, logSelector, onDone) {
  return new Promise((resolve) => {
    const tick = async () => {
      const res = await fetch(url);
      const data = await res.json();
      $(logSelector).textContent = data.log || '…';
      $(logSelector).scrollTop = $(logSelector).scrollHeight;
      if (data.is_loading) {
        setTimeout(tick, 1000);
      } else {
        if (data.is_loaded) {
          setStatusBanner(`🟢 Active model: ${data.current_model_id}`, 'ok');
          showToast('✅ Model loaded!');
        } else if (data.load_error) {
          setStatusBanner('🔴 Model load failed', 'error');
          showToast(`Load failed: ${data.load_error}`, true);
        }
        onDone && onDone();
        resolve(data);
      }
    };
    tick();
  });
}

function initSetupTab() {
  $('#btn-load-qwen').addEventListener('click', async () => {
    const btn = $('#btn-load-qwen');
    const modelId = $('#qwen-model-choice').value;
    setBusy(btn, true, 'Loading…');
    setStatusBanner(`🟡 Loading ${modelId}…`, 'loading');
    $('#qwen-load-log').textContent = `Starting load for ${modelId}…`;
    try {
      await fetch('/api/models/qwen/load', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: modelId,
          dtype: $('#qwen-dtype').value,
          attn_impl: $('#qwen-attn').value,
          compile: $('#qwen-compile').checked,
        }),
      });
      await pollStatus('/api/models/qwen/status', '#qwen-load-log', () => setBusy(btn, false));
    } catch (e) {
      $('#qwen-load-log').textContent = `❌ ${e.message}`;
      setBusy(btn, false);
    }
  });

  $('#btn-test-ollama').addEventListener('click', async () => {
    const url = $('#setup-ollama-url').value;
    $('#ollama-test-result').textContent = 'Testing…';
    const res = await fetch(`/api/ollama/test?url=${encodeURIComponent(url)}`);
    const data = await res.json();
    $('#ollama-test-result').textContent = (data.ok ? '🟢 ' : '🔴 ') + data.message;
  });
}

// ── History ──────────────────────────────────────────────────────────────
async function loadHistory() {
  const el = $('#history-list');
  el.textContent = 'Loading…';
  const res = await fetch('/api/outputs');
  const files = await res.json();
  if (!files.length) { el.textContent = 'No generations yet.'; return; }
  el.innerHTML = '';
  files.forEach((f) => {
    const row = document.createElement('div');
    row.className = 'history-item';
    row.innerHTML = `<span class="name">${f.name}</span><audio controls src="${f.url}" style="width:280px;margin:0;"></audio>`;
    el.appendChild(row);
  });
}

function initHistoryTab() {
  $('#btn-refresh-history').addEventListener('click', loadHistory);
}

// ── Batch narration ──────────────────────────────────────────────────────
// Mirrors server.py's _safe_folder so the "Saves to …" preview matches.
const safeFolder = (s) => (s || '').replace(/[^A-Za-z0-9._-]+/g, '-').replace(/^[-.]+|[-.]+$/g, '') || 'narration';

function el(tag, className, text) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text != null) e.textContent = text;
  return e;
}

async function initBatchTab(cfg) {
  const opts = await (await fetch('/api/narration/options')).json();
  const REF_RANGES = { clone: { recMin: 5, recMax: 15, quick: 15 }, fish: { recMin: 10, recMax: 30, quick: 20 } };
  const trimmer = createTrimmer($('#batch-trimmer'), REF_RANGES.clone);
  const folderInput = $('#batch-folder');
  let scriptFile = null, project = '', pollTimer = null, wasRunning = false;

  cfg.speakers.forEach((s) => $('#batch-voice').appendChild(new Option(s.label, s.id)));
  cfg.languages.forEach((l) => $('#batch-language').appendChild(new Option(l, l)));

  function showFolderPath() {
    const root = opts.output_root;
    const sep = root.includes('\\') ? '\\' : '/';
    $('#batch-folder-path').textContent = `Saves to ${root}${sep}${safeFolder(folderInput.value || project)}`;
  }

  function applyModel() {
    const m = $('#batch-model').value;
    $('#batch-voice-field').hidden = m !== 'emotion';
    $('#batch-language-field').hidden = m === 'fish';
    $('#batch-emotion-field').hidden = m === 'clone';
    $('#batch-ref-section').hidden = m === 'emotion';
    const sel = $('#batch-emotion');
    const prev = sel.value;
    sel.innerHTML = '';
    sel.appendChild(new Option('None', ''));
    (m === 'emotion' ? opts.qwen_emotions : m === 'fish' ? opts.fish_emotions : [])
      .forEach((e) => sel.appendChild(new Option(e, e)));
    if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
    $('#batch-emotion-note').textContent = m === 'clone'
      ? 'The Qwen voice-clone model has no emotions — "emotion" fields in the JSON are skipped.'
      : 'A scene\'s own "emotion" field in the JSON overrides this.';
    if (m !== 'emotion') trimmer.setRange(REF_RANGES[m]);
  }

  async function loadScript(f) {
    scriptFile = null;
    project = '';
    const preview = $('#batch-preview');
    preview.hidden = false;
    preview.innerHTML = '';
    let data;
    try {
      data = JSON.parse(await f.text());
    } catch (e) {
      preview.appendChild(el('div', 'error-banner', `⚠️ Not valid JSON: ${e.message}`));
      return;
    }
    const scenes = Array.isArray(data?.scenes) ? data.scenes : [];
    if (!scenes.length) {
      preview.appendChild(el('div', 'error-banner', '⚠️ This file has no "scenes" list.'));
      return;
    }
    scriptFile = f;
    project = typeof data.project === 'string' ? data.project : '';
    folderInput.placeholder = safeFolder(project);
    preview.appendChild(el('div', 'batch-preview-head',
      `${scenes.length} scene${scenes.length === 1 ? '' : 's'}${project ? ` · ${project}` : ''}`));
    const list = el('div', 'batch-scenes');
    scenes.forEach((s) => {
      const row = el('div', 'batch-scene');
      row.appendChild(el('span', 'name', s?.file ?? '(no file)'));
      row.appendChild(el('span', 'text', (s?.emotion ? `(${s.emotion}) ` : '') + (s?.text ?? '')));
      list.appendChild(row);
    });
    preview.appendChild(list);
    showFolderPath();
  }

  function renderResults(s) {
    const box = $('#batch-results');
    box.innerHTML = '';
    if (!s.files.length) return;
    const head = el('div', 'batch-results-head');
    head.appendChild(el('div', 'stats-line', `📁 ${s.output_dir}`));
    const zip = el('a', 'btn btn-secondary', '⬇ Download all (ZIP)');
    zip.href = `/api/narration/zip?folder=${encodeURIComponent(s.folder)}`;
    head.appendChild(zip);
    box.appendChild(head);
    const list = el('div', 'card');
    s.files.forEach((f) => {
      const row = el('div', 'history-item');
      row.appendChild(el('span', 'name', f.name));
      const audio = document.createElement('audio');
      audio.controls = true;
      audio.preload = 'none';
      audio.src = f.url;
      audio.style.cssText = 'width:260px;margin:0;';
      row.appendChild(audio);
      list.appendChild(row);
    });
    box.appendChild(list);
  }

  function render(s) {
    const btn = $('#btn-batch-start');
    btn.disabled = s.running;
    btn.innerHTML = s.running ? '<span class="spinner"></span> Generating…' : '🎬 Generate All Clips';
    if (!s.folder) return; // nothing has run since the server started

    $('#batch-progress').hidden = false;
    $('#batch-progress').firstElementChild.style.width = `${s.total ? (100 * s.done) / s.total : 0}%`;
    const text = $('#batch-progress-text');
    if (s.error) text.textContent = `🔴 ${s.error}`;
    else if (s.running) text.textContent = `${s.done} / ${s.total} scenes${s.done ? '' : ' — loading model / first clip…'}`;
    else if (s.counts) text.textContent = `✅ Finished: ${s.counts.generated} generated, ${s.counts.skipped} skipped, ${s.counts.failed} failed`;

    const log = $('#batch-log');
    log.hidden = false;
    const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
    log.innerHTML = '';
    s.lines.forEach((l) => log.appendChild(el('div', `log-line ${l.level}`, l.text)));
    if (atBottom) log.scrollTop = log.scrollHeight;

    if (!s.running) renderResults(s);
  }

  async function poll() {
    clearTimeout(pollTimer);
    const s = await (await fetch('/api/narration/status')).json();
    render(s);
    if (s.running) {
      wasRunning = true;
      pollTimer = setTimeout(poll, 1500);
    } else if (wasRunning) {
      wasRunning = false;
      if (s.error) showToast(s.error, true);
      else if (s.counts?.failed) showToast(`${s.counts.failed} scene(s) failed — see the log`, true);
      else showToast('🎬 All clips ready!');
    }
  }

  initDropzone('batch-dropzone', 'batch-file', 'batch-filename', loadScript);
  initDropzone('batch-ref-dropzone', 'batch-ref-file', 'batch-ref-filename', (f) => trimmer.load(f));
  $('#batch-model').addEventListener('change', applyModel);
  folderInput.addEventListener('input', showFolderPath);

  $('#btn-batch-start').addEventListener('click', async () => {
    const m = $('#batch-model').value;
    if (!scriptFile) { showToast('Choose a narration JSON first', true); return; }
    const form = new FormData();
    form.append('script', scriptFile);
    form.append('folder', folderInput.value.trim());
    form.append('model', m);
    form.append('voice', $('#batch-voice').value);
    form.append('language', $('#batch-language').value);
    form.append('emotion', m === 'clone' ? '' : $('#batch-emotion').value);
    form.append('overwrite', $('#batch-overwrite').checked);
    form.append('seed', $('#batch-seed').value || '1234');
    if (m !== 'emotion') {
      const ref = await trimmer.getFile();
      const refText = $('#batch-ref-text').value.trim();
      if (!ref) { showToast('Upload a reference clip to clone', true); return; }
      if (!refText) { showToast('Enter the reference transcript', true); return; }
      form.append('ref_audio', ref);
      form.append('ref_text', refText);
    }
    $('#btn-batch-start').disabled = true;
    try {
      const res = await fetch('/api/narration/start', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Could not start the batch');
      $('#batch-results').innerHTML = '';
      poll();
    } catch (e) {
      $('#btn-batch-start').disabled = false;
      showToast(e.message, true);
    }
  });

  applyModel();
  showFolderPath();
  poll(); // pick up a batch that's still running after a page reload
}

// ── Boot ───────────────────────────────────────────────────────────────
async function boot() {
  initNav();
  initSliders();
  initEmotionTab();
  initCloneTab();
  initFishTab();
  initSetupTab();
  initHistoryTab();
  const cfg = await loadConfig();
  await initBatchTab(cfg);
}

boot();
