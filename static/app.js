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

// ── Dropzones ──────────────────────────────────────────────────────────
function initDropzone(zoneId, fileInputId, filenameId, previewId) {
  const zone = $(`#${zoneId}`);
  const input = $(`#${fileInputId}`);
  const filenameEl = $(`#${filenameId}`);
  const preview = $(`#${previewId}`);

  const setFile = (file) => {
    if (!file) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    filenameEl.textContent = file.name;
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';
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
  initDropzone('clone-dropzone', 'clone-file', 'clone-filename', 'clone-ref-preview');

  $('#clone-text').addEventListener('input', () => {
    $('#clone-char-count').textContent = `${$('#clone-text').value.length} characters`;
  });

  $('#btn-generate-clone').addEventListener('click', async () => {
    const btn = $('#btn-generate-clone');
    const text = $('#clone-text').value;
    const refFile = $('#clone-file').files[0];
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
  initDropzone('fish-dropzone', 'fish-file', 'fish-filename', 'fish-ref-preview');

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
    const refFile = $('#fish-file').files[0];
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

// ── Boot ───────────────────────────────────────────────────────────────
async function boot() {
  initNav();
  initSliders();
  initEmotionTab();
  initCloneTab();
  initFishTab();
  initSetupTab();
  initHistoryTab();
  await loadConfig();
}

boot();
