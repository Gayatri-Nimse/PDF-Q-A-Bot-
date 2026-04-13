/* ── PDF Q&A Bot — Frontend (large-PDF + SSE edition) ─────────────────────── */

const $ = id => document.getElementById(id);

// ── Elements ──────────────────────────────────────────────────────────────────
const uploadZone     = $('uploadZone');
const fileInput      = $('fileInput');
const uploadBtn      = $('uploadBtn');
const uploadProgress = $('uploadProgress');
const progressFill   = $('progressFill');
const progressText   = $('progressText');
const progressPct    = $('progressPct');
const progressStage  = $('progressStage');
const docInfo        = $('docInfo');
const docName        = $('docName');
const docStats       = $('docStats');
const resetBtn       = $('resetBtn');
const chatArea       = $('chatArea');
const questionInput  = $('questionInput');
const sendBtn        = $('sendBtn');
const statusDot      = $('statusDot');
const statusText     = $('statusText');
const sidebarToggle  = $('sidebarToggle');
const sidebar        = $('sidebar');

let isProcessing = false;
let hasDocument  = false;
let sseSource    = null;    // EventSource for progress stream

// ── Sidebar ───────────────────────────────────────────────────────────────────
sidebarToggle.addEventListener('click', () => {
  sidebar.classList.toggle('collapsed');
  sidebar.classList.toggle('open');
});

// ── Status ────────────────────────────────────────────────────────────────────
function setStatus(state, text) {
  statusDot.className = 'status-dot ' + state;
  statusText.textContent = text;
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function showToast(msg, type = 'info', duration = 4000) {
  const t = document.createElement('div');
  t.className = 'toast ' + type;
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), duration);
}

// ── Textarea auto-resize ──────────────────────────────────────────────────────
questionInput.addEventListener('input', () => {
  questionInput.style.height = 'auto';
  questionInput.style.height = Math.min(questionInput.scrollHeight, 160) + 'px';
});
questionInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!sendBtn.disabled) sendMessage(); }
});
sendBtn.addEventListener('click', sendMessage);

// ── Drag & Drop ───────────────────────────────────────────────────────────────
uploadZone.addEventListener('dragover',  e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFileUpload(file);
});
uploadZone.addEventListener('click', () => fileInput.click());
uploadBtn.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFileUpload(fileInput.files[0]); });

// ── File upload ───────────────────────────────────────────────────────────────
async function handleFileUpload(file) {
  if (!file.name.toLowerCase().endsWith('.pdf')) { showToast('Only PDF files are supported.', 'error'); return; }
  if (file.size > 500 * 1024 * 1024) { showToast('File exceeds 500 MB limit.', 'error'); return; }

  uploadZone.style.display = 'none';
  uploadProgress.style.display = 'block';
  setStatus('loading', 'Uploading…');
  setProgress(2, 'Uploading file to server…', 'UPLOAD');

  try {
    const form = new FormData();
    form.append('file', file);

    const res  = await fetch('/api/upload', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Upload failed');

    // Connect SSE stream to track processing progress
    listenProgress(data.job_id, data.filename);

  } catch (err) {
    resetUploadUI();
    showToast('Upload failed: ' + err.message, 'error');
    setStatus('', 'Awaiting document');
  }
}

// ── SSE progress listener ─────────────────────────────────────────────────────
function listenProgress(jobId, filename) {
  if (sseSource) sseSource.close();

  sseSource = new EventSource(`/api/progress/${jobId}`);

  sseSource.onmessage = (e) => {
    const d = JSON.parse(e.data);

    if (d.status === 'error') {
      sseSource.close(); sseSource = null;
      resetUploadUI();
      showToast('Processing failed: ' + d.error, 'error');
      setStatus('', 'Awaiting document');
      return;
    }

    // Determine stage label from pct
    const stage = d.pct < 5   ? 'UPLOAD'
                : d.pct < 50  ? 'CHUNKING'
                : d.pct < 90  ? 'EMBEDDING'
                : d.pct < 100 ? 'INDEXING'
                :               'READY';
    setProgress(d.pct, d.message, stage);

    if (d.status === 'done') {
      sseSource.close(); sseSource = null;
      onProcessingDone(filename, d.stats || {});
    }
  };

  sseSource.onerror = () => {
    if (sseSource) { sseSource.close(); sseSource = null; }
  };
}

function setProgress(pct, message, stage) {
  progressFill.style.width  = pct + '%';
  if (progressText)  progressText.textContent  = message  || '';
  if (progressPct)   progressPct.textContent   = Math.round(pct) + '%';
  if (progressStage) progressStage.textContent = stage    || '';

  // Highlight pipeline steps
  const order = ['UPLOAD','CHUNKING','EMBEDDING','INDEXING','READY'];
  const curIdx = order.indexOf(stage);
  document.querySelectorAll('.ps-step').forEach(el => {
    const idx = order.indexOf(el.dataset.step);
    el.classList.remove('active','done');
    if (idx === curIdx) el.classList.add('active');
    else if (idx < curIdx) el.classList.add('done');
  });
}

function onProcessingDone(filename, stats) {
  uploadProgress.style.display = 'none';
  docInfo.style.display = 'block';
  docName.textContent = filename;
  const chunks = stats.total_chunks || '?';
  const pages  = stats.total_pages  || stats.unique_pages || '?';
  const avg    = stats.avg_chunk_size || '?';
  docStats.textContent = `${chunks} chunks · ${pages} pages · ~${avg} chars/chunk`;

  hasDocument = true;
  questionInput.disabled = false;
  sendBtn.disabled = false;
  questionInput.focus();
  setStatus('ready', filename);
  clearChat();
  addSystemMessage(`◈  Document indexed. ${chunks} chunks across ${pages} pages are ready. Ask me anything!`);
  showToast(`✓ Ready — ${chunks} chunks indexed`, 'success');
}

function resetUploadUI() {
  uploadProgress.style.display = 'none';
  uploadZone.style.display = 'block';
  fileInput.value = '';
  setProgress(0, '', '');
}

// ── Reset session ─────────────────────────────────────────────────────────────
resetBtn.addEventListener('click', async () => {
  if (sseSource) { sseSource.close(); sseSource = null; }
  try { await fetch('/api/reset', { method: 'POST' }); } catch (_) {}
  hasDocument = false;
  docInfo.style.display = 'none';
  uploadZone.style.display = 'block';
  fileInput.value = '';
  questionInput.disabled = true;
  sendBtn.disabled = true;
  setStatus('', 'Awaiting document');
  clearChat(true);
  showToast('Session cleared. Upload a new PDF.', 'info', 3000);
});

// ── Send message ──────────────────────────────────────────────────────────────
async function sendMessage() {
  const question = questionInput.value.trim();
  if (!question || isProcessing || !hasDocument) return;

  isProcessing = true;
  questionInput.value = '';
  questionInput.style.height = 'auto';
  sendBtn.disabled = true;
  sendBtn.classList.add('loading');
  sendBtn.querySelector('.send-arrow').textContent = '⟳';
  setStatus('loading', 'Thinking…');

  const emptyEl = document.getElementById('emptyState');
  if (emptyEl) emptyEl.style.display = 'none';

  appendMessage('user', question, []);
  const typingEl = appendTyping();

  try {
    const res  = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    typingEl.remove();
    if (!res.ok || data.error) throw new Error(data.error || 'Request failed');
    appendMessage('assistant', data.answer, data.sources || []);
    setStatus('ready', `Ready — ${data.model}`);
  } catch (err) {
    typingEl.remove();
    appendMessage('assistant', `⚠ Error: ${err.message}`, []);
    setStatus('ready', 'Ready');
    showToast(err.message, 'error');
  }

  isProcessing = false;
  sendBtn.disabled = false;
  sendBtn.classList.remove('loading');
  sendBtn.querySelector('.send-arrow').textContent = '↑';
  questionInput.focus();
}

// ── DOM helpers ───────────────────────────────────────────────────────────────
function appendMessage(role, content, sources = []) {
  const div   = document.createElement('div');
  div.className = `message ${role}`;
  const icon  = role === 'user' ? '◉' : '◈';
  const label = role === 'user' ? 'YOU' : 'DOCMIND AI';

  let srcHTML = '';
  if (sources && sources.length > 0) {
    const chips = sources.map(s => `
      <div class="source-chip">
        <div class="src-head">📄 ${s.file} — Page ${s.page}</div>
        <div class="src-snippet">${escHtml(s.snippet)}</div>
      </div>`).join('');
    srcHTML = `<div class="sources-section"><p class="sources-label">▸ SOURCES RETRIEVED</p><div class="sources-list">${chips}</div></div>`;
  }

  div.innerHTML = `
    <div class="msg-header"><span class="msg-icon">${icon}</span><span class="msg-role">${label}</span></div>
    <div class="msg-content">${escHtml(content)}</div>
    ${srcHTML}`;
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
  return div;
}

function addSystemMessage(text) {
  const div = document.createElement('div');
  div.className = 'message assistant';
  div.innerHTML = `
    <div class="msg-header"><span class="msg-icon">◈</span><span class="msg-role">SYSTEM</span></div>
    <div class="msg-content" style="color:var(--text-dim);font-size:13px;">${escHtml(text)}</div>`;
  chatArea.appendChild(div);
}

function appendTyping() {
  const div = document.createElement('div');
  div.className = 'message assistant';
  div.innerHTML = `
    <div class="msg-header"><span class="msg-icon">◈</span><span class="msg-role">DOCMIND AI</span></div>
    <div class="typing-indicator">
      <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
    </div>`;
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
  return div;
}

function clearChat(showEmpty = false) {
  chatArea.innerHTML = '';
  if (showEmpty) {
    chatArea.innerHTML = `
      <div class="empty-state" id="emptyState">
        <div class="empty-icon">◈</div>
        <h2 class="empty-title">Upload a PDF to begin</h2>
        <p class="empty-sub">Your document will be chunked, embedded, and indexed.<br/>Then ask anything — the AI reasons over the content.</p>
        <div class="empty-features">
          <div class="feature-pill">⚡ Semantic search</div>
          <div class="feature-pill">🧠 LLM reasoning</div>
          <div class="feature-pill">💬 Multi-turn memory</div>
          <div class="feature-pill">📄 Source citations</div>
          <div class="feature-pill">📚 Large PDF support</div>
        </div>
      </div>`;
  }
}

function escHtml(str) {
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Init ──────────────────────────────────────────────────────────────────────
(async () => {
  try {
    const res  = await fetch('/api/status');
    const data = await res.json();
    if (data.has_document) {
      hasDocument = true;
      questionInput.disabled = false;
      sendBtn.disabled = false;
      docInfo.style.display = 'block';
      uploadZone.style.display = 'none';
      docName.textContent = data.filename;
      setStatus('ready', data.filename);

      const hRes  = await fetch('/api/history');
      const hData = await hRes.json();
      if (hData.messages && hData.messages.length > 0) {
        const es = document.getElementById('emptyState');
        if (es) es.style.display = 'none';
        const s = hData.stats || {};
        if (s.total_chunks)
          docStats.textContent = `${s.total_chunks} chunks · ${s.total_pages || s.unique_pages || '?'} pages`;
        hData.messages.forEach(m => appendMessage(m.role, m.content, m.sources || []));
      }
    }
  } catch (_) {}
})();
