// Loaded by site-a and site-b from the CDN origin. Injects the cache-owning
// iframe, logs everything the frame reports, and runs the top-level tests
// (T4 HTTP cache, Probe).

const CDN = 'https://dianome-cdn.pages.dev';

const CDN_ORIGIN = new URL(CDN).origin;
const CHUNK_URL = `${CDN_ORIGIN}/chunks/test.bin`;

// ---- page skeleton ---------------------------------------------------------

const style = document.createElement('style');
style.textContent = `
  body { font-family: system-ui, sans-serif; margin: 16px; }
  iframe { width: 100%; height: 160px; border: 1px solid #999; }
  button { margin: 8px 6px 8px 0; }
  table { border-collapse: collapse; width: 100%; }
  th, td { border: 1px solid #ccc; padding: 4px 6px; text-align: left; vertical-align: top; font-size: 13px; }
  td.notes { font-family: ui-monospace, monospace; white-space: pre-wrap; word-break: break-word; }
`;
document.head.appendChild(style);

const info = document.createElement('p');
info.innerHTML = `Page origin: <code>${location.origin}</code> · CDN origin: <code>${CDN_ORIGIN}</code>`;
document.body.appendChild(info);

const iframe = document.createElement('iframe');
iframe.src = `${CDN_ORIGIN}/frame.html`;
iframe.setAttribute('allow', 'storage-access');
document.body.appendChild(iframe);

const controls = document.createElement('div');
controls.innerHTML = `
  <button id="t4">T4 HTTP cache</button>
  <button id="probe">Probe</button>
  <button id="copy">Copy results as Markdown</button>
`;
document.body.appendChild(controls);

const table = document.createElement('table');
table.innerHTML = `
  <thead><tr><th>time</th><th>test</th><th>source</th><th>ms</th><th>notes</th></tr></thead>
  <tbody></tbody>
`;
document.body.appendChild(table);
const tbody = table.querySelector('tbody');

// ---- log -------------------------------------------------------------------

const rows = []; // { time, test, source, ms, notes } — appended, never overwritten

function fmtMs(ms) {
  return typeof ms === 'number' && Number.isFinite(ms) ? ms.toFixed(1) : '';
}

function log(test, source, ms, notes) {
  const row = {
    time: new Date().toLocaleTimeString([], { hour12: false }),
    test: String(test ?? ''),
    source: String(source ?? ''),
    ms: fmtMs(ms),
    notes: String(notes ?? ''),
  };
  rows.push(row);
  const tr = document.createElement('tr');
  for (const key of ['time', 'test', 'source', 'ms', 'notes']) {
    const td = document.createElement('td');
    if (key === 'notes') td.className = 'notes';
    td.textContent = row[key];
    tr.appendChild(td);
  }
  tbody.appendChild(tr);
}

function kv(obj) {
  return Object.entries(obj)
    .map(([k, v]) => `${k}=${v && typeof v === 'object' ? JSON.stringify(v) : v}`)
    .join(' ');
}

// ---- messages from the frame ----------------------------------------------

window.addEventListener('message', (event) => {
  if (event.origin !== CDN_ORIGIN || event.source !== iframe.contentWindow) return;
  const m = event.data;
  if (!m || typeof m !== 'object') return;
  switch (m.type) {
    case 'chunk':
      log(m.test, m.source, m.ms, `bytes=${m.bytes}`);
      break;
    case 'fill':
      log('T5', '', null, `ceilingMB=${m.ceilingMB}${m.error ? ` error=${m.error}` : ''}`);
      break;
    case 'probe': {
      const { type, ...rest } = m;
      log('probe(frame)', '', null, kv(rest));
      break;
    }
    case 'error':
      log(m.test, 'error', null, m.message);
      break;
    default:
      log('?', '', null, JSON.stringify(m));
  }
});

// ---- T4: HTTP cache from the top level ------------------------------------

document.getElementById('t4').addEventListener('click', async () => {
  try {
    const res = await fetch(CHUNK_URL, { cache: 'default' });
    const buf = await res.arrayBuffer(); // entry lands in the timeline once the body is read
    const entry = performance.getEntriesByName(CHUNK_URL).at(-1);
    if (!entry) {
      log('T4', 'error', null, `no PerformanceResourceTiming entry for ${CHUNK_URL}; bytes=${buf.byteLength}`);
      return;
    }
    const transferSize = entry.transferSize;
    const source = transferSize === 0 ? 'cache' : 'network';
    log('T4', source, entry.duration,
      `transferSize=${transferSize} bytes=${buf.byteLength} status=${res.status}` +
      ` (transferSize 0 = HTTP cache; also 0 if Timing-Allow-Origin is missing on the response)`);
  } catch (e) {
    log('T4', 'error', null, e && e.message ? `${e.name}: ${e.message}` : String(e));
  }
});

// ---- Probe: frame + top level ---------------------------------------------

document.getElementById('probe').addEventListener('click', async () => {
  iframe.contentWindow.postMessage({ type: 'probe' }, CDN_ORIGIN);
  try {
    const { probe } = await import(`${CDN_ORIGIN}/probe.js`);
    const r = await probe();
    log('probe(top)', '', null, kv(r));
  } catch (e) {
    log('probe(top)', 'error', null, e && e.message ? `${e.name}: ${e.message}` : String(e));
  }
});

// ---- Copy results as Markdown ---------------------------------------------

function toMarkdown() {
  const esc = (s) => String(s).replace(/\|/g, '\\|').replace(/\n/g, ' ');
  const lines = ['| time | test | source | ms | notes |', '| --- | --- | --- | --- | --- |'];
  for (const r of rows) {
    lines.push(`| ${esc(r.time)} | ${esc(r.test)} | ${esc(r.source)} | ${esc(r.ms)} | ${esc(r.notes)} |`);
  }
  return lines.join('\n') + '\n';
}

document.getElementById('copy').addEventListener('click', async () => {
  const md = toMarkdown();
  try {
    await navigator.clipboard.writeText(md);
    log('copy', '', null, `copied ${rows.length} rows as Markdown`);
  } catch (e) {
    log('copy', 'error', null, e && e.message ? `${e.name}: ${e.message}` : String(e));
  }
});
