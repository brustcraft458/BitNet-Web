// chat.js — chat interface logic.
import { loadModel, generate } from "./aidecoder.js";

const $ = (s) => document.querySelector(s);

const historyEl     = $("#history");
const inputEl       = $("#input");
const sendBtn       = $("#send");
const statusSizeEl  = $("#status-size");
const statusInfoEl  = $("#status-info");
const tempEl        = $("#temperature");
const topkEl        = $("#topk");
const maxTokEl      = $("#maxtok");
const loaderEl      = $("#loader");
const loaderStageEl = $("#loader-stage");
const loaderFillEl  = $("#loader-fill");
const loaderPctEl   = $("#loader-pct");
const skeletonEl    = $("#skeleton");

let model = null;
let tokenizer = null;
let busy = false;

// Label ukuran yang ditampilkan di kanan header.
// Contoh:
//   fileMB  = "13.42 MB"                 (file aimodel.js)
//   wireMB  = "11.20 MB"                 (kalau server kompres)
//   sizeLabel = "11.20 MB ↓ / 13.42 MB"  (kalau kompresi terdeteksi)
//               "13.42 MB"               (kalau tidak)
let sizeLabel = "—";

function fmtMB(bytes) {
  if (!bytes || bytes <= 0) return null;
  return (bytes / 1024 / 1024).toFixed(2) + " MB";
}

function buildSizeLabel(res) {
  const fileMB = fmtMB(res.fileBytes);
  const wireMB = fmtMB(res.wireBytes);

  // Kalau wireBytes tidak tersedia (cached / API kosong) → tampilkan file saja.
  // Kalau wireBytes tersedia & signifikan lebih kecil → tampilkan keduanya.
  if (fileMB && wireMB && res.wireBytes < res.fileBytes * 0.95) {
    return `${wireMB} ↓ / ${fileMB}`;
  }
  if (fileMB) return fileMB;
  if (wireMB) return wireMB;
  // Last resort: kalau semuanya gagal, fallback ke ukuran binary model.
  const modelMB = fmtMB(res.modelBytes);
  return modelMB || "—";
}

function appendMsg(role, text) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.textContent = text;
  historyEl.appendChild(div);
  historyEl.scrollTop = historyEl.scrollHeight;
  return div;
}

function setBusy(b) {
  busy = b;
  inputEl.disabled = b;
  sendBtn.disabled = b;
}

/* ---------- Textarea auto-grow ---------- */
function autoGrow() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + "px";
}

/* ---------- Loader helpers ---------- */
function setLoader(p, stage) {
  const pct = Math.max(0, Math.min(1, p || 0));
  loaderFillEl.style.width = (pct * 100).toFixed(1) + "%";
  loaderPctEl.textContent  = (pct * 100).toFixed(0) + "%";
  if (stage) loaderStageEl.textContent = stage;
}

function hideLoader() {
  loaderEl.classList.add("hidden");
  setTimeout(() => {
    if (skeletonEl && skeletonEl.parentNode) skeletonEl.remove();
  }, 500);
}

/* ---------- Init ---------- */
async function init() {
  statusSizeEl.textContent = "—";
  statusInfoEl.textContent = "Loading…";
  setLoader(0, "Menyiapkan…");
  await new Promise(r => setTimeout(r, 60));

  try {
    const res = await loadModel({
      onProgress: (p, stage) => setLoader(p, stage),
    });
    model = res.model;
    tokenizer = res.tokenizer;
    sizeLabel = buildSizeLabel(res);

    const cfg = res.model.cfg;
    statusSizeEl.textContent = sizeLabel;
    statusInfoEl.textContent =
      `${cfg.d_model}d · ${cfg.n_layers}L · ` +
      `${cfg.vocab_size.toLocaleString()} vocab · ` +
      `${res.loadSeconds.toFixed(2)}s`;

    inputEl.disabled = false;
    sendBtn.disabled = false;
    hideLoader();
    autoGrow();
    inputEl.focus();
  } catch (e) {
    console.error(e);
    loaderStageEl.textContent = "Load error: " + e.message;
    statusInfoEl.textContent = "Load error";
  }
}

/* ---------- Send / generate ---------- */
async function send() {
  if (busy || !model) return;
  const prompt = inputEl.value.trim();
  if (!prompt) return;

  setBusy(true);
  inputEl.value = "";
  autoGrow();
  appendMsg("user", prompt);

  const botEl = appendMsg("bot", "");
  botEl.classList.add("thinking");
  botEl.innerHTML = '<span class="dots"><span></span><span></span><span></span></span>';

  let text = "";
  let firstToken = true;
  const t0 = performance.now();

  try {
    await generate(model, tokenizer, prompt, {
      max_new_tokens: parseInt(maxTokEl.value) || 128,
      temperature: parseFloat(tempEl.value) || 0.7,
      top_k: parseInt(topkEl.value) || 40,
      onToken: (delta) => {
        if (firstToken) {
          botEl.classList.remove("thinking");
          botEl.textContent = "";
          firstToken = false;
        }
        text += delta;
        botEl.textContent = text;
        historyEl.scrollTop = historyEl.scrollHeight;
      },
    });
    if (firstToken) {
      botEl.classList.remove("thinking");
      botEl.textContent = "(empty response)";
    }
    const dt = (performance.now() - t0) / 1000;
    statusSizeEl.textContent = sizeLabel;
    statusInfoEl.textContent = `${text.length} chars · ${dt.toFixed(2)}s`;
  } catch (e) {
    console.error(e);
    botEl.classList.remove("thinking");
    botEl.textContent = "Error: " + e.message;
  }

  setBusy(false);
  inputEl.focus();
}

export function boot() {
  sendBtn.addEventListener("click", send);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  inputEl.addEventListener("input", autoGrow);
  init();
}