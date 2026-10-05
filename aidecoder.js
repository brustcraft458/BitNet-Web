// aidecoder.js — BitNet b1.58 inference (pure JS).
// KV cache + per-token absmax + async loading dengan progress callback.
import { CONFIG, TOKENIZER, MODEL_B64 } from "./aimodel.js";

/* ============================ ByteLevel BPE ============================ */
function _bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const bsSet = new Set(bs);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bsSet.has(b)) { bs.push(b); cs.push(256 + n); n++; }
  const b2c = new Map(), c2b = new Map();
  for (let i = 0; i < bs.length; i++) {
    const ch = String.fromCodePoint(cs[i]);
    b2c.set(bs[i], ch); c2b.set(cs[i], bs[i]);
  }
  return { b2c, c2b };
}

export class BPETokenizer {
  constructor(data) {
    this.vocab = new Map(Object.entries(data.vocab));
    this.merges = data.merges;
    this.added = data.added || [];
    this.mergeRank = new Map();
    this.merges.forEach((m, i) => this.mergeRank.set(m, i));
    this.specialIds = {};
    for (const a of this.added) this.specialIds[a.content] = a.id;
    this.specialKeys = Object.keys(this.specialIds).sort((a, b) => b.length - a.length);
    this.idToToken = new Array(this.vocab.size);
    for (const [t, i] of this.vocab) this.idToToken[i] = t;
    for (const a of this.added) if (this.idToToken[a.id] === undefined) this.idToToken[a.id] = a.content;
    const { b2c, c2b } = _bytesToUnicode();
    this.b2c = b2c; this.c2b = c2b; this.bpeCache = new Map();
    this.pat = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
  }
  _textToMappedBytes(text) {
    const bytes = new TextEncoder().encode(text);
    let s = "";
    for (const b of bytes) s += this.b2c.get(b);
    return s;
  }
  _bpe(mapped) {
    if (this.bpeCache.has(mapped)) return this.bpeCache.get(mapped);
    let word = Array.from(mapped);
    if (word.length === 0) return [];
    while (word.length > 1) {
      let bestRank = Infinity, bestI = -1;
      for (let i = 0; i < word.length - 1; i++) {
        const r = this.mergeRank.get(word[i] + " " + word[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; bestI = i; }
      }
      if (bestI < 0) break;
      const next = []; let i = 0;
      while (i < word.length) {
        if (i === bestI) { next.push(word[i] + word[i + 1]); i += 2; }
        else { next.push(word[i]); i += 1; }
      }
      word = next;
    }
    const ids = [];
    for (const t of word) {
      const id = this.vocab.get(t);
      ids.push(id !== undefined ? id : (this.specialIds["<unk>"] ?? 1));
    }
    this.bpeCache.set(mapped, ids);
    return ids;
  }
  _encodePlain(text, out) {
    const pieces = text.match(this.pat) || [];
    for (const p of pieces) {
      const ids = this._bpe(this._textToMappedBytes(p));
      for (const id of ids) out.push(id);
    }
  }
  encode(text) {
    const out = [];
    let rest = text;
    while (rest.length > 0) {
      let bestPos = -1, bestTok = null;
      for (const sp of this.specialKeys) {
        const p = rest.indexOf(sp);
        if (p >= 0 && (bestPos < 0 || p < bestPos)) { bestPos = p; bestTok = sp; }
      }
      if (bestPos < 0) { this._encodePlain(rest, out); break; }
      if (bestPos > 0) this._encodePlain(rest.slice(0, bestPos), out);
      out.push(this.specialIds[bestTok]);
      rest = rest.slice(bestPos + bestTok.length);
    }
    return out;
  }
  decode(ids) {
    let s = "";
    for (const id of ids) {
      const t = this.idToToken[id];
      if (t === undefined) continue;
      if (this.specialIds[t] !== undefined) continue;
      s += t;
    }
    const bytes = [];
    for (const ch of s) {
      const b = this.c2b.get(ch.codePointAt(0));
      if (b !== undefined) bytes.push(b);
    }
    return new TextDecoder().decode(new Uint8Array(bytes));
  }
  getSpecial(name) { return this.specialIds[name]; }
}

/* ============================ Binary decoder ============================ */
function _b64ToBytes(b64) {
  const raw = atob(b64);
  const n = raw.length;
  const bytes = new Uint8Array(n);
  for (let i = 0; i < n; i++) bytes[i] = raw.charCodeAt(i);
  return bytes;
}

const _yield = () => new Promise(r => setTimeout(r, 0));

// Chunked base64 decode — biar UI tidak freeze dan progress bisa di-report.
// Chunk selalu kelipatan 4 char supaya atob tidak error di tengah.
async function _b64ToBytesAsync(b64, onProgress) {
  const n = b64.length;
  const CHUNK = 4 * 32768;   // 128 KB b64 chars per iterasi
  const YIELD_EVERY = 2;     // yield tiap 2 chunk (~192 KB output)
  const parts = [];
  let total = 0;
  let iter = 0;
  for (let i = 0; i < n; i += CHUNK) {
    const end = Math.min(i + CHUNK, n);
    const bin = atob(b64.slice(i, end));
    const arr = new Uint8Array(bin.length);
    for (let j = 0; j < bin.length; j++) arr[j] = bin.charCodeAt(j);
    parts.push(arr);
    total += arr.length;
    if (onProgress) onProgress(end / n);
    if (++iter % YIELD_EVERY === 0) await _yield();
  }
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) { out.set(p, off); off += p.length; }
  return out;
}

function _f16(h) {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7C00) >> 10;
  const f = h & 0x03FF;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1F) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

// Sync version (fallback / backward compat)
export function decodeModel(b64) {
  const bytes = _b64ToBytes(b64);
  const parsed = _parseBlob(bytes);
  return { ...parsed, byteLength: bytes.byteLength };
}

// Async version dengan progress + yield
async function decodeModelAsync(b64, onProgress) {
  // Stage 1: base64 → bytes  (progress 0 → 0.40)
  const bytes = await _b64ToBytesAsync(b64, p => {
    onProgress?.(p * 0.40, `Membaca weights… ${(p * 100).toFixed(0)}%`);
  });
  // Stage 2: parse header + tensors (progress 0.40 → 1.0)
  const parsed = await _parseBlobAsync(bytes, (p, msg) => {
    onProgress?.(0.40 + p * 0.60, msg);
  });
  return { ...parsed, byteLength: bytes.byteLength };
}

function _parseBlobHeader(bytes) {
  const view = new DataView(bytes.buffer);
  const td = new TextDecoder();
  let p = 0;
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]); p += 4;
  if (magic !== "BTN2") throw new Error("bad magic: " + magic);
  const version = view.getUint16(p, true); p += 2;
  const cfgLen = view.getUint32(p, true); p += 4;
  const cfg = JSON.parse(td.decode(bytes.subarray(p, p + cfgLen))); p += cfgLen;
  const nTensors = view.getUint32(p, true); p += 4;
  return { view, td, p, version, cfg, nTensors };
}

function _parseOneTensor(bytes, view, td, p) {
  const nameLen = view.getUint16(p, true); p += 2;
  const name = td.decode(bytes.subarray(p, p + nameLen)); p += nameLen;
  const dtype = bytes[p]; p += 1;
  const ndim = bytes[p]; p += 1;
  const shape = [];
  for (let j = 0; j < ndim; j++) { shape.push(view.getUint32(p, true)); p += 4; }
  let gamma = null;
  if (dtype === 2) { gamma = view.getFloat32(p, true); p += 4; }
  const dataLen = view.getUint32(p, true); p += 4;
  const dataStart = p; p += dataLen;
  const n = shape.reduce((a, b) => a * b, 1);
  let tensor;
  if (dtype === 2) {
    const out = new Int8Array(n);
    for (let k = 0; k < n; k++) {
      const code = (bytes[dataStart + (k >> 2)] >> ((k & 3) * 2)) & 3;
      out[k] = code === 1 ? 1 : code === 2 ? -1 : 0;
    }
    tensor = { dtype: "ternary", shape, gamma, data: out };
  } else if (dtype === 0) {
    const dv = new DataView(bytes.buffer, dataStart, dataLen);
    const out = new Float32Array(n);
    for (let k = 0; k < n; k++) out[k] = _f16(dv.getUint16(k * 2, true));
    tensor = { dtype: "fp16", shape, data: out };
  } else {
    const dv = new DataView(bytes.buffer, dataStart, dataLen);
    const out = new Float32Array(n);
    for (let k = 0; k < n; k++) out[k] = dv.getFloat32(k * 4, true);
    tensor = { dtype: "fp32", shape, data: out };
  }
  return { name, tensor, p };
}

function _parseBlob(bytes) {
  const { view, td, version, cfg, nTensors, p: pStart } = _parseBlobHeader(bytes);
  let p = pStart;
  const tensors = {};
  for (let i = 0; i < nTensors; i++) {
    const r = _parseOneTensor(bytes, view, td, p);
    tensors[r.name] = r.tensor;
    p = r.p;
  }
  return { version, config: cfg, tensors };
}

async function _parseBlobAsync(bytes, onProgress) {
  const { view, td, p: pStart, version, cfg, nTensors } = _parseBlobHeader(bytes);
  let p = pStart;
  const tensors = {};
  for (let i = 0; i < nTensors; i++) {
    const r = _parseOneTensor(bytes, view, td, p);
    tensors[r.name] = r.tensor;
    p = r.p;
    if (i % 4 === 3 || i === nTensors - 1) {
      await _yield();
      onProgress?.((i + 1) / nTensors, `Parse tensor ${i + 1}/${nTensors}…`);
    }
  }
  return { version, config: cfg, tensors };
}

/* ============================ BitNet model ============================ */
export class BitNetModel {
  constructor(tensors, config) {
    this.t = tensors;
    this.cfg = config;
    this.D = config.d_model;
    this.L = config.n_layers;
    this.H = config.n_heads;
    this.HD = this.D / this.H;
    this.FF = config.d_ff;
    this.V = config.vocab_size;
    this.TMAX = config.max_seq_len;
    this.residual_scale = Math.pow(2 * this.L, -0.5);
    this.eps = 1e-6;
    this.attnScale = 1 / Math.sqrt(this.HD);
    this.Qb = 127.0;

    this.tokEmb = tensors["tok_emb.weight"].data;
    this.posEmb = tensors["pos_emb.weight"].data;

    // KV cache — per layer, flat [TMAX * D]
    this.kCache = new Array(this.L).fill(null);
    this.vCache = new Array(this.L).fill(null);
  }

  resetCache() {
    const size = this.TMAX * this.D;
    for (let l = 0; l < this.L; l++) {
      this.kCache[l] = new Float32Array(size);
      this.vCache[l] = new Float32Array(size);
    }
  }

  // startPos = posisi token pertama di `ids` dalam sequence global.
  //   forward(wholePrompt, 0) → prefill, isi cache posisi [0, T)
  //   forward([nextTok], pos) → decode, isi cache posisi [pos, pos+1)
  forward(ids, startPos = 0) {
    if (startPos === 0 || this.kCache[0] === null) this.resetCache();

    const T = ids.length, D = this.D, H = this.H, HD = this.HD, FF = this.FF, L = this.L;
    const Qb = this.Qb, eps = this.eps;
    const totalT = startPos + T;

    const h = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      const tb = ids[t] * D, pb = (startPos + t) * D, hb = t * D;
      for (let i = 0; i < D; i++) h[hb + i] = this.tokEmb[tb + i] + this.posEmb[pb + i];
    }

    const x1 = new Float32Array(T * D);
    const q = new Float32Array(T * D);
    const k = new Float32Array(T * D);
    const v = new Float32Array(T * D);
    const attnCat = new Float32Array(T * D);
    const attnOut = new Float32Array(T * D);
    const f1 = new Float32Array(T * FF);
    const f3 = new Float32Array(T * FF);
    const fAct = new Float32Array(T * FF);
    const fOut = new Float32Array(T * D);
    const xq = new Float32Array(Math.max(D, FF));
    const scores = new Float32Array(totalT);

    const rmsNormRow = (src, sOff, w, dst, dOff) => {
      let ss = 0;
      for (let i = 0; i < D; i++) { const val = src[sOff + i]; ss += val * val; }
      const inv = 1 / Math.sqrt(ss / D + eps);
      for (let i = 0; i < D; i++) dst[dOff + i] = src[sOff + i] * inv * w[i];
    };

    // Per-token absmax int8 quantization (konsisten prefill ↔ decode).
    const quantRow = (src, off, n) => {
      let m = 0;
      for (let i = 0; i < n; i++) { const a = Math.abs(src[off + i]); if (a > m) m = a; }
      if (m < 1e-5) m = 1e-5;
      const s = Qb / m, iv = m / Qb;
      for (let i = 0; i < n; i++) {
        let z = Math.round(src[off + i] * s);
        if (z > Qb) z = Qb; else if (z < -Qb) z = -Qb;
        xq[i] = z * iv;
      }
    };
    const mm = (dst, dOff, w, wg, inDim, outDim) => {
      for (let m = 0; m < outDim; m++) {
        const wb = m * inDim;
        let s = 0;
        for (let kk = 0; kk < inDim; kk++) s += xq[kk] * w[wb + kk];
        dst[dOff + m] = s * wg;
      }
    };

    for (let l = 0; l < L; l++) {
      const p = "blocks." + l + ".";
      const n1 = this.t[p + "norm1.weight"].data;
      const n2 = this.t[p + "norm2.weight"].data;
      const qw = this.t[p + "attn.q_proj.weight"];
      const kw = this.t[p + "attn.k_proj.weight"];
      const vw = this.t[p + "attn.v_proj.weight"];
      const ow = this.t[p + "attn.o_proj.weight"];
      const w1 = this.t[p + "ffn.w1.weight"];
      const w2 = this.t[p + "ffn.w2.weight"];
      const w3 = this.t[p + "ffn.w3.weight"];

      // ---- Attention ----
      for (let t = 0; t < T; t++) rmsNormRow(h, t * D, n1, x1, t * D);

      for (let t = 0; t < T; t++) {
        const xb = t * D;
        quantRow(x1, xb, D);
        mm(q, xb, qw.data, qw.gamma, D, D);
        mm(k, xb, kw.data, kw.gamma, D, D);
        mm(v, xb, vw.data, vw.gamma, D, D);
      }

      // Append K/V ke cache
      const kc = this.kCache[l];
      const vc = this.vCache[l];
      for (let t = 0; t < T; t++) {
        const src = t * D;
        const dst = (startPos + t) * D;
        kc.set(k.subarray(src, src + D), dst);
        vc.set(v.subarray(src, src + D), dst);
      }

      // Causal attention per query token, baca K/V dari cache [0, totalT)
      for (let t = 0; t < T; t++) {
        const globalT = startPos + t;
        const qb = t * D;
        for (let hh = 0; hh < H; hh++) {
          const off = hh * HD;
          let maxS = -Infinity;
          for (let tt = 0; tt <= globalT; tt++) {
            const kb = tt * D + off;
            let s = 0;
            for (let d = 0; d < HD; d++) s += q[qb + off + d] * kc[kb + d];
            s *= this.attnScale;
            scores[tt] = s;
            if (s > maxS) maxS = s;
          }
          let sum = 0;
          for (let tt = 0; tt <= globalT; tt++) {
            const e = Math.exp(scores[tt] - maxS);
            scores[tt] = e; sum += e;
          }
          const invSum = 1 / sum;
          for (let d = 0; d < HD; d++) attnCat[qb + off + d] = 0;
          for (let tt = 0; tt <= globalT; tt++) {
            const wgt = scores[tt] * invSum;
            const vb = tt * D + off;
            for (let d = 0; d < HD; d++) attnCat[qb + off + d] += wgt * vc[vb + d];
          }
        }
      }

      for (let t = 0; t < T; t++) {
        const xb = t * D;
        quantRow(attnCat, xb, D);
        mm(attnOut, xb, ow.data, ow.gamma, D, D);
      }
      for (let i = 0; i < T * D; i++) h[i] += this.residual_scale * attnOut[i];

      // ---- SwiGLU FFN ----
      for (let t = 0; t < T; t++) rmsNormRow(h, t * D, n2, x1, t * D);

      for (let t = 0; t < T; t++) {
        const xb = t * D, fb = t * FF;
        quantRow(x1, xb, D);
        mm(f1, fb, w1.data, w1.gamma, D, FF);
        mm(f3, fb, w3.data, w3.gamma, D, FF);
      }
      for (let i = 0; i < T * FF; i++) {
        const vv = f1[i];
        fAct[i] = (vv / (1 + Math.exp(-vv))) * f3[i];
      }

      for (let t = 0; t < T; t++) {
        const fb = t * FF, ob = t * D;
        quantRow(fAct, fb, FF);
        mm(fOut, ob, w2.data, w2.gamma, FF, D);
      }
      for (let i = 0; i < T * D; i++) h[i] += this.residual_scale * fOut[i];
    }

    // ---- Final norm + lm_head (tied ke tok_emb) ----
    const nf = this.t["norm_f.weight"].data;
    const hNorm = new Float32Array(T * D);
    for (let t = 0; t < T; t++) rmsNormRow(h, t * D, nf, hNorm, t * D);

    const V = this.V;
    const out = new Float32Array(V);
    const hb = (T - 1) * D;   // hidden state token terakhir
    for (let i = 0; i < V; i++) {
      const base = i * D;
      let s = 0;
      for (let d = 0; d < D; d++) s += hNorm[hb + d] * this.tokEmb[base + d];
      out[i] = s;
    }
    return out;
  }
}

/* ============================ Sampling ============================ */
function sampleLogits(logits, temperature, topK, V) {
  const invT = 1 / Math.max(temperature, 1e-6);
  const scaled = new Float32Array(V);
  let maxS = -Infinity;
  for (let i = 0; i < V; i++) {
    const v = logits[i] * invT;
    scaled[i] = v;
    if (v > maxS) maxS = v;
  }
  let thr = -Infinity;
  if (topK > 0 && topK < V) {
    const copy = Float32Array.from(scaled).sort((a, b) => b - a);
    thr = copy[topK - 1];
  }
  let sum = 0;
  const probs = new Float32Array(V);
  for (let i = 0; i < V; i++) {
    if (scaled[i] < thr) { probs[i] = 0; continue; }
    const e = Math.exp(scaled[i] - maxS);
    probs[i] = e;
    sum += e;
  }
  const inv = 1 / sum;
  let r = Math.random(), acc = 0;
  for (let i = 0; i < V; i++) {
    acc += probs[i] * inv;
    if (r < acc) return i;
  }
  return V - 1;
}

/* ============================ generate (prefill + decode) ============================ */
export async function generate(model, tokenizer, prompt, opts = {}) {
  const maxNew = opts.max_new_tokens ?? 128;
  const temperature = opts.temperature ?? 0.7;
  const topK = opts.top_k ?? 40;
  const onToken = opts.onToken;

  const endId = tokenizer.getSpecial("<|end|>");
  const userTok = tokenizer.getSpecial("<|user|>");
  const astTok = tokenizer.getSpecial("<|assistant|>");

  let ids = [userTok, ...tokenizer.encode(prompt), astTok];
  if (ids.length > model.TMAX) ids = ids.slice(-model.TMAX);

  // PRE-FILL: satu forward untuk seluruh prompt, sekaligus isi KV cache.
  model.resetCache();
  let logits = model.forward(ids, 0);
  let pos = ids.length;

  const outIds = [];
  let lastText = "";
  for (let s = 0; s < maxNew; s++) {
    if (pos >= model.TMAX) break;
    const next = sampleLogits(logits, temperature, topK, model.V);
    if (next === endId) break;
    outIds.push(next);
    if (onToken) {
      const full = tokenizer.decode(outIds);
      const delta = full.slice(lastText.length);
      lastText = full;
      if (delta) { onToken(delta, next); await new Promise(r => setTimeout(r, 0)); }
    }
    // DECODE: satu token saja, KV cache dipakai.
    logits = model.forward([next], pos);
    pos += 1;
  }
  return tokenizer.decode(outIds);
}

/* ============================ File size helpers ============================ */
// Ambil statistik file aimodel.js dari Resource Timing API.
// decodedBodySize = ukuran file .js setelah decompress (= ukuran di disk).
// encodedBodySize = ukuran body yang benar-benar di-download (compressed kalau server gzip).
function _getAimodelFileStats() {
  try {
    const url = new URL("./aimodel.js", import.meta.url).href;
    const list = performance.getEntriesByType("resource");
    let e = list.find(x => x.name === url);
    if (!e) e = list.find(x => x.name.includes("aimodel.js"));
    if (!e || !e.decodedBodySize) return null;
    return {
      fileBytes: e.decodedBodySize || 0,
      wireBytes: e.encodedBodySize || 0,
      transferBytes: e.transferSize || 0,
    };
  } catch {
    return null;
  }
}

// Fallback kalau Resource Timing tidak tersedia. Estimasi dari panjang string
// base64 + header JS (CONFIG & TOKENIZER). Selalu > 0.
function _estimateAimodelFileBytes() {
  // MODEL_B64 = base64 ASCII → panjang string == jumlah byte.
  const b64Bytes = MODEL_B64.length;
  // save_model_web.py wrap base64 tiap 100 char + '\n'.
  const newlines = Math.floor(b64Bytes / 100);
  // Header: export CONFIG + export TOKENIZER + wrapper MODEL_B64.
  const enc = new TextEncoder();
  const headerText =
    "export const CONFIG = " + JSON.stringify(CONFIG) +
    ";\nexport const TOKENIZER = " + JSON.stringify(TOKENIZER) +
    ";\nexport const MODEL_B64 =\n";
  const headerBytes = enc.encode(headerText).length;
  // Beberapa baris statis (komentar, spasi) — over-estimate kecil, aman.
  return Math.max(1, b64Bytes + newlines + headerBytes);
}

/* ============================ Loader (async + progress) ============================ */
let _model = null;
let _tokenizer = null;
let _cachedStats = null;

export async function loadModel(opts = {}) {
  if (_model) {
    return {
      model: _model,
      tokenizer: _tokenizer,
      loadSeconds: 0,
      modelBytes: _cachedStats.modelBytes,
      fileBytes: _cachedStats.fileBytes,
      wireBytes: _cachedStats.wireBytes,
    };
  }
  const onProgress = opts.onProgress;
  const t0 = performance.now();

  onProgress?.(0.0, "Menyiapkan…");
  await _yield();

  const { tensors, byteLength } = await decodeModelAsync(MODEL_B64, onProgress);

  onProgress?.(0.97, "Membangun model…");
  await _yield();
  _model = new BitNetModel(tensors, CONFIG);

  onProgress?.(0.99, "Membangun tokenizer…");
  await _yield();
  _tokenizer = new BPETokenizer(TOKENIZER);

  onProgress?.(1.0, "Siap");
  await _yield();

  // Hitung ukuran file aimodel.js. Prioritas: Resource Timing → fallback estimasi.
  // Jaminan: fileBytes selalu > 0.
  const fileStats = _getAimodelFileStats();
  const fileBytes = (fileStats && fileStats.fileBytes > 0)
    ? fileStats.fileBytes
    : _estimateAimodelFileBytes();
  const wireBytes = (fileStats && fileStats.wireBytes > 0)
    ? fileStats.wireBytes
    : 0;   // 0 = tidak diketahui (bisa jadi cached atau API tidak tersedia)

  _cachedStats = {
    modelBytes: byteLength,
    fileBytes,
    wireBytes,
  };

  const dt = (performance.now() - t0) / 1000;
  return {
    model: _model,
    tokenizer: _tokenizer,
    loadSeconds: dt,
    modelBytes: byteLength,
    fileBytes,
    wireBytes,
  };
}