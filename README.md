# BitNet b1.58 — Chat in the Browser

A chatbot that runs entirely in your browser. No server. No API. No data leaves your device.
Open `index.html`, wait for the model to load, and start chatting.

> **⚠️ Warning**
> This model has only **~22M parameters** — it is a tiny demonstration model, not a production assistant.
> It **will hallucinate**, sometimes confidently. It may invent facts, contradict itself, or produce
> nonsense. Do not rely on it for factual information, medical advice, legal advice, or any decision
> that matters. Treat every response as a rough draft, not as truth.

---

## Why this matters

Any cloud AI sends your messages to a server. That means:

- Your data lives in someone else's logs.
- You pay per token — or with your data.
- Server goes down, you can't chat.
- The model can be changed or shut off without notice.

This project does the opposite. The model is shipped to your browser and runs locally. Once loaded, you can turn off Wi-Fi and it still works.

---

## What makes this different

**No server, no cost.** The model is a single `.js` file. Host it on GitHub Pages, Cloudflare Pages, a USB stick — hosting cost is zero. Inference cost is zero. Per-user cost is zero.

**Privacy by architecture.** There is no code that sends data anywhere. Not because we promise not to log, but because there is nowhere to send it. Open DevTools → Network → chat → you will see zero requests.

**Runs on tiny RAM.** Under 500 MB at inference time. Works on a mid-range Android phone, an old laptop, a Raspberry Pi. A conventional 7B model needs ~14 GB VRAM and a GPU costing thousands of dollars.

**Small by design, not by compression.** BitNet b1.58 forces every weight to be one of `{-1, 0, +1}` during training. The result is a model ~10x smaller than FP16 with nearly identical accuracy.

---

## Comparison

| | Cloud AI | **This project** |
|---|---|---|
| Requires a server | Yes | No |
| Data leaves device | Yes | No |
| Requires an account | Yes | No |
| Cost per token | Yes | No |
| Works offline | No | Yes |

---

## FAQ

**Is this really running locally?**
Yes. Open DevTools → Network → chat. Zero requests.

**Why is loading slow?**
The file is ~15 MB. After browser caching, next load is under a second.

**Is the output as good as any cloud AI?**
No. This model is ~22M parameters. Cloud AI models are orders of magnitude larger. The point of this project is not raw quality — it is to show that a useful AI can run entirely in a browser with no server. Expect hallucination, repetition, and nonsense at times.

**Why BitNet?**
Because a regular 7B model is 14 GB — not realistic for a browser. BitNet is small by design: every weight is ternary {-1, 0, +1}, packed into ~2 bits instead of 16, so a useful model fits in a browser tab.

---

## Reference

The BitNet b1.58 architecture belongs to Microsoft Research.
https://www.microsoft.com/en-us/research/publication/the-era-of-1-bit-llms-all-large-language-models-are-in-1-58-bits/
