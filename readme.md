# Kasanoma — On-Device English–Twi Code-Switched ASR

**Challenge track:** Mobile AI

## What we built

Kasanoma is a compressed, on-device speech recognition system for
**English–Twi code-switching** — the everyday pattern of Ghanaian speakers
moving between English and Twi within a single sentence. Standard monolingual
ASR models fail at these switch points, and general multilingual models are
too large and slow to run locally on a phone.

We fine-tuned Whisper Small on **KasaSpeech** (54,855 recordings, 95.6 hours
of English–Twi code-switched audio), then compressed it — via structural
depth reduction and INT8 quantization — for local, offline inference on
Arm-based mobile/edge hardware. On-device inference means no audio ever
has to leave the phone, which matters for both privacy and for users on
unreliable connectivity.

## Repo / artifacts

```
quant/
  test/
    whisper_cpp_eval.py     # whisper.cpp (Arm-optimized CPU inference) eval harness
    onnx_whisper_eval.py    # ONNX Runtime eval harness (comparison baseline)
  model_export_depth_6/     # GGML export for whisper.cpp
  whisper_depth_6_onnx_int8/# ONNX INT8 export
```

Both harnesses measure WER, CER, inference time, RTF, CPU utilization, and
peak RAM per utterance and in aggregate, using an identical text-normalization
step so results are directly comparable across runtimes.

## Setup

**Requirements:** Python 3.11, a compiled `whisper-cli` (whisper.cpp) binary,
and/or an ONNX Runtime environment.

```bash
pip install -r requirements.txt   # jiwer, psutil, soundfile, datasets, optimum, transformers
```

**Run the whisper.cpp (Arm-targeted) evaluation:**
```bash
python quant/test/whisper_cpp_eval.py
```
Set `WHISPER_CLI` and `MODEL_PATH` at the top of the script's `__main__`
block to your compiled binary and GGML model.

**Run the ONNX Runtime comparison:**
```bash
export HF_TOKEN=hf_...   # do not hardcode
python quant/test/onnx_whisper_eval.py
```

Each run prints a summary and writes a per-utterance CSV.

## Arm relevance

`whisper.cpp` runs entirely on CPU via GGML, with no GPU dependency — this is
the deployment path we target for Arm-based Android devices. Our results
show `whisper.cpp` is not just a convenience runtime but the one that
actually realizes the memory savings from quantization on this kind of
hardware, which is the core reason it's the right fit for Arm mobile
inference specifically (see benchmarks below).

## What we optimized

- **Model size:** structural depth reduction (12→6 encoder/decoder layers)
  cuts parameters from 244M to 142.5M; INT8 quantization further shrinks the
  on-disk artifact from 543.6 MB to **151 MB** (whisper.cpp/GGML export).
- **Peak RAM:** the critical mobile constraint. Depth-6 INT8 via whisper.cpp
  reaches **429.9 MB** peak RSS, a **73% reduction** from the 1,607 MB
  uncompressed baseline.
- **Latency/RTF:** all compressed configurations stay comfortably faster
  than real time (RTF 0.19–0.27), suitable for near-real-time transcription
  on-device.
- **Accuracy retention:** Depth-6 INT8 (whisper.cpp) holds WER at 0.132 vs.
  0.116 for the uncompressed baseline — a 1.6-point cost for a 73% memory
  reduction.

## Benchmarks (fixed 1,681-utterance test set)

Two compression paths, both fine-tuned on KasaSpeech: **structural pruning**
of our own Whisper Small model (Depth-6, 12→6 layers) and a **natively small
architecture** (Whisper Tiny) trained from scratch on the same data as a
comparison point.

| System | Size | Peak RAM | WER | RTF |
|---|---|---|---|---|
| Whisper Small baseline (uncompressed) | — | 1,607 MB | 0.116 | 0.195 |
| Depth-6 FP32 (pruned) | 543.6 MB | — | 0.173 | — |
| **Depth-6 INT8 — whisper.cpp (Arm-targeted)** | **151 MB** | **429.9 MB** | **0.132** | **0.229** |
| Depth-6 INT8 — ONNX Runtime | 229.1 MB | 1,464 MB | 0.252 | 0.273 |
| Whisper Tiny FP32 (native, fine-tuned) | 144.1 MB | 1,682 MB | 0.161 | 0.023 |
| Whisper Tiny INT8 — whisper.cpp (Arm-targeted) | — | — | **0.081** | — |
| Whisper Tiny INT8 — ONNX Runtime | 66.8 MB | — | 0.248 | — |

Two findings stand out:
1. **Runtime matters more than the nominal precision label.** Every INT8
   model is dramatically more accurate under whisper.cpp than under ONNX
   Runtime — for Tiny, 0.081 vs. 0.248 WER on the *same* quantized weights.
   Quantization only pays off if the inference runtime actually realizes it.
2. **Smaller-on-disk isn't smaller-in-memory.** Tiny's FP32 checkpoint is
   144 MB — 4x smaller than Depth-6 FP32 — yet its peak RAM (1,682 MB)
   actually exceeds the full 244M-parameter baseline. Depth-6 INT8 via
   whisper.cpp is the only configuration that meaningfully cuts *runtime*
   memory (429.9 MB, a 73% reduction), which is what actually determines
   whether a model fits on a phone.

## Confirmation

All model compression (depth reduction, INT8 export), evaluation harnesses,
and benchmark runs in this submission were built and run during the
challenge period.
