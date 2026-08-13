# Kasa-Noma — On-Device English–Twi Code-Switched ASR

**Challenge track:** Mobile AI

## What we built

Kasa-Noma is a compressed, on-device speech recognition system for
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

| System | Size | Peak RAM | WER | RTF |
|---|---|---|---|---|
| Whisper Small baseline (uncompressed) | — | 1,607 MB | 0.116 | 0.195 |
| Depth-6 FP32 | 543.6 MB | — | 0.173 | — |
| **Depth-6 INT8 — whisper.cpp (Arm-targeted)** | **151 MB** | **429.9 MB** | **0.132** | **0.229** |
| Depth-6 INT8 — ONNX Runtime | 229.1 MB | 1,464 MB | 0.252 | 0.273 |

The whisper.cpp export delivers both the best accuracy *and* the smallest
runtime memory footprint among compressed candidates — the ONNX Runtime
version, despite a smaller file on disk, barely reduces peak RAM at all.
This runtime gap is our key finding: **quantization only pays off if the
inference runtime actually realizes it in memory**, which matters directly
for what's viable on real Arm mobile hardware.

## Confirmation

All model compression (depth reduction, INT8 export), evaluation harnesses,
and benchmark runs in this submission were built and run during the
challenge period.
