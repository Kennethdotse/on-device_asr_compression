# Kasanoma - On-Device English-Twi Code-Switched ASR

**Challenge Track: Mobile AI**

## What we built

Kasanoma is an on-device speech recognition system for English-Twi code-switched speech.

The goal is to make ASR for Ghanaian speech practical on mobile and edge devices, where memory, compute, and connectivity can be limited.

We fine-tuned Whisper on KasaSpeech, our English-Twi code-switching speech dataset, then explored model compression and lightweight inference runtimes to get it running locally on a phone.

For this challenge, we focused on two compression approaches:

1. **Whisper Tiny** - using the smaller Whisper architecture.
2. **Depth-6 Whisper** - reducing Whisper Small from 12 layers to 6 layers.

For both, we evaluated INT8 quantized models using two inference runtimes: `whisper.cpp` and ONNX Runtime.

The main deployment target is Arm-powered mobile devices, using CPU-based, offline inference.

## Results

All results are from the same 1,681-utterance test set. Full numbers (parameters, CPU%, per-runtime comparisons) are in the `quant/` eval CSVs; here's what matters for deployment.

| Model | Runtime | Size | Peak RAM | WER |
| --- | --- | --- | --- | --- |
| Whisper Small (baseline, uncompressed) | - | 922MB | 1,607 MB | 0.116 |
| **Whisper Tiny INT8** | **whisper.cpp** | **41 MB** | **237.5 MB** | **0.081** |
| Whisper Tiny INT8 | ONNX Runtime | 66.8 MB | 1,104 MB | 0.248 |
| **Depth-6 INT8** | **whisper.cpp** | **151 MB** | **429.9 MB** | **0.132** |
| Depth-6 INT8 | ONNX Runtime | 229.1 MB | 1,464 MB | 0.252 |

Two things stand out:

**The runtime matters as much as the compression.** The same INT8 weights perform very differently depending on how they're run. Tiny INT8 goes from 237.5 MB peak RAM and 0.081 WER under `whisper.cpp` to 1,104 MB and 0.248 WER under ONNX Runtime, on identical model weights. A checkpoint being small on disk doesn't mean it stays small in memory once loaded.

**Whisper Tiny INT8 with `whisper.cpp` is our strongest result overall.** It beats the uncompressed Whisper Small baseline on WER (0.081 vs 0.116) while using a fraction of the memory (237.5 MB vs 1,607 MB) and a 41 MB checkpoint. Depth-6 INT8 with `whisper.cpp` is a second viable option, closer in behavior to the original Whisper Small architecture, at 151 MB and 429.9 MB peak RAM.

## Why `whisper.cpp`

For both Tiny and Depth-6, the `whisper.cpp` INT8 build used substantially less memory than the equivalent ONNX Runtime build, sometimes by 4-5x, despite starting from the same quantized weights.

This matters for the Mobile AI track because checkpoint size is only part of the deployment problem. Runtime memory and inference behavior also determine whether a model is actually practical on a phone, and those can vary a lot by runtime even when the model itself doesn't change.

## Arm and Mobile AI

Kasanoma is designed around CPU-based local inference rather than cloud inference or a dedicated GPU.

Our target deployment path:

```text
Microphone
    -> Audio preprocessing
    -> Kasanoma INT8 model
    -> whisper.cpp
    -> English-Twi transcription
```

`whisper.cpp` provides a lightweight C/C++ inference path suited to CPU-based deployment on Arm-powered mobile and edge devices.

The current benchmarks were collected during model development on desktop hardware. The next step is benchmarking directly on Arm mobile hardware for device-specific latency, memory, and power measurements.

## Dataset

KasaSpeech is our English-Twi code-switching speech dataset:

* 54,855 recordings
* approximately 95.6 hours of speech
* English-Twi code-switched utterances
* multiple speakers

The dataset is built around natural language switching between English and Twi, rather than treating the two languages as separate ASR tasks.

## Model Artifacts

The compressed models are available on Hugging Face:

* **Kasanoma Whisper Tiny, whisper.cpp**
  https://huggingface.co/Kennethdot/kasanoma_whisper_tiny.cpp

* **Kasanoma Whisper Depth-6, whisper.cpp**
  https://huggingface.co/Kennethdot/kasanoma_whisper_depth_6_whisper.cpp

* **Kasanoma Whisper Depth-6, ONNX INT8**
  `Kennethdot/kasanoma_whisper_depth6_onnx`

## Repository Structure

```text
quant/
├── test/
│   ├── whisper_cpp_eval.py
│   └── onnx_whisper_eval.py
│
├── model_export_depth_6/
└── whisper_depth_6_onnx_int8/
```

The evaluation scripts measure Word Error Rate (WER), Character Error Rate (CER), Real-Time Factor (RTF), inference time, CPU utilization, and peak RAM, per utterance and in aggregate.

## Setup

### Requirements

* Python 3.11
* Compiled `whisper-cli` from whisper.cpp
* Python dependencies in `requirements.txt`

```bash
pip install -r requirements.txt
```

### whisper.cpp

Set the paths to the `whisper-cli` binary and model in:

```text
quant/test/whisper_cpp_eval.py
```

Then run:

```bash
python quant/test/whisper_cpp_eval.py
```

### ONNX Runtime

```bash
export HF_TOKEN=hf_...
python quant/test/onnx_whisper_eval.py
```

Do not hard-code the Hugging Face token in the source code.

## What was done during the challenge

The work submitted for this project was developed and meaningfully updated during the challenge period, including:

* Fine-tuning Whisper for English-Twi code-switched speech
* Reducing Whisper Small from 12 to 6 layers
* Evaluating Whisper Tiny as a smaller architecture
* INT8 quantization
* Exporting models for `whisper.cpp` and ONNX Runtime
* Building evaluation scripts for memory, latency, and ASR accuracy
* Comparing the two inference runtimes
* Preparing compressed model artifacts for mobile deployment

## Summary

Kasanoma explores how English-Twi ASR can be compressed for mobile deployment without losing most of its recognition performance.

The strongest result was Whisper Tiny INT8 running with `whisper.cpp`: a 41 MB model, 237.5 MB peak RAM, and 0.081 WER, beating the uncompressed baseline on accuracy while using a fraction of the memory. Depth-6 INT8 with `whisper.cpp` offers a second option closer to the original Whisper Small architecture, at 151 MB and 429.9 MB peak RAM.

These results give a practical basis for deploying English-Twi ASR locally on Arm-powered mobile and edge devices.
