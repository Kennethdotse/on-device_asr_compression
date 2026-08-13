# Project Kasa — Mobile English–Twi ASR

> **Arm Create Challenge 2026 — Mobile AI**

Project Kasa is an English–Twi code-switched speech recognition system designed for Ghanaian speech.

We started with a fine-tuned Whisper Small model and are optimizing it for **on-device inference on Arm64 mobile devices**. The focus is reducing model size and memory usage while maintaining useful recognition accuracy.

## The Problem

English–Twi code-switching is common in everyday speech, but existing ASR systems often struggle when speakers switch between the two languages.

Our original fine-tuned Whisper Small model was around **922 MB** and used over **1.6 GB of RAM** during inference, making direct mobile deployment challenging.

## What We Built

We explored:

- **Whisper depth reduction** — reducing the encoder/decoder layers of the trained model.
- **INT8 quantization** — using ONNX Runtime and `whisper.cpp`.
- **Lightweight inference** — investigating deployment through `whisper.cpp`.
- **Arm64 deployment** — targeting Android and other Arm-powered devices.

Our current strongest compressed Whisper Small candidate is a **6-layer model with INT8 quantization using `whisper.cpp`**.

## Current Results

| Model | Runtime | WER | CER |
|---|---|---:|---:|
| Whisper Small | Baseline | -- | -- |
| Depth-6 + INT8 | `whisper.cpp` | **0.1320** | **0.0940** |
| Depth-6 + INT8 | ONNX Runtime | 0.2516 | 0.2070 |

The original CPU baseline already ran at approximately **5.1× real time**, so our main optimization target is **memory and model footprint**, rather than latency alone.

## Architecture

```text
English–Twi Speech
        │
        ▼
 Fine-tuned Whisper
        │
        ▼
  Depth Reduction
        │
        ▼
   Quantization
        │
   ┌────┴────┐
   ▼         ▼
ONNX      whisper.cpp
   │         │
   └────┬────┘
        ▼
     Arm64
     Mobile





