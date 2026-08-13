"""
ONNX Runtime Whisper evaluation harness.
Measures WER, CER, RTF, CPU utilization, and peak RAM per utterance and in aggregate.
"""

import re
import time
import threading
import psutil
import os
import csv

import jiwer
from transformers import GenerationConfig
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from datasets import load_dataset


# Text normalization (English–Twi safe)
def normalize_cs(text):
    text = text.lower()
    # keep English letters + Twi chars (ɔ, ɛ)
    text = re.sub(r"[^a-z0-9ɔɛ\s']", "", text)
    text = re.sub(r"'", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Background peak-RAM sampler (single inference calls are too short/uneven
# to catch peak RSS with a single before/after read, so we poll)
class PeakMemoryMonitor:
    def __init__(self, interval=0.02):
        self.process = psutil.Process(os.getpid())
        self.interval = interval
        self._peak_rss = 0
        self._stop_flag = False
        self._thread = None

    def _poll(self):
        while not self._stop_flag:
            rss = self.process.memory_info().rss
            if rss > self._peak_rss:
                self._peak_rss = rss
            time.sleep(self.interval)

    def start(self):
        self._peak_rss = self.process.memory_info().rss
        self._stop_flag = False
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag = True
        self._thread.join()
        return self._peak_rss  # bytes


# Transcription
def transcribe_from_dataset(dataset_sample, whisper_model, processor, max_new_tokens=128):
    input_features = processor.feature_extractor(
        dataset_sample["array"],
        sampling_rate=dataset_sample["sampling_rate"],
        return_tensors="pt"
    ).input_features
    # no .to(model.device) — ORTModel handles device/session placement internally

    predicted_ids = whisper_model.generate(
        input_features,
        max_new_tokens=max_new_tokens,
        task="transcribe",
        forced_decoder_ids=None
    )
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0].strip()


# Main evaluation loop
def run_eval(
    dataset,
    whisper_model,
    processor,
    system_name="onnx_model",
    split="test",
    max_new_tokens=128,
    normalize_for_wer_calc=True,
    limit=None,
    output_csv=None,
):
    process = psutil.Process(os.getpid())
    n = len(dataset[split]) if limit is None else min(limit, len(dataset[split]))
    print(f"[{system_name}] number of test examples to process: {n}")

    predictions, references = [], []
    per_utt_rows = []
    failed_indices = []

    total_infer_time = 0.0
    total_audio_duration = 0.0

    # warm-up run (excluded from timing) — avoids first-call overhead skewing RTF
    if n > 0:
        warm_sample = dataset[split][0]["audio"]
        try:
            _ = transcribe_from_dataset(warm_sample, whisper_model, processor, max_new_tokens)
        except Exception as e:
            print(f"[{system_name}] WARNING: warm-up run failed (continuing anyway): {e}")

    process.cpu_percent(interval=None)  # prime cpu_percent baseline

    for idx in range(n):
        print(f"[{system_name}] inference on example: {idx}")
        sample = dataset[split][idx]["audio"]
        ref_text = dataset[split][idx]["transcript"]

        audio_duration = len(sample["array"]) / sample["sampling_rate"]

        mem_monitor = PeakMemoryMonitor()
        mem_monitor.start()

        t0 = time.perf_counter()
        try:
            pred_text = transcribe_from_dataset(sample, whisper_model, processor, max_new_tokens)
        except Exception as e:
            mem_monitor.stop()
            print(f"[{system_name}] WARNING: example {idx} failed, skipping. Error: {e}")
            failed_indices.append(idx)
            continue
        t1 = time.perf_counter()

        peak_rss_bytes = mem_monitor.stop()
        cpu_pct = process.cpu_percent(interval=None)  # since last call

        infer_time = t1 - t0
        rtf = infer_time / audio_duration if audio_duration > 0 else float("nan")

        total_infer_time += infer_time
        total_audio_duration += audio_duration

        predictions.append(pred_text)
        references.append(ref_text)

        per_utt_rows.append({
            "idx": idx,
            "audio_duration_s": round(audio_duration, 4),
            "infer_time_s": round(infer_time, 4),
            "rtf": round(rtf, 4),
            "cpu_percent": round(cpu_pct, 2),
            "peak_rss_mb": round(peak_rss_bytes / (1024 ** 2), 2),
            "reference": ref_text,
            "prediction": pred_text,
        })

    if failed_indices:
        print(f"\n[{system_name}] {len(failed_indices)} example(s) failed and were skipped: {failed_indices}\n")

    n_scored = len(predictions)

    # ---------------- WER / CER ----------------
    if normalize_for_wer_calc:
        norm_refs = [normalize_cs(r) for r in references]
        norm_preds = [normalize_cs(p) for p in predictions]
    else:
        norm_refs = references
        norm_preds = predictions

    wer = jiwer.wer(norm_refs, norm_preds) if n_scored > 0 else float("nan")
    cer = jiwer.cer(norm_refs, norm_preds) if n_scored > 0 else float("nan")

    mean_rtf = total_infer_time / total_audio_duration if total_audio_duration > 0 else float("nan")
    mean_infer_time = total_infer_time / n_scored if n_scored > 0 else float("nan")
    mean_cpu = sum(r["cpu_percent"] for r in per_utt_rows) / n_scored if n_scored > 0 else float("nan")
    peak_rss_overall_mb = max((r["peak_rss_mb"] for r in per_utt_rows), default=float("nan"))

    summary = {
        "system": system_name,
        "n_requested": n,
        "n_scored": n_scored,
        "n_failed": len(failed_indices),
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "mean_infer_time_s": round(mean_infer_time, 4),
        "mean_rtf": round(mean_rtf, 4),
        "mean_cpu_percent": round(mean_cpu, 2),
        "peak_rss_mb": round(peak_rss_overall_mb, 2),
    }

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if output_csv:
        if per_utt_rows:
            # encoding="utf-8" — Twi characters (ɔ, ɛ) in reference/prediction
            # text aren't representable in Windows' default cp1252 codepage,
            # so writing without an explicit encoding raises UnicodeEncodeError.
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(per_utt_rows[0].keys()))
                writer.writeheader()
                writer.writerows(per_utt_rows)
            print(f"\nPer-utterance results written to {output_csv}")
        else:
            print(f"\n[{system_name}] No successful utterances — skipping CSV write.")

    return summary, per_utt_rows


# Example usage
if __name__ == "__main__":
    from transformers import WhisperProcessor
    from datasets import load_dataset

    import os

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    QUANT_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "whisper_tiny_onnx_int8"))
    assert os.path.isdir(QUANT_PATH), f"Path does not exist: {QUANT_PATH}"
    MODEL_ID = "kennethdot/kasanoma_whisper"  # or any other Hugging Face model ID

    # Read the HF token from an environment variable instead of hardcoding it.
    # Set it once per shell session with:
    #   PowerShell:  $env:HF_TOKEN = "hf_..."
    #   cmd.exe:     set HF_TOKEN=hf_...
    HF_TOKEN = os.environ.get("HF_TOKEN")

    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    quantized_model = ORTModelForSpeechSeq2Seq.from_pretrained(
        QUANT_PATH,
        encoder_file_name="encoder_model_quantized.onnx",
        decoder_file_name="decoder_model_quantized.onnx",
        decoder_with_past_file_name="decoder_with_past_model_quantized.onnx",
    )

    fresh_gen_config = GenerationConfig.from_pretrained(MODEL_ID, token=HF_TOKEN)
    quantized_model.generation_config = fresh_gen_config
    quantized_model.generation_config.save_pretrained(QUANT_PATH)

    dataset = load_dataset("Kennethdot/Ghana_English-Twi_Code-switching_Speech")

    summary, rows = run_eval(
        dataset=dataset,
        whisper_model=quantized_model,
        processor=processor,
        system_name="tiny_int8_onnx",
        split="test",
        max_new_tokens=128,
        normalize_for_wer_calc=True,
        limit=200,          # or e.g. 1681 to match your fixed eval subset
        output_csv="tiny_int8_onnx_results.csv",
    )
