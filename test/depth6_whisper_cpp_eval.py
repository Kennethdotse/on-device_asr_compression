"""
whisper.cpp evaluation harness.
Measures WER, CER, RTF, CPU utilization, and peak RAM per utterance and in aggregate.
Calls the compiled whisper-cli.exe (whisper.cpp) via subprocess.
"""

import os
import re
import csv
import json
import time
import wave
import psutil
import subprocess
import threading
import numpy as np
import soundfile as sf

import jiwer


# Text normalization (same as ONNX eval — keep identical across runtimes)
def normalize_cs(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9ɔɛ\s']", "", text)
    text = re.sub(r"'", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Peak-RAM sampler for a CHILD process (whisper-cli.exe), not this script
class ChildPeakMemoryMonitor:
    def __init__(self, pid, interval=0.02):
        self.pid = pid
        self.interval = interval
        self._peak_rss = 0
        self._stop_flag = False
        self._thread = None

    def _poll(self):
        try:
            proc = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return
        while not self._stop_flag:
            try:
                rss = proc.memory_info().rss
                if rss > self._peak_rss:
                    self._peak_rss = rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(self.interval)

    def start(self):
        self._stop_flag = False
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag = True
        self._thread.join()
        return self._peak_rss  # bytes


# Write a dataset sample (numpy array) to a temp 16kHz mono WAV file,
# since whisper-cli.exe reads audio files, not in-memory arrays
def write_temp_wav(dataset_sample, out_path, target_sr=16000):
    audio = dataset_sample["array"]
    sr = dataset_sample["sampling_rate"]

    if sr != target_sr:
        # simple resample fallback; swap for librosa.resample if available/needed
        import librosa
        audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    sf.write(out_path, audio, sr, subtype="PCM_16")
    return out_path


# Run whisper-cli.exe on a single WAV file, parse transcript + timing
def transcribe_with_whisper_cpp(
    whisper_cli_path,
    model_path,
    wav_path,
    language=None,
    extra_args=None,
):
    out_prefix = wav_path.rsplit(".", 1)[0]  # whisper-cli writes <prefix>.txt with -otxt

    cmd = [
        whisper_cli_path,
        "-m", model_path,
        "-f", wav_path,
        "-l", language if language is not None else "auto",
        "-otxt",
        "-of", out_prefix,
        "-nt",          # no timestamps in output text
        "-np",          # no progress printing (keeps stdout clean)
    ]
    if extra_args:
        cmd.extend(extra_args)

    t0 = time.perf_counter()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",   # don't crash the reader thread on undecodable bytes
    )

    mem_monitor = ChildPeakMemoryMonitor(proc.pid)
    mem_monitor.start()

    stdout, stderr = proc.communicate()
    t1 = time.perf_counter()

    peak_rss_bytes = mem_monitor.stop()
    infer_time = t1 - t0

    if proc.returncode != 0:
        raise RuntimeError(f"whisper-cli.exe failed:\n{stderr}")

    txt_path = out_prefix + ".txt"
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Expected output not found: {txt_path}\nstderr:\n{stderr}")

    # Read raw bytes and decode defensively — whisper-cli's output file encoding
    # isn't guaranteed to be UTF-8 on Windows, and Twi text (ɔ, ɛ, etc.) makes
    # mis-decodes more likely than with plain ASCII.
    with open(txt_path, "rb") as f:
        raw = f.read()

    try:
        transcription = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        transcription = raw.decode("cp1252", errors="replace").strip()

    # cleanup temp output file
    try:
        os.remove(txt_path)
    except OSError:
        pass

    return transcription, infer_time, peak_rss_bytes


# Main evaluation loop
def run_eval_whisper_cpp(
    dataset,
    whisper_cli_path,
    model_path,
    system_name="whispercpp_model",
    split="test",
    language=None,
    normalize_for_wer_calc=True,
    limit=None,
    tmp_dir="tmp_wavs",
    output_csv=None,
):
    os.makedirs(tmp_dir, exist_ok=True)

    n = len(dataset[split]) if limit is None else min(limit, len(dataset[split]))
    print(f"[{system_name}] number of test examples to process: {n}")

    predictions, references = [], []
    per_utt_rows = []
    failed_indices = []
    total_infer_time = 0.0
    total_audio_duration = 0.0

    # warm-up run — excluded from timing
    if n > 0:
        warm_sample = dataset[split][0]["audio"]
        warm_wav_path = os.path.join(tmp_dir, "warmup.wav")
        write_temp_wav(warm_sample, warm_wav_path)

        try:
            _ = transcribe_with_whisper_cpp(
                whisper_cli_path,
                model_path,
                warm_wav_path,
                language,
            )
        except Exception as e:
            print(f"[{system_name}] WARNING: warm-up run failed (continuing anyway): {e}")

        try:
            os.remove(warm_wav_path)
        except OSError:
            pass

    for idx in range(n):
        print(f"[{system_name}] inference on example: {idx}")
        sample = dataset[split][idx]["audio"]
        ref_text = dataset[split][idx]["transcript"]

        audio_duration = len(sample["array"]) / sample["sampling_rate"]

        wav_path = os.path.join(tmp_dir, f"utt_{idx}.wav")
        write_temp_wav(sample, wav_path)

        try:
            pred_text, infer_time, peak_rss_bytes = transcribe_with_whisper_cpp(
                whisper_cli_path, model_path, wav_path, language
            )
        except Exception as e:
            print(f"[{system_name}] WARNING: example {idx} failed, skipping. Error: {e}")
            failed_indices.append(idx)
            try:
                os.remove(wav_path)
            except OSError:
                pass
            continue

        try:
            os.remove(wav_path)
        except OSError:
            pass

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
        "peak_rss_mb": round(peak_rss_overall_mb, 2),
    }

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if output_csv:
        if per_utt_rows:
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
    from datasets import load_dataset

    WHISPER_CLI = os.path.abspath("quant/whisper.cpp/build/bin/Release/whisper-cli.exe")
    MODEL_PATH = os.path.abspath("quant/model_export_depth_6/ggml-model_q_8.bin")

    assert os.path.exists(WHISPER_CLI), f"whisper-cli.exe not found at {WHISPER_CLI}"
    assert os.path.exists(MODEL_PATH), f"model not found at {MODEL_PATH}"

    dataset = load_dataset("Kennethdot/Ghana_English-Twi_Code-switching_Speech")

    summary, rows = run_eval_whisper_cpp(
        dataset=dataset,
        whisper_cli_path=WHISPER_CLI,
        model_path=MODEL_PATH,
        system_name="depth6_int8_whispercpp",
        split="test",
        language=None,
        normalize_for_wer_calc=True,
        limit=200,  # or 1681 to match your fixed eval subset
        output_csv="depth6_int8_whispercpp_results.csv",
    )
