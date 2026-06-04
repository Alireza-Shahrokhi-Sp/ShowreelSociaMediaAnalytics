"""Transcribe every video slide under carousel_video/ (frames already exist).
Run with the conda env that has faster_whisper:
    D:\\conda_envs\\ma_env\\python.exe _transcribe_carousel_video.py
"""
import re
import subprocess
from pathlib import Path

BASE = Path(__file__).parent
ROOT = BASE / "multimodal_dataset_fixed" / "carousel_video"

_model = None
def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        print("Loading Whisper large-v3-turbo (CPU)...", flush=True)
        _model = WhisperModel("large-v3-turbo", device="cpu", compute_type="float32")
    return _model

def has_audio(mp4):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(mp4)],
        capture_output=True, text=True)
    return "audio" in r.stdout

def main():
    total = done = 0
    for sc_dir in sorted(ROOT.iterdir()):
        if not sc_dir.is_dir():
            continue
        mp4s = {int(re.search(r"_(\d+)\.mp4$", p.name).group(1)): p
                for p in sc_dir.glob("*.mp4") if re.search(r"_(\d+)\.mp4$", p.name)}
        slide_dirs = sorted(sc_dir.glob("slide_*"))
        print(f"\n=== {sc_dir.name}  ({len(slide_dirs)} slides) ===", flush=True)
        for sd in slide_dirs:
            n = int(sd.name.split("_")[1])
            mp4 = mp4s.get(n)
            out = sd / "transcription.txt"
            total += 1
            if not mp4 or not mp4.exists():
                out.write_text("", encoding="utf-8")
                print(f"  slide {n:>2}: no mp4", flush=True)
                continue
            if not has_audio(mp4):
                out.write_text("", encoding="utf-8")
                print(f"  slide {n:>2}: no audio stream", flush=True)
                continue
            try:
                segs, _ = get_model().transcribe(str(mp4), beam_size=5)
                lines = [f"[{s.start:.2f}s -> {s.end:.2f}s] {s.text}" for s in segs]
                out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                done += 1
                preview = (lines[0][:70] + "...") if lines else "(empty)"
                print(f"  slide {n:>2}: {len(lines)} segs  {preview}", flush=True)
            except Exception as e:
                out.write_text("", encoding="utf-8")
                print(f"  slide {n:>2}: FAILED {e}", flush=True)
    print(f"\nDONE  transcribed={done}/{total}")

if __name__ == "__main__":
    main()
