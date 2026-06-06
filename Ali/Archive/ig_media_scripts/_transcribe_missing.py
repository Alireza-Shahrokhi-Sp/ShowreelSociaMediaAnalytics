"""Transcribe the 5 reel/feed videos that are missing transcription.txt.
Run with D:\\conda_envs\\ma_env\\python.exe"""
import subprocess
from pathlib import Path

BASE = Path(__file__).parent
FIXED = BASE / "multimodal_dataset_fixed"
TARGETS = [("reel","ClJUlu4j7yD"),("feed","BMtWg1VjRYo"),("feed","BPZ0taWDwY6"),
           ("feed","By2ds26Do4M"),("feed","BZdlVVxAnwQ")]

def has_audio(mp4):
    r = subprocess.run(["ffprobe","-v","error","-select_streams","a",
        "-show_entries","stream=codec_type","-of","csv=p=0",str(mp4)],
        capture_output=True, text=True)
    return "audio" in r.stdout

from faster_whisper import WhisperModel
print("Loading Whisper large-v3-turbo...", flush=True)
model = WhisperModel("large-v3-turbo", device="cpu", compute_type="float32")

for form, sc in TARGETS:
    d = FIXED/form/sc
    mp4 = d/f"{sc}.mp4"
    out = d/"transcription.txt"
    if not mp4.exists():
        print(f"{sc}: NO MP4"); continue
    if not has_audio(mp4):
        out.write_text("", encoding="utf-8"); print(f"{sc}: no audio -> empty"); continue
    segs,_ = model.transcribe(str(mp4), beam_size=5)
    lines = [f"[{s.start:.2f}s -> {s.end:.2f}s] {s.text}" for s in segs]
    out.write_text("\n".join(lines)+("\n" if lines else ""), encoding="utf-8")
    print(f"{form}/{sc}: {len(lines)} segs", flush=True)
print("DONE")
