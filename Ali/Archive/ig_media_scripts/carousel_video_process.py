"""
Promote the 4 video-carousel posts into a dedicated 'carousel_video' category
with FULL per-slide processing (frames + transcription), mirroring the reel
pipeline but one level deeper (one sub-block per slide).

Output layout:
  multimodal_dataset_fixed/carousel_video/<sc>/
      <date>_UTC.json               post-level metadata (caption, counts,
                                     tagged accounts, coauthors, music note)
      <date>_UTC_1.mp4 ... _N.mp4    the slide videos (in carousel order)
      slide_01/frames/frame_*.jpg    frames for slide 1
      slide_01/transcription.txt     transcript for slide 1
      slide_02/...                   ...

Source videos come from playwright_downloads/carousel/<sc>/*.mp4 (yt-dlp, already
complete muxed mp4s). The old multimodal_dataset_fixed/carousel/<sc>/ folders are
removed so the post lives in exactly one place.

Music NOTE: Instagram's per-slide music *metadata* (song title/artist) is not
reachable from the web (private API endpoint is gated; web embed strips it). The
music AUDIO is preserved inside each slide's mp4 audio track.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

BASE   = Path(__file__).parent
PW     = BASE / "playwright_downloads" / "carousel"
FIXED  = BASE / "multimodal_dataset_fixed"
DEST_ROOT = FIXED / "carousel_video"
DUMP   = BASE / "_api_dump"
PARQ   = BASE / "Output" / "ig_posts_multimodal_enriched.parquet"
SCENE_THRESHOLD = 0.4

SHORTCODES = ["Cu6xsKttA4T", "DVqnRDaiP2_", "BpCOgjTi6r_", "BocKi58Cgol"]

# ---- metadata sources -------------------------------------------------------
df = pd.read_parquet(PARQ)
prow = {r["shortcode"]: r for _, r in df.iterrows()
        if r["shortcode"] in SHORTCODES}


def tags_and_coauthors(sc):
    fp = DUMP / f"{sc}.json"
    if not fp.exists():
        return [], []
    node = json.loads(fp.read_text(encoding="utf-8"))
    tagged = set()
    for u in ((node.get("usertags") or {}).get("in")) or []:
        un = (u.get("user") or {}).get("username")
        if un: tagged.add(un)
    for s in node.get("carousel_media") or []:
        for u in ((s.get("usertags") or {}).get("in")) or []:
            un = (u.get("user") or {}).get("username")
            if un: tagged.add(un)
    coauth = [c.get("username") for c in (node.get("coauthor_producers") or [])
              if c.get("username")]
    return sorted(tagged), coauth


def has_audio(mp4):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(mp4)],
        capture_output=True, text=True)
    return "audio" in r.stdout


def extract_frames(video, frames_dir):
    frames_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(video), "-ss", "00:00:00",
                    "-vframes", "1", "-q:v", "2", str(frames_dir / "frame_00_first.jpg")],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["ffmpeg", "-y", "-i", str(video), "-ss", "00:00:01",
                    "-vframes", "1", "-q:v", "2", str(frames_dir / "frame_01_second.jpg")],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["ffmpeg", "-y", "-i", str(video),
                    "-vf", f"select='gt(scene,{SCENE_THRESHOLD})'", "-vsync", "vfr",
                    "-q:v", "2", str(frames_dir / "frame_%04d_scene.jpg")],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


_WH = {"m": None}
def transcribe(video, out_txt):
    if not has_audio(video):
        out_txt.write_text("", encoding="utf-8")
        return "no-audio"
    try:
        if _WH["m"] is None:
            from faster_whisper import WhisperModel
            print("    loading Whisper large-v3-turbo (first time)...")
            _WH["m"] = WhisperModel("large-v3-turbo", device="cpu", compute_type="float32")
        segments, _ = _WH["m"].transcribe(str(video), beam_size=5)
        with open(out_txt, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}\n")
        return "ok"
    except Exception as e:
        out_txt.write_text("", encoding="utf-8")
        return f"failed: {e}"


def slide_idx(p):
    try:
        return int(p.stem.rsplit("_", 1)[-1])
    except ValueError:
        return 999


def date_str(sc):
    r = prow.get(sc)
    return pd.to_datetime(r["timestamp"], utc=True).strftime("%Y-%m-%d_%H-%M-%S_UTC")


def meta_dict(sc, n_slides):
    r = prow.get(sc)
    tagged, coauth = tags_and_coauthors(sc)
    permalink = f"https://www.instagram.com/p/{sc}/"
    d = {
        "id": sc, "shortcode": sc, "webpage_url": permalink,
        "source": "yt-dlp+playwright", "content_form": "carousel_video",
        "n_slides": n_slides,
        "tagged_usernames": tagged, "n_tagged": len(tagged),
        "coauthors": coauth, "n_coauthors": len(coauth),
        "music_note": ("music audio is embedded in each slide's mp4 audio track; "
                       "per-slide song title/artist not web-reachable"),
    }
    if r is not None:
        for k in ("media_id", "caption", "media_type", "timestamp",
                  "like_count", "comments_count"):
            v = r.get(k)
            if v is not None and pd.notna(v):
                d[k] = v if isinstance(v, (int, float, str, bool)) else str(v)
    return d


def main():
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    for sc in SHORTCODES:
        src = PW / sc
        mp4s = sorted(src.glob("*.mp4"), key=slide_idx) if src.exists() else []
        if not mp4s:
            print(f"{sc}: no source mp4s, skip")
            continue

        dest = DEST_ROOT / sc
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)
        ds = date_str(sc)

        print(f"\n=== {sc}  ({len(mp4s)} slides) -> carousel_video/ ===")
        for seq, src_mp4 in enumerate(mp4s, 1):
            dst_mp4 = dest / f"{ds}_{seq}.mp4"
            shutil.copy(src_mp4, dst_mp4)
            slide_dir = dest / f"slide_{seq:02d}"
            extract_frames(dst_mp4, slide_dir / "frames")
            status = transcribe(dst_mp4, slide_dir / "transcription.txt")
            nframes = len(list((slide_dir / "frames").glob("*.jpg")))
            print(f"  slide {seq:>2}: {nframes} frames, transcript {status}")

        (dest / f"{ds}.json").write_text(
            json.dumps(meta_dict(sc, len(mp4s)), ensure_ascii=False, indent=2),
            encoding="utf-8")

        # Remove the old image-carousel copy so the post lives only in carousel_video/
        old = FIXED / "carousel" / sc
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
            print(f"  removed old carousel/{sc}/")

    print("\nDONE")


if __name__ == "__main__":
    main()
