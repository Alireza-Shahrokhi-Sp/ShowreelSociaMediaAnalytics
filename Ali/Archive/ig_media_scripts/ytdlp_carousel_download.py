"""Download 4 video-carousel posts via yt-dlp with Netscape cookie auth."""
import shutil
import subprocess
from pathlib import Path

BASE    = Path(__file__).parent
PW      = BASE / "playwright_downloads"
COOKIES = str(BASE / "instagram_cookies_netscape.txt")

TARGETS = [
    "Cu6xsKttA4T",
    "DVqnRDaiP2_",
    "BpCOgjTi6r_",
    "BocKi58Cgol",
]

for sc in TARGETS:
    dest = PW / "carousel" / sc
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)

    url      = f"https://www.instagram.com/p/{sc}/"
    out_tmpl = str(dest / f"{sc}_%(playlist_index)s.%(ext)s")

    print(f"\n=== {sc} ===")
    subprocess.run(
        ["yt-dlp",
         "--cookies", COOKIES,
         "--output", out_tmpl,
         "--no-playlist-reverse",
         "--sleep-requests", "3",
         url],
        check=False,
    )
    files = sorted(dest.iterdir())
    print(f"  -> {len(files)} file(s): {[f.name for f in files]}")

print("\nAll done.")
