"""Download and extract `omniretarget/OmniRetarget_Dataset` (`robot-object` & `largebox` model).

Zero-token standard-library streaming download from HuggingFace public dataset repository.
Supports caching `robot-object.zip` inside a shared directory (e.g. Google Drive on Colab).

Usage (Windows PowerShell):
    D:\\anaconda\\envs\\DRL_env\\python.exe scripts/sugar_rl/download_omniretarget.py `
        --out_dir data_omniretarget_raw
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import urllib.request
import zipfile

HF_BASE_URL = "https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset/resolve/main"
FILES_TO_FETCH = (
    "robot-object.zip",
    "models/largebox/largebox.obj",
    "models/largebox/largebox.urdf",
)


def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    print(f"[Download] {url} -> {dst}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        next_report = 25 * 1024 * 1024
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total > 0 and done >= next_report:
                print(f"  ... {done / (1024**2):.1f} / {total / (1024**2):.1f} MB ({100.0 * done / total:.0f}%)", flush=True)
                next_report += 25 * 1024 * 1024
    shutil.move(str(tmp), str(dst))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=str, default="data_omniretarget_raw", help="Output directory for raw dataset.")
    ap.add_argument("--cache_dir", type=str, default=None, help="Optional cache directory for `robot-object.zip`.")
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_dir).resolve() if args.cache_dir else out_root

    npz_dir = out_root / "robot-object"
    largebox_obj = out_root / "models" / "largebox" / "largebox.obj"

    # 1. Fetch model assets
    for rel in FILES_TO_FETCH[1:]:
        dst = out_root / rel
        if not dst.is_file():
            _download_file(f"{HF_BASE_URL}/{rel}", dst)

    # 2. Fetch and extract robot-object.zip if needed
    if npz_dir.is_dir() and any(npz_dir.glob("*.npz")):
        print(f"[Skip] Found existing extracted trajectories in {npz_dir}")
    else:
        zip_cached = cache_root / "robot-object.zip"
        if not zip_cached.is_file():
            _download_file(f"{HF_BASE_URL}/robot-object.zip", zip_cached)
        else:
            print(f"[Cache Hit] Reusing {zip_cached}")
        print(f"[Extract] {zip_cached} -> {out_root}", flush=True)
        with zipfile.ZipFile(zip_cached, "r") as zf:
            zf.extractall(out_root)

    n_npz = len(list(npz_dir.glob("*.npz")))
    assert largebox_obj.is_file() and n_npz > 0, (
        f"Download incomplete: largebox_obj={largebox_obj.is_file()}, npz_count={n_npz}"
    )
    print(f"[OK] OmniRetarget raw dataset ready at `{out_root}` ({n_npz} .npz clips).")


if __name__ == "__main__":
    main()
