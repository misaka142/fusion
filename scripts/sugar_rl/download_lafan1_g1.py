"""Download Unitree G1 29-DoF LAFAN1 walking trajectories (`g1/walk*.csv`).

Source repository: `lvhaidong/LAFAN1_Retargeting_Dataset` (HuggingFace public dataset, 30 FPS CSV).
Each row has 36 columns: root pos (3, XYZ) + root quat (4, QX QY QZ QW) + 29 SDK joint angles.

Usage (Windows PowerShell):
    D:\\anaconda\\envs\\DRL_env\\python.exe scripts/sugar_rl/download_lafan1_g1.py `
        --out_dir data_lafan1_raw
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

_FUSION_ROOT = Path(__file__).resolve().parents[2]
if str(_FUSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_FUSION_ROOT))

from scripts.sugar_rl.download_omniretarget import _download_file

HF_LAFAN1_G1_BASE = "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/g1"

LAFAN1_WALK_FILES: tuple[str, ...] = (
    "walk1_subject1.csv",
    "walk1_subject2.csv",
    "walk1_subject5.csv",
    "walk2_subject1.csv",
    "walk2_subject3.csv",
    "walk2_subject4.csv",
    "walk3_subject1.csv",
    "walk3_subject2.csv",
    "walk3_subject3.csv",
    "walk3_subject4.csv",
    "walk3_subject5.csv",
    "walk4_subject1.csv",
)


def _is_valid_csv(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
    return len(first_line.split(",")) == 36


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=str, default="data_lafan1_raw", help="Output root directory (creates g1/*.csv).")
    ap.add_argument("--cache_dir", type=str, default=None, help="Optional cache directory for CSVs (e.g. Drive).")
    args = ap.parse_args()

    out_g1 = Path(args.out_dir).resolve() / "g1"
    out_g1.mkdir(parents=True, exist_ok=True)
    cache_g1 = (Path(args.cache_dir).resolve() / "lafan1_g1_walk") if args.cache_dir else None
    if cache_g1 is not None:
        cache_g1.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    cached = 0
    for fname in LAFAN1_WALK_FILES:
        dst = out_g1 / fname
        if _is_valid_csv(dst):
            cached += 1
            continue
        if cache_g1 is not None and _is_valid_csv(cache_g1 / fname):
            shutil.copy2(cache_g1 / fname, dst)
            cached += 1
            continue
        url = f"{HF_LAFAN1_G1_BASE}/{fname}"
        target_dl = (cache_g1 / fname) if cache_g1 is not None else dst
        _download_file(url, target_dl)
        if not _is_valid_csv(target_dl):
            raise RuntimeError(f"Downloaded CSV failed 36-column schema check: {target_dl}")
        if cache_g1 is not None:
            shutil.copy2(target_dl, dst)
        downloaded += 1

    print(f"[OK] LAFAN1 G1 walk CSVs ready at `{out_g1}` ({len(LAFAN1_WALK_FILES)} files, downloaded={downloaded}, cached={cached}).")


if __name__ == "__main__":
    main()
