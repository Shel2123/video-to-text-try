import datetime as dt
import pathlib
import subprocess
import torch
import argparse
from config import (
    DEFAULT_SCENE_THRESHOLD,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_ID
)
from scenedetect import ContentDetector, FrameTimecode, detect
from typing import Tuple

def _sec_to_hhmmss(sec: float) -> str:
    return str(dt.timedelta(seconds=int(sec))).rjust(8, "0")


def _grab_frame(video: str, timestamp: float, out_path: pathlib.Path) -> bool:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{timestamp:.3f}", "-i", video,
        "-frames:v", "1", str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as err:
        print(f"⚠️  ffmpeg error {err.returncode} — кадр пропущен")
        return False

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart scene captions generator (2025 edition)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Примеры:\n  python3 main.py clip.mp4\n  python3 main.py clip.mp4 0.15 --model Salesforce/blip-image-captioning-base""",
    )
    parser.add_argument("video")
    parser.add_argument("scene_thr", nargs="?", type=float, default=DEFAULT_SCENE_THRESHOLD,
                        help="порог: 0‑1 как доля или 0‑100 как %%")
    parser.add_argument("max_tokens", nargs="?", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID,
                        help="HF‑id модели image‑to‑text (pipeline)")
    parser.add_argument("--keep-all", action="store_true", help="не фильтровать одинаковые подписи")
    parser.add_argument("--debug", action="store_true", help="подробный лог и progress‑bar")
    return parser.parse_args()

# device select

def _select_device() -> str | int:
    if torch.cuda.is_available():
        print("🖥  CUDA GPU")
        return 0
    if torch.backends.mps.is_available():
        print("🖥  Apple M‑series (MPS)")
        return "mps"
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # Intel GPU
        print("🖥  Intel XPU")
        return "xpu"
    print("🖥  CPU")
    return -1

def _detect_scenes(video_path: str, threshold_raw: float, debug: bool) -> list[Tuple[float, float]]:
    # Порог принимаем либо как долю (<=1), либо как целое 0‑100
    thr_pct = int(threshold_raw * 100) if threshold_raw <= 1 else int(threshold_raw)
    detector = ContentDetector(threshold=thr_pct)
    scene_list = detect(video_path, detector, show_progress=debug)
    scenes_sec: list[Tuple[float, float]] = [
        (start.get_seconds(), end.get_seconds())  # type: ignore[attr-defined]
        for start, end in scene_list
    ]
    if debug:
        print(f"      Детектор вернул {len(scenes_sec)} сцен при threshold={thr_pct}")
    return scenes_sec

def _norm(txt: str) -> str:
        return " ".join(txt.casefold().split())
