from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import warnings

from PIL import Image
from transformers import pipeline

from utils import (
    _parse_args,
    _detect_scenes,
    _select_device,
    _sec_to_hhmmss,
    _grab_frame,
    _norm
)
from config import (
    DEFAULT_STEP_SEC,
    OFFLINE,
    FRAME_EXT
)


def main() -> None:
    args = _parse_args()
    video_path = pathlib.Path(args.video)
    if not video_path.is_file():
        print("File not found:", video_path)
        sys.exit(1)

    # ───────── scene detection ─────────
    print(f"\n🚧 Detecting scenes… (threshold = {args.scene_thr})")
    scenes = _detect_scenes(str(video_path), args.scene_thr, args.debug)

    if not scenes:
        print("   ⚠️  Detector is silent. Taking frames uniformly…")
        duration = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            text=True
        ).strip())
        scenes = [
            (t, min(t + DEFAULT_STEP_SEC, duration))
            for t in range(0, int(duration), DEFAULT_STEP_SEC)
        ]
        print(f"   Scenes (uniform): {len(scenes)}\n")
    else:
        print(f"   Scenes found: {len(scenes)}\n")

    # ───────── captioner ─────────
    if OFFLINE:
        print("🔌 OFFLINE mode: captioner stub → [stub caption]")
        captioner = lambda img: [{"generated_text": "[stub caption]"}]
    else:
        device = _select_device()
        captioner = pipeline(
            "image-to-text",
            model=args.model,
            device=device,
            max_new_tokens=args.max_tokens,
        )
        # Just in case, verify we didn't fall back to CPU without warning
        if device != -1 and hasattr(captioner, "device") and str(captioner.device).startswith("cpu"):
            print("⚠️  Model fell back to CPU — possibly insufficient VRAM/RAM.")

    # ───────── iterate scenes ─────────
    last_caption_norm: str | None = None
    printed_any = False
    extracted_any = False
    with tempfile.TemporaryDirectory() as tdir:
        tmp = pathlib.Path(tdir)
        for idx, (start, end) in enumerate(scenes, 1):
            mid_ts = (start + end) / 2
            frame_file = tmp / f"scene_{idx:04d}.{FRAME_EXT}"
            ok = _grab_frame(str(video_path), mid_ts, frame_file)
            if not ok:
                if args.debug:
                    print("      » frame skipped")
                caption_raw = "(frame unavailable)"
            else:
                extracted_any = True
                img = Image.open(frame_file).convert("RGB")
                caption_raw = captioner(img)[0]["generated_text"].strip()

            caption_norm = _norm(caption_raw)
            should_print = args.keep_all or caption_norm != last_caption_norm
            if should_print:
                print(f"{_sec_to_hhmmss(start)} — (Live) {caption_raw}")
                last_caption_norm = caption_norm
                printed_any = True

            if args.debug:
                head = caption_raw.replace("\n", " ")[:60]
                print(f"      » scene {idx} | {head}")

    if not extracted_any:
        print("❌ Failed to extract any frames.", file=sys.stderr)
        sys.exit(1)
    if not printed_any:
        print("00:00:00 — (Live) (no differences between frames)")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
