from __future__ import annotations
import sys, subprocess, tempfile, pathlib, warnings, datetime as dt, argparse
from typing import List, Tuple
from PIL import Image
from transformers import pipeline
import torch, os

DEFAULT_SCENE_THRESHOLD = 0.30
DEFAULT_MIN_THRESHOLD = 0.15
DEFAULT_STEP_SEC = 5
DEFAULT_MAX_TOKENS = 40
MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
OFFLINE = os.getenv("VSC_OFFLINE") == "1"
FRAME_EXT = "png"

try:
    import scenedetect
    HAVE_OPEN_VIDEO = hasattr(scenedetect, "open_video")
    from scenedetect.detectors import ContentDetector
except ImportError:
    print("❌ PySceneDetect not installed.  pip install scenedetect[opencv]", file=sys.stderr)
    sys.exit(1)


def get_device():
    if torch.cuda.is_available():
        print("🖥  CUDA GPU")
        return 0
    if torch.backends.mps.is_available():
        print("🖥  Apple M-series (MPS)")
        return "mps"
    print("🖥  CPU")
    return -1       


def get_duration_sec(video: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video],
        capture_output=True, text=True, check=True
    )
    return float(result.stdout.strip())


def detect_scenes(video_path: str, threshold: float, debug: bool = False) -> List[Tuple[float, float]]:
    scenes: List[Tuple[float, float]] = []
    thr_pct = threshold * 100 if threshold <= 1 else threshold
    if HAVE_OPEN_VIDEO:
        video = scenedetect.open_video(video_path)
        sm = scenedetect.SceneManager()
        sm.add_detector(ContentDetector(threshold=thr_pct))
        sm.detect_scenes(video)
        scene_list = sm.get_scene_list()
        scenes = [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]
    else:
        from scenedetect import VideoManager, SceneManager as SM
        vm = VideoManager([video_path])
        sm = SM()
        sm.add_detector(ContentDetector(threshold=thr_pct))
        vm.set_downscale_factor(1)
        vm.start()
        sm.detect_scenes(frame_source=vm)
        scenes = [(start.get_seconds(), end.get_seconds()) for start, end in sm.get_scene_list()]
        vm.release()

    if debug:
        print(f"      Detector returned {len(scenes)} scenes at threshold={threshold}")
    return scenes


def grab_frame(video: str, timestamp: float, out_path: pathlib.Path) -> bool:
    """Attempts to extract a frame; returns True on success, False if ffmpeg failed."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
       "-ss", f"{timestamp:.3f}", "-i", video,
       "-frames:v", "1", str(out_path)]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  ffmpeg error {e.returncode} - frame skipped")
        return False


def sec_to_hhmmss(sec: float) -> str:
    return str(dt.timedelta(seconds=int(sec))).rjust(8, "0")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart scene captions generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:\n  python3 main.py clip.mp4\n  python3 main.py clip.mp4 0.25 --keep-all""",
    )
    parser.add_argument("video")
    parser.add_argument("scene_thr", nargs="?", type=float, default=DEFAULT_SCENE_THRESHOLD)
    parser.add_argument("max_tokens", nargs="?", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--keep-all", action="store_true", help="do not filter duplicates")
    parser.add_argument("--debug", action="store_true", help="logs")
    return parser.parse_args()


def main():
    args = parse_args()
    video = pathlib.Path(args.video)
    if not video.is_file():
        print("File not found:", video)
        sys.exit(1)

    scene_thr = args.scene_thr
    max_tokens = args.max_tokens
    keep_all   = args.keep_all
    debug      = args.debug

    print(f"\n🚧 Detecting scenes… (threshold = {scene_thr})")
    scenes = detect_scenes(str(video), scene_thr, debug)

    if not scenes:
        print("   ⚠️  Detector failed. Uniform sampling…")
        duration = get_duration_sec(str(video))
        scenes = [(t, min(t + DEFAULT_STEP_SEC, duration)) for t in range(0, int(duration), DEFAULT_STEP_SEC)]
        print(f"   Scenes (uniform): {len(scenes)}\n")
    else:
        print(f"   Scenes found: {len(scenes)}\n")

    if OFFLINE:
        print("🔌 OFFLINE: using dummy captioner")
        captioner = lambda img: [{"generated_text": "[stub caption]"}]
        device = "offline"
    else:
        device = get_device()
        captioner = pipeline("image-to-text",
                             model=MODEL_NAME,
                             device=device,
                             max_new_tokens=max_tokens)
        if device != -1 and hasattr(captioner, "device") \
           and str(captioner.device).lower().startswith("cpu"):
            print("⚠️  Model fell to CPU; not enough VRAM / RAM.")
 
    last_caption = None
    def normalize(text: str) -> str:
        return " ".join(text.casefold().split())

    last_caption = None               # normalized
    first_printed = False
    any_frame_ok = False
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        for idx, (start, end) in enumerate(scenes, 1):
            mid_ts = (start + end) / 2
            frame_path = tmp / f"scene_{idx:04d}.{FRAME_EXT}"
            
            ok = grab_frame(str(video), mid_ts, frame_path)
            if not ok:
                if debug:
                    print("      » frame skipped")
                caption_raw = "(frame unavailable)"
            else:
                any_frame_ok = True 
                img = Image.open(frame_path).convert("RGB")
                caption_raw = captioner(img)[0]["generated_text"].strip()
                
            caption_norm = normalize(caption_raw)
        
            should_print = keep_all or caption_norm != last_caption
            if should_print:
                print(f"{sec_to_hhmmss(start)} - (Live) {caption_raw}")
                last_caption = caption_norm
                first_printed = True

            if debug:
                print("      » scene", idx, "|", caption_raw[:50].replace("\n", " "))

    if not any_frame_ok:
        print("❌ Failed to extract a single frame from the video.", file=sys.stderr)
        sys.exit(1)
    if not first_printed:
        print("00:00:00 - (Live) (no differences between frames)")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
