import os
import math
import argparse
from typing import List

import torch as _torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from transformers.utils.logging import set_verbosity_error

from config import (
    MAX_FRAMES_PER_SCENE,
    TILE,
    CONTACT_SHEET,
    USE_FP16,
    DEFAULT_SCENE_THRESHOLD
)
from utils import (
    to_square_canvas,
    downscale,
    make_contact_sheet,
    detect_scenes,
    hhmmss,
    grab_frame,
    sample_times_uniform
)

set_verbosity_error()

# Recomend of MPS
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class Describer:
    def __init__(self, model_id: str, force_cpu: bool = True, num_threads: int | None = None):
        # 1) force CPU
        device = "cpu"
        if num_threads:
            _torch.set_num_threads(max(1, num_threads))
        else:
            _torch.set_num_threads(max(1, (_os := os).cpu_count() - 1 or 1))

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=_torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()

        # fast processor
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.device = device

    def describe_scene(self, frames: list[Image.Image], scene_ts: str) -> str:
        if not frames:
            return f"{scene_ts} — (Live) No visual information."

        frames = [to_square_canvas(downscale(f, short=TILE), size=TILE) for f in frames]
        if CONTACT_SHEET and len(frames) > 1:
            frames = [to_square_canvas(make_contact_sheet(frames, cols=3, tile=TILE), size=TILE)]

        # short prompt for 2B model
        messages = [{
            "role": "user",
            "content": (
                [{"type": "image", "image": img} for img in frames] + [
                    {"type": "text", "text":
                        "Describe what is happening in the frame in one line in Russian, keeping the key points—e.g., which object is shown or what is taking place."
                        "Also try to preserve any labels, names, and people’s actions."
                        "Avoid line breaks."
                        "At the end, after a comma, you MUST indicate the shot size: wide, medium, or close-up."
                    }
                ]
            )
        }]

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt,
            images=frames,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        # 4) ban for bad words
        tok = self.processor.tokenizer
        bad_words = [
            "system","assistant","user","система","ассистент","помощник",
            "краткое описание","шаблон","...описание..."
        ]
        bad_ids = [tok.encode(w, add_special_tokens=False) for w in bad_words if w]

        gen_kwargs = dict(
            max_new_tokens=200,
            do_sample=True, # !!!
            temperature=0.7,
            top_p=0.9,
            use_cache=True,
            bad_words_ids=bad_ids
        )
        with _torch.inference_mode():
            out_ids = self.model.generate(**inputs, **gen_kwargs)

        # decode only new tokens
        prefix = inputs["input_ids"].shape[1]
        new_tokens = out_ids[:, prefix:]
        text = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0].strip().splitlines()[0]

        return f"{scene_ts} — (Live) {text}"

# main flow
def run(video_path: str,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        threshold: float = 30.0,
        max_frames_per_scene: int = MAX_FRAMES_PER_SCENE,
        force_cpu: bool = True) -> List[str]:

    scenes = detect_scenes(video_path, threshold=threshold)
    desc = Describer(model_id, force_cpu=force_cpu)

    lines: List[str] = []
    for sc in scenes:
        ts = hhmmss(sc.start_sec)
        times = sample_times_uniform(sc.start_sec, sc.end_sec, max_frames=max_frames_per_scene)
        frames: List[Image.Image] = [im for t in times if (im := grab_frame(video_path, t)) is not None]

        if not frames:
            line = f"{ts} — (Live) No visual informatio."
        else:
            # safety: limit for frames
            if len(frames) > max_frames_per_scene:
                step = math.ceil(len(frames) / max_frames_per_scene)
                frames = frames[::step][:max_frames_per_scene]
            line = desc.describe_scene(frames, ts)

        print(line, flush=True)   # realtime progress
        lines.append(line)

    return lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=str)
    ap.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--thr", type=float, default=DEFAULT_SCENE_THRESHOLD)
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES_PER_SCENE)
    ap.add_argument("--force-cpu", action="store_true", help="Start on CPU (stable)")
    args = ap.parse_args()

    run(args.video, model_id=args.model, threshold=args.thr,
        max_frames_per_scene=args.max_frames, force_cpu=args.force_cpu)

if __name__ == "__main__":
    main()
