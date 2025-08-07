# Smart captions for video

✔️  Flag `--keep-all` - rint every scene without deduplication.

✔️  Flag `--debug`    - detailed log: shows how many scenes, which fallback is used and the first words of each caption (so you can track progress).

✔️  Guaranteed minimum: at least one caption is always printed - even if all frames turn out to be identical.

## Installation:

    pip install scenedetect transformers pillow torch ffmpeg-python

Python ≥ 3.10, Torch ≥ 2.3, Transformers ≥ 4.53, PySceneDetect ≥ 0.6.6.

## Launch examples:

    python3 src/main.py movie.mp4                # automatic

    python3 src/main.py movie.mp4 0.25           # threshold 0.25

    python3 src/main.py movie.mp4 --keep-all     # without duplicate filtering

    python3 src/main.py movie.mp4 --debug        # detailed log

## Arguments:

    video_path (required)

    scene_thr  (opt) - 0.0‑1.0 or 0‑100, default=0.30 (30 %)

    max_tokens (opt) - default=40

    --model              - HF‑ID of the model caption‑pipeline (default - vit‑gpt2)

    --keep-all       - do not filter identical captions

    --debug          - print debug progress, progress bar
