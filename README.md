# Smart captions for video

✔️  Flag `--keep-all` - rint every scene without deduplication.

✔️  Flag `--debug`    - detailed log: shows how many scenes, which fallback is used and the first words of each caption (so you can track progress).

✔️  Guaranteed minimum: at least one caption is always printed - even if all frames turn out to be identical.

## Installation:

    pip install scenedetect transformers pillow torch ffmpeg-python

## Launch examples:

    python3 main.py movie.mp4                # automatic

    python3 main.py movie.mp4 0.25           # threshold 0.25

    python3 main.py movie.mp4 --keep-all     # without duplicate filtering

    python3 main.py movie.mp4 --debug        # detailed log

## Arguments:

    video_path (required)

    scene_thr  (opt) - 0..1, default=0.30

    max_tokens (opt) - default=40

    --keep-all       - do not filter identical captions

    --debug          - print debug progress
