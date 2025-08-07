import os

DEFAULT_SCENE_THRESHOLD = 0.30  # 30 %
DEFAULT_STEP_SEC = 5     # fallback sampling step
DEFAULT_MAX_TOKENS = 40
DEFAULT_MODEL_ID = "nlpconnect/vit-gpt2-image-captioning"
FRAME_EXT = "png"
OFFLINE = os.getenv("VSC_OFFLINE") == "1"