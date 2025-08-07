import scenedetect, transformers, torch
print(scenedetect.__version__, transformers.__version__, torch.__version__, sep='\n')
import importlib.util, pathlib, subprocess, os, sys, types, pytest

SCRIPT = pathlib.Path("main.py").resolve()

def run_cli(args, offline=True):
    env = dict(os.environ)
    if offline:
        env["VSC_OFFLINE"] = "1"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, env=env
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_no_crash_on_dummy_video(tmp_path):
    video = tmp_path / "black.mp4"
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=black",
        "-t", "1", "-pix_fmt", "yuv420p", str(video)
    ], check=True)

    rc, out, _ = run_cli([str(video), "0.5"])
    assert rc == 0
    assert "— (Лайф)" in out


def load_module_from_path(path):
    spec = importlib.util.spec_from_file_location("mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_detect_scenes_called_once(tmp_path):
    mod = load_module_from_path(SCRIPT)

    calls = {"n": 0}
    def fake_detect(*a, **kw):
        calls["n"] += 1
        return []

    # monkey-patch
    mod.detect_scenes = fake_detect
    mod.pipeline = lambda *a, **k: (lambda img: [{"generated_text": "x"}])
    mod.get_duration_sec = lambda _: 1.0

    dummy_video = tmp_path / "d.mp4"
    dummy_video.touch()

    sys.argv = ["prog", str(dummy_video)]
    with pytest.raises(SystemExit):
        mod.main()

    assert calls["n"] <= 1
