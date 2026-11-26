"""Model download manager: resumable downloads with progress, so a failed/partial download never breaks the app.
CLI backup (shows errors in the terminal):   python modelstore.py            (all models)
                                             python modelstore.py search vision translate
"""
import os
import sys
import threading
from pathlib import Path

HERE = Path(__file__).parent
LOCAL_VISION = HERE / "disease_model.pt"

MODELS = {
    "search": dict(repo="intfloat/multilingual-e5-small", name="Search model (multilingual-e5)", size_gb=0.5,
                   why="finds the right advice in any language", required=True,
                   markers=("model.safetensors", "pytorch_model.bin")),
    "vision": dict(repo=os.environ.get("AGRI_DISEASE_MODEL", "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"),
                   name="Leaf disease model (MobileNetV2)", size_gb=0.01, why="reads the leaf photo", required=True,
                   markers=("pytorch_model.bin", "model.safetensors")),
    "translate": dict(repo="facebook/nllb-200-distilled-600M", name="Translation model (NLLB-200)", size_gb=2.5,
                      why="answers in Hindi, Tamil, Telugu and 6 more languages", required=False,
                      markers=("pytorch_model.bin", "model.safetensors")),
    "speech": dict(repo=os.environ.get("AGRI_ASR", "openai/whisper-small"), name="Speech recognition (Whisper)",
                   size_gb=1.0, why="understands spoken questions", required=False,
                   markers=("model.safetensors", "pytorch_model.bin")),
}


def is_ready(key):
    m = MODELS[key]
    if key == "vision" and LOCAL_VISION.exists():
        return True
    from huggingface_hub import hf_hub_download
    for f in m["markers"]:
        try:
            hf_hub_download(m["repo"], f, local_files_only=True)
            return True
        except Exception:
            continue
    return False


def plan_files(repo):
    """Which files to download: skips ONNX/TF/Flax duplicates and, when safetensors exist, the .bin copy."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings]
    has_st = any(f.endswith(".safetensors") and "/" not in f for f, _ in files)
    skip_dirs = ("onnx/", "openvino/", "runs/", "coreml/", "tflite/")
    skip_ext = (".h5", ".msgpack", ".ot", ".tflite", ".onnx", ".mlmodel", ".gguf")
    skip_names = ("training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth")
    keep = []
    for f, s in files:
        low = f.lower()
        if low.startswith(skip_dirs) or low.endswith(skip_ext) or "tfevents" in low or low in skip_names:
            continue
        if low.endswith((".safetensors", ".bin")) and any(x in low for x in ("fp16", "int8", "quantized", "bnb")):
            continue
        if has_st and "/" not in low and low.endswith(".bin"):
            continue
        keep.append((f, s))
    return [f for f, _ in keep], sum(s for _, s in keep)


def cache_bytes(repo):
    """Bytes already on disk for this repo (including partial files) - used for the progress bar."""
    from huggingface_hub import constants
    root = Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "blobs"
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.iterdir() if p.is_file())


class Job:
    def __init__(self, key):
        self.key, self.repo = key, MODELS[key]["repo"]
        self.done, self.error, self.total = False, None, int(MODELS[key]["size_gb"] * 1e9) or 1
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            from huggingface_hub import snapshot_download
            files, total = plan_files(self.repo)
            self.total = total or self.total
            snapshot_download(self.repo, allow_patterns=files)
        except Exception as e:  # shown to the user with advice
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.done = True

    @property
    def progress(self):
        return min(cache_bytes(self.repo) / self.total, 0.99) if self.total else 0.0


def start(key):
    return Job(key)


if __name__ == "__main__":
    import time
    keys = sys.argv[1:] or list(MODELS)
    for k in keys:
        if is_ready(k):
            print(f"[ok] {MODELS[k]['name']} already downloaded")
            continue
        for attempt in range(1, 4):
            print(f"[..] downloading {MODELS[k]['name']} (attempt {attempt}/3)")
            job = start(k)
            while not job.done:
                print(f"     {job.progress:.0%}", end="\r")
                time.sleep(1)
            if job.error:
                print("     failed:", job.error)
                time.sleep(2)
            else:
                print("[ok] done" + " " * 20)
                break
