"""Split a downloaded image dataset into data/train and data/val (80/20).

Run:   python prepare_dataset.py raw_data
"""
import random
import shutil
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "raw_data")
DST = Path("data")
VAL_FRACTION = 0.2
EXT = {".jpg", ".jpeg", ".png"}
random.seed(42)

if not SRC.exists():
    sys.exit(f"Folder '{SRC}' not found. Put your downloaded class folders inside it.")

total = 0
for class_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
    images = [f for f in class_dir.iterdir() if f.suffix.lower() in EXT]
    if not images:
        continue
    random.shuffle(images)
    n_val = max(1, int(len(images) * VAL_FRACTION))
    for i, f in enumerate(images):
        split = "val" if i < n_val else "train"
        out = DST / split / class_dir.name
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out / f.name)
    total += len(images)
    print(f"{class_dir.name}: {len(images)} images ({n_val} val)")

print(f"\nDone. {total} images split into {DST}/train and {DST}/val")