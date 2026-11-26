"""Leaf-disease image classifier (deep learning, runs locally).

Default  : pretrained 38-class PlantVillage MobileNetV2 from Hugging Face (downloads once, no training needed).
Optional : train your own model on your own photos:
               python prepare_dataset.py raw_data      # split class folders into data/train and data/val
               python disease_model.py train           # creates disease_model.pt + disease_labels.json
           If those two files exist they are used instead of the pretrained model.
Test     : python disease_model.py predict leaf.jpg
"""
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).parent
WEIGHTS, LABELS = HERE / "disease_model.pt", HERE / "disease_labels.json"
HF_MODEL = os.environ.get("AGRI_DISEASE_MODEL", "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG = 224
_state = {}


def _tfms():
    from torchvision import transforms
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG, scale=(0.5, 1.0)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25), transforms.ColorJitter(0.4, 0.4, 0.4, 0.05),
        transforms.ToTensor(), norm])
    val_tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(), norm])
    return train_tf, val_tf


def _build(num_classes):
    import torch.nn as nn
    from torchvision import models
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m


def train(epochs=10, data="data"):
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets
    train_tf, val_tf = _tfms()
    tr = datasets.ImageFolder(f"{data}/train", train_tf)
    va = datasets.ImageFolder(f"{data}/val", val_tf)
    workers = 0 if os.name == "nt" else 2
    tl = DataLoader(tr, batch_size=32, shuffle=True, num_workers=workers)
    vl = DataLoader(va, batch_size=64, num_workers=workers)
    model = _build(len(tr.classes)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                correct += (model(x.to(DEVICE)).argmax(1).cpu() == y).sum().item()
                total += len(y)
        acc = correct / total
        print(f"epoch {ep + 1}/{epochs}  val_acc={acc:.3f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), WEIGHTS)
            json.dump(tr.classes, open(LABELS, "w"))
    print("best val acc:", best)


def _load():
    if "fn" in _state:
        return
    if WEIGHTS.exists() and LABELS.exists():
        classes = json.load(open(LABELS))
        model = _build(len(classes))
        model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
        model.to(DEVICE).eval()
        _, val_tf = _tfms()

        def fn(img):
            return model(val_tf(img).unsqueeze(0).to(DEVICE))
        _state.update(fn=fn, labels=classes, source="your trained model")
    else:
        from transformers import AutoImageProcessor, AutoModelForImageClassification, MobileNetV2ImageProcessor
        try:
            proc = AutoImageProcessor.from_pretrained(HF_MODEL)
        except Exception:
            # this repo ships no preprocessor_config.json: use MobileNetV2's own settings
            # (resize 256, centre-crop 224, normalise to [-1, 1]), the same ones the model was trained with
            proc = MobileNetV2ImageProcessor()
        model = AutoModelForImageClassification.from_pretrained(HF_MODEL).to(DEVICE).eval()

        def fn(img):
            return model(**proc(images=img, return_tensors="pt").to(DEVICE)).logits
        _state.update(fn=fn, labels=[model.config.id2label[i] for i in range(len(model.config.id2label))],
                      source="pretrained PlantVillage MobileNetV2")


def predict(image, top_k=3):
    """image: path or PIL image. Returns [(label, probability), ...] best first."""
    _load()
    img = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
    with torch.no_grad():
        probs = torch.softmax(_state["fn"](img), 1)[0]
    p, i = probs.topk(min(top_k, len(_state["labels"])))
    return [(_state["labels"][j], float(v)) for v, j in zip(p, i)]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif len(sys.argv) > 2 and sys.argv[1] == "predict":
        for label, p in predict(sys.argv[2]):
            print(f"{label}: {p:.2%}")
    else:
        print(__doc__)
