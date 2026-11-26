"""Measure the numbers for your results table.  Run from the AgriAssist folder.

  python evaluate.py photos  <folder>           accuracy; folder/<Class_name>/*.jpg  (e.g. Tomato___Late_blight)
  python evaluate.py reject  <folder>           how many poor photos (blurred / off-leaf) the app refuses
  python evaluate.py ids                        list knowledge-base ids to use in questions.csv
  python evaluate.py questions questions.csv    csv columns: question,expected_id,lang   (lang = en, hi, ta, te ...)
  python evaluate.py speech speech.csv          csv columns: audio_path,reference,lang
  python evaluate.py time                       response time for typed questions
"""
import csv
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from knowledge import norm

MIN_CONF = 0.60
IMG_EXT = {".jpg", ".jpeg", ".png"}


def wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)



def cmd_photos(folder):
    from disease_model import predict

    # Verified mapping: dataset folder name -> MobileNetV2 model label
    class_mapping = {
        "Pepper__bell___Bacterial_spot": "Bell Pepper with Bacterial Spot",
        "Pepper__bell___healthy": "Healthy Bell Pepper Plant",
        "Potato___Early_blight": "Potato with Early Blight",
        "Potato___healthy": "Healthy Potato Plant",
        "Potato___Late_blight": "Potato with Late Blight",
        "Tomato__Target_Spot": "Tomato with Target Spot",
        "Tomato__Tomato_mosaic_virus": "Tomato Mosaic Virus",
        "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato Yellow Leaf Curl Virus",
        "Tomato_Bacterial_spot": "Tomato with Bacterial Spot",
        "Tomato_Early_blight": "Tomato with Early Blight",
        "Tomato_healthy": "Healthy Tomato Plant",
        "Tomato_Late_blight": "Tomato with Late Blight",
        "Tomato_Leaf_Mold": "Tomato with Leaf Mold",
        "Tomato_Septoria_leaf_spot": "Tomato with Septoria Leaf Spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite":
            "Tomato with Spider Mites or Two-spotted Spider Mite",
    }

    ok = tot = acc_ok = acc_n = 0

    for cls in sorted(p for p in Path(folder).iterdir() if p.is_dir()):
        if cls.name not in class_mapping:
            print(f"Skipping unmapped folder: {cls.name}")
            continue

        expected_label = class_mapping[cls.name]

        for f in (x for x in cls.iterdir()
                  if x.is_file() and x.suffix.lower() in IMG_EXT):

            label, conf = predict(f, top_k=1)[0]

            tot += 1
            hit = norm(label) == norm(expected_label)
            ok += hit

            if conf >= MIN_CONF:
                acc_n += 1
                acc_ok += hit

    if tot == 0:
        print("No mapped images found for evaluation.")
        return

    print(f"images evaluated: {tot}")
    print(f"top-1 accuracy: {ok / tot:.2%} ({ok}/{tot})")
    print(
        f"photos accepted (confidence >= {MIN_CONF:.0%}): "
        f"{acc_n}/{tot}"
    )

    if acc_n:
        print(f"accuracy on accepted photos: {acc_ok / acc_n:.2%}")
    else:
        print("accuracy on accepted photos: N/A (no accepted photos)")



def cmd_reject(folder):
    from disease_model import predict
    files = [f for f in Path(folder).rglob("*") if f.suffix.lower() in IMG_EXT]
    rej = sum(predict(f, top_k=1)[0][1] < MIN_CONF for f in files)
    print(f"poor photos rejected: {rej} of {len(files)}")


def cmd_ids():
    from knowledge import load_kb
    for e in load_kb():
        print(f"{e['id']:55s} {e['name']}")


def cmd_questions(path, min_score=0.75):
    from engine import AgriBot
    bot, stats = AgriBot(), defaultdict(lambda: [0, 0])
    for row in csv.DictReader(open(path, encoding="utf-8")):
        lang = row.get("lang", "en")
        q = row["question"]
        hits = bot.retrieve(q, bot.translate(q, bot.detect_lang(q), "en"))
        good = hits[0][0] >= min_score and hits[0][1]["id"] == row["expected_id"]
        stats[lang][0] += good
        stats[lang][1] += 1
        if not good:
            print(f"  miss [{lang}] {q!r} -> {hits[0][1]['id']} ({hits[0][0]:.2f}), expected {row['expected_id']}")
    for lang, (a, n) in sorted(stats.items()):
        print(f"{lang}: {a}/{n} = {a / n:.1%}")
    a, n = sum(v[0] for v in stats.values()), sum(v[1] for v in stats.values())
    print(f"overall: {a}/{n} = {a / n:.1%}")


def cmd_speech(path):
    from engine import AgriBot
    bot, scores = AgriBot(), defaultdict(list)
    for row in csv.DictReader(open(path, encoding="utf-8")):
        hyp = bot.transcribe(Path(row["audio_path"]).read_bytes(), row["lang"])
        scores[row["lang"]].append(wer(row["reference"], hyp))
    for lang, v in sorted(scores.items()):
        print(f"{lang}: word error rate {statistics.mean(v):.1%} over {len(v)} clips")


def cmd_time():
    from engine import AgriBot
    bot = AgriBot()
    tests = [("How do I control late blight in tomato?", "en"), ("टमाटर में लेट ब्लाइट को कैसे रोकें?", "hi"),
             ("தக்காளியில் தாமதமான கருகல் நோயை எப்படிக் கட்டுப்படுத்துவது?", "ta")]
    for q, lang in tests:
        t = []
        for _ in range(3):
            s = time.time()
            bot.answer(q, lang)
            t.append(time.time() - s)
        print(f"{lang}: first {t[0]:.1f}s (includes loading/translating), repeat {statistics.mean(t[1:]):.1f}s")


if __name__ == "__main__":
    cmds = {"photos": cmd_photos, "reject": cmd_reject, "ids": cmd_ids, "questions": cmd_questions,
            "speech": cmd_speech, "time": cmd_time}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
    else:
        cmds[sys.argv[1]](*sys.argv[2:])
