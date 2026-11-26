"""Knowledge base loading + mapping a CNN class label to its disease advice (pure Python, no ML deps)."""
import difflib
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
SAFETY_EN = ("Always follow the product label and your local agriculture officer's advice "
             "before spraying any chemical, and wear gloves and a mask.")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def pretty(label):
    """'Tomato___Early_blight' -> ('Tomato', 'Early blight')"""
    crop, _, disease = label.partition("___")
    clean = lambda t: re.sub(r"\s+", " ", t.replace("_", " ")).strip()
    return clean(crop), clean(disease)


def _advice_text(e):
    return (f"{e['name']} ({e['crop']}). Symptoms: {e['symptoms']} "
            f"What to do: {e['treatment']} How to prevent it: {e['prevention']}")


def load_kb():
    """General Q&A entries + one entry per leaf-disease class. Each entry has a 'text' used for search."""
    general = json.load(open(HERE / "knowledge_base.json", encoding="utf-8"))
    diseases = json.load(open(HERE / "disease_kb.json", encoding="utf-8"))
    for e in diseases:
        e["text"] = _advice_text(e)
    kb = general + diseases
    for e in kb:
        e.setdefault("name", e["id"].replace("disease-", "").replace("-", " ").title())
        e["search"] = f"{e['name']}. {e['text']} {e.get('keywords', '')}"
        e["tokens"] = tokens(f"{e['name']} {e['crop']} {e['topic']} {e.get('keywords', '')}")
    return kb


def find_disease(label, kb):
    """Return (entry_or_None, is_healthy). Tolerant to different spellings of PlantVillage class names."""
    n = norm(label)
    crop, _ = pretty(label)
    if "healthy" in n:
        return {"name": f"Healthy {crop} leaf", "crop": crop, "source": "-",
                "text": (f"The {crop} leaf looks healthy and no disease was found. Keep checking your field every week. "
                         "Water and fertilise according to a soil test, remove weeds and fallen leaves, "
                         "and contact your KVK if you see spots, curling or yellowing.")}, True
    table = {norm(e["class"]): e for e in kb if e.get("class")}
    if n in table:
        return table[n], False
    close = difflib.get_close_matches(n, table, n=1, cutoff=0.85)
    return (table[close[0]], False) if close else (None, False)


_STOP = {"what", "when", "where", "which", "how", "does", "with", "from", "that", "this", "have", "your", "about",
         "should", "would", "could", "need", "want", "tell", "give", "show", "plant", "plants", "farm", "farming",
         "can", "use", "the", "and", "for", "are", "you", "please", "help", "leaf", "leaves", "disease",
         "diseases", "problem", "problems"}


def tokens(text):
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOP}


def lexical_boost(query_tokens, entry_tokens):
    """+0.03 per query word that (almost) matches a keyword of the entry, max +0.06. Helps short / misspelt questions."""
    hit = sum(1 for q in query_tokens if q in entry_tokens or difflib.get_close_matches(q, entry_tokens, 1, 0.82))
    return 0.03 * min(hit, 2)
