"""AgriAssist engine - local deep-learning pipeline (no API).

  Speech -> text : Whisper                 Text <-> English : NLLB-200 (one model, loaded once, cached results)
  Search         : multilingual-e5         Answer writer    : Qwen2.5-1.5B (optional)
  Text -> speech : Meta MMS-TTS
Every optional model fails soft: if translation / speech models are missing the portal still answers (in English).
"""
import io
import json
import os
import re
import unicodedata

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

import modelstore
from knowledge import find_disease, lexical_boost, load_kb, pretty, tokens

LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NLLB = {"en": "eng_Latn", "hi": "hin_Deva", "ta": "tam_Taml", "te": "tel_Telu",
        "kn": "kan_Knda", "ml": "mal_Mlym", "mr": "mar_Deva", "bn": "ben_Beng", "ur": "urd_Arab"}
MMS_TTS = {"en": "eng", "hi": "hin", "ta": "tam", "te": "tel", "kn": "kan", "ml": "mal", "mr": "mar", "bn": "ben"}
SCRIPT_LANG = {"TAMIL": "ta", "TELUGU": "te", "KANNADA": "kn", "MALAYALAM": "ml", "BENGALI": "bn", "ARABIC": "ur"}
SECTIONS = ("symptoms", "treatment", "prevention")

SYSTEM_PROMPT = """You are an agricultural advisor for farmers.
Use ONLY the context passages given. Use simple, short sentences and a short numbered list of steps.
If the context does not answer the question, say you are not sure and advise contacting the local KVK or agriculture officer.
Never invent pesticide names or doses. For chemicals say: follow the label and local recommendation, and use safety gear."""


class AgriBot:
    def __init__(self):
        self.kb = load_kb()
        self.embedder = SentenceTransformer(modelstore.MODELS["search"]["repo"])
        self.emb = self.embedder.encode(["passage: " + d["search"] for d in self.kb], normalize_embeddings=True)
        self._nmt_tok = self._nmt = self._llm = self._asr = None
        self.nmt_error = None
        self._tts, self._cache = {}, {}

    # ---------- language detection (script first, so even 3-word Indic questions work) ----------
    @staticmethod
    def detect_lang(text):
        counts = {}
        for ch in text:
            if ch.isalpha():
                script = unicodedata.name(ch, "").split(" ")[0]
                counts[script] = counts.get(script, 0) + 1
        if not counts:
            return "en"
        script = max(counts, key=counts.get)
        if script in SCRIPT_LANG:
            return SCRIPT_LANG[script]
        if script == "DEVANAGARI":  # Hindi vs Marathi share a script
            try:
                from langdetect import detect
                return "mr" if detect(text) == "mr" else "hi"
            except Exception:
                return "hi"
        return "en"

    # ---------- translation (soft-fails to the original text) ----------
    def can_translate(self):
        return self.nmt_error is None and modelstore.is_ready("translate")

    def _load_nmt(self):
        if self._nmt is not None:
            return True
        if not self.can_translate():
            return False
        try:
            repo = modelstore.MODELS["translate"]["repo"]
            self._nmt_tok = AutoTokenizer.from_pretrained(repo)
            self._nmt = AutoModelForSeq2SeqLM.from_pretrained(repo).to(DEVICE).eval()
            return True
        except Exception as e:
            self.nmt_error = f"{type(e).__name__}: {e}"
            return False

    def translate(self, text, src, tgt):
        if src == tgt or not text or not text.strip():
            return text
        key = (text, src, tgt)
        if key in self._cache:
            return self._cache[key]
        if not self._load_nmt():
            return text
        self._nmt_tok.src_lang = NLLB[src]
        bos = self._nmt_tok.convert_tokens_to_ids(NLLB[tgt])
        out_lines = []
        for line in text.split("\n"):
            parts = [p for p in re.split(r"(?<=[.!?।])\s+", line) if p.strip()]
            if not parts:
                out_lines.append(line)
                continue
            batch = self._nmt_tok(parts, return_tensors="pt", padding=True, truncation=True,
                                  max_length=256).to(DEVICE)
            with torch.no_grad():
                gen = self._nmt.generate(**batch, forced_bos_token_id=bos, num_beams=3, max_new_tokens=220)
            out_lines.append(" ".join(self._nmt_tok.batch_decode(gen, skip_special_tokens=True)))
        self._cache[key] = "\n".join(out_lines)
        return self._cache[key]

    # ---------- text questions ----------
    def retrieve(self, question, q_en=None, k=3):
        queries = ["query: " + question] + (["query: " + q_en] if q_en and q_en != question else [])
        q = self.embedder.encode(queries, normalize_embeddings=True)
        scores = (self.emb @ q.T).max(axis=1)
        qtok = tokens(question + " " + (q_en or ""))
        scores = scores + np.array([lexical_boost(qtok, d["tokens"]) for d in self.kb])
        idx = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.kb[i]) for i in idx]

    def _llm_answer(self, q_en, passages):
        if self._llm is None:
            self._llm = pipeline("text-generation", model=LLM_MODEL, device_map="auto", torch_dtype="auto")
        context = "\n\n".join(p["text"] for p in passages)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q_en}"}]
        return self._llm(msgs, max_new_tokens=350, do_sample=False)[0]["generated_text"][-1]["content"].strip()

    def _item(self, entry, lang):
        """One advice card, translated: {'title','sections'} for diseases, {'text'} for general advice."""
        tr = lambda s: self.translate(s, "en", lang)
        if entry.get("symptoms"):
            return {"title": tr(entry["name"]), "sections": {k: tr(entry[k]) for k in SECTIONS},
                    "source": entry["source"]}
        return {"text": tr(entry["text"]), "source": entry["source"]}

    def answer(self, question, lang="en", use_llm=False, min_score=0.75, context=""):
        """context = English disease name from an earlier leaf photo, used only for vague follow-ups."""
        src = self.detect_lang(question)
        q_en = self.translate(question, src, "en")
        hits = self.retrieve(question, q_en)
        if hits[0][0] < min_score and context:
            alt = self.retrieve(question, f"{q_en} {context}")
            if alt[0][0] > hits[0][0]:
                hits = alt
        good = [(s, d) for s, d in hits if s >= min_score]
        if not good:
            return {"items": [], "unsure": True, "top_score": hits[0][0],
                    "suggestions": [d["name"] for s, d in hits if s >= 0.60][:3]}
        items = []
        if use_llm:
            try:
                items = [{"text": self.translate(self._llm_answer(q_en, [d for _, d in good]), "en", lang),
                          "source": good[0][1]["source"]}]
            except Exception:
                items = []
        if not items:
            chosen = [good[0]] + [g for g in good[1:2] if g[0] >= good[0][0] - 0.02]
            items = [self._item(d, lang) for _, d in chosen]
        return {"items": items, "unsure": False, "top_score": hits[0][0]}

    # ---------- leaf photo ----------
    def diagnose(self, label, lang="en"):
        entry, healthy = find_disease(label, self.kb)
        crop, dis = pretty(label)
        if entry is None:
            return {"found": False, "healthy": False, "title": f"{crop}: {dis}", "crop": crop, "context": f"{crop} {dis}"}
        item = self._item(entry, lang)
        return {"found": True, "healthy": healthy, "crop": crop, "context": "" if healthy else entry["name"], **item}

    # ---------- speech ----------
    def transcribe(self, wav_bytes, lang=None):
        import soundfile as sf
        if self._asr is None:
            if not modelstore.is_ready("speech"):
                raise RuntimeError("Speech model is not downloaded yet.")
            self._asr = pipeline("automatic-speech-recognition", model=modelstore.MODELS["speech"]["repo"],
                                 chunk_length_s=30)
        data, sr = sf.read(io.BytesIO(wav_bytes))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:  # Whisper needs 16 kHz; resample here so torchaudio is not required
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(sr), 16000)
            data, sr = resample_poly(data, 16000 // g, int(sr) // g), 16000
        kw = {"task": "transcribe"}
        if lang:
            kw["language"] = lang
        return self._asr({"raw": data.astype("float32"), "sampling_rate": sr}, generate_kwargs=kw)["text"].strip()

    def speak(self, text, lang):
        """Returns WAV bytes, or None if the language has no voice / the voice model is unavailable."""
        if lang not in MMS_TTS:
            return None
        try:
            from scipy.io import wavfile
            from transformers import VitsModel
            code = MMS_TTS[lang]
            if code not in self._tts:
                name = f"facebook/mms-tts-{code}"
                self._tts[code] = (AutoTokenizer.from_pretrained(name), VitsModel.from_pretrained(name).eval())
            tok, model = self._tts[code]
            chunks, cur = [], ""
            for s in re.split(r"(?<=[.!?।])\s+", text.replace("\n", " ")):
                if len(cur) + len(s) > 250 and cur:
                    chunks.append(cur)
                    cur = ""
                cur += (" " if cur else "") + s
            if cur:
                chunks.append(cur)
            wavs = []
            with torch.no_grad():
                for c in chunks[:4]:
                    inputs = tok(c[:300], return_tensors="pt")
                    if inputs["input_ids"].shape[1] == 0:
                        continue

                    with torch.no_grad():
                         audio = model(**inputs).waveform[0].cpu().numpy()

                    wavs.append(audio)
                    wavs.append(np.zeros(int(model.config.sampling_rate * 0.25), dtype=np.float32))
            buf = io.BytesIO()
            wavfile.write(buf, model.config.sampling_rate, (np.concatenate(wavs) * 32767).astype(np.int16))
            return buf.getvalue()
        except Exception as e:
            print("TTS ERROR:", type(e).__name__, str(e))
            import traceback
            traceback.print_exc()
            return None
