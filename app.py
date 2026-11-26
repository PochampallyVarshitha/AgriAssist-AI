"""AgriAssist portal - leaf disease finder + farming advice in 9 languages.
Inputs: typed question, leaf photo (upload or camera), speech.      Run:  streamlit run app.py
"""
import hashlib
import html
import io
import time

import streamlit as st
from PIL import Image

import modelstore
from knowledge import pretty
from ui_text import EXAMPLES, LANGS, T

st.set_page_config(page_title="AgriAssist - Leaf Disease & Farming Advice", page_icon="🌾", layout="wide")
S = st.session_state
for _k, _v in {"results": [], "context": "", "last_photo": None, "last_audio": None}.items():
    S.setdefault(_k, _v)
MIN_CONF = 0.60

st.markdown("""<style>
html, body, [class*="css"], .stMarkdown, button, textarea, input {
  font-family: "Segoe UI","Nirmala UI","Noto Sans","Noto Sans Devanagari","Noto Naskh Arabic",Roboto,sans-serif !important;}
.stApp {background: linear-gradient(180deg,#f4f8f1 0%,#e9f3e4 100%);}
.block-container {max-width: 1080px; padding-top: 1.1rem;}
footer, #MainMenu {visibility: hidden;}
.rtl {direction: rtl; text-align: right;}
.hero {background: linear-gradient(120deg,#1b5e20 0%,#2e7d32 55%,#7cb342 100%); color:#fff; border-radius:22px;
  padding:26px 30px 20px; margin-bottom:16px; box-shadow:0 10px 28px rgba(27,94,32,.28);}
.hero-title {font-size:2.3rem; font-weight:800; letter-spacing:.3px; line-height:1.15;}
.hero-tag {font-size:1.08rem; opacity:.96; margin-top:6px;}
.pill {display:inline-block; background:rgba(255,255,255,.17); border:1px solid rgba(255,255,255,.4);
  padding:4px 13px; border-radius:999px; font-size:.86rem; margin:12px 8px 0 0; color:#fff;}
[data-testid="stVerticalBlockBorderWrapper"] {background:#fff; border-radius:18px !important;
  box-shadow:0 4px 18px rgba(30,70,30,.08); border:1px solid #dcead6 !important;}
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {box-shadow:none;}
.sec {border-radius:14px; padding:13px 15px; border-left:6px solid; height:100%; color:#1b2a1b; line-height:1.5; font-size:.97rem;}
.sec h4 {margin:0 0 6px; font-size:1.02rem; color:#1b2a1b;}
.sec.sym {border-color:#ef6c00; background:#fff6ec;}
.sec.act {border-color:#2e7d32; background:#eff8ef;}
.sec.pre {border-color:#1565c0; background:#eef4fd;}
.sec.rtl {border-left:none; border-right:6px solid;}
.dtitle {font-size:1.35rem; font-weight:700; color:#1b3a1b; margin-bottom:8px;}
.badge {display:inline-block; font-size:.8rem; padding:3px 10px; border-radius:999px; vertical-align:middle; margin-right:6px;}
.badge.ok {background:#dff3df; color:#1b5e20;} .badge.bad {background:#fde7d6; color:#b34700;}
.conf {height:10px; border-radius:999px; background:#e3ebdf; overflow:hidden; margin-top:6px;}
.conf > div {height:100%; background:linear-gradient(90deg,#fbc02d,#43a047);}
.confnote {font-size:.85rem; color:#4a5a4a; margin-top:3px;}
.q {background:#e8f3e4; border-radius:14px; padding:10px 14px; margin-bottom:10px; color:#1b2a1b; font-weight:600;}
.ans {background:#f7fbf5; border:1px solid #dcead6; border-radius:14px; padding:13px 16px; margin-bottom:8px;
  color:#1b2a1b; line-height:1.6;}
.safety {background:#fff4d6; border:1px solid #ffe08a; border-radius:12px; padding:10px 14px; margin-top:12px;
  font-size:.92rem; color:#4a3b00;}
.warn {background:#fff1ee; border:1px solid #ffc9bd; border-radius:12px; padding:12px 14px; color:#5a2a1f; line-height:1.55;}
.chips span {display:inline-block; background:#eef3ea; border:1px solid #d5e2cf; border-radius:999px;
  padding:3px 12px; margin:6px 6px 0 0; font-size:.85rem; color:#2c3d2c;}
.helpline {background:#e8f3e4; border-radius:12px; padding:10px 12px; font-size:.92rem; color:#1b3a1b;}
.stButton > button[kind="primary"], .stFormSubmitButton > button {border-radius:12px; font-weight:700;}
</style>""", unsafe_allow_html=True)


def esc(s):
    return html.escape(str(s)).replace("\n", "<br>")


# ------------------------------------------------------------------ language + hero
with st.sidebar:
    st.markdown("## 🌾 AgriAssist")
    lang_name = st.selectbox("🌐 Language / भाषा / மொழி", list(LANGS), key="lang_name")
lang = LANGS[lang_name]
t, RTL = T[lang], (" rtl" if lang == "ur" else "")

pills = "".join(f'<span class="pill">{esc(x)}</span>' for x in (t["tab_ask"], t["tab_photo"], t["tab_voice"], t["helpline"]))
st.markdown(f'<div class="hero{RTL}"><div class="hero-title">🌾 AgriAssist</div>'
            f'<div class="hero-tag">{esc(t["tagline"])}</div><div>{pills}</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ first-time model setup (never crashes)
def run_downloads(keys):
    for k in keys:
        m = modelstore.MODELS[k]
        bar = st.progress(0.0, text=f"{m['name']} …")
        job = modelstore.start(k)
        while not job.done:
            bar.progress(job.progress, text=f"{m['name']}: {job.progress:.0%}")
            time.sleep(0.7)
        if job.error:
            bar.empty()
            st.error(f"Download failed for **{m['name']}**\n\n`{job.error}`\n\n"
                     "• Check your internet and free disk space (about 5 GB needed)\n"
                     "• Click the button again - it resumes where it stopped\n"
                     "• On college Wi-Fi use a mobile hotspot, or run `python modelstore.py` in a terminal to see details")
            return False
        bar.progress(1.0, text=f"✅ {m['name']}")
    return True


def ensure_models():
    if not [k for k, m in modelstore.MODELS.items() if m["required"] and not modelstore.is_ready(k)]:
        return
    with st.container(border=True):
        st.markdown("### ⬇️ First-time setup")
        st.write("AgriAssist downloads its AI models once (internet needed). After that it works offline.")
        choice = {}
        for k, m in modelstore.MODELS.items():
            label = f"{m['name']} · ~{m['size_gb']:g} GB · {m['why']}"
            if modelstore.is_ready(k):
                st.markdown(f"✅ {label}")
            else:
                choice[k] = st.checkbox(label + (" **(required)**" if m["required"] else " (optional)"),
                                        value=True, disabled=m["required"], key=f"setup_{k}")
        total = sum(modelstore.MODELS[k]["size_gb"] for k, v in choice.items() if v)
        if st.button(f"⬇️ Download selected (~{total:.1f} GB)", type="primary"):
            if run_downloads([k for k, v in choice.items() if v]):
                st.rerun()
    st.stop()


ensure_models()


@st.cache_resource(show_spinner="Loading AI models… (first start takes a minute)")
def get_bot():
    from engine import AgriBot
    return AgriBot()


bot = get_bot()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    voice_out = st.toggle("🔊 " + t["aloud"], value=False)
    with st.expander("⚙️ " + t["adv"]):
        use_llm = st.checkbox("Write answers with local LLM (slower, extra ~3 GB download)", value=False)
        min_score = st.slider("Match strictness", 0.60, 0.95, 0.75, 0.01,
                              help="Higher = safer ('I am not sure' more often). Lower = answers more questions.")
    with st.expander("📦 AI models"):
        for k, m in modelstore.MODELS.items():
            ok = modelstore.is_ready(k)
            st.write(("✅ " if ok else "⬇️ ") + m["name"])
            if not ok and st.button(f"Download (~{m['size_gb']:g} GB)", key=f"side_{k}"):
                if run_downloads([k]):
                    bot.nmt_error = None
                    st.rerun()
    if st.button("🗑️ " + t["clear"], use_container_width=True):
        S.results, S.context, S.last_photo = [], "", None
        st.rerun()
    st.markdown(f'<div class="helpline">{esc(t["helpline"])}</div>', unsafe_allow_html=True)

if lang != "en" and not bot.can_translate():
    st.warning(t["no_translate"])
    if bot.nmt_error:
        st.caption(bot.nmt_error)

# ------------------------------------------------------------------ inputs
pending = None
speak_hint = lang
modes = [0, 1, 2]
mode_labels = [t["tab_ask"], t["tab_photo"], t["tab_voice"]]
mode = st.segmented_control("mode", modes, default=0, key="mode", label_visibility="collapsed",
                            format_func=lambda i: mode_labels[i])
mode = 0 if mode is None else mode

with st.container(border=True):
    if mode == 0:
        with st.form("ask_form", clear_on_submit=True, border=False):
            q = st.text_area("question", placeholder=t["ask_ph"], height=110, label_visibility="collapsed")
            go = st.form_submit_button(t["ask_btn"], type="primary", use_container_width=True)
        if go and q.strip():
            pending = ("q", q.strip(), "")
        st.caption(t["examples"])
        for i, ex in enumerate(EXAMPLES[lang]):
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                pending = ("q", ex, "")
    elif mode == 1:
        st.write(t["photo_hint"])
        up = st.file_uploader("photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        with st.expander("📸 " + t["camera"]):
            cam = st.camera_input("camera", label_visibility="collapsed")
        f = cam or up
        if f is not None:
            data = f.getvalue()
            if hashlib.md5(data).hexdigest() != S.last_photo:
                pending = ("photo", data, hashlib.md5(data).hexdigest())
    else:
        if not modelstore.is_ready("speech"):
            st.info("Speech recognition model (~1 GB) is not downloaded yet.")
            if st.button("⬇️ Download speech model", type="primary"):
                if run_downloads(["speech"]):
                    st.rerun()
        else:
            st.write(t["voice_hint"])
            SPEAK = {"Same as interface language": "same", "Auto-detect": None, **LANGS}
            sp = st.selectbox("Speaking language (pick the language you will speak)", list(SPEAK), key="speak_lang")
            speak_hint = lang if SPEAK[sp] == "same" else SPEAK[sp]
            aud = st.audio_input("voice", label_visibility="collapsed")
            with st.expander("📁 Microphone not working? Upload a recorded audio file instead"):
                afile = st.file_uploader("audio file", type=["wav", "mp3", "ogg", "flac"],
                                         label_visibility="collapsed", key="audio_file")
            for src in (aud, afile):
                if src is not None and pending is None:
                    data = src.getvalue()
                    if hashlib.md5(data).hexdigest() != S.last_audio:
                        pending = ("voice", data, hashlib.md5(data).hexdigest())


# ------------------------------------------------------------------ processing
def add_result(r):
    S.results.insert(0, r)
    del S.results[12:]


def narrate(text):
    if not voice_out:
        return None, False
    with st.spinner("🔊"):
        wav = bot.speak(text[:900], lang)
    return wav, wav is None


def do_question(q, via=""):
    with st.spinner(t["thinking"]):
        res = bot.answer(q, lang, use_llm, min_score, context=S.context)
    if res["unsure"]:
        spoken = t["unsure"]
    else:
        it = res["items"][0]
        spoken = (it["title"] + ". " + it["sections"]["treatment"]) if it.get("sections") else it["text"]
    wav, err = narrate(spoken)
    add_result({"kind": "qa", "lang": lang, "question": q, "via": via, "items": res["items"],
                "unsure": res["unsure"], "suggestions": res.get("suggestions", []), "audio": wav, "voice_err": err})


def do_photo(data):
    from disease_model import predict
    img = Image.open(io.BytesIO(data)).convert("RGB")
    with st.spinner(t["checking"]):
        top = predict(img, top_k=3)
        label, conf = top[0]
        if conf < MIN_CONF:
            add_result({"kind": "low", "lang": lang, "image": data, "conf": conf})
            return
        d = bot.diagnose(label, lang)
    S.context = d["context"]
    spoken = (d["title"] + ". " + d["sections"]["treatment"]) if d.get("sections") else d.get("text", d["title"])
    wav, err = narrate(spoken)
    alts = [(" · ".join(pretty(lb)), p) for lb, p in top[1:] if p >= 0.02]
    add_result({"kind": "diag", "lang": lang, "image": data, "conf": conf, "alts": alts, "audio": wav,
                "voice_err": err, **d})


if S.get("queued"):
    pending = ("q", S.pop("queued"), "")
if pending:
    kind, payload, digest = pending
    try:
        if kind == "q":
            do_question(payload)
        elif kind == "photo":
            S.last_photo = digest
            do_photo(payload)
        else:
            S.last_audio = digest
            with st.spinner(t["listening"]):
                text = bot.transcribe(payload, speak_hint)
            if text:
                do_question(text, via="voice")
    except Exception as e:
        st.error(f"Something went wrong: {type(e).__name__}: {e}")


# ------------------------------------------------------------------ results
def show_sections(sections, tt, d):
    cols = st.columns(3)
    for col, (cls, icon, key) in zip(cols, (("sym", "🔍", "symptoms"), ("act", "💊", "treatment"), ("pre", "🛡️", "prevention"))):
        col.markdown(f'<div class="sec {cls}{d}"><h4>{icon} {esc(tt[key])}</h4><div>{esc(sections[key])}</div></div>',
                     unsafe_allow_html=True)
    st.markdown(f'<div class="safety{d}">⚠️ {esc(tt["safety"])}</div>', unsafe_allow_html=True)


def render_result(r, idx=0):
    tt, d = T[r["lang"]], (" rtl" if r["lang"] == "ur" else "")
    with st.container(border=True):
        if r["kind"] in ("diag", "low"):
            c1, c2 = st.columns([1, 2.6], vertical_alignment="center")
            c1.image(r["image"])
            with c2:
                if r["kind"] == "low":
                    st.markdown(f'<div class="warn{d}">📷 {esc(tt["low_conf"])}</div>', unsafe_allow_html=True)
                else:
                    healthy = r.get("healthy")
                    title = f'{tt["healthy"]} · {r["crop"]}' if healthy else r["title"]
                    badge = '<span class="badge ok">🌿</span>' if healthy else '<span class="badge bad">🦠</span>'
                    st.markdown(f'<div class="dtitle{d}">{badge}{esc(title)}</div>'
                                f'<div class="conf"><div style="width:{r["conf"] * 100:.0f}%"></div></div>'
                                f'<div class="confnote{d}">{r["conf"]:.0%} {esc(tt["sure"])}</div>', unsafe_allow_html=True)
            if r["kind"] == "diag":
                if not r["found"]:
                    st.markdown(f'<div class="warn{d}">{esc(tt["unsure"])}</div>', unsafe_allow_html=True)
                elif r.get("sections"):
                    show_sections(r["sections"], tt, d)
                else:
                    st.markdown(f'<div class="ans{d}">{esc(r["text"])}</div>', unsafe_allow_html=True)
                if r.get("alts"):
                    chips = "".join(f"<span>{esc(n)} · {p:.0%}</span>" for n, p in r["alts"])
                    st.markdown(f'<div class="chips{d}"><b>{esc(tt["other"])}:</b> {chips}</div>', unsafe_allow_html=True)
        else:
            who = "🎤 " if r.get("via") == "voice" else "🧑‍🌾 "
            st.markdown(f'<div class="q{d}">{who}{esc(r["question"])}</div>', unsafe_allow_html=True)
            if r["unsure"]:
                st.markdown(f'<div class="warn{d}">{esc(tt["unsure"])}</div>', unsafe_allow_html=True)
                st.caption("💡 " + tt["tip"])
                sg = r.get("suggestions") or []
                if sg:
                    st.caption(tt["closest"])
                    for j, (c, nm) in enumerate(zip(st.columns(len(sg)), sg)):
                        c.button(nm, key=f"sg_{idx}_{j}", use_container_width=True,
                                 on_click=lambda n=nm: S.__setitem__("queued", n))
            if r.get("via") == "voice":
                st.caption("🎤 Heard this? If it is wrong, choose your speaking language above and record again.")
            for it in r["items"]:
                if it.get("sections"):
                    st.markdown(f'<div class="dtitle{d}"><span class="badge bad">🦠</span>{esc(it["title"])}</div>',
                                unsafe_allow_html=True)
                    show_sections(it["sections"], tt, d)
                else:
                    st.markdown(f'<div class="ans{d}">{esc(it["text"])}</div>', unsafe_allow_html=True)
        src = r.get("source") or next((i.get("source") for i in r.get("items", []) if i.get("source")), None)
        if src:
            st.caption(f'{tt["source"]}: {src}')
        if r.get("audio"):
            st.audio(r["audio"], format="audio/wav")
        elif r.get("voice_err"):
            st.caption("🔇 Voice is not available for this language (the first use needs internet).")


st.markdown(f"### 📋 {t['results']}")
if S.results:
    for n, res in enumerate(S.results):
        render_result(res, n)
else:
    st.info(t["no_results"])
