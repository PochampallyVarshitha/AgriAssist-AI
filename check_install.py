"""Run:  python check_install.py   - tells you which files in this folder are missing, wrong or out of date."""
import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).parent
EXPECT = {
    "app.py": [], "ui_text.py": ["T", "LANGS", "EXAMPLES"], "modelstore.py": ["MODELS", "is_ready", "start"],
    "knowledge.py": ["norm", "pretty", "load_kb", "find_disease", "tokens", "lexical_boost"],
    "engine.py": ["AgriBot"], "disease_model.py": ["predict", "train"],
    "knowledge_base.json": None, "disease_kb.json": None,
}
SHA = {
    "app.py": "05e2891b39cc",
    "engine.py": "c53ec0ffc7ad",
    "knowledge.py": "993b19c9be46",
    "ui_text.py": "db19d30a36d4",
    "modelstore.py": "536332982166",
    "disease_model.py": "a82c871d72b3",
    "knowledge_base.json": "b56ff134392b",
    "disease_kb.json": "659c699b3ba0"
}

bad = 0
for f, names in EXPECT.items():
    p = HERE / f
    if not p.exists():
        print(f"[MISSING] {f}"); bad += 1; continue
    problems = []
    if names:
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
            defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            for n in tree.body:
                if isinstance(n, ast.Assign):
                    defined |= {t.id for t in n.targets if isinstance(t, ast.Name)}
            problems = [x for x in names if x not in defined]
        except SyntaxError as e:
            problems = [f"syntax error line {e.lineno}"]
    same = hashlib.sha256(p.read_bytes()).hexdigest()[:12] == SHA[f]
    if problems:
        print(f"[WRONG FILE] {f}: missing {problems}  -> replace it with the one from AgriAssist_Portal.zip"); bad += 1
    elif not same:
        print(f"[DIFFERENT] {f}: differs from the shipped version (edited, or older/newer)")
    else:
        print(f"[ok] {f}")
print("\nAll required files look right." if not bad else f"\n{bad} file(s) need replacing: extract AgriAssist_Portal.zip over this folder (Replace all).")
