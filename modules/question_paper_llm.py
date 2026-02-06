# modules/question_paper_llm.py

import os
import re
import time
import json
import random
import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
import pandas as pd

# ✅ Gemini imports
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from db_utils import save_question_paper_upload, save_mcq_bank_items


# ============================================================
# SECRETS LOADER (NO .env NEEDED)
# ============================================================

def _load_local_secrets() -> dict:
    """
    Loads modules/config/secrets.local.json
    This file MUST be in .gitignore.
    """
    here = os.path.dirname(__file__)  # .../modules
    secrets_path = os.path.join(here, "config", "secrets.local.json")
    if not os.path.exists(secrets_path):
        return {}
    try:
        with open(secrets_path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


# ============================================================
# GEMINI CLIENT (BATCHED) + QUOTA SAFE
# ============================================================

_GEMINI_CLIENT = None
_GEMINI_MODEL = None
_GEMINI_BATCH_SIZE = 10
_GEMINI_SLEEP = 1.0

# Retry controls
_GEMINI_MAX_RETRIES = 3
_GEMINI_MAX_SLEEP_SECONDS = 6.0  # cap backoff so Streamlit doesn’t feel “stuck”

def get_gemini_client():
    """
    Creates Gemini client only once per process.
    Reads API key from:
      1) modules/config/secrets.local.json
      2) env var GEMINI_API_KEY (optional)
    """
    global _GEMINI_CLIENT, _GEMINI_MODEL, _GEMINI_BATCH_SIZE, _GEMINI_SLEEP
    global _GEMINI_MAX_RETRIES, _GEMINI_MAX_SLEEP_SECONDS

    if _GEMINI_CLIENT is not None:
        return _GEMINI_CLIENT

    secrets = _load_local_secrets()

    api_key = (secrets.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "❌ Gemini API key not found.\n"
            "Create: modules/config/secrets.local.json with {\"GEMINI_API_KEY\": \"...\"}\n"
            "and add it to .gitignore."
        )

    _GEMINI_CLIENT = genai.Client(api_key=api_key)

    _GEMINI_MODEL = (secrets.get("GEMINI_MODEL") or "").strip() or os.getenv("GEMINI_MODEL", "").strip()
    if not _GEMINI_MODEL:
        _GEMINI_MODEL = "models/gemini-flash-lite-latest"

    try:
        _GEMINI_BATCH_SIZE = int(secrets.get("GEMINI_BATCH_SIZE", _GEMINI_BATCH_SIZE))
    except Exception:
        _GEMINI_BATCH_SIZE = 10

    try:
        _GEMINI_SLEEP = float(secrets.get("GEMINI_SLEEP", _GEMINI_SLEEP))
    except Exception:
        _GEMINI_SLEEP = 1.0

    try:
        _GEMINI_MAX_RETRIES = int(secrets.get("GEMINI_MAX_RETRIES", _GEMINI_MAX_RETRIES))
    except Exception:
        _GEMINI_MAX_RETRIES = 3

    try:
        _GEMINI_MAX_SLEEP_SECONDS = float(secrets.get("GEMINI_MAX_SLEEP_SECONDS", _GEMINI_MAX_SLEEP_SECONDS))
    except Exception:
        _GEMINI_MAX_SLEEP_SECONDS = 6.0

    print(f"✅ Gemini client ready | model: {_GEMINI_MODEL} | batch: {_GEMINI_BATCH_SIZE}")
    return _GEMINI_CLIENT


def _is_quota_error(e: Exception) -> bool:
    msg = str(e)
    return ("429" in msg) or ("RESOURCE_EXHAUSTED" in msg) or ("quota" in msg.lower())


def _sleep_backoff(attempt: int):
    s = (2 ** attempt) + random.uniform(0.0, 0.6)
    s = min(_GEMINI_MAX_SLEEP_SECONDS, s)
    time.sleep(s)


# ============================================================
# TEXT CLEAN + STRUCTURE REPAIR (UNCHANGED OCR)
# ============================================================

def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]{8,}", "    ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _repair_structure_for_mcq(text: str) -> str:
    if not text:
        return ""

    t = text
    t = re.sub(r"\(\s*([A-D])\s*\)", r"\1)", t, flags=re.I)
    t = re.sub(r"\b([A-D])\s*\)", r"\1)", t, flags=re.I)
    t = re.sub(r"\b([A-D])\s*\.\s*", r"\1) ", t, flags=re.I)

    t = re.sub(r"\s+([A-D])\)\s*", r"\n\1) ", t, flags=re.I)

    t = re.sub(r"\s+(Q\s*\.?\s*\d{1,3}\s*[\.\)\:\-])\s+", r"\n\1 ", t, flags=re.I)
    t = re.sub(r"\s+(\d{1,3}\s*[\.\)\:\-])\s+", r"\n\1 ", t)

    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _score_text_for_mcq(text: str) -> int:
    if not text:
        return 0
    opt = len(re.findall(r"(?mi)^\s*[A-D]\)\s+", text))
    q1 = len(re.findall(r"(?mi)^\s*Q\s*\.?\s*\d{1,3}[\.\)\:\-]?", text))
    q2 = len(re.findall(r"(?mi)^\s*\d{1,3}[\.\)\:\-]\s+", text))
    return opt * 5 + max(q1, q2) * 3 + len(text) // 500


# ============================================================
# OCR (UNCHANGED - KEEP THIS)
# ============================================================

def _to_gray_from_pix(pix) -> np.ndarray:
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def _preprocess_for_ocr(gray: np.ndarray) -> np.ndarray:
    gray = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)
    return gray


def _ocr_try_psm(gray: np.ndarray, psm: int) -> str:
    cfg = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"
    txt = pytesseract.image_to_string(gray, config=cfg)
    txt = _clean_text(txt)
    txt = _repair_structure_for_mcq(txt)
    return txt


def ocr_pdf_to_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    try:
        for i in range(len(doc)):
            page = doc[i]
            native = _clean_text(page.get_text("text") or "")

            if len(native) >= 180:
                pages.append(_repair_structure_for_mcq(native))
                continue

            pix = page.get_pixmap(dpi=300)
            gray = _to_gray_from_pix(pix)
            gray = _preprocess_for_ocr(gray)

            candidates = []
            for psm in (6, 4, 11, 3):
                candidates.append(_ocr_try_psm(gray, psm))

            best = max(candidates, key=_score_text_for_mcq)
            pages.append(best)

    finally:
        doc.close()

    merged = "\n\n".join([p for p in pages if p.strip()])
    merged = _clean_text(merged)
    merged = _repair_structure_for_mcq(merged)
    return merged


# ============================================================
# ANSWER KEY (UNCHANGED)
# ============================================================

def load_answer_key_excel(answer_key_file) -> dict[int, str]:
    if answer_key_file is None:
        return {}

    df = pd.read_excel(answer_key_file)
    if df.shape[1] < 2:
        return {}

    q_col, a_col = df.columns[0], df.columns[1]
    key_map: dict[int, str] = {}

    for _, row in df.iterrows():
        q_val = row.get(q_col, None)
        opt_val = row.get(a_col, None)
        if pd.isna(q_val):
            continue
        try:
            qno = int(str(q_val).strip())
        except Exception:
            continue

        opt = "" if pd.isna(opt_val) else str(opt_val).strip().upper()
        opt = opt[:1] if opt else ""
        if opt in ["A", "B", "C", "D"]:
            key_map[qno] = opt

    return key_map


# ============================================================
# PARSING (UNCHANGED)
# ============================================================

_Q_RE = re.compile(r"(?mi)^\s*(?:Q\s*\.?\s*)?(\d{1,3})\s*[\.\)\:\-]\s+")
_OPT_RE = re.compile(r"(?mi)^\s*(?:[\(\[]\s*)?([A-D])(?:\s*[\)\]])?\s*[\)\.\:\-]\s*(.*)$")


def _looks_like_code(line: str) -> bool:
    if not line:
        return False
    s = line.rstrip("\n")
    if s.startswith(("    ", "\t")):
        return True
    if any(tok in s for tok in ("==", "!=", ">=", "<=", "::", "->")):
        return True
    if sum(sym in s for sym in "<>={}[]()") >= 2:
        return True
    if re.search(r"\b(def|class|import|from|return|print|for|while|if|else|elif|try|except|with)\b", s):
        return True
    if re.search(r"^\s*[A-Za-z_]\w*\s*=\s*.+$", s):
        return True
    return False


def parse_mcqs_from_text(text: str, key_map: dict[int, str] | None = None) -> list[dict]:
    if not text or len(text.strip()) < 30:
        return []

    matches = list(_Q_RE.finditer(text))
    if not matches:
        return []

    blocks = []
    for i in range(len(matches)):
        qno = int(matches[i].group(1))
        start = matches[i].end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append((qno, text[start:end].strip()))

    if key_map:
        wanted = set(key_map.keys())
        blocks = [(qno, blk) for (qno, blk) in blocks if qno in wanted]

    items = []
    for qno, block in blocks:
        raw_lines = [ln.rstrip("\n") for ln in block.split("\n")]
        lines = [ln for ln in raw_lines if ln.strip()]

        option_map = {"A": [], "B": [], "C": [], "D": []}
        question_buf = []
        current_opt = None

        for ln in lines:
            m = _OPT_RE.match(ln.strip())
            if m:
                current_opt = m.group(1).upper()
                first = m.group(2)
                option_map[current_opt].append(first if first is not None else "")
                continue

            if current_opt in option_map:
                option_map[current_opt].append(ln)
            else:
                question_buf.append(ln)

        has_code = any(_looks_like_code(x) for x in question_buf)
        if has_code:
            question_text = "\n".join(question_buf).strip()
        else:
            question_text = " ".join([x.strip() for x in question_buf]).strip()

        opt_final = {}
        for k in ["A", "B", "C", "D"]:
            v = "\n".join([x for x in option_map[k] if x is not None]).strip()
            opt_final[k] = v

        non_empty_opts = sum(1 for k in opt_final if opt_final[k].strip())
        if not question_text or non_empty_opts < 2:
            continue

        items.append({
            "question_no": qno,
            "question_text": question_text,
            "option_a": opt_final["A"],
            "option_b": opt_final["B"],
            "option_c": opt_final["C"],
            "option_d": opt_final["D"],
            "correct_option": "",
            "why_correct": "",
            "why_a_wrong": "",
            "why_b_wrong": "",
            "why_c_wrong": "",
            "why_d_wrong": "",
        })

    return items


# ============================================================
# GEMINI EXPLANATIONS (FIXED JSON: OBJECT BY QNO + JSON MODE)
# ============================================================

SYSTEM_PROMPT = """You are a computer science professor.

Return ONLY valid JSON (no markdown, no comments, no extra text).

Output MUST be a JSON object keyed by question number as STRING:

{
  "1": {
    "why_correct": "...",
    "why_a_wrong": "...",
    "why_b_wrong": "...",
    "why_c_wrong": "...",
    "why_d_wrong": "..."
  },
  "2": { ... }
}

Rules:
- Each value is EXACTLY one sentence.
- Technical facts only.
- Do NOT use "correct/wrong" wording.
- For the correct option letter, set its why_*_wrong to:
  "(Correct answer - see explanation above)"
"""

def _ensure_one_sentence(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = s.split("\n")[0].strip()
    if s and not s.endswith((".", "!", "?")):
        s += "."
    return s


def _safe_json_object(text: str) -> dict:
    """
    Parses a JSON object safely, even if model returns extra text (rare),
    by extracting first {...} block.
    """
    t = (text or "").strip()
    if not t:
        return {}
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def generate_fallback_explanations(item: dict) -> dict:
    correct_opt = (item.get("correct_option") or "").upper().strip()
    result = {
        "why_correct": f"Option {correct_opt} is technically correct." if correct_opt else "Explanation not available.",
        "why_a_wrong": "Explanation not available.",
        "why_b_wrong": "Explanation not available.",
        "why_c_wrong": "Explanation not available.",
        "why_d_wrong": "Explanation not available.",
    }
    if correct_opt in ["A", "B", "C", "D"]:
        result[f"why_{correct_opt.lower()}_wrong"] = "(Correct answer - see explanation above)"
    return result


def _apply_explanations(item: dict, data: dict) -> dict:
    correct_opt = (item.get("correct_option") or "").strip().upper()

    out = {
        "why_correct": _ensure_one_sentence(data.get("why_correct", "")) or "Explanation not available.",
        "why_a_wrong": _ensure_one_sentence(data.get("why_a_wrong", "")) or "Explanation not available.",
        "why_b_wrong": _ensure_one_sentence(data.get("why_b_wrong", "")) or "Explanation not available.",
        "why_c_wrong": _ensure_one_sentence(data.get("why_c_wrong", "")) or "Explanation not available.",
        "why_d_wrong": _ensure_one_sentence(data.get("why_d_wrong", "")) or "Explanation not available.",
    }

    if correct_opt in ["A", "B", "C", "D"]:
        out[f"why_{correct_opt.lower()}_wrong"] = "(Correct answer - see explanation above)"

    return out


def _build_batch_prompt(batch_items: list[dict]) -> str:
    parts = []
    for it in batch_items:
        qno = it["question_no"]
        corr = (it.get("correct_option") or "").strip().upper()
        parts.append(
            f"""QUESTION {qno}
Question: {it.get("question_text","")}

A) {it.get("option_a","")}
B) {it.get("option_b","")}
C) {it.get("option_c","")}
D) {it.get("option_d","")}

Correct answer: {corr}
"""
        )

    joined = "\n\n".join(parts)
    qkeys = ", ".join([str(it["question_no"]) for it in batch_items])

    return (
        f"MCQ numbers: {qkeys}\n\n"
        f"Return ONLY the JSON object as specified.\n\n"
        f"{joined}\n"
    )


def _gemini_generate_json(prompt_text: str, batch_size: int) -> str:
    client = get_gemini_client()
    model_name = _GEMINI_MODEL or "models/gemini-flash-lite-latest"

    # token sizing: safer if batch is bigger
    # (if your questions contain code, keep batch <= 10)
    max_tokens = 900 + int(batch_size * 180)  # ~ good for one-sentence fields

    resp = client.models.generate_content(
        model=model_name,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text=SYSTEM_PROMPT + "\n\n" + prompt_text)]
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",  # JSON
        ),
    )
    return (resp.text or "").strip()


def add_explanations_with_llm(items: list[dict]) -> list[dict]:
    print("\n" + "=" * 70)
    print("🤖 GENERATING EXPLANATIONS (GEMINI - BATCHED)")
    print("=" * 70)

    items_by_qno = {it["question_no"]: it for it in items}
    qnos = sorted(items_by_qno.keys())

    to_process = [q for q in qnos if (items_by_qno[q].get("correct_option") or "").upper() in ["A", "B", "C", "D"]]
    total = len(to_process)

    get_gemini_client()  # init
    batch_size = max(1, int(_GEMINI_BATCH_SIZE))
    sleep_s = max(0.0, float(_GEMINI_SLEEP))
    max_retries = max(0, int(_GEMINI_MAX_RETRIES))

    print(f"📊 Processing {total} questions in batches of {batch_size}\n")

    stop_gemini = False

    for start in range(0, total, batch_size):
        chunk_qnos = to_process[start:start + batch_size]
        batch_items = [items_by_qno[q] for q in chunk_qnos]

        label = f"{start+1:3d}-{min(start+batch_size,total):3d}/{total}"
        print(f"[{label}] Q{chunk_qnos[0]}..Q{chunk_qnos[-1]}: ", end="", flush=True)

        if stop_gemini:
            for q in chunk_qnos:
                items_by_qno[q].update(generate_fallback_explanations(items_by_qno[q]))
            print("⚠️ quota-hit earlier → fallback", flush=True)
            continue

        prompt = _build_batch_prompt(batch_items)

        out_text = ""
        last_err = None

        for attempt in range(max_retries + 1):
            try:
                out_text = _gemini_generate_json(prompt, batch_size=len(batch_items))
                break
            except genai_errors.ClientError as e:
                last_err = e
                if _is_quota_error(e):
                    print(f"⏳ 429. sleep...", end=" ", flush=True)
                    _sleep_backoff(attempt)
                    continue
                break
            except Exception as e:
                last_err = e
                _sleep_backoff(attempt)

        # If no output -> fallback
        if not out_text:
            if last_err and _is_quota_error(last_err):
                stop_gemini = True
                print("⚠️ 429/quota → fallback + stop calls", flush=True)
            else:
                print("❌ empty → fallback", flush=True)

            for q in chunk_qnos:
                items_by_qno[q].update(generate_fallback_explanations(items_by_qno[q]))
            continue

        parsed = _safe_json_object(out_text)
        if not parsed:
            print("❌ non-json → fallback", flush=True)
            for q in chunk_qnos:
                items_by_qno[q].update(generate_fallback_explanations(items_by_qno[q]))
            continue

        # Apply per question
        for q in chunk_qnos:
            it = items_by_qno[q]
            data = parsed.get(str(q), {})
            if isinstance(data, dict) and data:
                it.update(_apply_explanations(it, data))
            else:
                it.update(generate_fallback_explanations(it))

        print("✅", flush=True)

        if sleep_s:
            time.sleep(sleep_s)

    print("\n" + "=" * 70)
    print(f"✅ Completed {total} questions")
    print("=" * 70 + "\n")

    return [items_by_qno[q] for q in qnos]


# ============================================================
# MAIN PIPELINE (UNCHANGED)
# ============================================================

def run_question_paper_llm_flow(
    *,
    subject_id: int,
    qp_pdf_file,
    answer_key_file,
    uploads_dir: str,
):
    os.makedirs(uploads_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("📚 QUESTION PAPER PROCESSING")
    print("=" * 70)

    qp_path = os.path.join(uploads_dir, f"subject_{subject_id}_question_paper.pdf")
    with open(qp_path, "wb") as f:
        f.write(qp_pdf_file.getbuffer())
    print("✅ PDF saved")

    ak_path = None
    if answer_key_file is not None:
        ak_path = os.path.join(uploads_dir, f"subject_{subject_id}_answer_key.xlsx")
        with open(ak_path, "wb") as f:
            f.write(answer_key_file.getbuffer())
        print("✅ Answer key saved")

    question_paper_id = save_question_paper_upload(subject_id, qp_path, ak_path)
    print(f"✅ Question paper ID: {question_paper_id}")

    print("\n📄 Extracting text (native first, OCR fallback + multi-psm)...")
    t0 = time.time()
    text = ocr_pdf_to_text(qp_path)
    print(f"✅ Extracted {len(text):,} characters in {time.time() - t0:.1f}s")

    debug_txt = os.path.join(uploads_dir, f"subject_{subject_id}_ocr_debug.txt")
    with open(debug_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"🧾 OCR debug saved: {debug_txt}")

    key_map = load_answer_key_excel(answer_key_file)
    print(f"\n🔑 Answer key: {len(key_map)} answers")

    print("\n🔍 Parsing questions...")
    items = parse_mcqs_from_text(text, key_map=key_map if key_map else None)
    print(f"✅ Found {len(items)} questions")

    for it in items:
        qno = it["question_no"]
        it["correct_option"] = key_map.get(qno, "") if key_map else ""

    with_answers = sum(1 for it in items if it.get("correct_option") in ["A", "B", "C", "D"])
    print(f"✅ {with_answers}/{len(items)} have answers")

    if with_answers > 0:
        items = add_explanations_with_llm(items)
    else:
        print("\n⚠️  No answer key - skipping explanations")

    print("\n💾 Saving to database...")
    inserted = save_mcq_bank_items(question_paper_id, items)
    print(f"✅ Saved {inserted} questions")

    print("\n" + "=" * 70)
    print("✅ COMPLETE")
    print("=" * 70 + "\n")

    return {
        "question_paper_id": question_paper_id,
        "inserted_count": inserted,
        "questions": items,
        "ocr_debug_path": debug_txt,
    }
