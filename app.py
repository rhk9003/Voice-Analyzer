import os
import io
import re
import json
import datetime as dt
from typing import List, Tuple

import streamlit as st

try:
    import google.generativeai as genai
except Exception:
    genai = None

# Optional extractors
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    from PIL import Image
except Exception:
    Image = None


APP_TITLE = "Voice Analyzer | Persona & Voice Spec (Gemini)"
MODEL_NAME = "gemini-3-pro-preview"


# -----------------------------
# Helpers
# -----------------------------
def now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clamp_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[TRUNCATED]"
    return text


def read_plain_text(uploaded_file, max_chars: int) -> str:
    raw = uploaded_file.read()
    try:
        txt = raw.decode("utf-8", errors="ignore")
    except Exception:
        txt = str(raw)
    return clamp_text(txt, max_chars)


def extract_text_from_pdf(uploaded_file, max_chars: int) -> str:
    if PdfReader is None:
        return ""
    raw = uploaded_file.read()
    try:
        reader = PdfReader(io.BytesIO(raw))
        chunks = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                chunks.append(t)
        return clamp_text("\n\n".join(chunks), max_chars)
    except Exception:
        return ""


def extract_text_from_docx(uploaded_file, max_chars: int) -> str:
    if docx is None:
        return ""
    raw = uploaded_file.read()
    try:
        d = docx.Document(io.BytesIO(raw))
        paras = [p.text for p in d.paragraphs if p.text and p.text.strip()]
        return clamp_text("\n".join(paras), max_chars)
    except Exception:
        return ""


def extract_text_from_html(uploaded_file, max_chars: int) -> str:
    raw = uploaded_file.read()
    try:
        html = raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""
    if BeautifulSoup is None:
        # fallback: strip tags roughly
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        return clamp_text(text, max_chars)
    try:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return clamp_text(text, max_chars)
    except Exception:
        return ""


def extract_text_from_rtf(uploaded_file, max_chars: int) -> str:
    # Minimal RTF stripper (best-effort)
    raw = uploaded_file.read()
    try:
        rtf = raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""
    # remove rtf control words and braces
    text = re.sub(r"{\\.*?}|\\[a-zA-Z]+\d* ?", " ", rtf)
    text = re.sub(r"[{}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return clamp_text(text, max_chars)


def load_image(uploaded_file):
    if Image is None:
        return None
    raw = uploaded_file.read()
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def build_prompt(sample_blocks: List[str], notes: str, constraints: str, output_language: str, attachments_report: str) -> str:
    samples_joined = "\n\n".join([b for b in sample_blocks if b.strip()]).strip()
    notes = (notes or "").strip()
    constraints = (constraints or "").strip()

    lang_rule = {
        "繁體中文": "請用繁體中文輸出。",
        "English": "Please write in English.",
        "日本語": "日本語で出力してください。",
    }.get(output_language, "請用繁體中文輸出。")

    return f"""
你是一個「語感/風格/價值觀」分析器。你的任務：從樣本文本（以及必要時對圖片內容的理解）歸納作者的 Persona 與可執行的寫作規格（Voice Spec），並輸出可直接貼入另一個專案封包的 VOICE CONTEXT。

【重要規則】
1) 「可抽取文本的檔案」= 證據層：你只能從這裡歸納語感特徵、句法、立場、節奏，不可憑空捏造作者背景。
2) 「圖片檔」= 證據層：你可以根據圖片中可見文字/版面/語氣用法做歸納（不要臆測看不見的內容）。
3) 「我的筆記」= 語境校準層：可以補足作者定位/價值觀脈絡；若與證據層衝突，必須指出衝突並給出兩套版本（V-A: 以證據為準；V-B: 以筆記為準）。
4) 禁止引用任何樣本文本原句超過 25 字；不得大量抄錄。
5) 你的輸出必須可執行：要能讓另一個 AI 按規格穩定模仿寫作。
6) {lang_rule}

【你收到的附件狀態（供你判斷證據強度）】
{attachments_report}

【輸出格式（嚴格）】
A) Persona Brief（可讀）
- 作者世界觀一句話：
- 對讀者的定位（上對下/並肩/挑釁/對話）：
- 核心信念/價值觀（3–7條，句型化）：
- 動機邊界（他為什麼寫、他不做什麼）：
- 允許的模糊與留白（哪些可以不講死）：
- 禁語/禁套路（含理由）：

B) Voice Spec（可執行）
- tone_mix（%）：冷靜__ / 犀利__ / 幽默__ / 溫度__
- sentence_rhythm：短句比例__%；每段__–__行；轉折頻率__
- stance_rules：如何下結論/如何留白/如何反問
- lexical_rules：常用詞/避免詞/禁詞
- structure_rules：常用推理順序（例：現象→對照→推論→選項）
- do_not：絕對禁止事項
- sample_lines（<=5句，每句<=25字，模仿用、不可引用原文）：

C) 可直接貼入封包的 VOICE CONTEXT（請用一個 Markdown code block 包起來）
內容需長得像這樣（你要填滿欄位）：
=== [VOICE CONTEXT | EDITABLE] ===
[PERSONA LOG]
...
[VOICE SPEC]
...
=== [/VOICE CONTEXT] ===

D) 快速驗收清單（讓總編輯快速檢查是否像他）
- 3 個「像」的判準
- 3 個「不像」的警戒

【額外限制/偏好（若有，必須遵守）】
{constraints if constraints else "（無）"}

【樣本文本（證據層）】
{samples_joined if samples_joined else "（目前沒有可抽取文本。請先指出不足，並在可推論範圍內給一版『低信心』規格，提醒需要更多樣本或請我貼關鍵段落。）"}

【我的筆記（語境校準層）】
{notes if notes else "（未提供）"}
""".strip()


def call_gemini_multimodal(api_key: str, prompt_text: str, images: List, temperature: float, max_output_tokens: int) -> str:
    if genai is None:
        raise RuntimeError("google-generativeai 未安裝或匯入失敗。請確認 requirements.txt。")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)

    parts = [prompt_text]
    # Attach images as additional parts (multimodal)
    for img in images:
        if img is not None:
            parts.append(img)

    resp = model.generate_content(
        parts,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        },
    )

    text = getattr(resp, "text", None)
    if text:
        return text

    try:
        return resp.candidates[0].content.parts[0].text
    except Exception:
        return str(resp)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🧬", layout="wide")
st.title("🧬 Voice Analyzer（多格式 + 圖檔支援）")
st.caption(f"固定模型：{MODEL_NAME}｜輸入：多種檔案格式 + 你的筆記｜輸出：Persona Brief + Voice Spec + VOICE CONTEXT（可貼入封包）")

with st.sidebar:
    st.subheader("🔑 Gemini API Key")
    api_key = st.text_input("GEMINI_API_KEY", type="password", value=os.getenv("GEMINI_API_KEY", ""))

    st.divider()
    st.subheader("⚙️ 生成參數")
    temperature = st.slider("temperature", 0.0, 1.0, 0.4, 0.05)
    max_output_tokens = st.slider("max_output_tokens", 512, 8192, 4096, 256)

    st.divider()
    st.subheader("🧠 證據限制（避免 prompt 爆炸）")
    max_chars_per_file = st.slider("每個可抽文本檔的最大字元數", 5000, 200000, 60000, 5000)
    max_total_chars = st.slider("全部文本合計最大字元數", 20000, 400000, 180000, 10000)

    st.divider()
    output_language = st.selectbox("輸出語言", ["繁體中文", "English", "日本語"], index=0)

    st.divider()
    save_history = st.toggle("保存到本機 Session History", value=True)


col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1) 上傳樣本（支援多格式）")
    st.write("支援：txt / md / pdf / docx / html / rtf / csv + png/jpg/jpeg/webp（圖檔會走多模態，不靠 OCR）")

    uploads = st.file_uploader(
        "上傳檔案（可多檔）",
        type=["txt", "md", "pdf", "docx", "html", "htm", "rtf", "csv", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )

    pasted = st.text_area(
        "或直接貼樣本文本（可與上傳同時用）",
        height=220,
        placeholder="貼上 1–3 篇代表性文字…"
    )

    st.subheader("2) 你的筆記（語境校準層）")
    notes = st.text_area(
        "你對這個人的理解：價值觀、禁忌、讀者關係、寫作目的、不可碰觸的點…",
        height=220,
        placeholder="例：他討厭雞湯；寫作要推動產業改革；對讀者是並肩討論而非教學…"
    )

with col2:
    st.subheader("3) 額外限制 / 偏好（可選）")
    constraints = st.text_area(
        "例如：不要情緒勒索、不要教條式結論、要多反問、避免『你應該』句型…",
        height=180,
        placeholder="可留空"
    )

    st.subheader("4) 一鍵產出")
    run = st.button("🚀 開始語感分析", type="primary", use_container_width=True)

    st.info(
        "策略：能抽字就抽字；圖檔直接附給模型。\n"
        "若某些檔案抽不到字（掃描PDF/特殊格式），會在『附件狀態』標記，並以你的筆記補足語境。"
    )


# -----------------------------
# Prepare evidence
# -----------------------------
sample_blocks: List[str] = []
images: List = []
report_lines: List[str] = []
total_chars = 0

if uploads:
    for f in uploads:
        name = f.name
        ext = name.lower().split(".")[-1] if "." in name else ""
        mime = getattr(f, "type", "")

        # Image files
        if ext in ("png", "jpg", "jpeg", "webp"):
            img = load_image(f)
            if img is not None:
                images.append(img)
                report_lines.append(f"- ✅ 圖檔：{name}（多模態已附加）")
            else:
                report_lines.append(f"- ⚠️ 圖檔：{name}（讀取失敗，請換格式或重傳）")
            continue

        # Text-extractable formats
        extracted = ""
        if ext in ("txt", "md", "csv"):
            extracted = read_plain_text(f, max_chars=max_chars_per_file)
        elif ext in ("pdf",):
            extracted = extract_text_from_pdf(f, max_chars=max_chars_per_file)
        elif ext in ("docx",):
            extracted = extract_text_from_docx(f, max_chars=max_chars_per_file)
        elif ext in ("html", "htm"):
            extracted = extract_text_from_html(f, max_chars=max_chars_per_file)
        elif ext in ("rtf",):
            extracted = extract_text_from_rtf(f, max_chars=max_chars_per_file)
        else:
            # fallback: try reading as text
            extracted = read_plain_text(f, max_chars=max_chars_per_file)

        extracted = (extracted or "").strip()
        if extracted:
            remaining = max_total_chars - total_chars
            if remaining <= 0:
                report_lines.append(f"- ⏭️ {name}（可抽字但已達總字元上限，略過）")
                continue

            if len(extracted) > remaining:
                extracted = extracted[:remaining] + "\n\n[TRUNCATED_BY_TOTAL_LIMIT]"

            total_chars += len(extracted)
            sample_blocks.append(f"=== [FILE: {name} | {mime}] ===\n{extracted}\n=== [/FILE] ===")
            report_lines.append(f"- ✅ 可抽字：{name}（納入 {len(extracted):,} 字）")
        else:
            report_lines.append(f"- ⚠️ {name}（抽不到文字；可能是掃描/特殊格式。請改傳可選取文字的版本，或貼關鍵段落）")

if pasted.strip():
    paste_txt = clamp_text(pasted.strip(), max_total_chars - total_chars if max_total_chars - total_chars > 0 else 0)
    if paste_txt:
        sample_blocks.append(f"=== [PASTED] ===\n{paste_txt}\n=== [/PASTED] ===")
        report_lines.append(f"- ✅ 直接貼上文本（納入 {len(paste_txt):,} 字）")
        total_chars += len(paste_txt)
    else:
        report_lines.append("- ⏭️ 直接貼上文本（已達總字元上限，略過）")

attachments_report = "\n".join(report_lines) if report_lines else "- （未上傳任何附件）"

prompt_text = build_prompt(
    sample_blocks=sample_blocks,
    notes=notes,
    constraints=constraints,
    output_language=output_language,
    attachments_report=attachments_report
)

# history
if "history" not in st.session_state:
    st.session_state.history = []

if run:
    if not api_key.strip():
        st.error("缺少 GEMINI_API_KEY。請在側欄貼上。")
    else:
        with st.expander("📎 附件狀態（會影響證據強度）", expanded=True):
            st.write(attachments_report)
            st.caption(f"文本合計納入：{total_chars:,} 字｜圖片附加：{len(images)} 張")

        with st.spinner("生成中…"):
            try:
                output = call_gemini_multimodal(
                    api_key=api_key.strip(),
                    prompt_text=prompt_text,
                    images=images,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            except Exception as e:
                st.error(f"呼叫 Gemini 失敗：{e}")
                output = ""

        if output:
            st.subheader("✅ 生成結果")
            st.write(output)

            if save_history:
                st.session_state.history.insert(
                    0,
                    {
                        "ts": now_str(),
                        "model": MODEL_NAME,
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens,
                        "output_language": output_language,
                        "constraints": constraints,
                        "total_text_chars": total_chars,
                        "images_count": len(images),
                        "output": output,
                    }
                )

            st.divider()
            st.subheader("⬇️ 匯出")
            export_payload = {
                "meta": {
                    "ts": now_str(),
                    "model": MODEL_NAME,
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "output_language": output_language,
                },
                "evidence": {
                    "attachments_report": attachments_report,
                    "total_text_chars": total_chars,
                    "images_count": len(images),
                },
                "input": {
                    "constraints": constraints,
                    "notes": notes,
                },
                "output": output,
            }

            st.download_button(
                "下載 JSON（含輸入/證據摘要/輸出）",
                data=json.dumps(export_payload, ensure_ascii=False, indent=2),
                file_name=f"voice_analysis_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
            st.download_button(
                "下載 TXT（只含輸出）",
                data=output,
                file_name=f"voice_analysis_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain; charset=utf-8",
                use_container_width=True
            )

st.divider()
st.subheader("📚 Session History（本次瀏覽器期間）")
if st.session_state.history:
    for i, item in enumerate(st.session_state.history[:10], start=1):
        with st.expander(f"{i}. {item['ts']} | {item['model']} | temp={item['temperature']} | img={item['images_count']}"):
            st.write(item["output"])
else:
    st.caption("尚無紀錄。")
