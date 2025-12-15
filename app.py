import os
import json
import datetime as dt
import streamlit as st

try:
    import google.generativeai as genai
except Exception:
    genai = None


APP_TITLE = "Voice Analyzer | Persona & Voice Spec (Gemini)"
MODEL_NAME = "gemini-3-pro-preview"


# -----------------------------
# Helpers
# -----------------------------
def now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_read_text_file(uploaded_file, max_chars=120_000) -> str:
    """Read text from uploaded file with a hard cap to avoid huge prompts."""
    if not uploaded_file:
        return ""
    raw = uploaded_file.read()
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        text = str(raw)
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[TRUNCATED]"
    return text


def build_prompt(sample_texts: list[str], notes: str, constraints: str, output_language: str) -> str:
    samples_joined = "\n\n".join(sample_texts).strip()
    notes = (notes or "").strip()
    constraints = (constraints or "").strip()

    lang_rule = {
        "繁體中文": "請用繁體中文輸出。",
        "English": "Please write in English.",
        "日本語": "日本語で出力してください。",
    }.get(output_language, "請用繁體中文輸出。")

    return f"""
你是一個「語感/風格/價值觀」分析器。你的任務：從樣本文本中歸納作者的 Persona 與可執行的寫作規格（Voice Spec），並產出可直接貼入另一個專案封包的 VOICE CONTEXT。

【重要規則】
1) 「樣本文本」是證據層：你只能從樣本文本歸納語感特徵、句法、立場、節奏，不可憑空捏造作者背景。
2) 「我的筆記」是語境校準層：可以補足作者定位/價值觀脈絡；若與樣本文本衝突，必須指出衝突並給出兩套版本（V-A: 以樣本為準；V-B: 以筆記為準）。
3) 禁止引用樣本文本原句超過 25 字；不得大量抄錄。
4) 你的輸出必須可執行：要能讓另一個 AI 按規格穩定模仿寫作。
5) {lang_rule}

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
{samples_joined if samples_joined else "（未提供樣本文本，請先指出不足，並在可推論範圍內給一版『低信心』規格，提醒需要更多樣本）"}

【我的筆記（語境校準層）】
{notes if notes else "（未提供）"}
""".strip()


def call_gemini(api_key: str, prompt: str, temperature: float, max_output_tokens: int) -> str:
    if genai is None:
        raise RuntimeError("google-generativeai 未安裝或匯入失敗。請確認 requirements.txt 與安裝流程。")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)

    resp = model.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        },
    )

    # SDK 版本差異：盡量兼容
    text = getattr(resp, "text", None)
    if text:
        return text

    # fallback
    try:
        return resp.candidates[0].content.parts[0].text
    except Exception:
        return str(resp)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🧬", layout="wide")
st.title("🧬 Voice Analyzer（獨立工具）")
st.caption(f"固定模型：{MODEL_NAME}｜輸入：樣本文本 + 你的筆記｜輸出：Persona Brief + Voice Spec + 可貼入封包的 VOICE CONTEXT")

with st.sidebar:
    st.subheader("🔑 Gemini API Key")
    api_key = st.text_input("GEMINI_API_KEY", type="password", value=os.getenv("GEMINI_API_KEY", ""))

    st.divider()
    st.subheader("⚙️ 生成參數")
    temperature = st.slider("temperature", 0.0, 1.0, 0.4, 0.05)
    max_output_tokens = st.slider("max_output_tokens", 512, 8192, 4096, 256)

    st.divider()
    output_language = st.selectbox("輸出語言", ["繁體中文", "English", "日本語"], index=0)

    st.divider()
    st.subheader("📦 輸出保存")
    save_history = st.toggle("保存到本機 Session History", value=True)


col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1) 樣本文本（證據層）")
    st.write("上傳或貼上這個人的過去文章/貼文/腳本。建議至少 1,000–3,000 字，多篇更好。")

    uploads = st.file_uploader(
        "上傳 txt / md（可多檔）",
        type=["txt", "md"],
        accept_multiple_files=True
    )

    pasted = st.text_area(
        "或直接貼樣本文本（可與上傳同時用）",
        height=260,
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
        "輸出會包含：Persona Brief、Voice Spec、VOICE CONTEXT（可貼入封包）、驗收清單。\n\n"
        "注意：樣本文本越少，規格會越『低信心』。"
    )


# Prepare inputs
sample_texts = []
if uploads:
    for f in uploads:
        t = safe_read_text_file(f)
        if t:
            sample_texts.append(f"=== [FILE: {f.name}] ===\n{t}\n=== [/FILE] ===")

if pasted.strip():
    sample_texts.append(f"=== [PASTED] ===\n{pasted.strip()}\n=== [/PASTED] ===")

prompt = build_prompt(sample_texts, notes, constraints, output_language)

# history store
if "history" not in st.session_state:
    st.session_state.history = []

if run:
    if not api_key.strip():
        st.error("缺少 GEMINI_API_KEY。請在側欄貼上。")
    else:
        with st.spinner("生成中…"):
            try:
                output = call_gemini(
                    api_key=api_key.strip(),
                    prompt=prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens
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
                        "has_samples": bool(sample_texts),
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
                "input": {
                    "constraints": constraints,
                    "notes": notes,
                    "samples_count": len(sample_texts),
                },
                "output": output,
            }
            st.download_button(
                "下載 JSON（含輸入/輸出）",
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
        with st.expander(f"{i}. {item['ts']} | {item['model']} | temp={item['temperature']}"):
            st.write(item["output"])
else:
    st.caption("尚無紀錄。")
