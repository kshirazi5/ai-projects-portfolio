
import re, io
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Optional parsers (guard-import so app still loads)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import docx
except Exception:
    docx = None

from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="AI Resume ↔ JD Matcher", page_icon="🧠", layout="wide")
st.title("🧠 AI Resume ↔ Job Description Matcher")
st.caption("Upload or paste both sides, then click **Analyze**. If a file can’t be parsed, paste the text instead.")

def read_pdf(file_bytes: bytes) -> str:
    if not fitz:
        raise RuntimeError("PDF support requires PyMuPDF. Run: pip install pymupdf")
    text_parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)

def read_docx(file_bytes: bytes) -> str:
    if not docx:
        raise RuntimeError("DOCX support requires python-docx. Run: pip install python-docx")
    f = io.BytesIO(file_bytes)
    d = docx.Document(f)
    return "\n".join(p.text for p in d.paragraphs)

def read_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

def read_any(uploaded_file) -> str:
    if not uploaded_file:
        return ""
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.read()
    if suffix == ".pdf":
        return read_pdf(data)
    if suffix in [".docx", ".doc"]:
        return read_docx(data)
    if suffix in [".txt", ".md"]:
        return read_txt(data)
    return read_txt(data)

def normalize(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

left, right = st.columns(2)
with left:
    resume_file = st.file_uploader("Resume (PDF/DOCX/TXT)", type=["pdf","docx","doc","txt","md"])
    resume_text_manual = st.text_area("…or paste Resume text", height=180, placeholder="Paste resume here (optional)")
with right:
    jd_file = st.file_uploader("Job Description (PDF/DOCX/TXT)", type=["pdf","docx","doc","txt","md"])
    jd_text_manual = st.text_area("…or paste JD text", height=180, placeholder="Paste JD here (optional)")

def get_text(label, uploaded, pasted):
    txt = ""
    err = None
    if uploaded is not None:
        try:
            txt = read_any(uploaded)
        except Exception as e:
            err = f"{label} parse error: {e}"
    if not txt and pasted.strip():
        txt = pasted
    return normalize(txt), err

resume_text, resume_err = get_text("Resume", resume_file, resume_text_manual)
jd_text, jd_err = get_text("JD", jd_file, jd_text_manual)
if resume_err: st.warning(resume_err)
if jd_err: st.warning(jd_err)

st.divider()
run = st.button("🔎 Analyze", type="primary", use_container_width=True)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def keyword_tables(jd_text: str, resume_text: str, top_k=30):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
    X = vec.fit_transform([jd_text])
    vocab = np.array(vec.get_feature_names_out())
    scores = X.toarray()[0]
    idx = np.argsort(-scores)[:top_k]
    terms = [(vocab[i], float(scores[i])) for i in idx if scores[i] > 0]
    res_low = resume_text.lower()
    present = [(t, s) for t, s in terms if t.lower() in res_low]
    missing = [(t, s) for t, s in terms if t.lower() not in res_low]
    return terms, present, missing

if run:
    if not resume_text or not jd_text:
        st.error("Please provide BOTH a resume and a job description (upload or paste).")
        st.stop()

    with st.spinner("Loading model & computing similarity… (first run may take ~10–20s)"):
        model = load_model()
        res_emb = model.encode(resume_text, convert_to_tensor=True)
        jd_emb  = model.encode(jd_text, convert_to_tensor=True)
        score = float(util.cos_sim(res_emb, jd_emb).item())

    st.subheader("✨ Overall Match Score")
    st.metric("Cosine Similarity (0–1)", f"{score:.2f}")
    st.caption("Rule of thumb: ~0.60+ = decent, ~0.75+ = strong (depends on content length/quality).")

    st.subheader("🔑 Keyword Insights (from JD TF-IDF)")
    try:
        terms, present, missing = keyword_tables(jd_text, resume_text, top_k=30)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Found in Resume**")
            st.dataframe(pd.DataFrame(present, columns=["Keyword","JD Importance"]), use_container_width=True)
        with c2:
            st.markdown("**Missing from Resume**")
            st.dataframe(pd.DataFrame(missing, columns=["Keyword","JD Importance"]), use_container_width=True)
    except Exception as e:
        st.info(f"Keyword table unavailable: {e}")

    st.subheader("📚 Paragraph Similarity (quick view)")
    res_paras = [p for p in re.split(r"\n\s*\n", resume_text) if p.strip()]
    jd_paras  = [p for p in re.split(r"\n\s*\n", jd_text) if p.strip()]
    if res_paras and jd_paras:
        jd_embs = model.encode(jd_paras, convert_to_tensor=True)
        rows = []
        for i, rp in enumerate(res_paras[:20]):
            r_emb = model.encode(rp, convert_to_tensor=True)
            sims = util.cos_sim(r_emb, jd_embs).cpu().numpy().flatten()
            j_idx = int(np.argmax(sims))
            rows.append({
                "Resume paragraph #": i+1,
                "Best JD paragraph #": j_idx+1,
                "Similarity": float(sims[j_idx])
            })
        st.dataframe(pd.DataFrame(rows).round(2), use_container_width=True)
    else:
        st.caption("Not enough paragraphs to compare.")

    st.success("Done. Upload new files or tweak text and click **Analyze** again.")
else:
    st.info("Upload or paste both Resume and JD, then click **Analyze**.")
