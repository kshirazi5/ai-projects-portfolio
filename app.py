# app.py  (root)
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import NMF

# Import Resume ↔ JD Matcher from ai_projects.py
try:
    from ai_projects import resume_matcher_app
except Exception as e:
    import os, sys, textwrap
    st.title("Startup error")
    st.error(f"Couldn't import `resume_matcher_app` from `ai_projects.py`.\n\n{e}")
    st.caption("Diagnostics")
    st.code(textwrap.dedent(f"""
        cwd: {os.getcwd()}
        files in cwd: {os.listdir('.')}
        pythonpath (first 8): {sys.path[:8]}
        branch hint: (app should be on refactor/root-router)
        expected files: app.py, ai_projects.py, requirements.txt
    """))
    st.stop()

st.set_page_config(page_title="AI Projects Portfolio", page_icon="🤖", layout="wide")

st.sidebar.caption("Build: desc-v2")
st.caption(f"Loaded file: {globals().get('__file__', 'interactive')}")

# -----------------------------
# Project: JD Summarizer & Skill-Gap Highlighter
# -----------------------------
def proj_jd_summarizer():
    st.header("JD Summarizer & Skill-Gap Highlighter")
    st.markdown(
        "Quickly assess **how well your resume fits a job posting**. "
        "We compare TF-IDF features to surface **missing keywords** and show a quick similarity score."
    )

    c1, c2 = st.columns(2)
    jd = c1.text_area("Job Description", height=260, placeholder="Paste JD here…")
    resume = c2.text_area("Your Resume", height=260, placeholder="Paste your resume/profile…")
    if st.button("Analyze Match", type="primary", use_container_width=True):
        if not jd.strip() or not resume.strip():
            st.warning("Please paste both JD and Resume.")
            return
        vec = TfidfVectorizer(stop_words="english", max_features=2000)
        X = vec.fit_transform([jd, resume])
        score = cosine_similarity(X[0], X[1])[0, 0]
        st.subheader(f"Similarity: {score:.2f}")
        vocab = np.array(vec.get_feature_names_out())
        jd_top = np.asarray(X[0].toarray()).ravel().argsort()[-25:][::-1]
        rs_top = np.asarray(X[1].toarray()).ravel().argsort()[-25:][::-1]
        missing = sorted(set(vocab[jd_top]) - set(vocab[rs_top]))
        st.markdown("**Suggested keywords to add:** " + (", ".join(missing) if missing else "None"))

# -----------------------------
# Project: Anomaly Detection
# -----------------------------
def proj_anomaly_detection():
    st.header("Anomaly Detection Sandbox (Isolation Forest)")
    st.markdown(
        "Upload a dataset and flag **outliers** using Isolation Forest. "
        "Great for **fraud detection, data QA,** and **sensor anomalies**."
    )

    f = st.file_uploader("Upload CSV", type=["csv"])
    contamination = st.slider("Expected outlier proportion", 0.01, 0.20, 0.05, 0.01)
    if f is not None:
        df = pd.read_csv(f)
        st.write("**Preview**", df.head())
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.error("No numeric columns found.")
            return
        feats = st.multiselect("Features", num_cols, default=num_cols)
        if st.button("Run", type="primary"):
            X = df[feats].fillna(df[feats].median())
            iso = IsolationForest(random_state=42, contamination=contamination)
            out = iso.fit_predict(X)
            df_out = df.copy()
            df_out["anomaly"] = (out == -1).astype(int)
            st.success(f"Found {df_out['anomaly'].sum()} anomalies / {len(df_out)} rows.")
            st.dataframe(df_out.head(100), use_container_width=True)

# -----------------------------
# Project: Time-Series Forecaster
# -----------------------------
def proj_time_series():
    st.header("Time-Series Forecaster (Moving Average)")
    st.markdown(
        "Visualize your time series and produce a **simple moving-average forecast**. "
        "Useful as a **baseline** before advanced models."
    )

    f = st.file_uploader("Upload CSV", type=["csv"], key="ts")
    date_col = st.text_input("Date column", "date")
    val_col  = st.text_input("Value column", "value")
    horizon  = st.number_input("Horizon", 1, 60, 12)
    window   = st.number_input("MA window", 1, 52, 3)
    if f is not None:
        df = pd.read_csv(f)
        if date_col not in df or val_col not in df:
            st.error("Column names not found.")
            return
        ts = df[[date_col, val_col]].dropna().copy()
        ts[date_col] = pd.to_datetime(ts[date_col])
        ts = ts.sort_values(date_col).set_index(date_col)[val_col]
        st.line_chart(ts)
        ma = ts.rolling(window=window, min_periods=1).mean()
        last = float(ma.iloc[-1])
        freq = pd.infer_freq(ts.index) or "D"
        future_idx = pd.date_range(ts.index[-1], periods=horizon + 1, freq=freq)[1:]
        future = pd.Series([last] * horizon, index=future_idx, name="forecast")
        st.subheader("MA Forecast (flat)")
        st.line_chart(pd.concat([ts.rename("history"), future], axis=1))

# -----------------------------
# Project: Churn Predictor
# -----------------------------
def proj_churn_playground():
    st.header("Churn Predictor Playground (RandomForest)")
    st.markdown(
        "Train a **quick churn model** on your labeled dataset. Adjust parameters, "
        "view **accuracy** and a **classification report**."
    )

    f = st.file_uploader("Upload labeled CSV", type=["csv"], key="churn")
    if f is not None:
        df = pd.read_csv(f)
        target = st.selectbox("Target column", df.columns)
        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
        max_depth = st.slider("Max depth (None when 50)", 2, 50, 10)
        if st.button("Train", type="primary"):
            X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
            y = df[target]
            if y.dtype == "O":
                y = y.astype("category").cat.codes
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=test_size, random_state=42,
                stratify=y if y.nunique() > 1 else None
            )
            clf = RandomForestClassifier(
                n_estimators=200, random_state=42,
                max_depth=None if max_depth == 50 else max_depth
            )
            clf.fit(Xtr, ytr)
            preds = clf.predict(Xte)
            st.write("Accuracy:", round(accuracy_score(yte, preds), 4))
            st.code(classification_report(yte, preds))

# -----------------------------
# Project: Topic Explorer
# -----------------------------
def proj_topic_explorer():
    st.header("Topic Explorer (NMF)")
    st.markdown(
        "Discover **latent topics** in a small corpus using **NMF** on TF-IDF vectors. "
        "Great for quick **thematic exploration** of documents."
    )

    docs = st.text_area("Documents (one per line)", height=220)
    n_topics = st.slider("Number of topics", 2, 12, 5)
    topn = st.slider("Top words per topic", 3, 15, 8)
    if st.button("Extract Topics", type="primary"):
        corpus = [d.strip() for d in docs.splitlines() if d.strip()]
        if len(corpus) < 3:
            st.warning("Please enter at least 3 docs.")
            return
        vec = TfidfVectorizer(stop_words="english", max_features=5000)
        X = vec.fit_transform(corpus)
        nmf = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=400)
        W = nmf.fit_transform(X)
        H = nmf.components_
        vocab = np.array(vec.get_feature_names_out())
        for k in range(n_topics):
            top_idx = H[k].argsort()[-topn:][::-1]
            st.markdown(f"**Topic {k+1}:** " + ", ".join(vocab[top_idx]))
        st.dataframe(pd.DataFrame({
            "document": corpus,
            "topic": W.argmax(axis=1) + 1,
            "strength": W.max(axis=1)
        }))

# -----------------------------
# Router
# -----------------------------
def home():
    st.title("AI Projects Portfolio")
    st.markdown(
        "A multi-demo, recruiter-friendly portfolio. Use the **sidebar** to open each project. "
        "Each page includes a short description and interactive widgets."
    )

PAGES = {
    "🏠 Home": home,
    "🧾 Resume Matcher (SBERT)": resume_matcher_app,
    "📝 JD Summarizer": proj_jd_summarizer,
    "🛰️ Anomaly Detection": proj_anomaly_detection,
    "📈 Time-Series Forecast": proj_time_series,
    "🎯 Churn Predictor": proj_churn_playground,
    "🧩 Topic Explorer": proj_topic_explorer,
}

with st.sidebar:
    st.title("AI Projects")
    choice = st.radio("Navigate", list(PAGES.keys()))
    st.caption("Built with Streamlit • scikit-learn • numpy • pandas")

PAGES[choice]()
