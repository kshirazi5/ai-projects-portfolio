# 🤖 AI Projects Portfolio

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-URL.streamlit.app)

A collection of hands-on AI and data apps built with Python. Each project is self-contained, easy to run locally, and (where possible) available as a one-click web demo.

---

## 🌟 Featured App: AI Resume ↔ Job Description Matcher

**What it does:**  
Compares a resume to a job description using sentence embeddings and cosine similarity. Shows:
- ✅ Overall **match score** (0–1)
- 🔑 **Keyword insights** (found vs missing from the JD)
- 📚 **Paragraph-level alignment** (which resume sections match which parts of the JD)

**Stack:** Python, Streamlit, `sentence-transformers`, scikit-learn, PyMuPDF (PDF), python-docx (DOCX)

### Run locally
```bash
# clone
git clone https://github.com/kshirazi5/ai-projects-portfolio.git
cd ai-projects-portfolio

# (recommended) use a fresh env
# conda create -n ai-projects python=3.10 -y && conda activate ai-projects

# install deps (root requirements.txt)
pip install -r requirements.txt

# run app
streamlit run ai-resume-matcher/app.py
