# 🤖 AI Projects Portfolio

[![Open in Streamlit.](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-projects-portfolio.streamlit.app)

url:https://ai-projects-portfolio.streamlit.app

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
# clone
git clone https://github.com/github-username/ai-projects-portfolio.git
cd ai-projects-portfolio

# (recommended) use a fresh env
# conda create -n ai-projects python=3.10 -y && conda activate ai-projects

# install deps (root requirements.txt)
pip install -r requirements.txt

# run app
streamlit run ai-resume-matcher/app.py


Then open http://localhost:8501

Tip: If PDF/DOCX parsing fails locally, install parsers (already in requirements):

pip install pymupdf python-docx

📂 Repository Structure

ai-projects-portfolio/
├─ ai-resume-matcher/
│  ├─ app.py                  # Streamlit app
│  ├─ requirements.txt        # (optional copy for this subapp)
│  └─ test_files/             # sample resumes & JDs
├─ pages/                     # (optional) multipage wrappers
│  ├─ 1_Resume_Matcher.py     # links into the matcher app
│  ├─ 2_Interview_Generator.py (coming soon)
│  └─ 3_Job_Skill_Extractor.py (coming soon)
├─ Home.py                    # portfolio landing page (for multipage)
├─ requirements.txt           # ✅ used by Streamlit Cloud
└─ README.md

☁️ Deploying on Streamlit Cloud
You can make each project accessible via its own public URL.

Option A – Single app (this matcher only)

Repo: kshirazi5/ai-projects-portfolio

Branch: main

Main file path: ai-resume-matcher/app.py

Option B – Multipage portfolio

Make Home.py your landing page

Put each project as a page under /pages

Main file path: Home.py

🧪 Sample Files
Use the included test files to try the matcher quickly:

ai-resume-matcher/test_files/resume1.txt ↔ job1.txt (Data Analyst)

ai-resume-matcher/test_files/resume2.txt ↔ job2.txt (ML Engineer)

Cross-match them to see lower scores.

🛠 Tech Summary
Languages: Python, SQL

Libraries: Streamlit, sentence-transformers, scikit-learn, pandas, numpy, PyMuPDF, python-docx

Concepts: Embeddings, cosine similarity, TF-IDF keyword weighting, simple paragraph alignment

🐞 Troubleshooting
“Main module does not exist” on Streamlit Cloud:
Update the Main file path to ai-resume-matcher/app.py (avoid spaces in folder names).

Deps won’t install on Cloud:
Keep a root requirements.txt with:

streamlit>=1.28
sentence-transformers>=2.2.2
pymupdf>=1.23.0
python-docx>=0.8.11
pandas>=1.5
numpy>=1.23
scikit-learn>=1.1

PDF/DOCX text is empty: Some documents are scanned images (no text layer). Convert to text-based PDF or paste text directly.

📬 Contact
GitHub: https://github.com/kshirazi5

LinkedIn: https://www.linkedin.com/in/kshirazi5

⭐ If you find this useful, please star the repo!

### What to do next (1 min)
- Replace `https://YOUR-APP-URL.streamlit.app` with your live Streamlit URL.
- Save this as your root `README.md` and push:
  
  git add README.md
  git commit -m "docs: polished README with live badge and run instructions"
  git push
