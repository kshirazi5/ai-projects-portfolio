
# AI Resume ↔ Job Description Matcher (Streamlit)

Upload a resume + job description (PDF/DOCX/TXT) and get:
- Overall **semantic match score** (0–1) using `sentence-transformers`
- **Keyword insights** (found/missing via TF‑IDF)
- **Paragraph similarity** table

## Quickstart
```bash
conda activate ai-projects
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501

## Requirements
- Python 3.9+
- See `requirements.txt`

## Notes
- If PDF/DOCX parsing fails, paste text manually and click **Analyze**.
- First run downloads a small embedding model (~MBs), so it may take ~10–20s.
