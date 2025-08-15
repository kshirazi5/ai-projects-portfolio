# AI Projects Portfolio

An interactive **Streamlit-based portfolio** featuring multiple AI demos that showcase data science, NLP, and machine learning techniques. 

---

## 🚀 Live App  
Explore it online: [https://kshiraziai.streamlit.app](https://kshiraziai.streamlit.app)

---

## 📂 Projects Overview

| Project | Description |
|---------|-------------|
| **Resume ↔ Job Matcher** | Upload a resume and job description to compute SBERT-based semantic similarity, keyword overlap, and paragraph matching insightfully. |
| **JD Summarizer & Skill Gap Highlighter** | Quickly identifies resume gaps by computing TF-IDF similarity and recommending missing keywords from a job description. |
| **Anomaly Detection Sandbox** | Upload any CSV dataset to detect outliers using Isolation Forests and explore results interactively. |
| **Time-Series Forecaster** | Upload time-series data to visualize, apply moving averages, and generate basic flat forecasts. |
| **Churn Predictor Playground** | Train a Random Forest to predict customer churn interactively. Customize test size, max depth, and view performance metrics. |
| **Topic Explorer** | Extract latent topics from a corpus using NMF and preview document-topic distributions. |

---

## 🛠 Getting Started

Clone, install dependencies, and run locally:

```bash
git clone https://github.com/***/ai-projects-portfolio.git
cd ai-projects-portfolio
pip install -r requirements.txt
streamlit run app.py
