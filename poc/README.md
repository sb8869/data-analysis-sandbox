# Interactive Demo (Streamlit POC)

A two-tab Streamlit app that lets you interact with both trained models
directly in a browser.

| Tab | Notebook | What it does |
|---|---|---|
| 📰 Article Performance Predictor | `online_news_regression_analysis.ipynb` | Adjust ~11 of the most impactful content attributes (channel, publish day, length, images, keyword performance, etc.) and get a predicted share count. Remaining 47 features use "typical article" defaults. |
| 💰 Income Bracket Predictor | `adult_income_classification_analysis.ipynb` | Fill in all 12 demographic/employment fields and get a predicted income bracket plus the underlying probability. |

This app loads the same `.joblib` artifacts as the API (`../api/model_artifacts/`)
directly — no API server needs to be running.

---

## 1. Generate the model artifacts

Same prerequisite as the API: run the **"CELL 6-A2"** cell at the end of
each notebook to produce:
```
api/model_artifacts/regression_model.joblib
api/model_artifacts/classification_model.joblib
```

---

## 2. Install dependencies and run

```bash
cd poc
pip install -r requirements.txt
streamlit run app.py
```

This opens the app in your browser (default: http://localhost:8501).

**Note on caching**: artifacts are loaded once via `st.cache_resource`. If
you generate the `.joblib` files *after* starting the app, restart it (or
press "Rerun" with cache cleared from Streamlit's menu) for the new files
to be picked up.

---

## 3. Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (including the `.joblib` artifacts — see the
   note in the main `requirements.txt` about pinned versions).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your
   GitHub account, and create a new app.
3. Set:
   - **Repository**: your repo
   - **Branch**: `main`
   - **Main file path**: `poc/app.py`
   - **Requirements file**: Streamlit Cloud should auto-detect
     `poc/requirements.txt` since it's in the same folder as `app.py`.
4. Deploy — you'll get a public `*.streamlit.app` URL suitable for sharing
   as a portfolio link.

---

## Notes

- The Article Predictor's "vs. typical-article baseline" delta shows how
  much your specific inputs move the prediction relative to an article
  with every feature at its default value — useful for seeing which
  adjustments matter most.
- The Income Bracket Predictor's framing (probability rather than a hard
  yes/no) intentionally matches the BRD's "outreach prioritization"
  use case (Section 3.5) rather than a gatekeeping decision.
- If `model_used` shows something like `"XGBoost (tuned)"` or
  `"LightGBM (tuned)"`, make sure the corresponding package is installed
  (see the commented-out lines in `requirements.txt`) — `joblib.load()`
  needs it to reconstruct the pipeline even though this app never calls
  it directly for training.
  