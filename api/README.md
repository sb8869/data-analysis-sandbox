# Model API

A small FastAPI service that exposes the two trained models from the
analysis notebooks as REST endpoints.

| Endpoint | Notebook | Description |
|---|---|---|
| `POST /predict/article-shares` | `online_news_regression_analysis.ipynb` | Predicts expected social-media share count for an article from its content/metadata features |
| `POST /predict/income-bracket` | `adult_income_classification_analysis.ipynb` | Predicts whether an individual's income is likely `<=50K` or `>50K` from demographic/employment attributes |
| `GET /health` | — | Reports whether each model artifact loaded successfully |

---

## 1. Generate the model artifacts

The API loads pre-trained models from `model_artifacts/*.joblib`. These
are **not** committed to the repo (they're generated, not source) — you
create them by running one new cell at the end of each notebook:

- In `online_news_regression_analysis.ipynb`, run the cell titled
  **"CELL 6-A2: Save the best model as a deployable artifact"** (near the
  end, after the test-set evaluation cell). This writes
  `api/model_artifacts/regression_model.joblib`.
- In `adult_income_classification_analysis.ipynb`, run the equivalent
  **"CELL 6-A2"** cell. This writes
  `api/model_artifacts/classification_model.joblib`.

Both notebooks must be run from the repo root (same requirement as for
`data/`) so the relative path `api/model_artifacts/` resolves correctly.

The API will start even if one or both artifacts are missing — the
corresponding endpoint just returns `503` with a message telling you which
notebook cell to run.

---

## 2. Install API dependencies

```bash
cd api
pip install -r requirements.txt
```

(This is separate from the root `requirements.txt` so the API can be run
or deployed without the full notebook/EDA stack.)

---

## 3. Run the API

```bash
cd api
uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000/docs** for interactive Swagger UI — every
field has a description and a sensible default, so you can hit
**"Try it out" → "Execute"** on either endpoint with no changes and get a
real prediction back.

---

## 4. Example requests

### Article share prediction
```bash
curl -X POST http://127.0.0.1:8000/predict/article-shares \
  -H "Content-Type: application/json" \
  -d '{"n_tokens_content": 1200, "num_imgs": 8, "data_channel_is_tech": 1}'
```
Any omitted fields fall back to the defaults shown in `/docs` (representing
a "typical" article). Response:
```json
{
  "predicted_shares": 1842.3,
  "predicted_log_shares": 7.52,
  "model_used": "Random Forest (tuned)"
}
```

### Income bracket prediction
```bash
curl -X POST http://127.0.0.1:8000/predict/income-bracket \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "education": "Masters", "occupation": "Exec-managerial", "hours_per_week": 55}'
```
Response:
```json
{
  "predicted_income": ">50K",
  "probability_over_50k": 0.81,
  "model_used": "LightGBM (tuned)"
}
```

---

## Notes

- `predicted_shares` is the inverse-transformed prediction (the model is
  trained on `log(shares + 1)`; `predicted_log_shares` is the raw,
  log-scale value).
- `probability_over_50k` is provided alongside the binary
  `predicted_income` so callers can apply their own threshold — see the
  Business Requirements Document for why this matters for the
  "outreach prioritization" use case described there.
- `model_used` reflects whichever model won on the test set when you ran
  the notebook (e.g., `"Random Forest (tuned)"`, `"LightGBM (tuned)"`) —
  this can change if you re-run the tuning cells, since
  `RandomizedSearchCV` involves some randomness.
