"""
FastAPI service exposing both trained models from the analysis notebooks:

    POST /predict/article-shares  -> regression (Online News Popularity)
    POST /predict/income-bracket  -> classification (Adult / Census Income)
    GET  /health                  -> readiness check

Run locally:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.

Model artifacts are produced by the "CELL 6-A2" cells at the end of each
analysis notebook and expected at:
    model_artifacts/regression_model.joblib
    model_artifacts/classification_model.joblib
"""

import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from schemas import (
    ArticleFeatures,
    ArticleSharesResponse,
    PersonFeatures,
    IncomeBracketResponse,
)

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")
REGRESSION_PATH = os.path.join(ARTIFACT_DIR, "regression_model.joblib")
CLASSIFICATION_PATH = os.path.join(ARTIFACT_DIR, "classification_model.joblib")

# Populated at startup by the lifespan handler below.
models = {"regression": None, "classification": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load whatever artifacts are available. Missing artifacts don't crash
    # the app -- the corresponding endpoint just returns a 503 until the
    # notebook has been run to produce them.
    if os.path.exists(REGRESSION_PATH):
        models["regression"] = joblib.load(REGRESSION_PATH)
        print(f"Loaded regression model: {models['regression']['model_name']}")
    else:
        print(f"WARNING: {REGRESSION_PATH} not found. "
              f"/predict/article-shares will return 503 until it exists.")

    if os.path.exists(CLASSIFICATION_PATH):
        models["classification"] = joblib.load(CLASSIFICATION_PATH)
        print(f"Loaded classification model: {models['classification']['model_name']}")
    else:
        print(f"WARNING: {CLASSIFICATION_PATH} not found. "
              f"/predict/income-bracket will return 503 until it exists.")

    yield


app = FastAPI(
    title="Predictive Analytics API",
    description=(
        "Serves the two models from the regression and classification "
        "analysis notebooks: article share-count prediction and "
        "income-bracket classification."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["meta"])
def health():
    """Reports whether each model artifact loaded successfully."""
    return {
        "status": "ok",
        "models_loaded": {
            "regression": models["regression"] is not None,
            "classification": models["classification"] is not None,
        },
    }


@app.post("/predict/article-shares", response_model=ArticleSharesResponse, tags=["regression"])
def predict_article_shares(features: ArticleFeatures):
    """
    Predict the expected number of social media shares for an article,
    given its content and metadata features.

    The underlying model was trained on log(shares + 1); this endpoint
    applies the inverse transform (expm1) so the response is in actual
    share counts.
    """
    artifact = models["regression"]
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Regression model artifact not found. Run the "
                "'CELL 6-A2' cell at the end of "
                "online_news_regression_analysis.ipynb to generate "
                "model_artifacts/regression_model.joblib, then restart the API."
            ),
        )

    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_order = artifact["features"]

    # Build a single-row DataFrame in the exact column order the model expects.
    row = pd.DataFrame([features.model_dump()])[feature_order]

    row_scaled = scaler.transform(row)
    log_pred = model.predict(row_scaled)[0]

    # Defensive clip before the inverse transform. log(shares+1)=20 already
    # corresponds to ~485 million shares -- far beyond anything realistic --
    # so this only guards against numeric overflow (which would otherwise
    # serialize as `null` in the JSON response), not against legitimate
    # predictions.
    log_pred_clipped = float(np.clip(log_pred, -50, 20))
    shares_pred = float(np.expm1(log_pred_clipped))

    return ArticleSharesResponse(
        predicted_shares=round(shares_pred, 1),
        predicted_log_shares=round(log_pred_clipped, 4),
        model_used=artifact["model_name"],
    )


@app.post("/predict/income-bracket", response_model=IncomeBracketResponse, tags=["classification"])
def predict_income_bracket(person: PersonFeatures):
    """
    Predict whether an individual's annual income is likely above or below
    $50K, given demographic and employment attributes.

    Returns both the predicted class and the underlying probability, so
    callers can apply their own threshold if a 0.5 cutoff isn't appropriate
    for their use case (see the Business Requirements Document for the
    recommended "outreach prioritization" framing of this score).
    """
    artifact = models["classification"]
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Classification model artifact not found. Run the "
                "'CELL 6-A2' cell at the end of "
                "adult_income_classification_analysis.ipynb to generate "
                "model_artifacts/classification_model.joblib, then restart the API."
            ),
        )

    pipeline = artifact["model"]
    feature_order = artifact["features"]

    # The saved artifact is a full Pipeline (preprocessing + model), so we
    # pass it raw column values using the ORIGINAL (hyphenated) column names
    # it was trained on.
    raw = {
        "age": person.age,
        "workclass": person.workclass.value,
        "education": person.education.value,
        "marital-status": person.marital_status.value,
        "occupation": person.occupation.value,
        "relationship": person.relationship.value,
        "race": person.race.value,
        "sex": person.sex.value,
        "capital-gain": person.capital_gain,
        "capital-loss": person.capital_loss,
        "hours-per-week": person.hours_per_week,
        "native-country": person.native_country,
    }
    row = pd.DataFrame([raw])[feature_order]

    pred = int(pipeline.predict(row)[0])
    proba_over_50k = float(pipeline.predict_proba(row)[0][1])

    return IncomeBracketResponse(
        predicted_income=">50K" if pred == 1 else "<=50K",
        probability_over_50k=round(proba_over_50k, 4),
        model_used=artifact["model_name"],
    )
