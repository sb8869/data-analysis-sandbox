"""
Generates MOCK model artifacts with the correct schema/shape so the API can
be tested end-to-end without the real trained models.

These are placeholders only -- run the "CELL 6-A2" cells in each notebook
(with real data) to produce the actual artifacts, which will overwrite these.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

rng = np.random.default_rng(42)

# ============================================================
# Mock regression artifact
# ============================================================
REGRESSION_FEATURES = [
    'n_tokens_title', 'n_tokens_content', 'n_unique_tokens', 'n_non_stop_words',
    'n_non_stop_unique_tokens', 'num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos',
    'average_token_length', 'num_keywords', 'data_channel_is_lifestyle',
    'data_channel_is_entertainment', 'data_channel_is_bus', 'data_channel_is_socmed',
    'data_channel_is_tech', 'data_channel_is_world', 'kw_min_min', 'kw_max_min', 'kw_avg_min',
    'kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg', 'kw_avg_avg',
    'self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_sharess',
    'weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday',
    'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday', 'is_weekend',
    'LDA_00', 'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04', 'global_subjectivity',
    'global_sentiment_polarity', 'global_rate_positive_words', 'global_rate_negative_words',
    'rate_positive_words', 'rate_negative_words', 'avg_positive_polarity',
    'min_positive_polarity', 'max_positive_polarity', 'avg_negative_polarity',
    'min_negative_polarity', 'max_negative_polarity', 'title_subjectivity',
    'title_sentiment_polarity', 'abs_title_subjectivity', 'abs_title_sentiment_polarity',
]
assert len(REGRESSION_FEATURES) == 58

n = 200
X_reg = pd.DataFrame(rng.normal(size=(n, len(REGRESSION_FEATURES))), columns=REGRESSION_FEATURES)
y_reg = rng.normal(loc=6.0, scale=1.0, size=n)  # plausible log(shares+1) range

scaler = StandardScaler().fit(X_reg)
reg_model = LinearRegression().fit(scaler.transform(X_reg), y_reg)

joblib.dump({
    "model": reg_model,
    "scaler": scaler,
    "features": REGRESSION_FEATURES,
    "model_name": "Linear Regression (MOCK - replace via notebook)",
}, os.path.join(ARTIFACT_DIR, "regression_model.joblib"))
print("Wrote mock regression_model.joblib")

# ============================================================
# Mock classification artifact
# ============================================================
NUMERIC_FEATURES = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
CATEGORICAL_FEATURES = ['workclass', 'education', 'marital-status', 'occupation',
                         'relationship', 'race', 'sex', 'native-country']
CLASSIFICATION_FEATURES = ['age', 'workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                            'hours-per-week', 'native-country']

n = 200
X_cls = pd.DataFrame({
    'age': rng.integers(18, 70, n),
    'workclass': rng.choice(['Private', 'Self-emp-not-inc', 'Federal-gov'], n),
    'education': rng.choice(['Bachelors', 'HS-grad', 'Masters'], n),
    'marital-status': rng.choice(['Married-civ-spouse', 'Never-married', 'Divorced'], n),
    'occupation': rng.choice(['Prof-specialty', 'Sales', 'Exec-managerial'], n),
    'relationship': rng.choice(['Husband', 'Not-in-family', 'Wife'], n),
    'race': rng.choice(['White', 'Black', 'Asian-Pac-Islander'], n),
    'sex': rng.choice(['Male', 'Female'], n),
    'capital-gain': rng.choice([0, 0, 0, 5000], n),
    'capital-loss': rng.choice([0, 0, 0, 200], n),
    'hours-per-week': rng.integers(20, 60, n),
    'native-country': rng.choice(['United-States', 'Mexico'], n),
})[CLASSIFICATION_FEATURES]
y_cls = rng.integers(0, 2, n)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), NUMERIC_FEATURES),
    ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
])
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', LogisticRegression(max_iter=1000)),
])
pipeline.fit(X_cls, y_cls)

joblib.dump({
    "model": pipeline,
    "features": CLASSIFICATION_FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "model_name": "Logistic Regression (MOCK - replace via notebook)",
}, os.path.join(ARTIFACT_DIR, "classification_model.joblib"))
print("Wrote mock classification_model.joblib")
