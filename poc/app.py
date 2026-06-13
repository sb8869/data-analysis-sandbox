"""
Streamlit POC: interactive demo for both trained models.

    Tab 1: Article Performance Predictor (regression)
    Tab 2: Income Bracket Predictor (classification)

Loads the same .joblib artifacts used by the FastAPI service in ../api/,
so this app is self-contained -- no API needs to be running.

Run locally:
    streamlit run app.py

Deploy: point Streamlit Community Cloud at this file (poc/app.py) in the repo.
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# Load model artifacts
# ============================================================

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "api", "model_artifacts")
REGRESSION_PATH = os.path.join(ARTIFACT_DIR, "regression_model.joblib")
CLASSIFICATION_PATH = os.path.join(ARTIFACT_DIR, "classification_model.joblib")


@st.cache_resource
def load_artifact(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


regression_artifact = load_artifact(REGRESSION_PATH)
classification_artifact = load_artifact(CLASSIFICATION_PATH)


# ============================================================
# Regression: feature list, descriptions, and defaults
# ============================================================
# Same 58 features / defaults as api/schemas.py. Only ~11 are exposed as
# interactive controls below; everything else uses these "typical article"
# defaults.

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

REGRESSION_DEFAULTS = {
    'n_tokens_title': 10.0, 'n_tokens_content': 500.0, 'n_unique_tokens': 0.55,
    'n_non_stop_words': 1.0, 'n_non_stop_unique_tokens': 0.7, 'num_hrefs': 10.0,
    'num_self_hrefs': 3.0, 'num_imgs': 4.0, 'num_videos': 1.0,
    'average_token_length': 4.5, 'num_keywords': 7.0,
    'data_channel_is_lifestyle': 0, 'data_channel_is_entertainment': 0,
    'data_channel_is_bus': 0, 'data_channel_is_socmed': 0,
    'data_channel_is_tech': 1, 'data_channel_is_world': 0,
    'kw_min_min': -1.0, 'kw_max_min': 500.0, 'kw_avg_min': 200.0,
    'kw_min_max': 0.0, 'kw_max_max': 700000.0, 'kw_avg_max': 250000.0,
    'kw_min_avg': 1000.0, 'kw_max_avg': 5000.0, 'kw_avg_avg': 3000.0,
    'self_reference_min_shares': 1000.0, 'self_reference_max_shares': 5000.0,
    'self_reference_avg_sharess': 3000.0,
    'weekday_is_monday': 0, 'weekday_is_tuesday': 1, 'weekday_is_wednesday': 0,
    'weekday_is_thursday': 0, 'weekday_is_friday': 0, 'weekday_is_saturday': 0,
    'weekday_is_sunday': 0, 'is_weekend': 0,
    'LDA_00': 0.2, 'LDA_01': 0.2, 'LDA_02': 0.2, 'LDA_03': 0.2, 'LDA_04': 0.2,
    'global_subjectivity': 0.45, 'global_sentiment_polarity': 0.1,
    'global_rate_positive_words': 0.04, 'global_rate_negative_words': 0.02,
    'rate_positive_words': 0.7, 'rate_negative_words': 0.3,
    'avg_positive_polarity': 0.35, 'min_positive_polarity': 0.05, 'max_positive_polarity': 0.7,
    'avg_negative_polarity': -0.25, 'min_negative_polarity': -0.5, 'max_negative_polarity': -0.1,
    'title_subjectivity': 0.3, 'title_sentiment_polarity': 0.1,
    'abs_title_subjectivity': 0.2, 'abs_title_sentiment_polarity': 0.1,
}

WEEKDAY_FIELDS = [
    'weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday',
    'weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday',
]
CHANNEL_FIELDS = {
    'Lifestyle': 'data_channel_is_lifestyle',
    'Entertainment': 'data_channel_is_entertainment',
    'Business': 'data_channel_is_bus',
    'Social Media': 'data_channel_is_socmed',
    'Tech': 'data_channel_is_tech',
    'World': 'data_channel_is_world',
}


def predict_shares(overrides: dict):
    """Build a full 58-feature row (defaults + overrides), scale, and predict."""
    row = dict(REGRESSION_DEFAULTS)
    row.update(overrides)
    df = pd.DataFrame([row])[REGRESSION_FEATURES]

    model = regression_artifact["model"]
    scaler = regression_artifact["scaler"]

    scaled = scaler.transform(df)
    log_pred = model.predict(scaled)[0]
    log_pred = float(np.clip(log_pred, -50, 20))
    return float(np.expm1(log_pred)), log_pred


# ============================================================
# Classification: feature list and category options
# ============================================================

CLASSIFICATION_FEATURES = [
    'age', 'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country',
]

WORKCLASS_OPTIONS = [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked",
]
EDUCATION_OPTIONS = [
    "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm",
    "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th",
    "Doctorate", "5th-6th", "Preschool",
]
MARITAL_STATUS_OPTIONS = [
    "Married-civ-spouse", "Divorced", "Never-married", "Separated",
    "Widowed", "Married-spouse-absent", "Married-AF-spouse",
]
OCCUPATION_OPTIONS = [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-Forces",
]
RELATIONSHIP_OPTIONS = ["Husband", "Wife", "Own-child", "Not-in-family", "Other-relative", "Unmarried"]
RACE_OPTIONS = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
SEX_OPTIONS = ["Male", "Female"]


def predict_income(values: dict):
    df = pd.DataFrame([values])[CLASSIFICATION_FEATURES]
    pipeline = classification_artifact["model"]
    pred = int(pipeline.predict(df)[0])
    proba = float(pipeline.predict_proba(df)[0][1])
    return pred, proba


# ============================================================
# Page layout
# ============================================================

st.set_page_config(page_title="Predictive Analytics POC", layout="wide")
st.title("📊 Predictive Analytics POC")
st.caption(
    "Interactive demo for the two models from the analysis notebooks. "
    "See the Business Requirements Document for the full business case "
    "behind each model."
)

tab1, tab2 = st.tabs(["📰 Article Performance Predictor", "💰 Income Bracket Predictor"])

# ------------------------------------------------------------
# Tab 1: Article Performance Predictor
# ------------------------------------------------------------
with tab1:
    st.subheader("Predict expected social-media shares for an article")
    st.write(
        "Adjust the content attributes below. All other features "
        "(sentiment scores, additional keyword stats, etc.) are held at "
        "typical-article values."
    )

    if regression_artifact is None:
        st.error(
            "Regression model artifact not found at "
            f"`{REGRESSION_PATH}`.\n\n"
            "Run the 'CELL 6-A2' cell at the end of "
            "`online_news_regression_analysis.ipynb` to generate it, then "
            "restart this app."
        )
    else:
        col1, col2 = st.columns(2)

        with col1:
            channel = st.selectbox(
                "Content channel",
                ["Other / unspecified"] + list(CHANNEL_FIELDS.keys()),
                index=list(CHANNEL_FIELDS.keys()).index("Tech") + 1,
                help="Which Mashable content channel the article belongs to.",
            )
            weekday = st.selectbox(
                "Publish day",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                index=1,
                help="Day of the week the article was published.",
            )
            n_tokens_content = st.slider(
                "Article length (words)", 0, 5000, 500, step=50,
                help="Number of words in the content.",
            )
            num_imgs = st.slider("Number of images", 0, 30, 4)
            num_videos = st.slider("Number of videos", 0, 10, 1)
            num_hrefs = st.slider("Number of links", 0, 80, 10)

        with col2:
            num_keywords = st.slider("Number of keywords", 1, 10, 7)
            kw_avg_avg = st.slider(
                "Average keyword performance (kw_avg_avg)", 0, 10000, 3000, step=100,
                help="Historical average shares earned by this article's keywords. "
                     "One of the strongest predictors in the dataset.",
            )
            self_reference_avg_sharess = st.slider(
                "Avg. shares of linked articles", 0, 20000, 3000, step=100,
                help="Average share count of other articles this one links to.",
            )
            global_subjectivity = st.slider(
                "Subjectivity (0 = objective, 1 = subjective)", 0.0, 1.0, 0.45, step=0.01,
            )
            title_sentiment_polarity = st.slider(
                "Title sentiment (negative to positive)", -1.0, 1.0, 0.1, step=0.01,
            )

        if st.button("Predict shares", type="primary"):
            overrides = {
                'n_tokens_content': float(n_tokens_content),
                'num_imgs': float(num_imgs),
                'num_videos': float(num_videos),
                'num_hrefs': float(num_hrefs),
                'num_keywords': float(num_keywords),
                'kw_avg_avg': float(kw_avg_avg),
                'self_reference_avg_sharess': float(self_reference_avg_sharess),
                'global_subjectivity': float(global_subjectivity),
                'title_sentiment_polarity': float(title_sentiment_polarity),
            }

            # Weekday -> one-hot + is_weekend
            for field in WEEKDAY_FIELDS:
                overrides[field] = 0
            weekday_field = f"weekday_is_{weekday.lower()}"
            overrides[weekday_field] = 1
            overrides['is_weekend'] = 1 if weekday in ("Saturday", "Sunday") else 0

            # Channel -> one-hot
            for field in CHANNEL_FIELDS.values():
                overrides[field] = 0
            if channel in CHANNEL_FIELDS:
                overrides[CHANNEL_FIELDS[channel]] = 1

            shares_pred, log_pred = predict_shares(overrides)
            baseline_shares, _ = predict_shares({})

            c1, c2 = st.columns(2)
            c1.metric("Predicted shares", f"{shares_pred:,.0f}")
            c2.metric(
                "vs. typical-article baseline",
                f"{shares_pred:,.0f}",
                delta=f"{shares_pred - baseline_shares:,.0f}",
            )
            st.caption(
                f"Model used: **{regression_artifact['model_name']}** "
                f"(predicted on log(shares+1) scale: {log_pred:.2f})"
            )

# ------------------------------------------------------------
# Tab 2: Income Bracket Predictor
# ------------------------------------------------------------
with tab2:
    st.subheader("Estimate likelihood of income > $50K")
    st.write(
        "This score is intended for **prioritizing outreach** for "
        "income-based assistance programs, not for gatekeeping access "
        "to services -- see the Business Requirements Document, "
        "Section 3.5, for the fairness and use-case framing behind this model."
    )

    if classification_artifact is None:
        st.error(
            "Classification model artifact not found at "
            f"`{CLASSIFICATION_PATH}`.\n\n"
            "Run the 'CELL 6-A2' cell at the end of "
            "`adult_income_classification_analysis.ipynb` to generate it, "
            "then restart this app."
        )
    else:
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 17, 90, 38)
            sex = st.radio("Sex", SEX_OPTIONS, horizontal=True)
            race = st.selectbox("Race", RACE_OPTIONS)
            native_country = st.text_input("Native country", value="United-States")

        with col2:
            education = st.selectbox("Education", EDUCATION_OPTIONS, index=EDUCATION_OPTIONS.index("HS-grad"))
            marital_status = st.selectbox("Marital status", MARITAL_STATUS_OPTIONS)
            relationship = st.selectbox("Relationship", RELATIONSHIP_OPTIONS)
            workclass = st.selectbox("Workclass", WORKCLASS_OPTIONS)
            occupation = st.selectbox("Occupation", OCCUPATION_OPTIONS, index=OCCUPATION_OPTIONS.index("Prof-specialty"))

        hours_per_week = st.slider("Hours worked per week", 1, 99, 40)
        c1, c2 = st.columns(2)
        capital_gain = c1.number_input("Capital gain (USD)", min_value=0.0, value=0.0, step=100.0)
        capital_loss = c2.number_input("Capital loss (USD)", min_value=0.0, value=0.0, step=100.0)

        if st.button("Predict income bracket", type="primary"):
            values = {
                'age': age,
                'workclass': workclass,
                'education': education,
                'marital-status': marital_status,
                'occupation': occupation,
                'relationship': relationship,
                'race': race,
                'sex': sex,
                'capital-gain': capital_gain,
                'capital-loss': capital_loss,
                'hours-per-week': hours_per_week,
                'native-country': native_country,
            }
            pred, proba = predict_income(values)

            label = ">$50K" if pred == 1 else "<=$50K"
            st.metric("Predicted income bracket", label)
            st.progress(proba, text=f"Probability of >$50K: {proba:.1%}")
            st.caption(f"Model used: **{classification_artifact['model_name']}**")
            