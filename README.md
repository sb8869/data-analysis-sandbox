# Machine Learning Analysis — Regression & Classification

This repository contains two end-to-end machine learning notebooks, each covering the full analysis lifecycle: exploratory data analysis, assumptions validation, train/test splitting, cross-validation, hyperparameter tuning, and results validation.

| Notebook | Task | Dataset |
|---|---|---|
| `online_news_regression_analysis.ipynb` | Regression | [UCI Online News Popularity](https://archive.ics.uci.edu/dataset/332/online+news+popularity) |
| `adult_income_classification_analysis.ipynb` | Classification | [UCI Adult / Census Income](https://archive.ics.uci.edu/dataset/2/adult) |

Both notebooks download their datasets automatically via `ucimlrepo` — no manual downloads needed.

---

## 📦 Prerequisites

Python 3.9+ required. Check with: `python --version`

```bash
pip install -r requirements.txt
```

---

## 🖥️ Running Locally

### Jupyter Notebook
```bash
jupyter notebook
```
Then open either notebook from the Jupyter file browser.

### VS Code
1. Install the **Jupyter** extension (by Microsoft)
2. Open either `.ipynb` file
3. Select a Python kernel, then **Run All**

---

## 📓 Notebook 1: Online News Popularity (Regression)

**Goal**: Predict the number of social media shares an article will receive based on its content and metadata.

**Models compared**: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting

**Highlights**:
- Built-in feature dictionary — every column name (e.g., `kw_avg_avg`, `LDA_02`) is mapped to a plain-English description, shown throughout the notebook
- Log-transform applied to the heavily right-skewed target (`shares`)
- Full assumptions validation (linearity, homoscedasticity, normality, multicollinearity, independence)

**Expected runtime**: ~20-30 minutes

---

## 📓 Notebook 2: Adult Income (Classification)

**Goal**: Predict whether a person's annual income exceeds $50K based on census attributes.

**Models compared**: SVC, Random Forest, XGBoost, LightGBM

**Highlights**:
- Same feature-dictionary pattern as Notebook 1
- Stratified train/test split and stratified cross-validation (preserves the ~76%/24% class ratio)
- ColumnTransformer pipeline (StandardScaler + OneHotEncoder) shared across all 4 models
- Confusion matrices, ROC curves, and feature importance comparisons

**Expected runtime**: ~20-30 minutes (SVC hyperparameter tuning is the slowest step)
