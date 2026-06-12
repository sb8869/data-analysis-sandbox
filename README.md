# Machine Learning Analysis — Regression & Classification

This repository contains two end-to-end machine learning notebooks, each covering the full analysis lifecycle: exploratory data analysis, assumptions validation, train/test splitting, cross-validation, hyperparameter tuning, and results validation.

| Notebook | Task | Dataset |
|---|---|---|
| `online_news_regression_analysis.ipynb` | Regression | [UCI Online News Popularity](https://archive.ics.uci.edu/dataset/332/online+news+popularity) |
| `adult_income_classification_analysis.ipynb` | Classification | [UCI Adult / Census Income](https://archive.ics.uci.edu/dataset/2/adult) |

Both datasets are included in this repository as CSV files (see **Data Setup** below) — no internet access to UCI is required to run either notebook.

---

## 📦 Prerequisites

Python 3.9+ required. Check with: `python --version`

```bash
pip install -r requirements.txt
```

---

## 🗂️ Data Setup

The `data/` folder contains pre-downloaded copies of both datasets:

```
data/online_news_popularity.csv   (~40K rows, 61 columns)
data/adult_income.csv             (~49K rows, 15 columns)
```

Both notebooks load directly from these CSVs (`pd.read_csv('data/...')`), so they:
- Don't depend on UCI's servers being reachable
- Run identically across Jupyter, VS Code, Spyder, and cloud notebook environments
- Run faster (no download on every execution)

**Important**: run notebooks from the repository root, so the relative path `data/...` resolves correctly. In Jupyter/VS Code this is the default if you open the project folder normally. In Spyder, make sure the working directory (top-right of the IDE) is set to the repo root before running.

### Regenerating the CSVs (optional)

The CSVs are committed to the repo, so most people will never need this. If you do want to regenerate them from UCI directly:

```bash
pip install ucimlrepo
python fetch_data.py
```

This re-downloads both datasets and overwrites the files in `data/`.

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

### Spyder
1. Open the project folder in Spyder so the working directory is the repo root
2. Open either `.ipynb` file (Spyder 6+ has built-in notebook support) or convert to a `.py` script
3. Run all cells

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

### Speeding things up while learning
- In **Cell 4-B**, reduce the SVC sample size (`8000` → `2000`)
- In **Cells 5-A through 5-D**, reduce `n_iter` (e.g., `20` → `5`) and `cv` (e.g., `3` → `2`)

---

## 📁 Repository Structure

```
.
├── README.md
├── requirements.txt
├── fetch_data.py
├── data/
│   ├── online_news_popularity.csv
│   └── adult_income.csv
├── online_news_regression_analysis.ipynb
└── adult_income_classification_analysis.ipynb
```
