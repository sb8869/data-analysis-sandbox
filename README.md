# Online News Popularity — Regression Analysis
## Setup & Running Guide

---

## 📦 Prerequisites

Python 3.9+ required. Check with: `python --version`

### Install dependencies

Using a requirements file:
```bash
pip install -r requirements.txt
```

---

## 🖥️ Running Locally

### Option A: Jupyter Notebook (Recommended for learning)
```bash
cd online_news_popularity
jupyter notebook online_news_regression_analysis.ipynb
```
Then open the URL that appears in terminal (e.g., http://localhost:8888)

### Option B: VS Code
1. Open VS Code
2. Install extension: **Jupyter** (by Microsoft)
3. Open `online_news_regression_analysis.ipynb`
4. Click **Select Kernel** → choose your Python environment
5. Click **Run All** or run cells one by one with Shift+Enter

### Option C: Spyder
1. Open Spyder
2. In the file browser (left panel), navigate to this folder
3. Double-click the `.ipynb` file — Spyder opens it in notebook mode
4. Or: copy cell content into a `.py` script and run via F5

---

## ☁️ Running on Cloud Platforms

### Google Colab (GCP)
1. Go to https://colab.research.google.com
2. File → Upload notebook → select `online_news_regression_analysis.ipynb`
3. All packages are pre-installed on Colab (just run cells)
4. OR open directly from GitHub if you push this repo

### Azure Machine Learning
1. Go to Azure ML Studio → Notebooks
2. Upload this `.ipynb` file
3. Select a compute instance (Standard_DS3_v2 is sufficient)
4. Run all cells

### AWS SageMaker
1. Go to AWS Console → SageMaker → Notebook Instances
2. Create instance → Open JupyterLab
3. Upload `online_news_regression_analysis.ipynb`
4. Select kernel: `conda_python3`
5. Run all cells

---

## 📁 Outputs

The notebook generates these plots:
- `eda_target_distribution.png` — shares distribution before/after log transform
- `eda_feature_distributions.png` — key feature histograms
- `eda_boxplots.png` — outlier visualization
- `eda_correlation_heatmap.png` — feature correlation matrix
- `eda_publishing_day.png` — shares by day of week
- `eda_channels.png` — shares by content category
- `assumption1_linearity.png`
- `assumption3_homoscedasticity.png`
- `assumption4_normality.png`
- `assumption5_vif.png`
- `split_verification.png`
- `cv_model_comparison.png`
- `tuning_ridge.png`
- `results_predicted_vs_actual.png`
- `results_residual_analysis.png`
- `results_feature_importance.png`

---

## ⏱️ Expected Runtime

| Section | Approx. Time |
|---|---|
| EDA | ~30 seconds |
| Assumptions | ~2 minutes |
| Train/Test Split | ~5 seconds |
| Cross-Validation | ~5-10 minutes |
| Hyperparameter Tuning | ~10-15 minutes |
| Results Validation | ~2 minutes |

**Total: ~20-30 minutes on a standard laptop**

Tip: Set `n_iter=10` in RandomizedSearchCV cells to speed things up during learning.
