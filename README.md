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
