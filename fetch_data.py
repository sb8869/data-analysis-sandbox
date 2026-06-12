"""
One-time setup script: downloads both datasets from UCI and saves them
as local CSV files in data/.

WHY THIS EXISTS
---------------
Both notebooks originally pulled their data live from UCI's servers via
the `ucimlrepo` package on every run. That works fine in Jupyter/VS Code
(which have internet access), but creates problems for:
  - Spyder environments where installing/using ucimlrepo is awkward
  - Cloud notebook environments where outbound access to UCI may be
    restricted or slow
  - Live demos, where a flaky UCI connection could fail the very first cell

Running this script ONCE produces two CSV files that the notebooks load
directly with pandas. No internet access to UCI is needed afterwards.

USAGE
-----
    pip install ucimlrepo pandas
    python fetch_data.py

This creates:
    data/online_news_popularity.csv   (~40K rows, 61 columns)
    data/adult_income.csv             (~49K rows, 15 columns)

These CSVs are committed to the repository, so most people will never
need to run this script. It's provided for reproducibility, or in case
the data needs to be regenerated.
"""

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo


def fetch_and_save(uci_id, output_path, dataset_name):
    print(f"Fetching {dataset_name} (UCI id={uci_id})...")
    dataset = fetch_ucirepo(id=uci_id)

    X = dataset.data.features
    y = dataset.data.targets

    # Known quirk: ucimlrepo can return column names with leading/trailing
    # whitespace (e.g., ' shares' instead of 'shares'). Strip them so the
    # CSV header is clean.
    X.columns = X.columns.str.strip()
    y.columns = y.columns.str.strip()

    df = pd.concat([X, y], axis=1)
    df.columns = df.columns.str.strip()

    df.to_csv(output_path, index=False)
    print(f"  Saved {df.shape[0]:,} rows x {df.shape[1]} columns -> {output_path}")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    fetch_and_save(332, "data/online_news_popularity.csv", "Online News Popularity")
    fetch_and_save(2, "data/adult_income.csv", "Adult / Census Income")

    print("\nDone! Both CSVs are saved in the data/ folder.")
    print("The notebooks will now load from these local files automatically.")
