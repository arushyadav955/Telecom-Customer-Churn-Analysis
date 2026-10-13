# ============================================================
#  STEP 1: Load & Explore the Data
#  Telco Customer Churn Project
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── 1.1 Load ─────────────────────────────────────────────
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("=" * 55)
print("  DATASET OVERVIEW")
print("=" * 55)
print(f"  Rows       : {df.shape[0]:,}")
print(f"  Columns    : {df.shape[1]}")
print(f"  Memory     : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

# ── 1.2 Column types ─────────────────────────────────────
print("\n--- Column Info ---")
print(df.dtypes)

# ── 1.3 Missing values ───────────────────────────────────
print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "  No nulls found (yet!)")

# ── 1.4 Target distribution ──────────────────────────────
print("\n--- Churn Distribution ---")
churn_counts = df["Churn"].value_counts()
churn_pct    = df["Churn"].value_counts(normalize=True) * 100
for label in churn_counts.index:
    print(f"  {label:5s} → {churn_counts[label]:,} customers  ({churn_pct[label]:.1f}%)")

# ── 1.5 Numeric summary ──────────────────────────────────
print("\n--- Numeric Columns Summary ---")
print(df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(2))

# ── 1.6 Unique values per categorical column ─────────────
print("\n--- Categorical Columns (unique values) ---")
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols.remove("customerID")
for col in cat_cols:
    print(f"  {col:<20} → {df[col].unique().tolist()}")