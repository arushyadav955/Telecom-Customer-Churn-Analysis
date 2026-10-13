# ============================================================
#  STEP 2: Data Cleaning & Fixing Issues
#  Telco Customer Churn Project
# ============================================================

import pandas as pd
import numpy as np

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("=" * 55)
print("  STEP 2: DATA CLEANING")
print("=" * 55)

# ── 2.1 Fix TotalCharges (object → float, blank → NaN) ───
print("\n--- Before fix: TotalCharges dtype:", df["TotalCharges"].dtype)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("--- After fix : TotalCharges dtype:", df["TotalCharges"].dtype)
print("--- NaNs created (were blank strings):", df["TotalCharges"].isnull().sum())

# ── 2.2 Inspect the rows with NaN TotalCharges ───────────
print("\n--- Rows with missing TotalCharges ---")
missing_rows = df[df["TotalCharges"].isnull()]
print(missing_rows[["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]])

# ── 2.3 Fix: tenure=0 means brand new customer ───────────
# TotalCharges = 0 for them makes business sense → fill with 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)
print(f"\n--- After fillna(0): NaNs remaining: {df['TotalCharges'].isnull().sum()}")

# ── 2.4 Fix Churn column (Yes/No → 1/0) ─────────────────
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
print(f"\n--- Churn encoded: {df['Churn'].value_counts().to_dict()}")

# ── 2.5 Fix SeniorCitizen (already 0/1 but confirm) ─────
print(f"\n--- SeniorCitizen values: {df['SeniorCitizen'].value_counts().to_dict()}")

# ── 2.6 Inspect tenure = 0 rows ─────────────────────────
print(f"\n--- Customers with tenure = 0 months: {(df['tenure'] == 0).sum()}")
print(df[df["tenure"] == 0][["customerID","tenure","MonthlyCharges","TotalCharges","Contract","Churn"]].head(10))

# ── 2.7 Final shape & dtypes check ──────────────────────
print("\n--- Final dtypes after cleaning ---")
print(df.dtypes)

print("\n--- Final missing values ---")
print(df.isnull().sum().sum(), "total missing values")

# ── 2.8 Save cleaned version ────────────────────────────
df.to_csv("outputs/telco_cleaned.csv", index=False)
print("\n✅ Cleaned data saved → outputs/telco_cleaned.csv")
print(f"   Shape: {df.shape}")