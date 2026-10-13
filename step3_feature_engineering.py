# ============================================================
#  STEP 3: Feature Engineering
#  Telco Customer Churn Project
# ============================================================

import pandas as pd
import numpy as np

df = pd.read_csv("outputs/telco_cleaned.csv")

print("=" * 55)
print("  STEP 3: FEATURE ENGINEERING")
print("=" * 55)

# ── 3.1 Binary encode simple Yes/No columns ──────────────
binary_cols = [
    "Partner", "Dependents", "PhoneService",
    "PaperlessBilling"
]
for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})
    print(f"  ✅ {col} encoded")

# ── 3.2 Encode gender ────────────────────────────────────
df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
print(f"  ✅ gender encoded (Male=1, Female=0)")

# ── 3.3 Smart encode 3-value service columns ─────────────
# "No internet service" / "No phone service" → treated as "No" (0)
# These aren't truly different from No — customer just doesn't have base service
three_val_cols = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]
for col in three_val_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0,
                            "No internet service": 0,
                            "No phone service": 0})
    print(f"  ✅ {col} encoded")

# ── 3.4 Ordinal encode Contract ──────────────────────────
# Month-to-month = highest churn risk → 0
# One year = medium → 1
# Two year = lowest churn risk → 2
df["Contract"] = df["Contract"].map({
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
})
print(f"  ✅ Contract encoded (0=Month-to-month, 1=One year, 2=Two year)")

# ── 3.5 One-hot encode InternetService & PaymentMethod ───
df = pd.get_dummies(df, columns=["InternetService", "PaymentMethod"], drop_first=False)

# Convert the new bool columns to int
bool_cols = df.select_dtypes(include="bool").columns.tolist()
df[bool_cols] = df[bool_cols].astype(int)
print(f"  ✅ InternetService → one-hot encoded")
print(f"  ✅ PaymentMethod   → one-hot encoded")

# ── 3.6 Create new business-driven features ──────────────
print("\n--- Creating new features ---")

# Avg monthly spend consistency
df["AvgMonthlySpend"] = (df["TotalCharges"] /
                          df["tenure"].replace(0, 1)).round(2)
print("  ✅ AvgMonthlySpend = TotalCharges / tenure")

# Total number of services subscribed
service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies"]
df["TotalServices"] = df[service_cols].sum(axis=1)
print(f"  ✅ TotalServices = sum of all {len(service_cols)} service flags")
print(f"     Range: {df['TotalServices'].min()} – {df['TotalServices'].max()}")

# Has internet service flag
df["HasInternet"] = (df["InternetService_DSL"] | 
                     df["InternetService_Fiber optic"]).astype(int)
print("  ✅ HasInternet flag")

# Charge per service (spend efficiency)
df["ChargePerService"] = (df["MonthlyCharges"] /
                           df["TotalServices"].replace(0, 1)).round(2)
print("  ✅ ChargePerService = MonthlyCharges / TotalServices")

# Tenure groups (customer lifecycle stage)
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["New (0–12m)", "Growing (13–24m)",
            "Mature (25–48m)", "Loyal (49–72m)"]
)
print("  ✅ TenureGroup (lifecycle stage bins)")

# ── 3.7 Drop customerID (not a feature) ─────────────────
df.drop(columns=["customerID"], inplace=True)
print("\n  ✅ customerID dropped")

# ── 3.8 Summary ─────────────────────────────────────────
print("\n--- Feature Engineering Summary ---")
print(f"  Original columns : 21")
print(f"  Final columns    : {df.shape[1]}")
print(f"  Final shape      : {df.shape}")

print("\n--- New columns added ---")
new_cols = ["AvgMonthlySpend", "TotalServices",
            "HasInternet", "ChargePerService", "TenureGroup"]
for c in new_cols:
    print(f"  {c:<25} → sample values: {df[c].value_counts().head(3).to_dict()}")

print("\n--- All columns in final dataset ---")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:02d}. {col}")

# ── 3.9 Save ─────────────────────────────────────────────
df.to_csv("outputs/telco_features.csv", index=False)
print("\n✅ Feature-engineered data saved → outputs/telco_features.csv")