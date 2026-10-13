# ============================================================
#  STEP 6: CLV Estimation + Business Recommendations
#  Telco Customer Churn Project
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("=" * 55)
print("  STEP 6: CLV + BUSINESS RECOMMENDATIONS")
print("=" * 55)

# ── Load ─────────────────────────────────────────────────
df = pd.read_csv("outputs/telco_with_predictions.csv")
print(f"\n  Loaded {df.shape[0]:,} rows ✅")

# ─────────────────────────────────────────────────────────
#  PART A: CLV
# ─────────────────────────────────────────────────────────
MARGIN = 0.30
df["monthly_churn_prob"]       = np.clip(df["churn_proba_gb"] / 12, 0.005, 0.99)
df["expected_lifetime_months"] = (1 / df["monthly_churn_prob"]).round(1)
df["CLV"]                      = (df["MonthlyCharges"] * MARGIN *
                                   df["expected_lifetime_months"]).round(2)

clv_min = df["CLV"].min()
clv_25  = df["CLV"].quantile(0.25)
clv_50  = df["CLV"].quantile(0.50)
clv_75  = df["CLV"].quantile(0.75)
clv_max = df["CLV"].max()

df["CLV_Segment"] = pd.cut(
    df["CLV"],
    bins  =[clv_min - 1, clv_25, clv_50, clv_75, clv_max + 1],
    labels=["Low Value", "Mid Value", "High Value", "Elite"]
)

print("\n--- CLV Summary ---")
print(f"  Average CLV         : ${df['CLV'].mean():,.2f}")
print(f"  Median CLV          : ${df['CLV'].median():,.2f}")
print(f"  Max CLV             : ${df['CLV'].max():,.2f}")
print(f"  Min CLV             : ${df['CLV'].min():,.2f}")
print(f"  Total Portfolio CLV : ${df['CLV'].sum():,.2f}")

print("\n--- CLV by Segment ---")
clv_seg = df.groupby("CLV_Segment", observed=True).agg(
    Customers   =("CLV",            "count"),
    Avg_CLV     =("CLV",            "mean"),
    Total_CLV   =("CLV",            "sum"),
    Avg_Monthly =("MonthlyCharges", "mean"),
    Avg_Tenure  =("tenure",         "mean"),
    Churn_Rate  =("Churn",          "mean"),
).round(2)
print(clv_seg.to_string())

# ─────────────────────────────────────────────────────────
#  PART B: Business Segmentation
# ─────────────────────────────────────────────────────────
def assign_segment(row):
    clv  = row["CLV_Segment"]
    prob = row["churn_proba_gb"]
    if clv == "Elite"      and prob <  0.30: return "Champions"
    if clv == "Elite"      and prob >= 0.30: return "High Value at Risk"
    if clv == "High Value" and prob <  0.30: return "Loyal High Value"
    if clv == "High Value" and prob >= 0.30: return "High Value at Risk"
    if clv == "Mid Value"  and prob <  0.40: return "Stable Mid Value"
    if clv == "Mid Value"  and prob >= 0.40: return "Mid Value Slipping"
    if clv == "Low Value"  and prob >= 0.60: return "Likely to Churn"
    return "Low Value Stable"

df["Business_Segment"] = df.apply(assign_segment, axis=1)

print("\n--- Business Segment Distribution ---")
seg_summary = df.groupby("Business_Segment").agg(
    Customers      =("CLV",            "count"),
    Avg_CLV        =("CLV",            "mean"),
    Avg_ChurnProba =("churn_proba_gb", "mean"),
    Total_CLV      =("CLV",            "sum"),
).sort_values("Avg_CLV", ascending=False).round(2)
print(seg_summary.to_string())

# ─────────────────────────────────────────────────────────
#  PART C: Recommendations
# ─────────────────────────────────────────────────────────
def recommend(row):
    seg = row["Business_Segment"]
    if seg == "Champions":          return "Reward and Upsell"
    if seg == "High Value at Risk": return "URGENT Priority Retention Offer"
    if seg == "Loyal High Value":   return "Nurture and Bundle Offers"
    if seg == "Stable Mid Value":   return "Grow and Upsell Services"
    if seg == "Mid Value Slipping": return "Win-back Campaign"
    if seg == "Likely to Churn":    return "No Action Save Budget"
    return "Standard Engagement"

df["Recommendation"] = df.apply(recommend, axis=1)

print("\n--- Recommendations Summary ---")
rec_summary = df.groupby("Recommendation").agg(
    Customers =("CLV", "count"),
    Total_CLV =("CLV", "sum"),
    Avg_CLV   =("CLV", "mean"),
).sort_values("Total_CLV", ascending=False).round(2)
print(rec_summary.to_string())

urgent       = len(df[df["Recommendation"].str.contains("URGENT")])
budget_saved = len(df[df["Recommendation"] == "No Action Save Budget"])
at_risk_clv  = df[df["churn_proba_gb"] >= 0.5]["CLV"].sum()

print(f"\n  Urgent retention cases     : {urgent:,} customers")
print(f"  Total CLV at risk (p>0.5)  : ${at_risk_clv:,.2f}")
print(f"  Budget saved (no action)   : {budget_saved:,} customers")

# ─────────────────────────────────────────────────────────
#  PART D: Visualisations
# ─────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

print("\n  Building charts...")

BG     = "#F8F7F4"
CARD   = "#FFFFFF"
BORDER = "#D3D1C7"
TEXT   = "#2C2C2A"
TEXT_S = "#5F5E5A"

SEG_COLORS = {
    "Champions":          "#1D9E75",
    "Loyal High Value":   "#378ADD",
    "Stable Mid Value":   "#7F77DD",
    "Mid Value Slipping": "#EF9F27",
    "High Value at Risk": "#D85A30",
    "Likely to Churn":    "#888780",
    "Low Value Stable":   "#B4B2A9",
}

plt.rcParams.update({
    "figure.facecolor":  BG,    "axes.facecolor":    CARD,
    "axes.edgecolor":    BORDER,"axes.labelcolor":   TEXT,
    "xtick.color":       TEXT_S,"ytick.color":       TEXT_S,
    "text.color":        TEXT,  "font.family":       "DejaVu Sans",
    "axes.spines.top":   False, "axes.spines.right": False,
    "axes.grid":         True,  "grid.color":        BORDER,
    "grid.linewidth":    0.5,   "grid.alpha":        0.6,
})

fig = plt.figure(figsize=(20, 18), facecolor=BG)
fig.suptitle("Telco Churn — CLV & Business Recommendations",
             fontsize=18, fontweight="bold", color=TEXT, y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.50, wspace=0.40,
                       left=0.06, right=0.97,
                       top=0.93, bottom=0.05)

# ── 1. CLV Distribution ──────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df["CLV"], bins=50, color="#378ADD", edgecolor="none", alpha=0.8)
ax1.axvline(df["CLV"].mean(), color="#D85A30", lw=2,
            linestyle="--", label=f"Mean ${df['CLV'].mean():,.0f}")
ax1.set_title("CLV Distribution", fontsize=12, fontweight="bold")
ax1.set_xlabel("Customer Lifetime Value ($)")
ax1.set_ylabel("Count")
ax1.legend(framealpha=0)

# ── 2. Avg CLV by Segment ────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
clv_avgs    = clv_seg["Avg_CLV"]
seg_c       = ["#B4B2A9", "#7F77DD", "#378ADD", "#1D9E75"]
bars = ax2.bar(clv_avgs.index, clv_avgs.values,
               color=seg_c, width=0.6, edgecolor="none")
for bar, v in zip(bars, clv_avgs.values):
    ax2.text(bar.get_x() + bar.get_width()/2,
             v + 5, f"${v:,.0f}",
             ha="center", fontsize=9, color=TEXT_S)
ax2.set_title("Average CLV by Segment", fontsize=12, fontweight="bold")
ax2.set_ylabel("Avg CLV ($)")
ax2.set_xticklabels(clv_avgs.index, rotation=10)

# ── 3. CLV vs Churn Probability ──────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
sc  = df["Churn"].map({0: "#378ADD", 1: "#D85A30"})
ax3.scatter(df["churn_proba_gb"], df["CLV"],
            c=sc, alpha=0.15, s=6)
ax3.axvline(0.5, color="#D85A30", lw=1.5,
            linestyle="--", alpha=0.7, label="Risk threshold")
ax3.set_xlabel("Churn Probability")
ax3.set_ylabel("CLV ($)")
ax3.set_title("CLV vs Churn Probability", fontsize=12, fontweight="bold")
ax3.legend(framealpha=0, fontsize=8)

# ── 4. Customer Count by Business Segment ────────────────
ax4 = fig.add_subplot(gs[1, :2])
seg_counts  = df["Business_Segment"].value_counts()
bc4 = [SEG_COLORS.get(s, "#888780") for s in seg_counts.index]
bars = ax4.barh(seg_counts.index, seg_counts.values,
                color=bc4, edgecolor="none", height=0.6)
for bar, v in zip(bars, seg_counts.values):
    ax4.text(v + 10, bar.get_y() + bar.get_height()/2,
             f"{v:,}", va="center", fontsize=9, color=TEXT_S)
ax4.set_title("Customer Count by Business Segment",
              fontsize=12, fontweight="bold")
ax4.set_xlabel("Number of Customers")
ax4.invert_yaxis()

# ── 5. Total CLV by Business Segment ─────────────────────
ax5 = fig.add_subplot(gs[1, 2])
seg_clv = df.groupby("Business_Segment")["CLV"].sum().sort_values(ascending=False)
bc5     = [SEG_COLORS.get(s, "#888780") for s in seg_clv.index]
bars = ax5.barh(seg_clv.index, seg_clv.values,
                color=bc5, edgecolor="none", height=0.6)
for bar, v in zip(bars, seg_clv.values):
    ax5.text(v + 50, bar.get_y() + bar.get_height()/2,
             f"${v:,.0f}", va="center", fontsize=8, color=TEXT_S)
ax5.set_title("Total CLV by Business Segment",
              fontsize=12, fontweight="bold")
ax5.set_xlabel("Total CLV ($)")
ax5.invert_yaxis()

# ── 6. Action Framework Table ────────────────────────────
ax6 = fig.add_subplot(gs[2, :])
ax6.axis("off")
table_data = [
    ["Champions",          "Low",    "Reward and Upsell",
     "Loyalty perks, early feature access, referral program"],
    ["High Value at Risk", "High",   "URGENT Retention",
     "Personal outreach, discount offer, contract upgrade"],
    ["Loyal High Value",   "Low",    "Nurture and Grow",
     "Bundle offers, personalized check-ins, upsell premium"],
    ["Stable Mid Value",   "Medium", "Upsell",
     "Add-on promotions, gradual contract upgrade push"],
    ["Mid Value Slipping", "High",   "Win-back Campaign",
     "Targeted offer before contract ends, service review call"],
    ["Likely to Churn",    "High",   "No Action",
     "Low CLV — cost of saving exceeds value, reallocate budget"],
    ["Low Value Stable",   "Low",    "Monitor",
     "Standard engagement, low-cost automated touchpoints only"],
]
col_labels = ["Segment", "Churn Risk", "Action", "Strategy"]
table = ax6.table(
    cellText=table_data, colLabels=col_labels,
    cellLoc="left", loc="center", bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.auto_set_column_width([0, 1, 2, 3])
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor(BORDER)
    if row == 0:
        cell.set_facecolor("#378ADD")
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor(BG)
    else:
        cell.set_facecolor(CARD)
ax6.set_title("Business Action Framework by Segment",
              fontsize=12, fontweight="bold", pad=15)

plt.savefig("outputs/plots/step6_clv_business.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✅ Chart saved → outputs/plots/step6_clv_business.png")

# ── Save final dataset ────────────────────────────────────
df.to_csv("outputs/telco_final.csv", index=False)
print("✅ Final dataset saved → outputs/telco_final.csv")
print("\n  All done! Ready for Step 7 — Final Report 📄")