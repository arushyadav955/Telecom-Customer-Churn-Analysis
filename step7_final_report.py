# ============================================================
#  STEP 7: Final Summary Report
#  Telco Customer Churn Project
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

print("=" * 55)
print("  STEP 7: FINAL SUMMARY REPORT")
print("=" * 55)

df = pd.read_csv("outputs/telco_final.csv")
print(f"\n  Loaded {df.shape[0]:,} rows ✅")

# ── Colours ───────────────────────────────────────────────
BG      = "#F8F7F4"
CARD    = "#FFFFFF"
BORDER  = "#D3D1C7"
TEXT    = "#2C2C2A"
TEXT_S  = "#5F5E5A"
GREEN   = "#1D9E75"
BLUE    = "#378ADD"
CORAL   = "#D85A30"
AMBER   = "#EF9F27"
PURPLE  = "#7F77DD"
GRAY    = "#888780"

SEG_COLORS = {
    "Champions":          GREEN,
    "Loyal High Value":   BLUE,
    "Stable Mid Value":   PURPLE,
    "Mid Value Slipping": AMBER,
    "High Value at Risk": CORAL,
    "Likely to Churn":    GRAY,
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

# ─────────────────────────────────────────────────────────
#  FIGURE — 4 rows × 3 cols
# ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 26), facecolor=BG)
fig.suptitle("Telco Customer Churn — Full Project Report",
             fontsize=22, fontweight="bold", color=TEXT, y=0.99)

gs = gridspec.GridSpec(4, 3, figure=fig,
                       hspace=0.55, wspace=0.38,
                       left=0.06, right=0.97,
                       top=0.96, bottom=0.03)

# ── ROW 0: KPI Metric Cards (text axes) ──────────────────
kpis = [
    ("Total Customers",      f"{len(df):,}",          BLUE),
    ("Overall Churn Rate",   f"{df['Churn'].mean():.1%}", CORAL),
    ("Avg Monthly Charge",   f"${df['MonthlyCharges'].mean():.2f}", PURPLE),
    ("Avg CLV",              f"${df['CLV'].mean():,.0f}", GREEN),
    ("Total Portfolio CLV",  f"${df['CLV'].sum()/1e6:.2f}M", AMBER),
    ("Urgent Retention",     f"{len(df[df['Recommendation']=='URGENT Priority Retention Offer']):,}", CORAL),
]

for i, (label, value, color) in enumerate(kpis):
    row, col = divmod(i, 3)
    ax = fig.add_subplot(gs[row, col]) if i >= 3 else fig.add_subplot(gs[0, i])
    ax.set_facecolor(CARD)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0,0), 1, 1,
                 transform=ax.transAxes,
                 facecolor=CARD, edgecolor=color,
                 linewidth=2, zorder=0))
    ax.text(0.5, 0.62, value, transform=ax.transAxes,
            ha="center", va="center",
            fontsize=26, fontweight="bold", color=color)
    ax.text(0.5, 0.25, label, transform=ax.transAxes,
            ha="center", va="center",
            fontsize=11, color=TEXT_S)

# ── ROW 1: Churn by Contract | Churn by Internet | Churn by Tenure
contract_map = {0: "Month-to-month", 1: "One year", 2: "Two year"}
df["Contract_label"] = df["Contract"].map(contract_map)

ax_c = fig.add_subplot(gs[1, 0])
ct = df.groupby(["Contract_label","Churn"]).size().unstack(fill_value=0)
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
order = ["Month-to-month","One year","Two year"]
ct_pct = ct_pct.reindex(order)
bars = ax_c.bar(ct_pct.index, ct_pct[1].values,
                color=[CORAL, AMBER, GREEN], width=0.5, edgecolor="none")
for bar, v in zip(bars, ct_pct[1].values):
    ax_c.text(bar.get_x() + bar.get_width()/2, v + 0.5,
              f"{v:.1f}%", ha="center", fontsize=9, color=TEXT_S)
ax_c.set_title("Churn Rate by Contract", fontsize=12, fontweight="bold")
ax_c.set_ylabel("Churn Rate (%)")
ax_c.set_xticklabels(order, rotation=10, ha="right")
ax_c.set_ylim(0, 55)

ax_i = fig.add_subplot(gs[1, 1])
internet_churn = {
    "No Internet": df[df["InternetService_No"]==1]["Churn"].mean()*100,
    "DSL":         df[df["InternetService_DSL"]==1]["Churn"].mean()*100,
    "Fiber Optic": df[df["InternetService_Fiber optic"]==1]["Churn"].mean()*100,
}
bars = ax_i.bar(internet_churn.keys(), internet_churn.values(),
                color=[GREEN, BLUE, CORAL], width=0.5, edgecolor="none")
for bar, v in zip(bars, internet_churn.values()):
    ax_i.text(bar.get_x() + bar.get_width()/2, v + 0.5,
              f"{v:.1f}%", ha="center", fontsize=9, color=TEXT_S)
ax_i.set_title("Churn Rate by Internet Service", fontsize=12, fontweight="bold")
ax_i.set_ylabel("Churn Rate (%)")
ax_i.set_ylim(0, 55)

ax_t = fig.add_subplot(gs[1, 2])
tenure_order = ["New (0-12m)","Growing (13-24m)","Mature (25-48m)","Loyal (49-72m)"]
df["TenureGroup2"] = pd.cut(
    df["tenure"], bins=[-1,12,24,48,72],
    labels=tenure_order
)
tg = df.groupby("TenureGroup2", observed=True)["Churn"].mean() * 100
tg = tg.reindex(tenure_order)
bar_colors_t = [CORAL, AMBER, BLUE, GREEN]
bars = ax_t.bar(tg.index, tg.values,
                color=bar_colors_t, width=0.6, edgecolor="none")
for bar, v in zip(bars, tg.values):
    ax_t.text(bar.get_x() + bar.get_width()/2, v + 0.5,
              f"{v:.1f}%", ha="center", fontsize=8, color=TEXT_S)
ax_t.set_title("Churn Rate by Tenure Group", fontsize=12, fontweight="bold")
ax_t.set_ylabel("Churn Rate (%)")
ax_t.set_xticklabels(tenure_order, rotation=12, ha="right", fontsize=8)
ax_t.set_ylim(0, 60)

# ── ROW 2: Feature Importance | CLV Segments | Recommendations
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

drop_cols = [c for c in ["TenureGroup","Contract_label","TenureGroup2",
                          "CLV_Segment","Business_Segment","Recommendation",
                          "monthly_churn_prob","expected_lifetime_months",
                          "CLV","churn_proba_gb","churn_proba_lr",
                          "churn_proba_rf","churn_predicted"] if c in df.columns]
X = df.drop(columns=drop_cols + ["Churn"])
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                 learning_rate=0.05, random_state=42)
gb.fit(X_train, y_train)
feat_imp = pd.Series(gb.feature_importances_,
                     index=X.columns).sort_values(ascending=False).head(10)

ax_f = fig.add_subplot(gs[2, 0])
colors_fi = [CORAL if i < 3 else BLUE for i in range(len(feat_imp))]
ax_f.barh(feat_imp.index[::-1], feat_imp.values[::-1],
          color=colors_fi[::-1], edgecolor="none")
ax_f.set_title("Top 10 Feature Importances", fontsize=12, fontweight="bold")
ax_f.set_xlabel("Importance Score")
for i, (v, name) in enumerate(zip(feat_imp.values[::-1],
                                   feat_imp.index[::-1])):
    ax_f.text(v + 0.002, i, f"{v:.3f}",
              va="center", fontsize=8, color=TEXT_S)

ax_clv = fig.add_subplot(gs[2, 1])
clv_seg_data = df.groupby("CLV_Segment", observed=True)["CLV"].sum()
seg_c = ["#B4B2A9","#7F77DD","#378ADD","#1D9E75"]
ax_clv.pie(clv_seg_data.values,
           labels=clv_seg_data.index,
           colors=seg_c,
           autopct="%1.1f%%",
           startangle=90,
           wedgeprops={"edgecolor": CARD, "linewidth": 2})
ax_clv.set_title("Portfolio CLV Distribution\nby Segment",
                 fontsize=12, fontweight="bold")

ax_r = fig.add_subplot(gs[2, 2])
rec_counts = df["Recommendation"].value_counts()
short_labels = {
    "Reward and Upsell":                "Reward & Upsell",
    "Nurture and Bundle Offers":        "Nurture",
    "Grow and Upsell Services":         "Grow",
    "Win-back Campaign":                "Win-back",
    "No Action Save Budget":            "No Action",
    "Standard Engagement":              "Standard",
    "URGENT Priority Retention Offer":  "URGENT",
}
rec_counts.index = [short_labels.get(x, x) for x in rec_counts.index]
rec_colors = [GREEN, BLUE, PURPLE, AMBER, GRAY, "#B4B2A9", CORAL]
bars = ax_r.barh(rec_counts.index, rec_counts.values,
                 color=rec_colors[:len(rec_counts)],
                 edgecolor="none", height=0.6)
for bar, v in zip(bars, rec_counts.values):
    ax_r.text(v + 5, bar.get_y() + bar.get_height()/2,
              f"{v:,}", va="center", fontsize=9, color=TEXT_S)
ax_r.set_title("Customers by Recommendation",
               fontsize=12, fontweight="bold")
ax_r.set_xlabel("Number of Customers")
ax_r.invert_yaxis()

# ── ROW 3: Full Business Summary Table ───────────────────
ax_tbl = fig.add_subplot(gs[3, :])
ax_tbl.axis("off")

summary = df.groupby("Business_Segment").agg(
    Customers      =("CLV",            "count"),
    Avg_CLV        =("CLV",            "mean"),
    Total_CLV      =("CLV",            "sum"),
    Avg_Churn_Prob =("churn_proba_gb", "mean"),
    Avg_Monthly    =("MonthlyCharges", "mean"),
    Avg_Tenure     =("tenure",         "mean"),
).sort_values("Avg_CLV", ascending=False).round(2)

table_data = []
for seg, row in summary.iterrows():
    rec = df[df["Business_Segment"]==seg]["Recommendation"].iloc[0]
    table_data.append([
        seg,
        f"{int(row['Customers']):,}",
        f"${row['Avg_CLV']:,.0f}",
        f"${row['Total_CLV']:,.0f}",
        f"{row['Avg_Churn_Prob']:.0%}",
        f"${row['Avg_Monthly']:.2f}",
        f"{row['Avg_Tenure']:.0f} mo",
        short_labels.get(rec, rec),
    ])

col_labels = ["Segment","Customers","Avg CLV","Total CLV",
              "Churn Prob","Avg Monthly","Avg Tenure","Action"]
table = ax_tbl.table(
    cellText=table_data, colLabels=col_labels,
    cellLoc="center", loc="center", bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.auto_set_column_width(list(range(len(col_labels))))

seg_color_map = {
    "Champions":          "#E1F5EE",
    "Loyal High Value":   "#E6F1FB",
    "Stable Mid Value":   "#EEEDFE",
    "Mid Value Slipping": "#FAEEDA",
    "High Value at Risk": "#FAECE7",
    "Likely to Churn":    "#F1EFE8",
    "Low Value Stable":   "#F1EFE8",
}
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor(BORDER)
    if row == 0:
        cell.set_facecolor(BLUE)
        cell.set_text_props(color="white", fontweight="bold")
    else:
        seg_name = table_data[row-1][0]
        cell.set_facecolor(seg_color_map.get(seg_name, CARD))

ax_tbl.set_title("Complete Business Segment Summary",
                 fontsize=13, fontweight="bold", pad=15)

plt.savefig("outputs/plots/step7_final_report.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✅ Final report saved → outputs/plots/step7_final_report.png")

df.to_csv("outputs/telco_final.csv", index=False)
print("✅ Final dataset saved → outputs/telco_final.csv")

# ─────────────────────────────────────────────────────────
#  PRINT FINAL SUMMARY
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  PROJECT COMPLETE — FINAL SUMMARY")
print("=" * 55)
print(f"\n  Dataset          : 7,043 Telco customers")
print(f"  Features used    : 28 (inc. 5 engineered)")
print(f"  Best model       : Gradient Boosting (AUC 0.842)")
print(f"  Churn rate       : 26.5%")
print(f"\n  --- Top Churn Drivers ---")
for feat, imp in feat_imp.head(5).items():
    print(f"    {feat:<35} {imp:.4f}")
print(f"\n  --- Portfolio ---")
print(f"    Total CLV            : ${df['CLV'].sum():>12,.2f}")
print(f"    Avg CLV              : ${df['CLV'].mean():>12,.2f}")
print(f"    CLV at risk (p>0.5)  : ${df[df['churn_proba_gb']>=0.5]['CLV'].sum():>12,.2f}")
print(f"\n  --- Actions ---")
for rec, grp in df.groupby("Recommendation"):
    label = short_labels.get(rec, rec)
    print(f"    {label:<25} : {len(grp):>5,} customers  |  CLV ${grp['CLV'].sum():>12,.2f}")
print("\n" + "=" * 55)
print("  ALL STEPS COMPLETE ✅")
print("=" * 55)