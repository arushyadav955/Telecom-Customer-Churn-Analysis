# ============================================================
#  STEP 4: EDA & Visualizations
#  Telco Customer Churn Project
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("outputs/telco_features.csv")
import matplotlib
matplotlib.use("Agg")  # ← add this line

# ── Colour palette ────────────────────────────────────────
NO_COLOR  = "#378ADD"   # blue  = stayed
YES_COLOR = "#D85A30"   # coral = churned
BG        = "#F8F7F4"
CARD      = "#FFFFFF"
BORDER    = "#D3D1C7"
TEXT      = "#2C2C2A"
TEXT_S    = "#5F5E5A"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT_S,
    "ytick.color":       TEXT_S,
    "text.color":        TEXT,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        BORDER,
    "grid.linewidth":    0.5,
    "grid.alpha":        0.6,
})

fig = plt.figure(figsize=(20, 24), facecolor=BG)
fig.suptitle("Telco Customer Churn — Exploratory Data Analysis",
             fontsize=20, fontweight="bold", color=TEXT, y=0.98)

gs = gridspec.GridSpec(4, 3, figure=fig,
                       hspace=0.50, wspace=0.38,
                       left=0.06, right=0.97,
                       top=0.94, bottom=0.04)

churn_labels = {0: "No Churn", 1: "Churned"}
colors       = [NO_COLOR, YES_COLOR]

# ── 1. Churn Distribution (pie) ───────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
counts = df["Churn"].value_counts().sort_index()
ax1.pie(counts, labels=["No Churn", "Churned"],
        colors=colors, autopct="%1.1f%%",
        startangle=90, wedgeprops={"edgecolor": CARD, "linewidth": 2})
ax1.set_title("Churn Distribution", fontsize=12, fontweight="bold")

# ── 2. Churn by Contract Type ────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
contract_map = {0: "Month-to-month", 1: "One year", 2: "Two year"}
df["Contract_label"] = df["Contract"].map(contract_map)
ct = df.groupby(["Contract_label", "Churn"]).size().unstack(fill_value=0)
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
ct_pct.plot(kind="bar", ax=ax2, color=colors, width=0.6,
            edgecolor="none")
ax2.set_title("Churn Rate by Contract Type", fontsize=12, fontweight="bold")
ax2.set_ylabel("Percentage (%)")
ax2.set_xlabel("")
ax2.set_xticklabels(ct_pct.index, rotation=15, ha="right")
ax2.legend(["No Churn", "Churned"], framealpha=0)
for container in ax2.containers:
    ax2.bar_label(container, fmt="%.0f%%", fontsize=8, padding=2)

# ── 3. Churn by Tenure Group ─────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
order = ["New (0–12m)", "Growing (13–24m)",
         "Mature (25–48m)", "Loyal (49–72m)"]
tg = df.groupby(["TenureGroup", "Churn"]).size().unstack(fill_value=0)
tg = tg.reindex(order)
tg_pct = tg.div(tg.sum(axis=1), axis=0) * 100
tg_pct.plot(kind="bar", ax=ax3, color=colors, width=0.6, edgecolor="none")
ax3.set_title("Churn Rate by Tenure Group", fontsize=12, fontweight="bold")
ax3.set_ylabel("Percentage (%)")
ax3.set_xlabel("")
ax3.set_xticklabels(order, rotation=15, ha="right", fontsize=8)
ax3.legend(["No Churn", "Churned"], framealpha=0)
for container in ax3.containers:
    ax3.bar_label(container, fmt="%.0f%%", fontsize=8, padding=2)

# ── 4. Monthly Charges Distribution ─────────────────────
ax4 = fig.add_subplot(gs[1, 0])
for churn_val, color, label in zip([0, 1], colors, ["No Churn", "Churned"]):
    subset = df[df["Churn"] == churn_val]["MonthlyCharges"]
    ax4.hist(subset, bins=30, alpha=0.65, color=color,
             label=label, edgecolor="none")
ax4.set_title("Monthly Charges Distribution", fontsize=12, fontweight="bold")
ax4.set_xlabel("Monthly Charges ($)")
ax4.set_ylabel("Count")
ax4.legend(framealpha=0)

# ── 5. Tenure Distribution ───────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
for churn_val, color, label in zip([0, 1], colors, ["No Churn", "Churned"]):
    subset = df[df["Churn"] == churn_val]["tenure"]
    ax5.hist(subset, bins=30, alpha=0.65, color=color,
             label=label, edgecolor="none")
ax5.set_title("Tenure Distribution", fontsize=12, fontweight="bold")
ax5.set_xlabel("Tenure (months)")
ax5.set_ylabel("Count")
ax5.legend(framealpha=0)

# ── 6. Churn by Internet Service ─────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
internet_churn = {
    "No Internet":  df[df["InternetService_No"] == 1]["Churn"].mean() * 100,
    "DSL":          df[df["InternetService_DSL"] == 1]["Churn"].mean() * 100,
    "Fiber Optic":  df[df["InternetService_Fiber optic"] == 1]["Churn"].mean() * 100,
}
bars = ax6.bar(internet_churn.keys(), internet_churn.values(),
               color=[NO_COLOR, NO_COLOR, YES_COLOR],
               width=0.5, edgecolor="none")
for bar, v in zip(bars, internet_churn.values()):
    ax6.text(bar.get_x() + bar.get_width()/2, v + 0.5,
             f"{v:.1f}%", ha="center", fontsize=10, color=TEXT_S)
ax6.set_title("Churn Rate by Internet Service", fontsize=12, fontweight="bold")
ax6.set_ylabel("Churn Rate (%)")
ax6.set_ylim(0, 60)

# ── 7. Total Services vs Churn ───────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
svc = df.groupby(["TotalServices", "Churn"]).size().unstack(fill_value=0)
svc_pct = svc.div(svc.sum(axis=1), axis=0) * 100
svc_pct[1].plot(kind="bar", ax=ax7, color=YES_COLOR,
                width=0.6, edgecolor="none")
ax7.set_title("Churn Rate by Number of Services", fontsize=12, fontweight="bold")
ax7.set_xlabel("Total Services Subscribed")
ax7.set_ylabel("Churn Rate (%)")
ax7.set_xticklabels(svc_pct.index, rotation=0)
for bar in ax7.patches:
    ax7.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f"{bar.get_height():.0f}%",
             ha="center", fontsize=8, color=TEXT_S)

# ── 8. Churn by Payment Method ───────────────────────────
ax8 = fig.add_subplot(gs[2, 1])
pay_cols = {
    "Electronic\nCheck":   "PaymentMethod_Electronic check",
    "Mailed\nCheck":       "PaymentMethod_Mailed check",
    "Bank\nTransfer":      "PaymentMethod_Bank transfer (automatic)",
    "Credit\nCard":        "PaymentMethod_Credit card (automatic)",
}
pay_churn = {k: df[df[v] == 1]["Churn"].mean() * 100
             for k, v in pay_cols.items()}
bar_colors = [YES_COLOR if v > 30 else NO_COLOR
              for v in pay_churn.values()]
bars = ax8.bar(pay_churn.keys(), pay_churn.values(),
               color=bar_colors, width=0.5, edgecolor="none")
for bar, v in zip(bars, pay_churn.values()):
    ax8.text(bar.get_x() + bar.get_width()/2, v + 0.5,
             f"{v:.1f}%", ha="center", fontsize=10, color=TEXT_S)
ax8.set_title("Churn Rate by Payment Method", fontsize=12, fontweight="bold")
ax8.set_ylabel("Churn Rate (%)")
ax8.set_ylim(0, 60)

# ── 9. Monthly Charges vs Tenure (scatter) ───────────────
ax9 = fig.add_subplot(gs[2, 2])
for churn_val, color, label, alpha in zip(
        [0, 1], colors, ["No Churn", "Churned"], [0.25, 0.5]):
    subset = df[df["Churn"] == churn_val]
    ax9.scatter(subset["tenure"], subset["MonthlyCharges"],
                c=color, alpha=alpha, s=8, label=label)
ax9.set_title("Monthly Charges vs Tenure", fontsize=12, fontweight="bold")
ax9.set_xlabel("Tenure (months)")
ax9.set_ylabel("Monthly Charges ($)")
ax9.legend(framealpha=0, markerscale=3)

# ── 10. Churn by Senior Citizen & Partner ────────────────
ax10 = fig.add_subplot(gs[3, 0])
demo_churn = {
    "Senior\nCitizen":    df[df["SeniorCitizen"] == 1]["Churn"].mean() * 100,
    "Non-Senior":         df[df["SeniorCitizen"] == 0]["Churn"].mean() * 100,
    "Has\nPartner":       df[df["Partner"] == 1]["Churn"].mean() * 100,
    "No\nPartner":        df[df["Partner"] == 0]["Churn"].mean() * 100,
    "Has\nDependents":    df[df["Dependents"] == 1]["Churn"].mean() * 100,
    "No\nDependents":     df[df["Dependents"] == 0]["Churn"].mean() * 100,
}
bar_colors = [YES_COLOR if v > 26.5 else NO_COLOR
              for v in demo_churn.values()]
bars = ax10.bar(demo_churn.keys(), demo_churn.values(),
                color=bar_colors, width=0.6, edgecolor="none")
for bar, v in zip(bars, demo_churn.values()):
    ax10.text(bar.get_x() + bar.get_width()/2, v + 0.3,
              f"{v:.1f}%", ha="center", fontsize=8, color=TEXT_S)
ax10.set_title("Churn Rate by Demographics", fontsize=12, fontweight="bold")
ax10.set_ylabel("Churn Rate (%)")
ax10.axhline(26.5, color="gray", linewidth=1,
             linestyle="--", alpha=0.7, label="Avg 26.5%")
ax10.legend(framealpha=0)
ax10.set_ylim(0, 50)

# ── 11. Correlation heatmap (numeric only) ───────────────
ax11 = fig.add_subplot(gs[3, 1:])
num_cols = ["tenure", "MonthlyCharges", "TotalCharges",
            "Contract", "TotalServices", "SeniorCitizen",
            "Partner", "Dependents", "PaperlessBilling",
            "HasInternet", "AvgMonthlySpend", "Churn"]
corr = df[num_cols].corr()
mask_upper = np.triu(np.ones_like(corr, dtype=bool), k=1)

import matplotlib.colors as mcolors
cmap = plt.cm.RdBu_r

im = ax11.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
ax11.set_xticks(range(len(num_cols)))
ax11.set_yticks(range(len(num_cols)))
ax11.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=8)
ax11.set_yticklabels(num_cols, fontsize=8)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        val = corr.iloc[i, j]
        color = "white" if abs(val) > 0.5 else TEXT
        ax11.text(j, i, f"{val:.2f}", ha="center",
                  va="center", fontsize=7, color=color)
plt.colorbar(im, ax=ax11, shrink=0.8)
ax11.set_title("Correlation Heatmap", fontsize=12, fontweight="bold")

# ── Save ─────────────────────────────────────────────────
plt.savefig("outputs/plots/step4_eda.png",
            dpi=150, bbox_inches="tight",
            facecolor=BG)
plt.close()
print("✅ EDA chart saved → outputs/plots/step4_eda.png")

# ── Print top churn insights ──────────────────────────────
print("\n" + "=" * 55)
print("  KEY INSIGHTS FROM EDA")
print("=" * 55)
print(f"\n  Contract type churn rates:")
for label, row in ct_pct.iterrows():
    print(f"    {label:<20} → Churn: {row[1]:.1f}%")

print(f"\n  Internet service churn rates:")
for k, v in internet_churn.items():
    print(f"    {k:<20} → Churn: {v:.1f}%")

print(f"\n  Payment method churn rates:")
for k, v in pay_churn.items():
    print(f"    {k.replace(chr(10),' '):<25} → Churn: {v:.1f}%")

print(f"\n  Tenure group churn rates:")
for label, row in tg_pct.iterrows():
    print(f"    {label:<25} → Churn: {row[1]:.1f}%")