# ============================================================
#  STEP 5: Churn Prediction Models
#  Telco Customer Churn Project
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection    import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing      import StandardScaler
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics            import (classification_report, roc_auc_score,
                                        confusion_matrix, roc_curve,
                                        precision_recall_curve, average_precision_score)


import matplotlib
matplotlib.use("Agg")


# ── Load ─────────────────────────────────────────────────
df = pd.read_csv("outputs/telco_features.csv")

# Drop non-numeric / leakage columns
drop_cols = ["TenureGroup", "Contract_label"] \
            if "Contract_label" in df.columns else ["TenureGroup"]
df = df.drop(columns=drop_cols, errors="ignore")

# ── Features & Target ────────────────────────────────────
X = df.drop(columns=["Churn"])
y = df["Churn"]

print("=" * 55)
print("  STEP 5: CHURN PREDICTION MODELS")
print("=" * 55)
print(f"\n  Features : {X.shape[1]}")
print(f"  Samples  : {X.shape[0]}")
print(f"  Churn %  : {y.mean():.1%}")

# ── Train / Test Split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"\n  Train size : {X_train.shape[0]}")
print(f"  Test size  : {X_test.shape[0]}")

# ── Scale for Logistic Regression ────────────────────────
scaler      = StandardScaler()
X_train_s   = scaler.fit_transform(X_train)
X_test_s    = scaler.transform(X_test)

# ── Define Models ─────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=42
    ),
}

# ── Train, Evaluate, Cross-validate ──────────────────────
results   = {}
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- Training & Evaluating Models ---\n")

for name, model in models.items():
    # Use scaled data for LR, raw for tree models
    Xtr = X_train_s if name == "Logistic Regression" else X_train
    Xte = X_test_s  if name == "Logistic Regression" else X_test
    Xcv = X_train_s if name == "Logistic Regression" else X_train

    model.fit(Xtr, y_train)

    y_pred  = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    auc     = roc_auc_score(y_test, y_proba)
    ap      = average_precision_score(y_test, y_proba)
    cv_auc  = cross_val_score(model, Xcv, y_train,
                               cv=cv, scoring="roc_auc", n_jobs=-1)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred, output_dict=True)

    results[name] = {
        "model":    model,
        "y_pred":   y_pred,
        "y_proba":  y_proba,
        "auc":      auc,
        "ap":       ap,
        "cv_auc":   cv_auc,
        "cm":       cm,
        "report":   report,
    }

    print(f"  ── {name} ──")
    print(f"     Test AUC          : {auc:.4f}")
    print(f"     Avg Precision     : {ap:.4f}")
    print(f"     CV AUC (5-fold)   : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    print(f"     Precision (churn) : {report['1']['precision']:.4f}")
    print(f"     Recall (churn)    : {report['1']['recall']:.4f}")
    print(f"     F1 (churn)        : {report['1']['f1-score']:.4f}")
    print()

# ── Feature Importance (Gradient Boosting) ───────────────
gb_model    = results["Gradient Boosting"]["model"]
feat_imp    = pd.Series(gb_model.feature_importances_, index=X.columns)
feat_imp    = feat_imp.sort_values(ascending=False).head(15)

# ── Visualisations ────────────────────────────────────────
BG      = "#F8F7F4"
CARD    = "#FFFFFF"
BORDER  = "#D3D1C7"
TEXT    = "#2C2C2A"
TEXT_S  = "#5F5E5A"
COLORS  = ["#378ADD", "#1D9E75", "#D85A30"]

plt.rcParams.update({
    "figure.facecolor":  BG, "axes.facecolor":  CARD,
    "axes.edgecolor":    BORDER, "axes.labelcolor": TEXT,
    "xtick.color":       TEXT_S, "ytick.color":    TEXT_S,
    "text.color":        TEXT,   "font.family":    "DejaVu Sans",
    "axes.spines.top":   False,  "axes.spines.right": False,
    "axes.grid":         True,   "grid.color":     BORDER,
    "grid.linewidth":    0.5,    "grid.alpha":     0.6,
})

fig = plt.figure(figsize=(20, 18), facecolor=BG)
fig.suptitle("Telco Churn — Model Evaluation",
             fontsize=18, fontweight="bold", color=TEXT, y=0.98)
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.50, wspace=0.38,
                        left=0.06, right=0.97,
                        top=0.93, bottom=0.05)

# ── Plot 1: ROC Curves ───────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for (name, res), color in zip(results.items(), COLORS):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax1.plot(fpr, tpr, color=color, lw=2,
             label=f"{name} (AUC={res['auc']:.3f})")
ax1.plot([0,1],[0,1], "k--", lw=1, alpha=0.4)
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curves", fontsize=12, fontweight="bold")
ax1.legend(fontsize=8, framealpha=0)

# ── Plot 2: Precision-Recall Curves ─────────────────────
ax2 = fig.add_subplot(gs[0, 1])
for (name, res), color in zip(results.items(), COLORS):
    prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
    ax2.plot(rec, prec, color=color, lw=2,
             label=f"{name} (AP={res['ap']:.3f})")
ax2.axhline(y.mean(), color="gray", linestyle="--",
            lw=1, alpha=0.6, label=f"Baseline ({y.mean():.2f})")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curves", fontsize=12, fontweight="bold")
ax2.legend(fontsize=8, framealpha=0)

# ── Plot 3: CV AUC Comparison ────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
names   = list(results.keys())
cv_means= [results[n]["cv_auc"].mean() for n in names]
cv_stds = [results[n]["cv_auc"].std()  for n in names]
bars    = ax3.bar(names, cv_means, color=COLORS,
                  width=0.5, edgecolor="none", yerr=cv_stds,
                  capsize=5, error_kw={"ecolor": TEXT_S})
ax3.set_ylim(0.70, 0.90)
ax3.set_title("Cross-Validated AUC (5-fold)", fontsize=12, fontweight="bold")
ax3.set_ylabel("AUC Score")
ax3.set_xticklabels(names, rotation=10, ha="right", fontsize=8)
for bar, v in zip(bars, cv_means):
    ax3.text(bar.get_x() + bar.get_width()/2,
             v + 0.003, f"{v:.4f}",
             ha="center", fontsize=9, color=TEXT_S)

# ── Plots 4-6: Confusion Matrices ────────────────────────
for idx, (name, res) in enumerate(results.items()):
    ax = fig.add_subplot(gs[1, idx])
    cm = res["cm"]
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nNo Churn", "Predicted\nChurn"])
    ax.set_yticklabels(["Actual\nNo Churn", "Actual\nChurn"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i,j] > cm.max()/2 else TEXT)
    ax.set_title(f"Confusion Matrix\n{name}",
                 fontsize=11, fontweight="bold")

# ── Plot 7: Feature Importance ───────────────────────────
ax7 = fig.add_subplot(gs[2, :2])
colors_fi = ["#D85A30" if i < 5 else "#378ADD"
             for i in range(len(feat_imp))]
bars = ax7.barh(feat_imp.index[::-1],
                feat_imp.values[::-1],
                color=colors_fi[::-1], edgecolor="none")
ax7.set_title("Top 15 Feature Importances — Gradient Boosting",
              fontsize=12, fontweight="bold")
ax7.set_xlabel("Importance Score")
for bar, v in zip(bars, feat_imp.values[::-1]):
    ax7.text(v + 0.001, bar.get_y() + bar.get_height()/2,
             f"{v:.4f}", va="center", fontsize=8, color=TEXT_S)

# ── Plot 8: Model Metrics Summary ────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis("off")
metrics_data = []
for name, res in results.items():
    r = res["report"]
    metrics_data.append([
        name.replace(" ", "\n"),
        f"{res['auc']:.3f}",
        f"{r['1']['precision']:.3f}",
        f"{r['1']['recall']:.3f}",
        f"{r['1']['f1-score']:.3f}",
    ])
table = ax8.table(
    cellText=metrics_data,
    colLabels=["Model", "AUC", "Precision", "Recall", "F1"],
    cellLoc="center", loc="center",
    bbox=[0, 0.2, 1, 0.75]
)
table.auto_set_font_size(False)
table.set_fontsize(9)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor(BORDER)
    if row == 0:
        cell.set_facecolor("#378ADD")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor(CARD if row % 2 == 0 else BG)
ax8.set_title("Model Metrics Summary",
              fontsize=12, fontweight="bold", pad=20)

plt.savefig("outputs/plots/step5_models.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✅ Model evaluation chart saved → outputs/plots/step5_models.png")

# ── Save predictions ─────────────────────────────────────
best_model  = results["Gradient Boosting"]
df_pred     = df.copy()
df_pred["churn_proba_gb"]  = results["Gradient Boosting"]["model"].predict_proba(X)[:, 1]
df_pred["churn_proba_lr"]  = results["Logistic Regression"]["model"].predict_proba(
                                scaler.transform(X))[:, 1]
df_pred["churn_proba_rf"]  = results["Random Forest"]["model"].predict_proba(X)[:, 1]
df_pred["churn_predicted"]  = results["Gradient Boosting"]["model"].predict(X)

df_pred.to_csv("outputs/telco_with_predictions.csv", index=False)
print("✅ Predictions saved → outputs/telco_with_predictions.csv")

print("\n--- Top 10 Most Important Features ---")
print(feat_imp.to_string())