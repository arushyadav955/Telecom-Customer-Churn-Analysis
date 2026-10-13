# 📊 Telco Customer Churn Analysis

A full end-to-end machine learning project to predict customer churn,
estimate Customer Lifetime Value (CLV), and generate actionable
business recommendations for a Telco company.

---

## 🗂️ Project Structure
```
Telco_churn_analysis/
├── step1_eda.py                  # Load & explore data
├── step2_cleaning.py             # Data cleaning & fixing
├── step3_feature_engineering.py  # Feature engineering
├── step4_eda_visualization.py    # EDA & visualizations
├── step5_churn_model.py          # Churn prediction models
├── step6_clv_business.py         # CLV & business segments
├── step7_final_report.py         # Final report dashboard
├── outputs/
│   ├── plots/                    # All charts & dashboards
│   └── telco_final.csv           # Final dataset with predictions
└── README.md
```

---

## 📁 Dataset

- **Source**: IBM Telco Customer Churn Dataset
- **Size**: 7,043 customers × 21 columns
- **Target**: `Churn` (Yes/No)

---

## 🔧 Steps Covered

| Step | Description |
|------|-------------|
| 1 | Load & explore raw data |
| 2 | Fix TotalCharges, encode target, handle missing values |
| 3 | Engineer 5 new features (CLV proxies, service counts) |
| 4 | EDA with 11 visualizations |
| 5 | Train 3 models — Logistic Regression, Random Forest, Gradient Boosting |
| 6 | CLV estimation + 7 business segments |
| 7 | Final report dashboard |

---

## 🤖 Models & Results

| Model | AUC | Precision | Recall | F1 |
|-------|-----|-----------|--------|----|
| Logistic Regression | 0.841 | 0.503 | 0.781 | 0.612 |
| Random Forest | 0.841 | 0.542 | 0.738 | 0.625 |
| **Gradient Boosting** | **0.842** | **0.666** | **0.521** | **0.585** |

---

## 🔑 Top Churn Drivers

1. **Contract type** — Month-to-month customers churn at 42.7%
2. **Tenure** — New customers (0–12m) churn at 47.4%
3. **Fiber Optic** — 41.9% churn rate vs 7.4% for no internet
4. **Electronic Check** — 45.3% churn rate
5. **Charge per Service** — Engineered feature, 6th most important

---

## 💰 Business Segments & Actions

| Segment | Customers | Avg CLV | Action |
|---------|-----------|---------|--------|
| Champions | 1,761 | $4,182 | Reward & Upsell |
| Loyal High Value | 1,728 | $1,458 | Nurture |
| Stable Mid Value | 1,207 | $884 | Grow & Upsell |
| Win-back | 557 | $674 | Win-back Campaign |
| No Action | 958 | $382 | Save Budget |
| URGENT | 29 | $1,222 | Priority Retention |

---

## 📦 Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## 🚀 How to Run
```bash
python step1_eda.py
python step2_cleaning.py
python step3_feature_engineering.py
python step4_eda_visualization.py
python step5_churn_model.py
python step6_clv_business.py
python step7_final_report.py
```
