 # 🛒 FMCG Customer Churn Intelligence Platform

> **Identifying ₹2.1Cr in at-risk wallet revenue across 250,000+ loyalty users using SQL, Python, and Tableau**

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791) ![Tableau](https://img.shields.io/badge/Tableau-Dashboard-orange) ![ML](https://img.shields.io/badge/ML-RandomForest-green) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 🎯 Business Problem

An FMCG company running a loyalty wallet program across 9 NCR Delhi hubs has **no visibility into which customers are silently churning** — and what that means in lost wallet revenue. The business needs a system to proactively flag at-risk users before they go dormant.

**Questions this project answers:**
- Which hubs have the highest churn concentration?
- What customer behaviours best predict churn?
- How much wallet revenue is at risk right now?
- Which users should the re-engagement team call first?

---

## 📊 Key Findings

- **75,210  users (30.10%)** are classified as High-Risk churn candidates
- **₹2.1 Crore** in wallet balance is sitting with users likely to churn
- **Noida Sector 18** has the highest absolute churn volume
- **Days inactive** and **wallet balance trend** are the two strongest churn predictors
- Random Forest model achieves **AUC-ROC: 1.00** on held-out test data

---

## 🗂️ Project Structure

```
fmcg-churn-intelligence/
│
├── raw/
│   ├── data/                      ← Simulated source data (250K users, 1M+ txns)
│      ├── users.csv               → 250,000 loyalty wallet users
│      ├── transactions.csv        → 1,010,061 wallet transactions
│      ├── Hubs.csv                → 9 geographic hub metadata
│      └── Skus.csv                → 14 product SKU catalog entries
│   
│├── processed/ 
│   └── data/                       ← Cleaned & ML-scored outputs
│       ├── users_clean.csv         → Deduplicated, validated users
│       ├── transactions_clean.csv  → 901,366 clean transactions
│       ├── users_with_churn_score.csv → Each user's churn probability (0–1)
│       ├── tableau_main.csv        → Pre-aggregated data for Tableau
│       ├── tableau_monthly.csv     → Monthly revenue trend data
│       ├── tableau_hub_kpis.csv    → Hub-level KPI summary
│       └── cleaning_report.txt     → Audit trail of cleaning steps
│
├── python/
│   ├── 01_data_simulation.py       ← Generates all raw CSVs from scratch
│   ├── 02_data_cleaning.py         ← Cleans, validates, removes outliers
│   └── 03_eda_and_ML.py            ← EDA charts + Random Forest churn model
│
├── sql/
│   ├── 01_schema.sql               ← PostgreSQL table definitions (DDL)
│   └── 02_analytics_queries.sql    ← 10 business analytics queries
│
├──tableau/
│   ├── FMCG-churn-Intelligence   ← tableau packaged workbook
│
├── assets/
│   └── screenshots/                ← All generated charts (PNG) + dashboard 
│       ├── 01_dashboard.png
│       ├── 01_revenue_by_hub.png
│       ├── 02_dashboard.png
│       ├── 02_monthly_trend.png
│       |── 03_monthly_trend.png
│       |── 03_segments.png
│       ├── 04_feature_importance.png
│
├──Requirements.txt                 ←  This lets anyone clone your project and install all dependencies with one command: pip install -r requirements.txt.
│
├──.gitignore                       ←   README will instruct anyone cloning the repo to run 01_data_simulation.py to regenerate them.
│
└── README.md
```

---

## 🚀 How to Run (Step by Step)

### Prerequisites
```bash
pip install pandas numpy faker scikit-learn matplotlib seaborn plotly openpyxl
```

### Step 1 — Generate Raw Data
```bash
python python/01_data_simulation.py
```
Creates 4 CSVs in `raw/data` — 250K users + 1M transactions.

### Step 2 — Clean the Data
```bash
python python/02_data_cleaning.py
```
Outputs cleaned files to `processed/data` .

### Step 3 — EDA + Train ML Model
```bash
python python/03_eda_and_ml.py
```
Generates 4 charts in `assets/screenshots/` + exports `users_score.csv`.

### Step 4 — SQL Analytics (PostgreSQL)
```bash
# Load schema first
psql -U postgres -d fmcg_db -f sql/01_schema.sql

# Then run analytics queries
psql -U postgres -d fmcg_db -f sql/02_analytics_queries.sql
```

### Step 5 — Tableau Dashboard
1. Open Tableau Desktop or Tableau Public
2. Connect to: `processed/data/users_scored.csv`
3. Also connect: `raw/data/Hubs.csv` 
4. Build dashboards using the layout guide below

---

## 🗄️ SQL Files Explained

### `01_schema.sql`
Defines 4 PostgreSQL tables matching the CSV structure:
- `users` — loyalty wallet user master
- `transactions` — all wallet events (recharge, purchase, debit)
- `Hubs` — geographic hub reference
- `Skus` — product catalog

### `02_analytics_queries.sql`
10 production-grade queries covering:

| Query | Business Purpose |
|-------|-----------------|
| Hub Revenue Ranking | RANK() OVER — which hubs drive most revenue |
| Wallet Bucket Analysis | CASE WHEN tiers — High/Mid/Low value users |
| 30/60/90 Day Churn Cohorts | DATEDIFF logic to find dormant users |
| Monthly Cohort Retention | CTE-based retention matrix |
| Debit Anomaly Detection | Users with suspicious debit/credit ratios |
| SKU Salience Analysis | Which products dominate each hub |
| Top 10 High-Value Users | Window function leaderboard |
| Revenue at Risk | Churning users × wallet balance |
| Hub MoM Growth | Month-over-month revenue change |
| Recharge Frequency Index | Avg days between recharges per segment |

---

## 📈 Tableau Dashboard Layout

**Dashboard 1 — Executive Overview**
- KPI Cards: Total Users / Active Users / Total Revenue / Avg Wallet Balance
- Bar chart: Revenue by Hub (sorted descending)
- Line chart: Monthly Revenue Trend with 3-month rolling average

**Dashboard 2 — Churn Risk Panel & Customer Segments**
- Scatter plot: Churn Probability vs Wallet Balance (bubble = user count)
- Heatmap: Churn Risk by Hub × Customer Segment
- Donut chart: Segment distribution (Champion / Loyal / At-Risk / Hibernating / Lost)
- Grouped bar: Avg wallet balance by segment
- Stacked bar: Active vs Churned % within each segment

> 💡 **Tip:** Use `churn_risk_tier` as a filter pill on all dashboards so stakeholders can toggle between High / Medium / Low risk views.

---

## 🤖 Machine Learning Details

**Model:** Random Forest Classifier (200 trees, max_depth=12)

**Features used:**
- `days_inactive` — strongest single predictor
- `wallet_balance` — high balance = lower churn risk
- `total_recharges` — frequency signal
- `avg_recharge_value` — spend quality signal
- `last_txn_days_ago` — recency signal
- `unique_skus` — product diversity (engaged users buy more categories)
- `purchase_count` / `recharge_count` — behavioral ratios
- `customer_segment`, `hub`, `tier` — categorical context

**Why Random Forest?**
- Handles class imbalance better than Logistic Regression
- Feature importance is interpretable for business stakeholders
- No need for feature scaling
- Robust to outliers in wallet balance data

---

## 💡 Business Impact

| Metric | Value |
|--------|-------|
| High-Risk Users Identified | 75,210 |
| Wallet Revenue at Risk | ₹2.1 Crore |
| Model AUC-ROC | 1.00 |
| Records Processed | 1,010,061 |
| Data Cleaned (removed) | 108,695 anomalous transactions |

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Data Generation | Python · Faker · NumPy |
| Data Wrangling | pandas |
| Machine Learning | scikit-learn (Random Forest) |
| Visualisation | matplotlib · seaborn |
| Database | PostgreSQL 15 · pgAdmin 4 |
| SQL Techniques | Window functions · CTEs · CASE WHEN · LAG · PARTITION BY |
| BI Dashboard | Tableau Public (2 dashboards) |
| Version Control | Git · GitHub |
---

## 👤 Author
 
**Sparsh Bhatnagar**
 Data Analytics Professional | MCA (ML & AI) — Amity University
[LinkedIn](https://www.linkedin.com/in/sparshbhatnagar/) | Ghaziabad, India
[![Email](https://img.shields.io/badge/Email-Reach%20Me-D14836?style=flat&logo=gmail&logoColor=white)](mailto:Sparsh.Bhatnagar13@gmail.com)
[![Tableau](https://img.shields.io/badge/Tableau-Live%20Dashboard-E97627?style=flat&logo=tableau&logoColor=white)]( https://public.tableau.com/views/FMCG-Churn-Intelligence/Dashboard3?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

---

*Dataset is fully simulated to mirror real FMCG loyalty wallet patterns. No proprietary data from DS Group or any other company is included.*
