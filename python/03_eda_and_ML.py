import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')           # saves charts to files instead of opening popups
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import os, warnings
warnings.filterwarnings("ignore")

os.makedirs("assets/screenshots", exist_ok=True)

# ── LOAD CLEAN DATA ───────────────────────────────────
print("Loading cleaned data...")
users = pd.read_csv("processed/data/users_clean.csv",
                    parse_dates=["registration_date","last_active_date"])
txns  = pd.read_csv("processed/data/transactions_clean.csv",
                    parse_dates=["txn_date"])
hubs  = pd.read_csv("raw/data/Hubs.csv")

# Merge hub names onto users so we have hub_name column
users = users.merge(hubs[["hub_id","hub_name"]], on="hub_id", how="left")

print(f"  Users: {len(users):,} | Transactions: {len(txns):,}")

# ── CHART 1: REVENUE BY HUB ───────────────────────────
print("\nGenerating Chart 1 — Revenue by Hub...")

purchases = txns[txns["txn_type"] == "purchase"].copy()

hub_revenue = (
    purchases
    .merge(hubs[["hub_id","hub_name"]], on="hub_id", how="left")
    .groupby("hub_name")["amount"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
hub_revenue["amount_lakhs"] = hub_revenue["amount"] / 100000

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(hub_revenue["hub_name"], hub_revenue["amount_lakhs"],
               color="#1B4F72")
ax.set_xlabel("Revenue (₹ Lakhs)")
ax.set_title("Total Purchase Revenue by Hub (2022–2024)", fontweight="bold")

# Add value labels on each bar
for bar, val in zip(bars, hub_revenue["amount_lakhs"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"₹{val:.0f}L", va="center", fontsize=9)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("assets/screenshots/01_revenue_by_hub.png", dpi=150)
plt.close()
print("  ✓ 01_revenue_by_hub.png saved")

# ── CHART 2: MONTHLY REVENUE TREND ───────────────────
print("Generating Chart 2 — Monthly Revenue Trend...")

monthly = (
    purchases
    .groupby("txn_month")["amount"]
    .sum()
    .reset_index()
    .sort_values("txn_month")
)
monthly["rolling_3m"] = monthly["amount"].rolling(window=3).mean()

fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(range(len(monthly)), monthly["amount"],
                alpha=0.15, color="#1B4F72")
ax.plot(range(len(monthly)), monthly["amount"],
        color="#1B4F72", linewidth=2, label="Monthly Revenue")
ax.plot(range(len(monthly)), monthly["rolling_3m"],
        color="#E74C3C", linewidth=2, linestyle="--", label="3-Month Rolling Avg")

# X axis labels — show every 3rd month to avoid crowding
ax.set_xticks(range(0, len(monthly), 3))
ax.set_xticklabels(monthly["txn_month"].iloc[::3], rotation=45, ha="right")
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"₹{x/100000:.0f}L")
)
ax.set_title("Monthly Purchase Revenue Trend", fontweight="bold")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("assets/screenshots/02_monthly_trend.png", dpi=150)
plt.close()
print("  ✓ 02_monthly_trend.png saved")

# ── CHART 3: SEGMENT BREAKDOWN ────────────────────────
print("Generating Chart 3 — Segment Distribution...")

seg_order  = ["Champion","Loyal","At-Risk","Hibernating","Lost"]
seg_colors = ["#1B4F72","#2ECC71","#F39C12","#E67E22","#E74C3C"]
seg_counts = users["customer_segment"].value_counts().reindex(seg_order)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: bar chart of user counts
axes[0].bar(seg_order, seg_counts.values / 1000, color=seg_colors)
axes[0].set_ylabel("Users (thousands)")
axes[0].set_title("Users per Segment", fontweight="bold")
for i, v in enumerate(seg_counts.values):
    axes[0].text(i, v/1000 + 0.3, f"{v/1000:.0f}K",
                 ha="center", fontsize=9, fontweight="bold")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Right: average wallet balance per segment
avg_wallet = users.groupby("customer_segment")["wallet_balance"].mean().reindex(seg_order)
axes[1].bar(seg_order, avg_wallet.values, color=seg_colors)
axes[1].set_ylabel("Avg Wallet Balance (₹)")
axes[1].set_title("Avg Wallet Balance by Segment", fontweight="bold")
for i, v in enumerate(avg_wallet.values):
    axes[1].text(i, v + 20, f"₹{v:.0f}", ha="center", fontsize=9, fontweight="bold")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.suptitle("Customer Segment Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("assets/screenshots/03_segments.png", dpi=150)
plt.close()
print("  ✓ 03_segments.png saved")

# ── CHURN PREDICTION MODEL ────────────────────────────
print("\nTraining churn prediction model...")

# Step 1: Build features from transaction history per user
user_txn_stats = txns.groupby("user_id").agg(
    total_txns        = ("txn_id",   "count"),
    total_spend       = ("amount",   "sum"),
    avg_txn_value     = ("amount",   "mean"),
    unique_skus       = ("sku_id",   "nunique"),
    purchase_count    = ("txn_type", lambda x: (x == "purchase").sum()),
    recharge_count    = ("txn_type", lambda x: (x == "recharge").sum()),
).reset_index()

# Step 2: Merge transaction stats onto users table
ml_df = users.merge(user_txn_stats, on="user_id", how="left")
ml_df = ml_df.fillna(0)   # users with no transactions get 0s

# Step 3: Encode the categorical segment column as a number
le = LabelEncoder()
ml_df["segment_encoded"] = le.fit_transform(ml_df["customer_segment"])

# Step 4: Define which columns are features (inputs to the model)
feature_cols = [
    "days_inactive", "wallet_balance", "total_recharges",
    "avg_recharge_value", "total_txns", "total_spend",
    "avg_txn_value", "unique_skus", "purchase_count",
    "recharge_count", "tenure_days", "segment_encoded"
]

X = ml_df[feature_cols]   # features matrix
y = ml_df["is_churned"]   # target column (what we want to predict)

print(f"  Features: {len(feature_cols)} | Samples: {len(X):,}")
print(f"  Churn rate in data: {y.mean()*100:.1f}%")

# Step 5: Split data — 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 6: Train the Random Forest model
print("\n  Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,    # 100 decision trees
    max_depth=10,        # each tree goes max 10 levels deep
    random_state=42,
    n_jobs=-1            # use all CPU cores to speed up training
)
model.fit(X_train, y_train)

# Step 7: Evaluate
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # probability of churn (class=1)
auc     = roc_auc_score(y_test, y_proba)

print(f"\n  AUC-ROC Score : {auc:.4f}  (1.0 = perfect, 0.5 = random)")
print(f"\n{classification_report(y_test, y_pred, target_names=['Active','Churned'])}")

# Step 8: Score ALL users with churn probability
ml_df["churn_probability"] = model.predict_proba(X)[:, 1]
ml_df["churn_risk_tier"]   = pd.cut(
    ml_df["churn_probability"],
    bins=[0, 0.33, 0.66, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

# Step 9: Save the scored file — this is what Tableau will read
output_cols = [
    "user_id", "hub_id", "hub_name", "customer_segment",
    "registration_date", "last_active_date", "days_inactive",
    "wallet_balance", "total_recharges", "is_churned",
    "tenure_days", "churn_probability", "churn_risk_tier",
    "total_txns", "total_spend", "avg_txn_value"
]
scored_df = ml_df[[c for c in output_cols if c in ml_df.columns]]
scored_df.to_csv("processed/data/users_scored.csv", index=False)

# Step 10: Feature importance chart
importances = pd.Series(model.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(importances.index, importances.values, color="#1B4F72")
ax.set_xlabel("Importance Score")
ax.set_title("Feature Importance — Churn Model", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("assets/screenshots/04_feature_importance.png", dpi=150)
plt.close()

# Final summary
high_risk = ml_df[ml_df["churn_risk_tier"] == "High Risk"]
print("\n" + "="*50)
print("  MODEL + EDA COMPLETE")
print("="*50)
print(f"  AUC-ROC Score          : {auc:.3f}")
print(f"  High-risk users        : {len(high_risk):,}")
print(f"  Revenue at risk        : ₹{high_risk['wallet_balance'].sum():,.0f}")
print(f"  Charts saved           : assets/screenshots/")
print(f"  Scored file saved      : data/processed/users_scored.csv")
print("="*50)
print("\n✅ Done! Next step → open Tableau and load users_scored.csv")

