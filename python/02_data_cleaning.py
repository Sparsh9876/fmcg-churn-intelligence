import pandas as pd
import numpy as np
import os
from datetime import datetime

# ── LOAD RAW DATA ─────────────────────────────────────
print("Loading raw data...")
users_df = pd.read_csv("raw/data/users.csv")
txn_df   = pd.read_csv("raw/data/transactions.csv")

print(f"  users.csv    : {len(users_df):,} rows, {users_df.shape[1]} columns")
print(f"  transactions : {len(txn_df):,} rows, {txn_df.shape[1]} columns")

# Always do this first — see what your data actually looks like
print("\nUsers — first 3 rows:")
print(users_df.head(3))
print("\nUsers — data types:")
print(users_df.dtypes)
print("\nUsers — missing values:")
print(users_df.isnull().sum())

# ── CLEAN USERS ───────────────────────────────────────
print("\n" + "="*45)
print("CLEANING USERS TABLE")
print("="*45)

rows_before = len(users_df)

# Step 1: Remove duplicate user IDs
duplicates = users_df.duplicated(subset=["user_id"]).sum()
users_df   = users_df.drop_duplicates(subset=["user_id"])
print(f"  Duplicate user_ids removed : {duplicates}")

# Step 2: Convert date columns from text → proper dates
# Without this, you cannot subtract dates or sort by date
users_df["registration_date"] = pd.to_datetime(users_df["registration_date"])
users_df["last_active_date"]  = pd.to_datetime(users_df["last_active_date"])
print(f"  Date columns converted to datetime ✓")

# Step 3: Fix impossible dates
# last_active cannot be BEFORE registration — that makes no sense
bad_dates = (users_df["last_active_date"] < users_df["registration_date"]).sum()
users_df  = users_df[users_df["last_active_date"] >= users_df["registration_date"]]
print(f"  Impossible date rows removed : {bad_dates}")

# Step 4: Clip negative wallet balances to 0
# A wallet cannot have negative balance in this system
negative_wallets = (users_df["wallet_balance"] < 0).sum()
users_df["wallet_balance"] = users_df["wallet_balance"].clip(lower=0)
print(f"  Negative wallet balances fixed : {negative_wallets}")

# Step 5: Add tenure column (how long has this user been registered)
users_df["tenure_days"] = (
    pd.Timestamp("2024-12-31") - users_df["registration_date"]
).dt.days

rows_after = len(users_df)
print(f"\n  Rows before : {rows_before:,}")
print(f"  Rows after  : {rows_after:,}")
print(f"  Removed     : {rows_before - rows_after:,}")

# ── CLEAN TRANSACTIONS ────────────────────────────────
print("\n" + "="*45)
print("CLEANING TRANSACTIONS TABLE")
print("="*45)

rows_before = len(txn_df)

# Step 1: Remove duplicate transaction IDs
dups = txn_df.duplicated(subset=["txn_id"]).sum()
txn_df = txn_df.drop_duplicates(subset=["txn_id"])
print(f"  Duplicate txn_ids removed : {dups}")

# Step 2: Drop rows where critical fields are missing
txn_df = txn_df.dropna(subset=["user_id", "amount", "txn_date"])
print(f"  Rows with missing key fields dropped ✓")

# Step 3: Remove zero or negative amounts
bad_amounts = (txn_df["amount"] <= 0).sum()
txn_df = txn_df[txn_df["amount"] > 0]
print(f"  Zero/negative amount rows removed : {bad_amounts}")

# Step 4: Remove extreme outliers (above 99.9th percentile)
# These are likely data entry errors
p999 = txn_df["amount"].quantile(0.999)
outliers = (txn_df["amount"] > p999).sum()
txn_df = txn_df[txn_df["amount"] <= p999]
print(f"  Outlier transactions removed : {outliers}  (above ₹{p999:.0f})")

# Step 5: Keep only completed transactions for analysis
# Failed/pending don't represent real business activity
failed = (txn_df["status"] != "completed").sum()
txn_clean = txn_df[txn_df["status"] == "completed"].copy()
print(f"  Non-completed transactions filtered : {failed}")

# Step 6: Convert date column
txn_clean["txn_date"] = pd.to_datetime(txn_clean["txn_date"])

rows_after = len(txn_clean)
print(f"\n  Rows before : {rows_before:,}")
print(f"  Rows after  : {rows_after:,}")
print(f"  Removed     : {rows_before - rows_after:,} ({(rows_before-rows_after)/rows_before*100:.1f}%)")

# ── SAVE CLEANED DATA ─────────────────────────────────
os.makedirs("processed/data", exist_ok=True)

users_df.to_csv("processed/data/users_clean.csv",           index=False)
txn_clean.to_csv("processed/data/transactions_clean.csv",   index=False)

print("\n" + "="*45)
print("CLEANING COMPLETE")
print("="*45)
print(f"  users_clean.csv        : {len(users_df):,} rows")
print(f"  transactions_clean.csv : {len(txn_clean):,} rows")
print(f"\n  Churn rate             : {users_df['is_churned'].mean()*100:.1f}%")
print(f"  Active users           : {(users_df['is_churned']==0).sum():,}")
print(f"  Churned users          : {users_df['is_churned'].sum():,}")
print("\n✅ Saved to data/processed/")
print("   Next step → run 03_eda_and_ml.py")
