"""
===============================================================
PROJECT 1: FMCG Customer Intelligence & Churn Prevention Engine
FILE: 01_data_simulation.py
PURPOSE: Generate a realistic FMCG loyalty wallet dataset
         simulating DS Group / OVINO-style user behaviour
         across 9 geographic hubs in NCR Delhi region.

WHY THIS FILE EXISTS:
  Real company data is confidential. This script generates
  statistically realistic data so the project is 100% shareable
  on GitHub and LinkedIn without any NDA concerns.
  
OUTPUT:
  - raw/data/users.csv          (250,000 users)
  - raw/data/transactions.csv   (1.2M+ transactions)
  - raw/data/Hubs.csv           (9 geographic hubs)
  - raw/data/Skus.csv           (product catalog)
===============================================================
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os

# ── SETUP ──────────────────────────────────────────
fake = Faker('en_IN')   # Indian locale — gives Indian-style names
random.seed(42)         # makes random numbers repeatable
np.random.seed(42)      # same for numpy random

# How many rows to generate
N_USERS = 250000
N_TRANSACTIONS = 800000

# Date range for our data
START_DATE = datetime(2022, 1, 1)
END_DATE   = datetime(2024, 12, 31)

# ── HUBS (9 geographic locations) ───────────────────
hubs_data = [
    {"hub_id": "H01", "hub_name": "Delhi Central",  "city": "Delhi",     "state": "Delhi",   "tier": "Tier-1"},
    {"hub_id": "H02", "hub_name": "Noida Sector",   "city": "Noida",     "state": "UP",      "tier": "Tier-1"},
    {"hub_id": "H03", "hub_name": "Gurgaon Hub",    "city": "Gurgaon",   "state": "Haryana", "tier": "Tier-1"},
    {"hub_id": "H04", "hub_name": "Ghaziabad Zone", "city": "Ghaziabad", "state": "UP",      "tier": "Tier-2"},
    {"hub_id": "H05", "hub_name": "Faridabad Hub",  "city": "Faridabad", "state": "Haryana", "tier": "Tier-2"},
    {"hub_id": "H06", "hub_name": "Greater Noida",  "city": "Gr. Noida", "state": "UP",      "tier": "Tier-2"},
    {"hub_id": "H07", "hub_name": "Meerut Zone",    "city": "Meerut",    "state": "UP",      "tier": "Tier-3"},
    {"hub_id": "H08", "hub_name": "Agra Hub",       "city": "Agra",      "state": "UP",      "tier": "Tier-3"},
    {"hub_id": "H09", "hub_name": "Lucknow Central","city": "Lucknow",   "state": "UP",      "tier": "Tier-1"},
]

# ── SKU CATALOG (products sold) ─────────────────────
skus_data = [
    {"sku_id": "S001", "product_name": "Cow Milk 1L",       "category": "Dairy",    "price": 62},
    {"sku_id": "S002", "product_name": "Toned Milk 500ml",  "category": "Dairy",    "price": 28},
    {"sku_id": "S003", "product_name": "Paneer 200g",       "category": "Dairy",    "price": 90},
    {"sku_id": "S004", "product_name": "Curd 400g",         "category": "Dairy",    "price": 45},
    {"sku_id": "S005", "product_name": "Premium Tea 250g",  "category": "Beverage", "price": 120},
    {"sku_id": "S006", "product_name": "Fruit Juice 1L",    "category": "Beverage", "price": 99},
    {"sku_id": "S007", "product_name": "Basmati Rice 5kg",  "category": "Staples",  "price": 340},
    {"sku_id": "S008", "product_name": "Atta Wheat 10kg",   "category": "Staples",  "price": 420},
    {"sku_id": "S009", "product_name": "Refined Oil 1L",    "category": "Staples",  "price": 135},
    {"sku_id": "S010", "product_name": "Namkeen 200g",      "category": "Snacks",   "price": 30},
]

# ── SAVE HUBS AND SKUS ───────────────────────────────
os.makedirs("raw/data", exist_ok=True)  # creates folder if it doesn't exist

hubs_df = pd.DataFrame(hubs_data)
skus_df = pd.DataFrame(skus_data)

hubs_df.to_csv("raw/data/Hubs.csv", index=False)
skus_df.to_csv("raw/data/Skus.csv", index=False)

print(f"✓ Hubs.csv saved  — {len(hubs_df)} rows")
print(f"✓ Skus.csv saved  — {len(skus_df)} rows")

# ── CUSTOMER SEGMENTS ────────────────────────────────
# Each segment has different behaviour ranges
# Format → "Segment": (min, max)

segment_config = {
    #              days_inactive   wallet_bal   total_recharges
    "Champion":   {"days": (1,  15), "wallet": (800, 5000), "recharges": (20, 60)},
    "Loyal":      {"days": (10, 30), "wallet": (300, 1200), "recharges": (10, 25)},
    "At-Risk":    {"days": (31, 60), "wallet": (80,  500),  "recharges": (4,  12)},
    "Hibernating":{"days": (61,120), "wallet": (20,  200),  "recharges": (2,  6)},
    "Lost":       {"days": (121,365),"wallet": (0,   50),   "recharges": (1,  3)},
}

# How many users in each segment (must add up to N_USERS)
segment_weights = [0.15, 0.20, 0.25, 0.25, 0.15]
segment_names   = list(segment_config.keys())

# ── GENERATE USERS ───────────────────────────────────
print(f"Generating {N_USERS:,} users...")

# Step 1: assign a segment to every user all at once
segments = np.random.choice(
    segment_names,
    size=N_USERS,
    p=segment_weights
)

# Step 2: assign a hub to every user (bigger cities get more users)
hub_ids = [h["hub_id"] for h in hubs_data]
hub_weights = [3200, 640, 876, 1720, 1414, 980, 1300, 1585, 2817]
hub_weights = [w / sum(hub_weights) for w in hub_weights]  # convert to probabilities

assigned_hubs = np.random.choice(hub_ids, size=N_USERS, p=hub_weights)

# Step 3: build each user's details based on their segment
user_rows = []

for i in range(N_USERS):
    seg  = segments[i]
    cfg  = segment_config[seg]

    # registration date — random day between 2022 and 2024
    reg_date = START_DATE + timedelta(days=random.randint(0, 900))

    # days inactive — pulled from this segment's range
    days_inactive = random.randint(*cfg["days"])

    # last active date — count backwards from end of dataset
    last_active = END_DATE - timedelta(days=days_inactive)

    # wallet balance and recharges from segment ranges
    wallet   = round(random.uniform(*cfg["wallet"]), 2)
    recharges = random.randint(*cfg["recharges"])

    # churned = 1 if inactive more than 60 days, else 0
    is_churned = 1 if days_inactive > 60 else 0

    user_rows.append({
        "user_id":           f"U{str(i+1).zfill(6)}",
        "hub_id":            assigned_hubs[i],
        "registration_date": reg_date.strftime("%Y-%m-%d"),
        "last_active_date":  last_active.strftime("%Y-%m-%d"),
        "days_inactive":     days_inactive,
        "wallet_balance":    wallet,
        "total_recharges":   recharges,
        "avg_recharge_value":round(wallet / max(recharges, 1), 2),
        "customer_segment":  seg,
        "is_churned":        is_churned,
        "gender":            random.choice(["M", "F"]),
        "preferred_sku":     random.choice([s["sku_id"] for s in skus_data]),
    })

print(f"  ✓ {N_USERS:,} user rows built")

# ── SAVE USERS CSV ───────────────────────────────────
users_df = pd.DataFrame(user_rows)
users_df.to_csv("raw/data/users.csv", index=False)

# Print a summary so you can verify it looks right
print(f"\n✓ users.csv saved — {len(users_df):,} rows")
print(f"\nSegment breakdown:")
print(users_df["customer_segment"].value_counts())
print(f"\nChurn rate: {users_df['is_churned'].mean()*100:.1f}%")
print(f"Avg wallet balance: ₹{users_df['wallet_balance'].mean():,.0f}")


# ── GENERATE TRANSACTIONS ────────────────────────────
# Each transaction = one wallet event (recharge, purchase, or debit)
print(f"\nGenerating {N_TRANSACTIONS:,} transactions...")

# SKU ids and their purchase probabilities
# Cow Milk is bought most — mirrors real FMCG data
sku_ids = [s["sku_id"] for s in skus_data]
sku_weights = [0.25, 0.15, 0.10, 0.08, 0.08,
               0.07, 0.07, 0.06, 0.07, 0.07]

# Transaction types and how often each occurs
txn_types   = ["purchase", "recharge", "debit_transfer"]
txn_weights = [0.50, 0.35, 0.15]

# Build a lookup: user_id → hub_id (so each txn knows its hub)
user_hub_map = dict(zip(users_df["user_id"], users_df["hub_id"]))

# Champions and Loyal users transact more — weight sampling accordingly
seg_txn_weight_map = {
    "Champion": 0.30, "Loyal": 0.25,
    "At-Risk": 0.20, "Hibernating": 0.15, "Lost": 0.10
}
user_txn_weights = users_df["customer_segment"].map(seg_txn_weight_map).values
user_txn_weights = user_txn_weights / user_txn_weights.sum()  # normalise to sum=1

# Sample which user each transaction belongs to
sampled_user_ids = np.random.choice(
    users_df["user_id"], size=N_TRANSACTIONS, p=user_txn_weights
)

txn_rows = []

for i in range(N_TRANSACTIONS):
    user_id  = sampled_user_ids[i]
    txn_type = random.choices(txn_types, weights=txn_weights, k=1)[0]
    sku_id   = np.random.choice(sku_ids, p=sku_weights)

    # Amount depends on transaction type
    if txn_type == "recharge":
        amount = random.choice([100, 200, 300, 500, 1000, 2000])
    elif txn_type == "purchase":
        sku_price = next(s["price"] for s in skus_data if s["sku_id"] == sku_id)
        amount    = round(sku_price * random.randint(1, 4), 2)
    else:  # debit_transfer
        amount = round(random.uniform(50, 500), 2)

    # Random date within our range
    txn_date = START_DATE + timedelta(days=random.randint(0, 1095),
                                      hours=random.randint(6, 22),
                                      minutes=random.randint(0, 59))

    txn_rows.append({
        "txn_id":    f"T{str(i+1).zfill(8)}",
        "user_id":   user_id,
        "hub_id":    user_hub_map[user_id],
        "sku_id":    sku_id,
        "txn_type":  txn_type,
        "txn_date":  txn_date.strftime("%Y-%m-%d"),
        "txn_month": txn_date.strftime("%Y-%m"),
        "amount":    amount,
        "status":    random.choices(["completed","failed","pending"],
                                     weights=[0.94, 0.04, 0.02], k=1)[0],
    })

txn_df = pd.DataFrame(txn_rows)
txn_df.to_csv("raw/data/transactions.csv", index=False)

print(f"✓ transactions.csv saved — {len(txn_df):,} rows")
print(f"\nTransaction type breakdown:")
print(txn_df["txn_type"].value_counts())
print(f"\nTotal purchase revenue: ₹{txn_df[txn_df['txn_type']=='purchase']['amount'].sum():,.0f}")

# ── FINAL SUMMARY ────────────────────────────────────
print("\n" + "="*50)
print("  DATA SIMULATION COMPLETE")
print("="*50)
print(f"  users.csv          : {len(users_df):>10,} rows")
print(f"  transactions.csv   : {len(txn_df):>10,} rows")
print(f"  Hubs.csv           : {len(hubs_df):>10} rows")
print(f"  Skus.csv           : {len(skus_df):>10} rows")
print(f"\n  Total wallet value : ₹{users_df['wallet_balance'].sum():>12,.0f}")
print(f"  Total revenue      : ₹{txn_df[txn_df['txn_type']=='purchase']['amount'].sum():>12,.0f}")
print("="*50)
print("\n✅ All files saved to raw/data")
print("   Next step → run 02_data_cleaning.py")

