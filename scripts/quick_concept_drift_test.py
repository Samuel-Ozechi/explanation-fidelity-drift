# scripts/quick_concept_drift_test.py
"""
Quick test for concept drift - run this first to verify
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load data
df = pd.read_csv("data/raw/ieee_cis/train_transaction.csv", low_memory=False)

# Simple chronological split
df_sorted = df.sort_values("TransactionDT").reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.6)

train = df_sorted.iloc[:split_idx]
test = df_sorted.iloc[split_idx:]

# Simple preprocessing - just use numeric columns
numeric_cols = train.select_dtypes(include=["float64", "int64"]).columns
exclude = ["TransactionDT", "TransactionID", "isFraud"]
features = [c for c in numeric_cols if c not in exclude]

# Handle missing values
X_train = train[features].fillna(train[features].median())
y_train = train["isFraud"]

X_test = test[features].fillna(train[features].median())
y_test = test["isFraud"]

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Performance
train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
test_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

print(f"Training ROC-AUC: {train_auc:.4f}")
print(f"Test ROC-AUC: {test_auc:.4f}")
print(f"Performance Drop: {train_auc - test_auc:.4f}")
print(f"Fraud rate - Train: {y_train.mean():.4%}")
print(f"Fraud rate - Test: {y_test.mean():.4%}")