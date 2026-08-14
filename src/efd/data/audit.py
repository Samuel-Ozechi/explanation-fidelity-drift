from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "isFraud"
TIME = "TransactionDT"
ID = "TransactionID"


def transaction_schema(path: str | Path, sample_rows: int = 10_000) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=sample_rows, low_memory=False)
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({
            "feature": c,
            "dtype": str(s.dtype),
            "sample_non_null": int(s.notna().sum()),
            "sample_missing_rate": float(s.isna().mean()),
            "n_unique_sample": int(s.nunique(dropna=True)),
            "is_numeric": bool(pd.api.types.is_numeric_dtype(s)),
        })
    return pd.DataFrame(rows)


def scan_transactions(path: str | Path, chunksize: int = 100_000, time_bins: int = 20) -> dict:
    path = Path(path)
    total_rows = 0
    fraud_count = 0
    dt_min = None
    dt_max = None
    missing_counts = None
    dtypes = None
    bin_rows = np.zeros(time_bins, dtype=np.int64)
    bin_fraud = np.zeros(time_bins, dtype=np.int64)

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        if dtypes is None:
            dtypes = {c: str(t) for c, t in chunk.dtypes.items()}
            missing_counts = pd.Series(0, index=chunk.columns, dtype="int64")
        total_rows += len(chunk)
        fraud_count += int(chunk[TARGET].fillna(0).sum())
        dt = pd.to_numeric(chunk[TIME], errors="coerce")
        cmin, cmax = dt.min(), dt.max()
        if pd.notna(cmin): dt_min = cmin if dt_min is None else min(dt_min, cmin)
        if pd.notna(cmax): dt_max = cmax if dt_max is None else max(dt_max, cmax)
        missing_counts = missing_counts.add(chunk.isna().sum(), fill_value=0)

    if dt_min is not None and dt_max is not None and dt_max > dt_min:
        edges = np.linspace(dt_min, dt_max, time_bins + 1)
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            dt = pd.to_numeric(chunk[TIME], errors="coerce").to_numpy()
            idx = np.searchsorted(edges, dt, side="right") - 1
            idx = np.clip(idx, 0, time_bins - 1)
            valid = np.isfinite(dt)
            np.add.at(bin_rows, idx[valid], 1)
            fraud = chunk[TARGET].fillna(0).to_numpy()
            np.add.at(bin_fraud, idx[valid], fraud[valid].astype(np.int64))

    temporal = []
    if dt_min is not None:
        edges = np.linspace(dt_min, dt_max, time_bins + 1)
        for i in range(time_bins):
            temporal.append({
                "window": i,
                "dt_start": float(edges[i]),
                "dt_end": float(edges[i + 1]),
                "rows": int(bin_rows[i]),
                "fraud": int(bin_fraud[i]),
                "fraud_rate": float(bin_fraud[i] / bin_rows[i]) if bin_rows[i] else None,
            })

    missingness = [
        {"feature": k, "missing": int(v), "missing_rate": float(v / total_rows)}
        for k, v in missing_counts.sort_values(ascending=False).items()
    ]
    return {
        "file": str(path),
        "rows": total_rows,
        "columns": len(dtypes or {}),
        "fraud_count": fraud_count,
        "fraud_rate": fraud_count / total_rows if total_rows else None,
        "transaction_dt_min": None if dt_min is None else float(dt_min),
        "transaction_dt_max": None if dt_max is None else float(dt_max),
        "dtypes": dtypes or {},
        "missingness": missingness,
        "temporal_summary": temporal,
    }


def identity_join_coverage(transactions_path: str | Path, identity_path: str | Path, chunksize: int = 100_000) -> dict:
    identity_ids = set()
    for chunk in pd.read_csv(identity_path, usecols=[ID], chunksize=chunksize, low_memory=False):
        identity_ids.update(chunk[ID].dropna().astype("int64").tolist())
    total = matched = 0
    for chunk in pd.read_csv(transactions_path, usecols=[ID], chunksize=chunksize, low_memory=False):
        ids = chunk[ID].dropna().astype("int64")
        total += len(ids)
        matched += int(ids.isin(identity_ids).sum())
    return {
        "transaction_rows_considered": total,
        "identity_unique_transaction_ids": len(identity_ids),
        "matched_transaction_rows": matched,
        "matched_rate": matched / total if total else None,
        "unmatched_transaction_rows": total - matched,
        "join_type_for_modeling": "left",
    }


def write_audit(audit: dict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in audit.items() if k not in {"missingness", "temporal_summary", "schema"}}
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(audit["missingness"]).to_csv(output_dir / "missingness.csv", index=False)
    pd.DataFrame(audit["temporal_summary"]).to_csv(output_dir / "temporal_summary.csv", index=False)
    audit["schema"].to_csv(output_dir / "feature_schema.csv", index=False)
    if "identity_join" in audit:
        (output_dir / "join_summary.json").write_text(json.dumps(audit["identity_join"], indent=2), encoding="utf-8")
