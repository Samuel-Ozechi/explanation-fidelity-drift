from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from efd.data.audit import identity_join_coverage, scan_transactions, transaction_schema, write_audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transactions", required=True)
    p.add_argument("--identity")
    p.add_argument("--output", default="results/audit")
    p.add_argument("--chunksize", type=int, default=100_000)
    p.add_argument("--time-bins", type=int, default=20)
    args = p.parse_args()

    audit = scan_transactions(args.transactions, args.chunksize, args.time_bins)
    audit["schema"] = transaction_schema(args.transactions)
    if args.identity:
        audit["identity_join"] = identity_join_coverage(args.transactions, args.identity, args.chunksize)
    write_audit(audit, args.output)

    print("\nIEEE-CIS audit complete")
    print(f"Rows: {audit['rows']:,}")
    print(f"Columns: {audit['columns']:,}")
    print(f"Fraud: {audit['fraud_count']:,} ({audit['fraud_rate']:.4%})")
    print(f"TransactionDT: {audit['transaction_dt_min']} -> {audit['transaction_dt_max']}")
    if "identity_join" in audit:
        print(f"Identity coverage: {audit['identity_join']['matched_rate']:.2%}")
    print(f"Results: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
