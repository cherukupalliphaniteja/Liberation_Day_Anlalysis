"""
upload_to_bigquery.py
=====================
Uploads all tables from liberation_day.duckdb into Google BigQuery.

Usage:
    python database/upload_to_bigquery.py --project liberation-day-analysis \
                                          --dataset liberation_day \
                                          --key path/to/service_account.json

Authentication (pick one):
  A) Service account key:  --key path/to/key.json
  B) Application Default:  gcloud auth application-default login  (then omit --key)

Tables uploaded (in order):
    small  : countries, tariffs, gdp, trade_matrix, ge_results,
             surrogate_training, surrogate_validation, baci_country_codes
    medium : gravity (325K rows)
    large  : baci_trade (11.2M), itpds_trade (10.3M)
    xlarge : oecd_icio (77M) — chunked, takes ~20–40 min

Requirements:
    pip install google-cloud-bigquery pyarrow duckdb
"""

import argparse
import os
import sys
import time
import warnings

import duckdb
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
except ImportError:
    sys.exit("Run: pip install google-cloud-bigquery pyarrow")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(SCRIPT_DIR, "liberation_day.duckdb")

CHUNK_SIZE = {
    "baci_trade":    500_000,
    "itpds_trade":   500_000,
    "oecd_icio":     500_000,
    "gravity":       100_000,
    "default":        50_000,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_client(project: str, key_path: str | None) -> bigquery.Client:
    if key_path:
        creds = service_account.Credentials.from_service_account_file(
            key_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(project=project, credentials=creds)
    return bigquery.Client(project=project)   # uses ADC


def ensure_dataset(client: bigquery.Client, project: str, dataset_id: str):
    full = f"{project}.{dataset_id}"
    try:
        client.get_dataset(full)
        print(f"  Dataset {full} already exists")
    except Exception:
        ds = bigquery.Dataset(full)
        ds.location = "US"
        client.create_dataset(ds)
        print(f"  Created dataset {full}")


def table_ref(project, dataset, table):
    return f"{project}.{dataset}.{table}"


def upload_table(client, project, dataset, table_name, con, chunk_size):
    """Stream a DuckDB table into BigQuery in chunks."""
    ref  = table_ref(project, dataset, table_name)
    total = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )
    append_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        autodetect=False,
    )

    t0 = time.time()
    loaded = 0
    first  = True
    offset = 0

    while offset < total:
        df = con.execute(
            f"SELECT * FROM {table_name} LIMIT {chunk_size} OFFSET {offset}"
        ).fetchdf()

        if df.empty:
            break

        # Fix object columns that BigQuery can't auto-detect
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str)

        cfg = job_config if first else append_config
        if not first:
            cfg.schema = client.get_table(ref).schema

        job = client.load_table_from_dataframe(df, ref, job_config=cfg)
        job.result()   # wait

        loaded += len(df)
        offset += chunk_size
        first   = False

        pct     = loaded / total * 100
        elapsed = time.time() - t0
        rate    = loaded / elapsed if elapsed > 0 else 0
        eta     = (total - loaded) / rate if rate > 0 else 0
        print(f"    {loaded:>12,} / {total:>12,}  ({pct:5.1f}%)  "
              f"{rate/1000:.0f}K rows/s  ETA {eta:.0f}s", end="\r")

    elapsed = time.time() - t0
    print(f"    {loaded:>12,} / {total:>12,}  (100.0%)  "
          f"done in {elapsed:.0f}s{' '*20}")
    return loaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Upload DuckDB to BigQuery")
    parser.add_argument("--project", default="liberation-day-analysis",
                        help="GCP project ID")
    parser.add_argument("--dataset", default="liberation_day",
                        help="BigQuery dataset name")
    parser.add_argument("--key",     default=None,
                        help="Path to service account JSON key (omit for ADC)")
    parser.add_argument("--tables",  default=None,
                        help="Comma-separated table names to upload (default: all)")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        sys.exit(f"DuckDB not found at {DB_PATH}\nRun build_database.py first.")

    print(f"\nConnecting to BigQuery project: {args.project}")
    client  = get_client(args.project, args.key)
    ensure_dataset(client, args.project, args.dataset)

    con = duckdb.connect(DB_PATH, read_only=True)

    # Table upload order — small first, xlarge last
    all_tables = [
        "countries", "tariffs", "gdp", "trade_matrix",
        "baci_country_codes", "ge_results",
        "surrogate_training", "surrogate_validation",
        "gravity",
        "baci_trade",
        "itpds_trade",
        "oecd_icio",
    ]

    tables = (
        [t.strip() for t in args.tables.split(",")]
        if args.tables else all_tables
    )

    # Verify requested tables exist in DB
    existing = {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='main'"
    ).fetchall()}

    grand_total = 0
    grand_t0 = time.time()

    for tbl in tables:
        if tbl not in existing:
            print(f"\n  [SKIP] {tbl} not in DuckDB")
            continue

        n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        chunk = CHUNK_SIZE.get(tbl, CHUNK_SIZE["default"])
        n_chunks = (n // chunk) + (1 if n % chunk else 0)

        print(f"\n[{tbl}]  {n:,} rows  ({n_chunks} chunk(s) of {chunk:,})")
        loaded = upload_table(client, args.project, args.dataset, tbl, con, chunk)
        grand_total += loaded

    con.close()
    elapsed = time.time() - grand_t0
    print(f"\n{'='*60}")
    print(f"  Done.  {grand_total:,} total rows in {elapsed/60:.1f} min")
    print(f"  BigQuery: {args.project}.{args.dataset}")
    print(f"  Console:  https://console.cloud.google.com/bigquery"
          f"?project={args.project}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
