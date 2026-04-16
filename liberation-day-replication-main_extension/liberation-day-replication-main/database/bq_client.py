"""
bq_client.py
============
Centralised BigQuery access layer for the Liberation Day project.

All tables live in:
    Project : liberation-day-analysis
    Dataset : liberation_day

Usage
-----
    from database.bq_client import query, get_countries, get_ge_results
    from database.bq_client import compute_icio_multipliers

    df = get_ge_results()          # 1,746-row DataFrame
    mults = compute_icio_multipliers(year=2022)  # dict sector->multiplier

Authentication
--------------
Run once per machine:
    gcloud auth application-default login

Environment variable overrides:
    BQ_PROJECT  (default: liberation-day-analysis)
    BQ_DATASET  (default: liberation_day)
"""

import os
import functools

import pandas as pd

try:
    from google.cloud import bigquery
except ImportError:
    raise ImportError("Run: pip install google-cloud-bigquery pyarrow")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT = os.environ.get("BQ_PROJECT", "liberation-day-analysis")
DATASET = os.environ.get("BQ_DATASET", "liberation_day")
_CLIENT: "bigquery.Client | None" = None


def _client() -> "bigquery.Client":
    """Return (or lazily create) the shared BigQuery client."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = bigquery.Client(project=PROJECT)
    return _CLIENT


def _tbl(table: str) -> str:
    """Full BigQuery table reference string."""
    return f"`{PROJECT}.{DATASET}.{table}`"


# ---------------------------------------------------------------------------
# Core query helper
# ---------------------------------------------------------------------------
def query(sql: str) -> pd.DataFrame:
    """
    Execute *sql* against BigQuery and return a pandas DataFrame.

    Example
    -------
    >>> df = query("SELECT iso3, welfare_pct FROM `liberation-day-analysis.liberation_day.ge_results` WHERE scenario_id = 0")
    """
    return _client().query(sql).to_dataframe()


# ---------------------------------------------------------------------------
# Table helpers (full-table fetches, suitable for small/medium tables)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def get_countries() -> pd.DataFrame:
    """
    Return the *countries* table (194 rows).

    Columns: iso3, country_name, region, income_group
    """
    return query(f"SELECT * FROM {_tbl('countries')} ORDER BY iso3")


@functools.lru_cache(maxsize=None)
def get_tariffs() -> pd.DataFrame:
    """
    Return the *tariffs* table (194 rows).

    Columns: iso3, country_name, tariff_pct
    """
    return query(f"SELECT * FROM {_tbl('tariffs')} ORDER BY tariff_pct DESC")


@functools.lru_cache(maxsize=None)
def get_gdp() -> pd.DataFrame:
    """Return the *gdp* table (194 rows)."""
    return query(f"SELECT * FROM {_tbl('gdp')} ORDER BY iso3")


@functools.lru_cache(maxsize=None)
def get_ge_results(scenario_id: int | None = None) -> pd.DataFrame:
    """
    Return GE model results.

    Parameters
    ----------
    scenario_id : int or None
        If given, filter to that scenario (0-8).  None = all 1,746 rows.

    Columns
    -------
    iso3, country_name, scenario_id, scenario_name,
    welfare_pct, cpi_pct, exports_pct, imports_pct,
    employment_pct, tariff_rev_pct, trade_deficit_pct
    """
    where = f"WHERE scenario_id = {scenario_id}" if scenario_id is not None else ""
    return query(
        f"SELECT * FROM {_tbl('ge_results')} {where} ORDER BY scenario_id, iso3"
    )


@functools.lru_cache(maxsize=None)
def get_surrogate_training() -> pd.DataFrame:
    """Return the surrogate training dataset (4,356 rows)."""
    return query(f"SELECT * FROM {_tbl('surrogate_training')} ORDER BY us_tariff")


@functools.lru_cache(maxsize=None)
def get_surrogate_validation() -> pd.DataFrame:
    """Return the surrogate validation table (9 rows)."""
    return query(
        f"SELECT * FROM {_tbl('surrogate_validation')} ORDER BY scenario_id"
    )


@functools.lru_cache(maxsize=None)
def get_baci_country_codes() -> pd.DataFrame:
    """Return BACI country code mapping (238 rows)."""
    return query(
        f"SELECT * FROM {_tbl('baci_country_codes')} ORDER BY country_iso3"
    )


# ---------------------------------------------------------------------------
# Larger-table helpers — return subsets / aggregates
# ---------------------------------------------------------------------------

def get_baci_trade(
    importer: str | None = None,
    exporter: str | None = None,
    year: int | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Fetch rows from baci_trade (11.2 M rows total).

    All parameters are optional filters.  Without any filter this pulls the
    full table — only do that if you need it.

    Example
    -------
    >>> us_imports = get_baci_trade(importer='USA')
    """
    conditions = []
    if importer:
        conditions.append(f"importer_iso3 = '{importer}'")
    if exporter:
        conditions.append(f"exporter_iso3 = '{exporter}'")
    if year:
        conditions.append(f"year = {year}")
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    lim   = f"LIMIT {limit}" if limit else ""
    return query(f"SELECT * FROM {_tbl('baci_trade')} {where} {lim}")


def get_top_trade_partners(
    importer: str = "USA",
    n: int = 20,
) -> pd.DataFrame:
    """
    Return the top *n* trade partners for *importer* by trade value (USD billions).
    """
    return query(f"""
        SELECT
            exporter_iso3,
            ROUND(SUM(value_1000usd) / 1e6, 2) AS trade_usd_billions
        FROM {_tbl('baci_trade')}
        WHERE importer_iso3 = '{importer}'
        GROUP BY exporter_iso3
        ORDER BY trade_usd_billions DESC
        LIMIT {n}
    """)


def get_itpds_trade(
    exporter: str | None = None,
    importer: str | None = None,
    sector: str | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    """
    Fetch filtered rows from itpds_trade (10.3 M rows total).

    Example
    -------
    >>> us_china = get_itpds_trade(exporter='USA', importer='CHN')
    """
    conditions = []
    if exporter:
        conditions.append(f"exporter_iso3 = '{exporter}'")
    if importer:
        conditions.append(f"importer_iso3 = '{importer}'")
    if sector:
        conditions.append(f"broad_sector = '{sector}'")
    if year:
        conditions.append(f"year = {year}")
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return query(f"SELECT * FROM {_tbl('itpds_trade')} {where}")


def get_gravity(
    origin: str | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    """
    Fetch rows from the gravity table (325 K rows).

    Example
    -------
    >>> usa_gravity = get_gravity(origin='USA', year=2019)
    """
    conditions = []
    if origin:
        conditions.append(f"iso3_o = '{origin}'")
    if year:
        conditions.append(f"year = {year}")
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return query(f"SELECT * FROM {_tbl('gravity')} {where}")


# ---------------------------------------------------------------------------
# OECD ICIO — derived metrics via SQL aggregation (avoids pulling 77M rows)
# ---------------------------------------------------------------------------

def compute_icio_multipliers(
    year: int = 2022,
    beta_labor: float = 0.49,
) -> dict:
    """
    Compute US sector-specific IO multipliers entirely in BigQuery.

    For each US model sector:
        import_share_interm = SUM(flows from non-USA sources) / SUM(all flows)
        io_multiplier       = 1 / (1 - (1 - beta_labor) * import_share_interm)

    Parameters
    ----------
    year : int
        ICIO year to use (default 2022, the latest available).
    beta_labor : float
        Labor share in value added (default 0.49).

    Returns
    -------
    dict  model_sector -> {'import_share_interm': float, 'io_multiplier': float}
    """
    sql = f"""
        WITH usa_dest AS (
            -- All flows into US sectors for the given year
            SELECT
                dest_sector                                          AS model_sector,
                SUM(flow_usd_millions)                               AS total_input,
                SUM(CASE WHEN source_country != 'USA'
                         THEN flow_usd_millions ELSE 0 END)          AS import_input
            FROM {_tbl('oecd_icio')}
            WHERE dest_country = 'USA'
              AND year = {year}
            GROUP BY dest_sector
        )
        SELECT
            model_sector,
            SAFE_DIVIDE(import_input, total_input)                   AS import_share_interm,
            1.0 / (1.0 - {1 - beta_labor}
                   * SAFE_DIVIDE(import_input, total_input))         AS io_multiplier
        FROM usa_dest
        ORDER BY model_sector
    """
    df = query(sql)
    result = {}
    for _, row in df.iterrows():
        sector = row["model_sector"]
        share  = float(row["import_share_interm"] or 0.0)
        mult   = float(row["io_multiplier"]       or 1.0)
        result[sector] = {
            "import_share_interm": share,
            "io_multiplier":       mult,
        }
    return result


def get_icio_bilateral_flows(
    source_country: str | None = None,
    dest_country: str | None = None,
    year: int = 2022,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Fetch aggregated bilateral IO supply-chain flows.

    Example
    -------
    >>> flows = get_icio_bilateral_flows(dest_country='USA', year=2022)
    """
    conditions = [f"year = {year}"]
    if source_country:
        conditions.append(f"source_country = '{source_country}'")
    if dest_country:
        conditions.append(f"dest_country = '{dest_country}'")
    where = "WHERE " + " AND ".join(conditions)
    lim   = f"LIMIT {top_n}" if top_n else ""
    return query(f"""
        SELECT
            source_country, dest_country,
            source_sector,  dest_sector,
            ROUND(SUM(flow_usd_millions), 2) AS flow_usd_millions
        FROM {_tbl('oecd_icio')}
        {where}
        GROUP BY source_country, dest_country, source_sector, dest_sector
        ORDER BY flow_usd_millions DESC
        {lim}
    """)


# ---------------------------------------------------------------------------
# Convenience: build the baseline_results array (194 x 7 x 9)
# from BigQuery ge_results — mirrors what the .npz files provide
# ---------------------------------------------------------------------------

def get_baseline_results_array() -> "tuple[pd.DataFrame, object]":
    """
    Return GE results in two convenient forms:

    1. A long-format DataFrame (all 1,746 rows).
    2. A numpy array of shape (194, 7, 9) matching the .npz layout used
       by the original analysis scripts.

    Array metric order (axis-1):
        0  welfare_pct
        1  trade_deficit_pct
        2  exports_pct
        3  imports_pct
        4  employment_pct
        5  cpi_pct
        6  tariff_rev_pct

    Returns
    -------
    df         : pd.DataFrame  long-format results
    arr        : np.ndarray    shape (194, 7, 9)
    """
    import numpy as np

    df = get_ge_results()

    metrics = [
        "welfare_pct", "trade_deficit_pct", "exports_pct",
        "imports_pct", "employment_pct", "cpi_pct", "tariff_rev_pct",
    ]

    countries = sorted(df["iso3"].unique())       # 194
    scenarios = sorted(df["scenario_id"].unique())  # 0-8

    N, M, S = len(countries), len(metrics), len(scenarios)
    arr = np.zeros((N, M, S))

    c_idx = {c: i for i, c in enumerate(countries)}
    s_idx = {s: i for i, s in enumerate(scenarios)}

    for _, row in df.iterrows():
        ci = c_idx[row["iso3"]]
        si = s_idx[row["scenario_id"]]
        for mi, m in enumerate(metrics):
            arr[ci, mi, si] = row[m] if pd.notna(row[m]) else 0.0

    return df, arr


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Project : {PROJECT}")
    print(f"Dataset : {DATASET}\n")

    print("[1] Countries (first 5):")
    print(get_countries().head())

    print("\n[2] US GE results (scenario 0 — Liberation Day):")
    us = get_ge_results(scenario_id=0).query("iso3 == 'USA'")
    print(us[["iso3", "scenario_name", "welfare_pct", "cpi_pct", "employment_pct"]].to_string(index=False))

    print("\n[3] ICIO multipliers (USA, 2022):")
    mults = compute_icio_multipliers(year=2022)
    for sec, v in sorted(mults.items()):
        print(f"  {sec:30s} import_share={v['import_share_interm']:.3f}  "
              f"multiplier={v['io_multiplier']:.3f}")

    print("\n[4] Top 5 US import partners:")
    print(get_top_trade_partners("USA", n=5).to_string(index=False))

    print("\n[Done]")
