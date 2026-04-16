"""
build_database.py
=================
Builds liberation_day.duckdb — a single-file analytical database
containing all project data sources.

Tables created:
  countries          — 194 country ISO codes + names
  tariffs            — US Liberation Day tariff rates by country
  trade_matrix       — 194×194 CEPII bilateral trade flows (USD millions)
  gdp                — GDP by country (USD millions)
  baci_trade         — 11M bilateral trade flows by HS6 product (2023)
  itpds_trade        — 10M bilateral trade flows by sector (1986–2019)
  gravity            — 325K bilateral gravity variables (2015–2019)
  oecd_icio          — Input-Output flows (country×sector, 2016–2022)
  ge_results         — GE model outputs: welfare/CPI/employment (9 scenarios)
  surrogate_training — 4,356-row surrogate model training set
  surrogate_results  — Bias-corrected surrogate predictions for key scenarios

Run:
  python database/build_database.py

Output:
  database/liberation_day.duckdb   (~400 MB estimated)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import duckdb

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR   = os.path.join(REPO_ROOT, 'project_data')
BASE_DATA  = os.path.join(REPO_ROOT, 'data', 'base_data')
PY_OUT     = os.path.join(REPO_ROOT, 'python_output')
SURR_DIR   = os.path.join(REPO_ROOT, 'surrogate_training')
DB_PATH    = os.path.join(SCRIPT_DIR, 'liberation_day.duckdb')

sys.path.insert(0, os.path.join(REPO_ROOT, 'code_python'))

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def ok(msg, t0):
    print(f"  [OK] {msg}  ({time.time()-t0:.1f}s)")


# ---------------------------------------------------------------------------
# Connect (overwrite if exists)
# ---------------------------------------------------------------------------
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
con = duckdb.connect(DB_PATH)
print(f"DuckDB {duckdb.__version__} — writing to {DB_PATH}")


# ===========================================================================
# TABLE 1: countries
# ===========================================================================
section("1/10  countries")
t0 = time.time()

countries_path = os.path.join(DATA_DIR, 'base_data', 'country_labels.csv')
con.execute(f"""
    CREATE TABLE countries AS
    SELECT
        iso3                           AS iso3,
        CAST(iso_code AS INTEGER)      AS country_id,
        CountryName                    AS country_name
    FROM read_csv_auto('{countries_path}', header=true)
    ORDER BY country_id
""")
n = con.execute("SELECT COUNT(*) FROM countries").fetchone()[0]
ok(f"countries — {n} rows", t0)


# ===========================================================================
# TABLE 2: tariffs (US Liberation Day tariff per trading partner)
# ===========================================================================
section("2/10  tariffs")
t0 = time.time()

tariffs_path = os.path.join(BASE_DATA, 'tariffs.csv')
# tariffs.csv has one column: applied_tariff (one value per country, same order as country_id)
tariffs_raw = pd.read_csv(tariffs_path)
tariffs_raw.columns = ['applied_tariff']
tariffs_raw['country_id'] = range(1, len(tariffs_raw) + 1)
tariffs_raw['tariff_pct'] = (tariffs_raw['applied_tariff'] * 100).round(2)
con.register('tariffs_df', tariffs_raw)
con.execute("""
    CREATE TABLE tariffs AS
    SELECT t.country_id, c.iso3, c.country_name,
           t.applied_tariff, t.tariff_pct
    FROM tariffs_df t
    LEFT JOIN countries c USING (country_id)
    ORDER BY country_id
""")
n = con.execute("SELECT COUNT(*) FROM tariffs").fetchone()[0]
ok(f"tariffs — {n} rows", t0)


# ===========================================================================
# TABLE 3: gdp
# ===========================================================================
section("3/10  gdp")
t0 = time.time()

gdp_path = os.path.join(BASE_DATA, 'gdp.csv')
gdp_raw = pd.read_csv(gdp_path, header=0)
gdp_raw.columns = ['gdp_raw']
gdp_raw['country_id'] = range(1, len(gdp_raw) + 1)
gdp_raw['gdp_usd_millions'] = pd.to_numeric(gdp_raw['gdp_raw'], errors='coerce')
con.register('gdp_df', gdp_raw[['country_id', 'gdp_usd_millions']])
con.execute("""
    CREATE TABLE gdp AS
    SELECT g.country_id, c.iso3, c.country_name,
           g.gdp_usd_millions
    FROM gdp_df g
    LEFT JOIN countries c USING (country_id)
    ORDER BY country_id
""")
n = con.execute("SELECT COUNT(*) FROM gdp").fetchone()[0]
ok(f"gdp — {n} rows", t0)


# ===========================================================================
# TABLE 4: trade_matrix (194×194 CEPII trade, wide→long)
# ===========================================================================
section("4/10  trade_matrix (CEPII 194×194)")
t0 = time.time()

trade_path = os.path.join(BASE_DATA, 'trade_cepii.csv')
trade_wide = pd.read_csv(trade_path, header=0)
N = trade_wide.shape[0]
# Columns are export_ij1 … export_ij194 — exporter rows, importer columns
trade_wide.columns = [f'imp_{j+1}' for j in range(N)]
trade_wide['exporter_id'] = range(1, N + 1)

trade_long = trade_wide.melt(
    id_vars='exporter_id',
    var_name='imp_col',
    value_name='trade_usd_millions'
)
trade_long['importer_id'] = trade_long['imp_col'].str.replace('imp_', '').astype(int)
trade_long = trade_long[['exporter_id', 'importer_id', 'trade_usd_millions']]
trade_long['trade_usd_millions'] = pd.to_numeric(
    trade_long['trade_usd_millions'], errors='coerce').fillna(0)
# Filter zeros to keep size manageable
trade_long = trade_long[trade_long['trade_usd_millions'] > 0].copy()

con.register('trade_df', trade_long)
con.execute("""
    CREATE TABLE trade_matrix AS
    SELECT
        t.exporter_id,
        e.iso3  AS exporter_iso3,
        e.country_name AS exporter_name,
        t.importer_id,
        m.iso3  AS importer_iso3,
        m.country_name AS importer_name,
        t.trade_usd_millions
    FROM trade_df t
    LEFT JOIN countries e ON t.exporter_id = e.country_id
    LEFT JOIN countries m ON t.importer_id = m.country_id
""")
n = con.execute("SELECT COUNT(*) FROM trade_matrix").fetchone()[0]
ok(f"trade_matrix — {n:,} rows (non-zero pairs)", t0)


# ===========================================================================
# TABLE 5: baci_trade (11M rows, bilateral by HS6)
# ===========================================================================
section("5/10  baci_trade (11M rows — takes ~30s)")
t0 = time.time()

baci_path = os.path.join(DATA_DIR, 'base_data', 'raw_data',
                         'BACI_HS22_Y2023_V202501.csv')
country_codes_path = os.path.join(DATA_DIR, 'base_data', 'raw_data',
                                  'country_codes_V202501.csv')

# Load country codes mapping (BACI numeric → ISO3)
if os.path.exists(country_codes_path):
    cc = pd.read_csv(country_codes_path)
    cc.columns = [c.strip() for c in cc.columns]
    # Actual columns: country_code, country_name, country_iso2, country_iso3
    con.register('baci_cc', cc)
    con.execute("""
        CREATE TABLE baci_country_codes AS
        SELECT * FROM baci_cc
    """)
    cc_join = """
        LEFT JOIN baci_country_codes ec ON t.i = ec.country_code
        LEFT JOIN baci_country_codes mc ON t.j = mc.country_code
    """
    iso_cols = "ec.country_iso3 AS exporter_iso3, mc.country_iso3 AS importer_iso3,"
else:
    cc_join = ""
    iso_cols = ""

con.execute(f"""
    CREATE TABLE baci_trade AS
    SELECT
        t.t                     AS year,
        t.i                     AS exporter_code,
        {iso_cols}
        t.j                     AS importer_code,
        t.k                     AS hs6_product_code,
        CAST(t.v AS DOUBLE)     AS value_1000usd,
        CAST(t.q AS DOUBLE)     AS quantity_tonnes
    FROM read_csv_auto('{baci_path}', header=true) t
    {cc_join}
""")
n = con.execute("SELECT COUNT(*) FROM baci_trade").fetchone()[0]
ok(f"baci_trade — {n:,} rows", t0)


# ===========================================================================
# TABLE 6: itpds_trade (10M rows, bilateral by sector)
# ===========================================================================
section("6/10  itpds_trade (10M rows — takes ~30s)")
t0 = time.time()

itpds_path = os.path.join(DATA_DIR, 'ITPDS', 'ITPD_S_R1.1_2019.csv')
con.execute(f"""
    CREATE TABLE itpds_trade AS
    SELECT
        exporter_dynamic_code,
        importer_dynamic_code,
        CAST(year AS INTEGER)           AS year,
        CAST(industry_id AS INTEGER)    AS industry_id,
        TRY_CAST(trade AS DOUBLE)       AS trade_usd_millions,
        exporter_iso3,
        exporter_name,
        importer_iso3,
        importer_name,
        industry_descr,
        broad_sector,
        flag_mirror,
        flag_zero,
        flag_itpds
    FROM read_csv_auto('{itpds_path}', header=true)
""")
n = con.execute("SELECT COUNT(*) FROM itpds_trade").fetchone()[0]
ok(f"itpds_trade — {n:,} rows", t0)


# ===========================================================================
# TABLE 7: gravity (bilateral gravity variables)
# ===========================================================================
section("7/10  gravity")
t0 = time.time()

grav_path = os.path.join(DATA_DIR, 'Dynamic_Gravity_Database',
                         'release_2.1_2015_2019.csv')
con.execute(f"""
    CREATE TABLE gravity AS
    SELECT *
    FROM read_csv_auto('{grav_path}', header=true)
""")
n  = con.execute("SELECT COUNT(*) FROM gravity").fetchone()[0]
cols = con.execute("DESCRIBE gravity").fetchdf()['column_name'].tolist()
ok(f"gravity — {n:,} rows, {len(cols)} columns", t0)


# ===========================================================================
# TABLE 8: oecd_icio (Input-Output, long format, all years stacked)
# ===========================================================================
section("8/10  oecd_icio (7 years x IO matrix -> long format)")
t0 = time.time()

icio_dir  = os.path.join(DATA_DIR, 'OECD_ICIO_SML_2016_2022')
icio_years = list(range(2016, 2023))
icio_frames = []

for yr in icio_years:
    fpath = os.path.join(icio_dir, f'{yr}_SML.csv')
    if not os.path.exists(fpath):
        print(f"  [SKIP] {yr}_SML.csv not found")
        continue
    print(f"  Loading ICIO {yr}...", end='', flush=True)
    df = pd.read_csv(fpath, index_col=0)
    # df: rows = source (country_sector), cols = destination (country_sector)
    # Convert to long: source, destination, value
    df_long = df.stack().reset_index()
    df_long.columns = ['source', 'destination', 'flow_usd_millions']
    df_long['year'] = yr
    df_long = df_long[df_long['flow_usd_millions'] != 0]
    # Parse country and sector from "AGO_A01" format
    df_long['source_country']  = df_long['source'].str[:3]
    df_long['source_sector']   = df_long['source'].str[4:]
    df_long['dest_country']    = df_long['destination'].str[:3]
    df_long['dest_sector']     = df_long['destination'].str[4:]
    icio_frames.append(df_long[['year','source_country','source_sector',
                                 'dest_country','dest_sector','flow_usd_millions']])
    print(f" {len(df_long):,} rows")

if icio_frames:
    icio_all = pd.concat(icio_frames, ignore_index=True)
    con.register('icio_df', icio_all)
    con.execute("""
        CREATE TABLE oecd_icio AS SELECT * FROM icio_df
    """)
    n = con.execute("SELECT COUNT(*) FROM oecd_icio").fetchone()[0]
    ok(f"oecd_icio — {n:,} rows across {len(icio_frames)} years", t0)
else:
    print("  [SKIP] No OECD ICIO files found")


# ===========================================================================
# TABLE 9: ge_results (from baseline_results.npz, long format)
# ===========================================================================
section("9/10  ge_results (GE model outputs)")
t0 = time.time()

SCENARIO_NAMES = {
    0: 'Liberation Day – No Retaliation',
    1: 'Liberation Day – Armington',
    2: 'Liberation Day – Eaton-Kortum',
    3: 'Optimal US – No Retaliation',
    4: 'Lib + Optimal Retaliation',
    5: 'Lib + Reciprocal Retaliation',
    6: 'Nash Equilibrium',
    7: 'Lib + Lump-Sum Rebate',
    8: 'Lib – High Elasticity',
}
METRIC_NAMES = ['welfare_pct', 'trade_deficit_pct', 'exports_pct',
                'imports_pct', 'employment_pct', 'cpi_pct', 'tariff_rev_share']

npz = np.load(os.path.join(PY_OUT, 'baseline_results.npz'), allow_pickle=True)
results = npz['results']   # (194, 7, 9)
countries_df = con.execute("SELECT country_id, iso3, country_name FROM countries").fetchdf()

rows = []
for sc in range(9):
    for ci in range(194):
        c_row = countries_df[countries_df['country_id'] == ci + 1]
        iso3  = c_row['iso3'].iloc[0]  if len(c_row) else 'UNK'
        cname = c_row['country_name'].iloc[0] if len(c_row) else 'Unknown'
        row = {
            'scenario_id':   sc,
            'scenario_name': SCENARIO_NAMES[sc],
            'country_id':    ci + 1,
            'iso3':          iso3,
            'country_name':  cname,
        }
        for mi, mname in enumerate(METRIC_NAMES):
            row[mname] = float(results[ci, mi, sc])
        rows.append(row)

ge_df = pd.DataFrame(rows)
con.register('ge_df', ge_df)
con.execute("CREATE TABLE ge_results AS SELECT * FROM ge_df")
n = con.execute("SELECT COUNT(*) FROM ge_results").fetchone()[0]
ok(f"ge_results — {n:,} rows (194 countries × 9 scenarios)", t0)


# ===========================================================================
# TABLE 10: surrogate_training + surrogate_results
# ===========================================================================
section("10/10  surrogate tables")
t0 = time.time()

# Training data
train_csv = os.path.join(SURR_DIR, 'training_data.csv')
con.execute(f"""
    CREATE TABLE surrogate_training AS
    SELECT * FROM read_csv_auto('{train_csv}', header=true)
""")
n = con.execute("SELECT COUNT(*) FROM surrogate_training").fetchone()[0]
ok(f"surrogate_training — {n:,} rows", t0)

# Validation table (bias-corrected predictions vs true GE)
val_csv = os.path.join(SURR_DIR, 'validation_table.csv')
if os.path.exists(val_csv):
    con.execute(f"""
        CREATE TABLE surrogate_validation AS
        SELECT * FROM read_csv_auto('{val_csv}', header=true)
    """)
    n2 = con.execute("SELECT COUNT(*) FROM surrogate_validation").fetchone()[0]
    ok(f"surrogate_validation — {n2} rows", t0)


# ===========================================================================
# Indexes for fast queries
# ===========================================================================
section("Creating indexes")
t0 = time.time()

indexes = [
    ("baci_trade",      "exporter_code"),
    ("baci_trade",      "importer_code"),
    ("baci_trade",      "hs6_product_code"),
    ("itpds_trade",     "exporter_iso3"),
    ("itpds_trade",     "importer_iso3"),
    ("itpds_trade",     "year"),
    ("itpds_trade",     "broad_sector"),
    ("gravity",         "iso3_o"),
    ("gravity",         "iso3_d"),
    ("gravity",         "year"),
    ("oecd_icio",       "year"),
    ("oecd_icio",       "source_country"),
    ("oecd_icio",       "dest_country"),
    ("ge_results",      "scenario_id"),
    ("ge_results",      "iso3"),
    ("surrogate_training", "us_tariff"),
]

for table, col in indexes:
    try:
        con.execute(f"CREATE INDEX idx_{table}_{col} ON {table} ({col})")
        print(f"  idx {table}.{col}")
    except Exception as e:
        print(f"  [SKIP] {table}.{col}: {e}")

ok("indexes done", t0)


# ===========================================================================
# Summary
# ===========================================================================
section("Database Summary")
print(f"\n  File: {DB_PATH}")
print(f"  Size: {os.path.getsize(DB_PATH)/1e6:.0f} MB\n")

tables = con.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'main'
    ORDER BY table_name
""").fetchdf()

print(f"  {'Table':<28} {'Rows':>12}")
print(f"  {'-'*28}  {'-'*12}")
for tbl in tables['table_name']:
    n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    print(f"  {tbl:<28} {n:>12,}")

print(f"\n  Total tables: {len(tables)}")
con.close()
print("\n[DONE] Database ready.")
print(f"  Connect with: con = duckdb.connect('{DB_PATH}')")
