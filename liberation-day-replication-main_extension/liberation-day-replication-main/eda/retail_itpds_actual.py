"""
retail_itpds_actual.py
======================
Hardest-hit retail sectors using ONLY data in our BigQuery database.

Sources (100% from our DB):
  - itpds_trade  : industry-level bilateral trade flows, 2019 (USD millions)
  - tariffs      : Liberation Day tariff rates by country

No external brand assumptions. Industry names come directly from ITPDS descriptions.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
FIG_DIR    = os.path.join(SCRIPT_DIR, "figures")
sys.path.insert(0, REPO_ROOT)
from database import bq_client as bq
os.makedirs(FIG_DIR, exist_ok=True)

DARK  = "#0d1117"; CARD  = "#161b22"
TW    = "#e6edf3"; TG    = "#8b949e"
RED   = "#f85149"; ORG   = "#fb8500"
YEL   = "#ffd166"; GRN   = "#3fb950"
BLU   = "#58a6ff"; PUR   = "#bc8cff"
TEAL  = "#39d353"

def dark_ax(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TG, labelsize=9)
    ax.xaxis.label.set_color(TG); ax.yaxis.label.set_color(TG)
    ax.title.set_color(TW)
    for s in ax.spines.values(): s.set_edgecolor("#30363d")
    ax.grid(color="#21262d", lw=0.5, alpha=0.9)
    return ax

def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"[saved] {name}")

# ---------------------------------------------------------------------------
# Pull all data from BigQuery
# ---------------------------------------------------------------------------
print("Fetching data from BigQuery...")

# Industry-level US imports by source country WITH tariff rates (SQL join)
raw = bq.query("""
    SELECT
        t.industry_id,
        t.industry_descr,
        t.broad_sector,
        t.exporter_iso3,
        t.exporter_name,
        tr.tariff_pct,
        ROUND(SUM(t.trade_usd_millions), 1)                              AS trade_usd_m,
        ROUND(SUM(t.trade_usd_millions) * tr.tariff_pct / 100.0, 1)     AS tariff_cost_m
    FROM `liberation-day-analysis.liberation_day.itpds_trade` t
    JOIN `liberation-day-analysis.liberation_day.tariffs` tr
        ON t.exporter_iso3 = tr.iso3
    WHERE t.importer_iso3 = 'USA'
      AND t.exporter_iso3 != 'USA'
      AND t.year = 2019
      AND t.trade_usd_millions > 0
    GROUP BY
        t.industry_id, t.industry_descr, t.broad_sector,
        t.exporter_iso3, t.exporter_name, tr.tariff_pct
    ORDER BY tariff_cost_m DESC
""")

print(f"  Rows fetched: {len(raw):,}")

# Aggregate by industry (across all source countries)
by_industry = raw.groupby(["industry_id","industry_descr","broad_sector"]).agg(
    trade_usd_m   = ("trade_usd_m",   "sum"),
    tariff_cost_m = ("tariff_cost_m", "sum"),
    n_countries   = ("exporter_iso3", "nunique"),
).reset_index()

# Weighted average tariff per industry
by_industry["wtd_tariff"] = by_industry["tariff_cost_m"] / by_industry["trade_usd_m"] * 100
by_industry = by_industry.sort_values("tariff_cost_m", ascending=False)

# Sector totals
by_sector = by_industry.groupby("broad_sector").agg(
    trade_usd_m   = ("trade_usd_m",   "sum"),
    tariff_cost_m = ("tariff_cost_m", "sum"),
).reset_index().sort_values("tariff_cost_m", ascending=False)

SECTOR_COL = {
    "Manufacturing":    BLU,
    "Agriculture":      GRN,
    "Mining and Energy":ORG,
    "Services":         PUR,
}

# Shorten long industry names for display
def shorten(name, maxlen=42):
    return name if len(name) <= maxlen else name[:maxlen-2] + ".."

top25 = by_industry.head(25).copy()
top25["label"] = top25["industry_descr"].apply(shorten)

# Top source country per industry
top_src = raw.sort_values("tariff_cost_m", ascending=False)\
             .groupby("industry_id").first().reset_index()[
                 ["industry_id","exporter_name","tariff_pct","trade_usd_m"]
             ].rename(columns={
                 "exporter_name":"top_country",
                 "tariff_pct":   "top_tariff",
                 "trade_usd_m":  "top_trade",
             })
top25 = top25.merge(top_src, on="industry_id", how="left")

# ===========================================================================
# FIGURE — 100% FROM OUR DATA (ITPDS + TARIFFS TABLES)
# ===========================================================================
fig = plt.figure(figsize=(22, 15), facecolor=DARK)
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.52, wspace=0.40,
                        left=0.04, right=0.97,
                        top=0.90, bottom=0.05)

fig.text(0.5, 0.956,
         "Hardest-Hit Sectors — Liberation Day Tariff Impact",
         ha="center", fontsize=22, fontweight="bold", color=TW)
fig.text(0.5, 0.925,
         "100% from our database  |  "
         "Source: ITPDS bilateral trade 2019 (itpds_trade) + Liberation Day tariff rates (tariffs)  |  "
         "Industry names = ITPDS official descriptions",
         ha="center", fontsize=10.5, color=TG)

# ---- Panel A: Top 25 industries by tariff cost (main chart) ----
ax_a = dark_ax(fig.add_subplot(gs[:2, :2]))

top25_s = top25.sort_values("tariff_cost_m", ascending=True)
cols_a = [SECTOR_COL.get(s, TG) for s in top25_s["broad_sector"]]
bars = ax_a.barh(top25_s["label"], top25_s["tariff_cost_m"] / 1000,
                 color=cols_a, edgecolor=DARK, linewidth=0.7,
                 height=0.72, zorder=3)

for bar, row in zip(bars, top25_s.itertuples()):
    val_bn = bar.get_width()
    ax_a.text(
        val_bn + 0.15,
        bar.get_y() + bar.get_height()/2,
        f"  ${val_bn:.1f}B cost  |  "
        f"${row.trade_usd_m/1000:.1f}B imports  @  "
        f"{row.wtd_tariff:.0f}% avg tariff  |  "
        f"top: {row.top_country} ({row.top_tariff:.0f}%)",
        va="center", fontsize=8, color=TW
    )

ax_a.set_xlabel("Annual Tariff Cost on US Imports (USD billions)", fontsize=11)
ax_a.set_title(
    "Top 25 Sectors by Liberation Day Tariff Cost\n"
    "(trade_usd_m × tariff_pct from ITPDS + tariffs tables in BigQuery)",
    fontsize=11, fontweight="bold", color=TW, pad=10
)
ax_a.set_xlim(0, top25_s["tariff_cost_m"].max() / 1000 * 1.75)
ax_a.tick_params(axis="y", colors=TW, labelsize=8.8)

legend_patches = [mpatches.Patch(color=v, label=k) for k, v in SECTOR_COL.items()]
ax_a.legend(handles=legend_patches, loc="lower right", fontsize=9,
            facecolor=CARD, labelcolor=TW, edgecolor="#30363d")

# ---- Panel B: Sector totals (row 0-1, col 2) ----
ax_b = dark_ax(fig.add_subplot(gs[:2, 2]))
bs = by_sector.sort_values("tariff_cost_m", ascending=True)
cols_b = [SECTOR_COL.get(s, TG) for s in bs["broad_sector"]]
bars_b = ax_b.barh(bs["broad_sector"], bs["tariff_cost_m"]/1000,
                   color=cols_b, edgecolor=DARK, lw=0.8, height=0.5, zorder=3)
for bar, row in zip(bars_b, bs.itertuples()):
    ax_b.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
              f"${bar.get_width():.0f}B", va="center",
              fontsize=11, fontweight="bold", color=TW)
ax_b.set_xlabel("Total Annual Tariff Cost (USD billions)", fontsize=10)
ax_b.set_title("Total Tariff Cost\nby Broad Sector",
               fontsize=12, fontweight="bold", color=TW, pad=10)
ax_b.tick_params(axis="y", colors=TW, labelsize=11)
ax_b.set_xlim(0, bs["tariff_cost_m"].max()/1000 * 1.4)

# ---- Panel C: Top 5 industries — source country breakdown (row 2, col 0-1) ----
ax_c = dark_ax(fig.add_subplot(gs[2, :2]))

top5_ids = by_industry.head(5)["industry_id"].tolist()
top5_names = by_industry.head(5)["industry_descr"].apply(shorten).tolist()

# For each top-5 industry, get breakdown by source country
top5_src = raw[raw["industry_id"].isin(top5_ids)].copy()
top5_src["industry_label"] = top5_src["industry_id"].map(
    dict(zip(top5_ids, top5_names))
)

# Top 5 countries per industry
country_colors = {
    "China":       RED,
    "Mexico":      ORG,
    "Vietnam":     YEL,
    "Japan":       BLU,
    "Korea, South":PUR,
    "India":       TEAL,
    "Ireland":     GRN,
    "Germany":     "#adbac7",
    "Malaysia":    "#f0b27a",
    "Canada":      "#85c1e9",
}

x = np.arange(5)
width = 0.14
countries_shown = ["China","Vietnam","Mexico","Japan","Korea, South"]
c_cols = [country_colors.get(c, TG) for c in countries_shown]

for ci, (country, col) in enumerate(zip(countries_shown, c_cols)):
    vals = []
    for ind_id in top5_ids:
        sub = raw[(raw["industry_id"]==ind_id) & (raw["exporter_name"]==country)]
        vals.append(sub["tariff_cost_m"].sum()/1000 if len(sub) else 0)
    offset = (ci - 2) * width
    bars_c = ax_c.bar(x + offset, vals, width, label=country,
                      color=col, edgecolor=DARK, lw=0.6, alpha=0.9, zorder=3)

ax_c.set_xticks(x)
ax_c.set_xticklabels(top5_names, fontsize=8.5, color=TW)
ax_c.set_ylabel("Tariff Cost (USD billions)", color=TG, fontsize=10)
ax_c.set_title("Top 5 Sectors — Tariff Cost by Source Country",
               fontsize=11, fontweight="bold", color=TW, pad=8)
ax_c.legend(fontsize=9, facecolor=CARD, labelcolor=TW, edgecolor="#30363d", ncol=5)

# ---- Panel D: Tariff rate vs trade volume scatter (row 2, col 2) ----
ax_d = dark_ax(fig.add_subplot(gs[2, 2]))

plot_df = by_industry[by_industry["trade_usd_m"] > 2000].copy()
cols_d = [SECTOR_COL.get(s, TG) for s in plot_df["broad_sector"]]
sc = ax_d.scatter(
    plot_df["wtd_tariff"],
    plot_df["trade_usd_m"] / 1000,
    s=plot_df["tariff_cost_m"] / plot_df["tariff_cost_m"].max() * 500 + 30,
    c=[list(SECTOR_COL.values()).index(SECTOR_COL.get(s, TG))
       for s in plot_df["broad_sector"]],
    cmap=matplotlib.colors.ListedColormap(list(SECTOR_COL.values())),
    alpha=0.80, edgecolors="white", lw=0.5, zorder=3
)
for _, row in by_industry.head(12).iterrows():
    if row["trade_usd_m"] > 2000:
        ax_d.annotate(
            shorten(row["industry_descr"], 22),
            (row["wtd_tariff"], row["trade_usd_m"]/1000),
            fontsize=7, color=TW, alpha=0.9
        )
ax_d.set_xlabel("Weighted Avg Tariff Rate (%)", color=TG, fontsize=10)
ax_d.set_ylabel("US Import Value (USD billions)", color=TG, fontsize=10)
ax_d.set_title("Import Volume vs\nTariff Rate (bubble=tariff cost)",
               fontsize=11, fontweight="bold", color=TW, pad=8)
ax_d.tick_params(axis="both", colors=TG)

fig.text(0.5, 0.022,
         "Data: ITPDS bilateral trade 2019 (itpds_trade table, BigQuery)  |  "
         "Tariff rates: Liberation Day schedule (tariffs table, BigQuery)  |  "
         "No external data or assumptions used",
         ha="center", fontsize=8.5, color=TG, style="italic")

save(fig, "retail_ACTUAL_itpds_sectors.png")

# ---- Print summary ----
print("\n=== TOP 15 HARDEST-HIT SECTORS (FROM OUR ITPDS DATA) ===")
print(f"{'Industry (ITPDS description)':<48} {'Imports':>10} {'Avg Tariff':>11} {'Tariff Cost':>12} {'Top Source':>15}")
print("-" * 100)
for _, row in by_industry.head(15).iterrows():
    src = top_src[top_src["industry_id"]==row["industry_id"]]["top_country"].values
    src_str = src[0] if len(src) else "n/a"
    print(f"  {shorten(row['industry_descr'], 46):<46}  "
          f"${row['trade_usd_m']/1000:>7.1f}B  "
          f"{row['wtd_tariff']:>9.1f}%  "
          f"${row['tariff_cost_m']/1000:>9.1f}B  "
          f"{src_str:>15}")

print(f"\n  Total tariff cost across all manufacturing imports: "
      f"${by_industry[by_industry['broad_sector']=='Manufacturing']['tariff_cost_m'].sum()/1000:.0f}B/yr")
