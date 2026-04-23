"""
eda_bigquery.py
===============
Complete EDA for the Liberation Day project pulling all data from BigQuery.

Sections
--------
  1. Tariff landscape          — rates by country, region, income group
  2. BACI trade structure      — US import partners, HS6 product concentration
  3. GE results                — welfare distribution, winners/losers, retaliation impact
  4. Surrogate training data   — feature & target distributions, correlation heatmap
  5. Surrogate bias analysis   — raw vs corrected predictions, CI coverage
  6. OECD ICIO supply chains   — sector import dependency, IO multipliers
  7. Gravity & trade structure — distance-trade relationship, WTO effect

Run
---
    python eda/eda_bigquery.py

Outputs  →  eda/figures/*.png  +  eda/eda_report.txt
"""

import os
import sys
import warnings
import textwrap
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
FIG_DIR    = os.path.join(SCRIPT_DIR, "figures")
REPORT_PATH = os.path.join(SCRIPT_DIR, "eda_report.txt")
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, REPO_ROOT)
from database import bq_client as bq

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})
PALETTE = ["#2c7bb6", "#d7191c", "#1a9641", "#e6550d", "#756bb1",
           "#fdae61", "#abd9e9", "#a6d96a", "#d9ef8b", "#fee08b"]

report_lines = []

def section(title):
    sep = "=" * 68
    block = f"\n{sep}\n  {title}\n{sep}"
    print(block)
    report_lines.append(block)

def note(text):
    print(f"  {text}")
    report_lines.append(f"  {text}")

def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {name}")


# ===========================================================================
# SECTION 1 — TARIFF LANDSCAPE
# ===========================================================================
section("1. TARIFF LANDSCAPE")

tariffs = bq.get_tariffs()
countries = bq.get_countries()
gdp = bq.get_gdp()

df_t = tariffs.merge(countries, on="iso3", how="left") \
              .merge(gdp,       on="iso3", how="left")

note(f"Countries with tariff data: {len(df_t)}")
note(f"Tariff range: {df_t['tariff_pct'].min():.1f}% – {df_t['tariff_pct'].max():.1f}%")
note(f"Mean tariff : {df_t['tariff_pct'].mean():.2f}%   Median: {df_t['tariff_pct'].median():.1f}%")

# --- 1a. Distribution histogram ---
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(df_t["tariff_pct"], bins=30, color=PALETTE[0], edgecolor="white", linewidth=0.6)
axes[0].axvline(df_t["tariff_pct"].mean(),   color="red",    lw=1.5, linestyle="--", label=f'Mean {df_t["tariff_pct"].mean():.1f}%')
axes[0].axvline(df_t["tariff_pct"].median(), color="orange", lw=1.5, linestyle=":",  label=f'Median {df_t["tariff_pct"].median():.0f}%')
axes[0].set_xlabel("Tariff Rate (%)")
axes[0].set_ylabel("Number of Countries")
axes[0].set_title("Distribution of Liberation Day Tariff Rates")
axes[0].legend()

# By income group
if "income_group" in df_t.columns and df_t["income_group"].notna().any():
    ig_means = df_t.groupby("income_group")["tariff_pct"].mean().sort_values(ascending=False)
    colors_ig = [PALETTE[i % len(PALETTE)] for i in range(len(ig_means))]
    bars = axes[1].barh(ig_means.index, ig_means.values, color=colors_ig)
    for bar, val in zip(bars, ig_means.values):
        axes[1].text(val + 0.3, bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}%", va="center", fontsize=9)
    axes[1].set_xlabel("Mean Tariff Rate (%)")
    axes[1].set_title("Mean Tariff by Income Group")
else:
    # fallback: region
    if "region" in df_t.columns and df_t["region"].notna().any():
        rg = df_t.groupby("region")["tariff_pct"].mean().sort_values(ascending=False)
    else:
        rg = df_t.nlargest(15, "tariff_pct").set_index("iso3")["tariff_pct"]
    colors_rg = [PALETTE[i % len(PALETTE)] for i in range(len(rg))]
    bars = axes[1].barh(rg.index, rg.values, color=colors_rg)
    for bar, val in zip(bars, rg.values):
        axes[1].text(val + 0.3, bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}%", va="center", fontsize=9)
    axes[1].set_xlabel("Mean Tariff Rate (%)")
    axes[1].set_title("Mean Tariff by Region")

fig.suptitle("Liberation Day Tariff Landscape", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
save(fig, "1a_tariff_distribution.png")

# --- 1b. Top 25 highest & lowest tariff countries ---
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

top25 = df_t.nlargest(25, "tariff_pct")
bot25 = df_t.nsmallest(25, "tariff_pct")

for ax, sub, title, color in [
    (axes[0], top25, "Top 25 Highest Tariffs", PALETTE[1]),
    (axes[1], bot25, "Top 25 Lowest Tariffs",  PALETTE[2]),
]:
    labels = sub.apply(lambda r: r.get("country_name_x", r.get("country_name", r["iso3"])), axis=1)
    bars = ax.barh(labels, sub["tariff_pct"], color=color, alpha=0.85)
    for bar, val in zip(bars, sub["tariff_pct"]):
        ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                f"{val:.0f}%", va="center", fontsize=8)
    ax.set_xlabel("Tariff Rate (%)")
    ax.set_title(title, fontweight="bold")
    ax.invert_yaxis()

plt.tight_layout()
save(fig, "1b_tariff_top_bottom25.png")

note(f"Top 5 tariff countries: {', '.join(df_t.nlargest(5,'tariff_pct')['iso3'].tolist())}")
note(f"Flat 10% floor countries: {(df_t['tariff_pct']==10).sum()}")


# ===========================================================================
# SECTION 2 — BACI TRADE STRUCTURE
# ===========================================================================
section("2. BACI TRADE STRUCTURE (US imports, 2023)")

# Top 30 US import partners
partners = bq.query("""
    SELECT exporter_iso3 AS iso3,
           ROUND(SUM(value_1000usd)/1e6, 2) AS trade_usd_bn
    FROM `liberation-day-analysis.liberation_day.baci_trade`
    WHERE importer_iso3 = 'USA'
    GROUP BY exporter_iso3
    ORDER BY trade_usd_bn DESC
    LIMIT 30
""")

# Join with tariff rates
partners = partners.merge(tariffs[["iso3","tariff_pct"]], on="iso3", how="left")

note(f"Total US imports (BACI 2023): ${partners['trade_usd_bn'].sum():.0f}B")
note(f"Top 5 partners: {', '.join(partners.head(5)['iso3'].tolist())}")
top5_share = partners.head(5)["trade_usd_bn"].sum() / partners["trade_usd_bn"].sum() * 100
note(f"Top 5 share of total: {top5_share:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Bar chart: trade value, coloured by tariff rate
norm = plt.Normalize(partners["tariff_pct"].min(), partners["tariff_pct"].max())
cmap = plt.cm.RdYlGn_r
colors_bar = [cmap(norm(v)) for v in partners["tariff_pct"].fillna(10)]
bars = axes[0].barh(partners["iso3"], partners["trade_usd_bn"], color=colors_bar)
axes[0].invert_yaxis()
axes[0].set_xlabel("US Imports (USD billions)")
axes[0].set_title("Top 30 US Import Partners\n(colour = tariff rate: green=low, red=high)")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=axes[0], label="Tariff %", shrink=0.7)

# Scatter: trade value vs tariff rate (bubble = trade size)
axes[1].scatter(
    partners["tariff_pct"], partners["trade_usd_bn"],
    s=partners["trade_usd_bn"] / partners["trade_usd_bn"].max() * 800 + 30,
    c=partners["trade_usd_bn"], cmap="Blues", alpha=0.75, edgecolors="grey", lw=0.5
)
for _, row in partners.head(10).iterrows():
    axes[1].annotate(row["iso3"], (row["tariff_pct"], row["trade_usd_bn"]),
                     fontsize=8, ha="left", va="bottom")
axes[1].set_xlabel("Liberation Day Tariff Rate (%)")
axes[1].set_ylabel("US Imports (USD billions)")
axes[1].set_title("Trade Value vs Tariff Rate\n(bubble size = trade volume)")

plt.tight_layout()
save(fig, "2a_baci_top_partners.png")

# --- 2b. HS6 product concentration ---
hs6_top = bq.query("""
    SELECT
        CAST(FLOOR(SAFE_CAST(hs6_product_code AS INT64) / 10000) AS INT64) AS hs_chapter,
        ROUND(SUM(value_1000usd)/1e6, 1) AS trade_usd_bn,
        COUNT(DISTINCT exporter_iso3) AS n_exporters
    FROM `liberation-day-analysis.liberation_day.baci_trade`
    WHERE importer_iso3 = 'USA'
      AND SAFE_CAST(hs6_product_code AS INT64) IS NOT NULL
    GROUP BY hs_chapter
    ORDER BY trade_usd_bn DESC
    LIMIT 20
""")

HS_LABELS = {
    84: "Machinery/Mech",    85: "Electrical Equip", 87: "Vehicles",
    27: "Mineral Fuels",     90: "Optical/Medical",   39: "Plastics",
    29: "Organic Chem",      30: "Pharma",            62: "Apparel (woven)",
    61: "Apparel (knit)",    72: "Iron/Steel",         76: "Aluminium",
    94: "Furniture",         73: "Iron articles",      38: "Misc Chemicals",
    71: "Gems/Jewellery",    88: "Aircraft",           40: "Rubber",
    48: "Paper/Board",       95: "Toys/Sports",
}
hs6_top["label"] = hs6_top["hs_chapter"].map(HS_LABELS).fillna(
    hs6_top["hs_chapter"].astype(str)
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors_hs = [PALETTE[i % len(PALETTE)] for i in range(len(hs6_top))]
bars = axes[0].bar(hs6_top["label"], hs6_top["trade_usd_bn"], color=colors_hs)
axes[0].set_xticklabels(hs6_top["label"], rotation=45, ha="right", fontsize=8)
axes[0].set_ylabel("US Imports (USD billions)")
axes[0].set_title("Top 20 HS Chapters — US Imports")

# Cumulative concentration curve
total_trade = hs6_top["trade_usd_bn"].sum()
cumulative  = hs6_top["trade_usd_bn"].cumsum() / total_trade * 100
axes[1].plot(range(1, len(cumulative)+1), cumulative, marker="o", color=PALETTE[0])
axes[1].axhline(80, color="red", linestyle="--", lw=1, label="80% threshold")
axes[1].set_xlabel("Number of HS Chapters (ranked by value)")
axes[1].set_ylabel("Cumulative Share of US Imports (%)")
axes[1].set_title("Product Concentration Curve\n(How many chapters = 80% of imports?)")
axes[1].legend()

n80 = int((cumulative < 80).sum()) + 1
note(f"Top {n80} HS chapters account for 80% of US imports")

plt.tight_layout()
save(fig, "2b_baci_hs_concentration.png")


# ===========================================================================
# SECTION 3 — GE RESULTS EDA
# ===========================================================================
section("3. GE RESULTS — WELFARE DISTRIBUTION ACROSS COUNTRIES")

ge = bq.get_ge_results()

SCENARIO_SHORT = {
    0: "Lib Day\nNo Ret.",    1: "Lib Day\nArmington",
    2: "Lib Day\nEK",         3: "Optimal US\nNo Ret.",
    4: "Lib+\nOpt Ret.",      5: "Lib+\nRecip Ret.",
    6: "Nash\nEquil.",         7: "Lib+\nLumpSum",
    8: "Lib\nHigh Elas.",
}

scenarios = sorted(ge["scenario_id"].unique())

# --- 3a. US metrics across all 9 scenarios ---
us = ge[ge["iso3"] == "USA"].set_index("scenario_id").sort_index()
metrics_us = ["welfare_pct", "cpi_pct", "employment_pct", "trade_deficit_pct"]
metric_labels = ["Welfare %", "CPI %", "Employment %", "Trade Deficit %"]

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
axes = axes.flatten()
for i, (m, lbl) in enumerate(zip(metrics_us, metric_labels)):
    vals = [us.loc[s, m] if s in us.index else 0 for s in scenarios]
    colors_sc = [PALETTE[1] if v < 0 else PALETTE[0] for v in vals]
    bars = axes[i].bar([SCENARIO_SHORT[s] for s in scenarios], vals, color=colors_sc)
    axes[i].axhline(0, color="black", lw=1)
    axes[i].set_title(f"US {lbl}", fontweight="bold")
    axes[i].set_xticklabels([SCENARIO_SHORT[s] for s in scenarios], fontsize=7.5)
    for bar, val in zip(bars, vals):
        yoff = 0.05 if val >= 0 else -0.15
        axes[i].text(bar.get_x() + bar.get_width()/2, val + yoff,
                     f"{val:+.2f}", ha="center", fontsize=7, fontweight="bold")

fig.suptitle("US Economic Outcomes Across All 9 Tariff Scenarios",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "3a_us_outcomes_all_scenarios.png")

# --- 3b. Global welfare distribution (boxplot per scenario) ---
fig, ax = plt.subplots(figsize=(13, 5))
data_by_sc = [
    ge[(ge["scenario_id"] == s) & (ge["iso3"] != "USA")]["welfare_pct"].dropna().values
    for s in scenarios
]
bp = ax.boxplot(data_by_sc, patch_artist=True, notch=False,
                medianprops=dict(color="black", lw=2))
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticklabels([SCENARIO_SHORT[s] for s in scenarios], fontsize=9)
ax.axhline(0, color="black", lw=1, linestyle="--")
ax.set_ylabel("Welfare Change (%)")
ax.set_title("Global Welfare Distribution (194 countries, excl. USA) — All Scenarios",
             fontweight="bold")
save(fig, "3b_global_welfare_boxplot.png")

note(f"Scenario 0 (Lib Day No Ret): median welfare {np.median(data_by_sc[0]):+.3f}%")
note(f"Scenario 4 (Opt Retaliation): median welfare {np.median(data_by_sc[4]):+.3f}%")

# --- 3c. Winners vs losers count per scenario ---
fig, ax = plt.subplots(figsize=(11, 4))
winners = [int((ge[(ge["scenario_id"]==s) & (ge["iso3"]!="USA")]["welfare_pct"] > 0).sum()) for s in scenarios]
losers  = [int((ge[(ge["scenario_id"]==s) & (ge["iso3"]!="USA")]["welfare_pct"] < 0).sum()) for s in scenarios]
x = np.arange(len(scenarios))
w = 0.35
ax.bar(x - w/2, winners, w, color=PALETTE[2], label="Winners",  alpha=0.85)
ax.bar(x + w/2, losers,  w, color=PALETTE[1], label="Losers",   alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([SCENARIO_SHORT[s] for s in scenarios], fontsize=9)
ax.set_ylabel("Number of Countries (excl. USA)")
ax.set_title("Winners vs Losers per Scenario", fontweight="bold")
ax.legend()
for i, (w_, l_) in enumerate(zip(winners, losers)):
    ax.text(i - w/2, w_ + 0.5, str(w_), ha="center", fontsize=8)
    ax.text(i + w/2, l_ + 0.5, str(l_), ha="center", fontsize=8)
plt.tight_layout()
save(fig, "3c_winners_losers.png")

# --- 3d. Heatmap: welfare by country (top 30 by |welfare| in sc0) ---
sc0 = ge[ge["scenario_id"] == 0].set_index("iso3")["welfare_pct"]
top30 = sc0.abs().nlargest(30).index.tolist()

heat_df = ge[ge["iso3"].isin(top30)].pivot(index="iso3", columns="scenario_id", values="welfare_pct")
heat_df.columns = [SCENARIO_SHORT[c].replace("\n"," ") for c in heat_df.columns]

fig, ax = plt.subplots(figsize=(14, 9))
vmax = heat_df.abs().max().max()
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
sns.heatmap(heat_df, ax=ax, cmap="RdYlGn", norm=norm, annot=True, fmt=".1f",
            annot_kws={"size": 7}, linewidths=0.3, linecolor="white",
            cbar_kws={"label": "Welfare Change (%)"})
ax.set_title("Welfare Change (%) — Top 30 Countries by |Impact|, All Scenarios",
             fontweight="bold", fontsize=11)
ax.set_ylabel("")
plt.tight_layout()
save(fig, "3d_welfare_heatmap_countries.png")

# --- 3e. Retaliation impact: sc0 vs sc4 vs sc5 ---
fig, ax = plt.subplots(figsize=(10, 5))
major = ["USA", "CHN", "CAN", "MEX", "DEU", "JPN", "KOR", "GBR", "FRA", "IND"]
for sc, lbl, color in [(0, "No Retaliation", PALETTE[0]),
                        (4, "Optimal Retaliation", PALETTE[1]),
                        (5, "Reciprocal Retaliation", PALETTE[3])]:
    sub = ge[(ge["scenario_id"] == sc) & (ge["iso3"].isin(major))]
    vals = [sub[sub["iso3"]==c]["welfare_pct"].values[0] if len(sub[sub["iso3"]==c]) else 0
            for c in major]
    x_pos = np.arange(len(major))
    offset = [-0.28, 0, 0.28][[0, 4, 5].index(sc)]
    ax.bar(x_pos + offset, vals, 0.28, label=lbl, color=color, alpha=0.85)

ax.set_xticks(np.arange(len(major)))
ax.set_xticklabels(major)
ax.axhline(0, color="black", lw=1)
ax.set_ylabel("Welfare Change (%)")
ax.set_title("Retaliation Impact on Major Economies", fontweight="bold")
ax.legend()
plt.tight_layout()
save(fig, "3e_retaliation_impact.png")


# ===========================================================================
# SECTION 4 — SURROGATE TRAINING DATA EDA
# ===========================================================================
section("4. SURROGATE TRAINING DATA EDA")

train = bq.get_surrogate_training()
note(f"Training rows: {len(train):,}")
note(f"Features: us_tariff, china_rate, eu_rate, canmex_rate")
note(f"Targets : welfare_us, cpi_us, employment_us, trade_deficit_us, hhi_pharma, welfare_china, welfare_eu")

features = ["us_tariff", "china_rate", "eu_rate", "canmex_rate"]
targets  = ["welfare_us", "cpi_us", "employment_us",
            "trade_deficit_us", "welfare_china", "welfare_eu"]

# --- 4a. Feature distributions ---
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for ax, feat in zip(axes, features):
    ax.hist(train[feat], bins=20, color=PALETTE[0], edgecolor="white", linewidth=0.5)
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.set_xlabel("Rate")
    ax.set_ylabel("Count")
    vals = train[feat].unique()
    note(f"  {feat}: {len(vals)} unique values  [{vals.min():.2f} – {vals.max():.2f}]")
fig.suptitle("Surrogate Training — Input Feature Distributions", fontsize=12, fontweight="bold")
plt.tight_layout()
save(fig, "4a_feature_distributions.png")

# --- 4b. Target distributions ---
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.flatten()
for ax, tgt in zip(axes, targets):
    ax.hist(train[tgt], bins=30, color=PALETTE[3], edgecolor="white", linewidth=0.5)
    ax.axvline(train[tgt].mean(),   color="red",   lw=1.5, linestyle="--",
               label=f'mean={train[tgt].mean():.2f}')
    ax.axvline(train[tgt].median(), color="blue",  lw=1.5, linestyle=":",
               label=f'median={train[tgt].median():.2f}')
    ax.set_title(tgt, fontsize=10, fontweight="bold")
    ax.set_xlabel("Value (%)")
    ax.legend(fontsize=7)
    note(f"  {tgt}: mean={train[tgt].mean():.3f}  std={train[tgt].std():.3f}  "
         f"min={train[tgt].min():.3f}  max={train[tgt].max():.3f}")
fig.suptitle("Surrogate Training — Target Variable Distributions", fontsize=12, fontweight="bold")
plt.tight_layout()
save(fig, "4b_target_distributions.png")

# --- 4c. Correlation heatmap (features + targets) ---
all_cols = features + targets
corr = train[all_cols].corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, ax=ax, cmap="coolwarm", vmin=-1, vmax=1,
            annot=True, fmt=".2f", annot_kws={"size": 8},
            linewidths=0.4, linecolor="white",
            cbar_kws={"label": "Pearson r"})
ax.set_title("Correlation Heatmap — Features & Targets (Surrogate Training Data)",
             fontweight="bold")
plt.tight_layout()
save(fig, "4c_correlation_heatmap.png")

strongest = corr.loc[targets, features].abs().stack().idxmax()
note(f"Strongest feature-target correlation: {strongest[1]} -> {strongest[0]}  "
     f"r={corr.loc[strongest[0], strongest[1]]:.3f}")

# --- 4d. Key nonlinear relationships ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
pairs = [("us_tariff", "welfare_us"), ("us_tariff", "cpi_us"), ("china_rate", "welfare_us")]
for ax, (xf, yt) in zip(axes, pairs):
    sub = train[train[["us_tariff","china_rate","eu_rate","canmex_rate"]]
                .drop(columns=xf).apply(lambda row: all(row == 0), axis=1)]
    if len(sub) < 5:
        sub = train
    ax.scatter(sub[xf], sub[yt], alpha=0.35, s=12, color=PALETTE[0])
    ax.set_xlabel(xf)
    ax.set_ylabel(yt)
    ax.set_title(f"{xf} → {yt}", fontweight="bold")
fig.suptitle("Key Input-Output Relationships (others held at 0)", fontsize=11, fontweight="bold")
plt.tight_layout()
save(fig, "4d_feature_target_scatter.png")


# ===========================================================================
# SECTION 5 — SURROGATE BIAS ANALYSIS
# ===========================================================================
section("5. SURROGATE BIAS ANALYSIS (raw vs corrected vs true)")

val = bq.get_surrogate_validation()

note(f"Validation scenarios: {len(val)}")
note(f"Columns: {list(val.columns)}")

targets_v = [c.replace("_true", "") for c in val.columns if c.endswith("_true")]

# --- 5a. Raw vs corrected vs true for welfare_us ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sc_labels = val["scenario_name"].str.replace("–","-").str.replace("—","-")

for ax, col in zip(axes, ["welfare_us", "cpi_us"]):
    if f"{col}_true" not in val.columns:
        continue
    true_v = val[f"{col}_true"]
    pred_v = val[f"{col}_pred"]

    x = np.arange(len(val))
    w = 0.28
    ax.bar(x - w,   true_v, w, label="True GE",       color=PALETTE[2], alpha=0.9)
    ax.bar(x,       pred_v, w, label="Surrogate Pred", color=PALETTE[0], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(sc_labels, rotation=40, ha="right", fontsize=7.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(f"{col} (%)")
    ax.set_title(f"{col}: True vs Surrogate", fontweight="bold")
    ax.legend(fontsize=8)

fig.suptitle("Surrogate Model Bias Analysis — True vs Predicted",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save(fig, "5a_bias_true_vs_pred.png")

# --- 5b. Error scatter (true vs pred for all targets in validation) ---
n_tgts = len(targets_v)
ncols = min(3, n_tgts)
nrows = (n_tgts + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
axes = np.array(axes).flatten() if n_tgts > 1 else [axes]

for ax, tgt in zip(axes, targets_v):
    tc = f"{tgt}_true"
    pc = f"{tgt}_pred"
    if tc not in val.columns or pc not in val.columns:
        ax.axis("off")
        continue
    ax.scatter(val[tc], val[pc], color=PALETTE[0], s=60, zorder=3)
    lims = [min(val[tc].min(), val[pc].min()) - 0.5,
            max(val[tc].max(), val[pc].max()) + 0.5]
    ax.plot(lims, lims, "r--", lw=1.2, label="Perfect")
    for _, row in val.iterrows():
        ax.annotate(row["scenario_name"][:12], (row[tc], row[pc]),
                    fontsize=6, alpha=0.7)
    mae = (val[tc] - val[pc]).abs().mean()
    ax.set_xlabel(f"True {tgt}")
    ax.set_ylabel(f"Predicted {tgt}")
    ax.set_title(f"{tgt}  MAE={mae:.3f}", fontweight="bold", fontsize=9)
    ax.legend(fontsize=7)

for ax in axes[n_tgts:]:
    ax.axis("off")

fig.suptitle("Surrogate Validation — True vs Predicted (all targets)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save(fig, "5b_bias_scatter_all_targets.png")

for tgt in targets_v:
    tc = f"{tgt}_true";  pc = f"{tgt}_pred"
    if tc in val.columns and pc in val.columns:
        mae = (val[tc] - val[pc]).abs().mean()
        bias = (val[pc] - val[tc]).mean()
        note(f"  {tgt:<22}  MAE={mae:.4f}  mean_bias={bias:+.4f}")

# --- 5c. Sign accuracy (do retaliation scenarios get correct direction?) ---
ret_sc = val[val["scenario_name"].str.contains("Retaliation|Nash", case=False, na=False)]
if "welfare_us_true" in val.columns and "welfare_us_pred" in val.columns and len(ret_sc):
    sign_match = (np.sign(ret_sc["welfare_us_true"]) == np.sign(ret_sc["welfare_us_pred"])).sum()
    note(f"\n  Retaliation scenarios: {len(ret_sc)} — sign correct in {sign_match}/{len(ret_sc)} cases")
    note(f"  (true welfare_us for retaliation: {ret_sc['welfare_us_true'].tolist()})")


# ===========================================================================
# SECTION 6 — OECD ICIO SUPPLY CHAIN EDA
# ===========================================================================
section("6. OECD ICIO SUPPLY CHAIN EDA (2022)")

note("Computing US sector IO multipliers via BigQuery aggregation...")
mults = bq.compute_icio_multipliers(year=2022)

mult_df = pd.DataFrame([
    {"sector": k, "import_share": v["import_share_interm"], "io_multiplier": v["io_multiplier"]}
    for k, v in mults.items()
]).sort_values("import_share", ascending=False)

note(f"US sectors analysed: {len(mult_df)}")
note(f"Most import-dependent: {mult_df.iloc[0]['sector']} "
     f"(share={mult_df.iloc[0]['import_share']:.3f})")
note(f"Least import-dependent: {mult_df.iloc[-1]['sector']} "
     f"(share={mult_df.iloc[-1]['import_share']:.3f})")

fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(mult_df)*0.35 + 1)))

colors_mult = [PALETTE[1] if v > mult_df["import_share"].median() else PALETTE[0]
               for v in mult_df["import_share"]]
axes[0].barh(mult_df["sector"], mult_df["import_share"], color=colors_mult, alpha=0.85)
axes[0].axvline(mult_df["import_share"].median(), color="black", lw=1.2,
                linestyle="--", label="Median")
axes[0].set_xlabel("Intermediate Import Share")
axes[0].set_title("US Sector — Intermediate Import Share\n(OECD ICIO 2022)",
                  fontweight="bold")
axes[0].legend()
axes[0].invert_yaxis()

axes[1].barh(mult_df["sector"], mult_df["io_multiplier"], color=PALETTE[2], alpha=0.85)
axes[1].axvline(1.0, color="black", lw=1, linestyle=":")
axes[1].set_xlabel("IO Multiplier")
axes[1].set_title("Leontief IO Multiplier\n(1 = no supply-chain amplification)",
                  fontweight="bold")
axes[1].invert_yaxis()

plt.tight_layout()
save(fig, "6a_icio_sector_multipliers.png")

# --- 6b. Top cross-country supply chain flows into USA ---
note("Fetching top bilateral IO flows into USA...")
flows_usa = bq.get_icio_bilateral_flows(dest_country="USA", year=2022, top_n=20)

if not flows_usa.empty:
    flows_usa["label"] = flows_usa["source_country"] + " → USA\n(" + flows_usa["source_sector"] + ")"
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(flows_usa["label"], flows_usa["flow_usd_millions"] / 1000,
            color=PALETTE[0], alpha=0.85)
    ax.set_xlabel("Flow (USD billions)")
    ax.set_title("Top 20 Supply Chain Flows Into USA (OECD ICIO 2022)",
                 fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    save(fig, "6b_icio_top_flows_usa.png")
    note(f"Largest flow: {flows_usa.iloc[0]['source_country']} -> {flows_usa.iloc[0]['dest_sector']}  "
         f"${flows_usa.iloc[0]['flow_usd_millions']/1000:.1f}B")


# ===========================================================================
# SECTION 7 — GRAVITY & TRADE STRUCTURE EDA
# ===========================================================================
section("7. GRAVITY & TRADE STRUCTURE EDA")

note("Fetching US gravity variables (2019)...")
grav = bq.get_gravity(origin="USA", year=2019)

note(f"US gravity pairs: {len(grav)}")

# Join with BACI import values
baci_us = bq.query("""
    SELECT exporter_iso3 AS iso3_d,
           ROUND(SUM(value_1000usd)/1e6, 2) AS imports_usd_bn
    FROM `liberation-day-analysis.liberation_day.baci_trade`
    WHERE importer_iso3 = 'USA'
    GROUP BY exporter_iso3
""")

grav = grav.merge(baci_us, on="iso3_d", how="left")
grav = grav[grav["imports_usd_bn"].notna() & (grav["imports_usd_bn"] > 0)]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Distance vs imports (log-log)
grav_plot = grav[grav["distance"].notna() & (grav["distance"] > 0)]
axes[0].scatter(np.log(grav_plot["distance"]),
                np.log(grav_plot["imports_usd_bn"]),
                alpha=0.5, s=18, color=PALETTE[0])
# Trend line
x_ = np.log(grav_plot["distance"].astype(float))
y_ = np.log(grav_plot["imports_usd_bn"].astype(float))
mask_ = np.isfinite(x_) & np.isfinite(y_)
if mask_.sum() > 5:
    coef = np.polyfit(x_[mask_], y_[mask_], 1)
    xfit = np.linspace(x_[mask_].min(), x_[mask_].max(), 100)
    axes[0].plot(xfit, np.polyval(coef, xfit), "r-", lw=2,
                 label=f"slope={coef[0]:.2f}")
    note(f"  Distance elasticity of trade: {coef[0]:.3f} (log-log slope)")
axes[0].set_xlabel("ln(Distance km)")
axes[0].set_ylabel("ln(US Imports USD bn)")
axes[0].set_title("Gravity: Distance vs US Imports (log-log)", fontweight="bold")
axes[0].legend()

# WTO membership effect
if "member_wto_d" in grav.columns:
    wto_yes = grav[grav["member_wto_d"] == True]["imports_usd_bn"].dropna()
    wto_no  = grav[grav["member_wto_d"] == False]["imports_usd_bn"].dropna()
    axes[1].boxplot([np.log1p(wto_yes), np.log1p(wto_no)],
                    labels=["WTO Member", "Non-Member"],
                    patch_artist=True,
                    boxprops=dict(facecolor=PALETTE[0], alpha=0.7))
    axes[1].set_ylabel("ln(1 + US Imports USD bn)")
    axes[1].set_title("WTO Membership vs US Import Volume", fontweight="bold")
    note(f"  WTO members median imports: ${wto_yes.median():.2f}B")
    note(f"  Non-WTO members median imports: ${wto_no.median():.2f}B")
else:
    axes[1].axis("off")

plt.tight_layout()
save(fig, "7a_gravity_distance_wto.png")

# --- 7b. Top partners: distance + tariff + trade (bubble chart) ---
top20_grav = grav.nlargest(20, "imports_usd_bn")
top20_grav = top20_grav.merge(tariffs[["iso3","tariff_pct"]],
                               left_on="iso3_d", right_on="iso3", how="left")

fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(
    top20_grav["distance"],
    top20_grav["tariff_pct"].fillna(10),
    s=top20_grav["imports_usd_bn"] / top20_grav["imports_usd_bn"].max() * 1200 + 40,
    c=top20_grav["imports_usd_bn"], cmap="YlOrRd",
    alpha=0.8, edgecolors="grey", lw=0.5
)
for _, row in top20_grav.iterrows():
    ax.annotate(row["iso3_d"], (row["distance"], row.get("tariff_pct", 10)),
                fontsize=8, ha="center", va="bottom")
plt.colorbar(sc, label="US Imports (USD bn)")
ax.set_xlabel("Distance from USA (km)")
ax.set_ylabel("Liberation Day Tariff Rate (%)")
ax.set_title("Top 20 US Trade Partners\n(bubble size = import volume)", fontweight="bold")
plt.tight_layout()
save(fig, "7b_gravity_bubble_chart.png")


# ===========================================================================
# WRITE REPORT
# ===========================================================================
section("EDA COMPLETE — FIGURES SAVED")

fig_files = sorted(os.listdir(FIG_DIR))
note(f"\n{len(fig_files)} figures saved to  eda/figures/:")
for f in fig_files:
    note(f"  {f}")

report_lines.append(
    "\n\nGenerated by eda/eda_bigquery.py — Liberation Day Replication Project\n"
    "All data sourced from BigQuery: liberation-day-analysis.liberation_day\n"
)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"\n[Done]  Report saved to: {REPORT_PATH}")
print(f"        Figures saved to: {FIG_DIR}")
