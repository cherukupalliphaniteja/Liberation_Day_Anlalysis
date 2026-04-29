"""
retail_brand_actual.py
======================
Brand impact chart built 100% from retail_prices table in BigQuery.
Source file: project_data/retail_prices_illustrative.csv (now in BQ).
19 brands, 1000 products, before/after tariff prices.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
FIG_DIR    = os.path.join(SCRIPT_DIR, "figures")
sys.path.insert(0, REPO_ROOT)
from database import bq_client as bq
os.makedirs(FIG_DIR, exist_ok=True)

DARK = "#0d1117"; CARD = "#161b22"
TW   = "#e6edf3"; TG   = "#8b949e"
RED  = "#f85149"; ORG  = "#fb8500"
YEL  = "#ffd166"; GRN  = "#3fb950"
BLU  = "#58a6ff"; PUR  = "#bc8cff"
TEAL = "#39d353"

SECTOR_COL = {
    "Electronics":     BLU,
    "Footwear":        RED,
    "Clothing":        PUR,
    "Home Appliances": ORG,
    "Grocery":         GRN,
}

def dark_ax(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TG, labelsize=9)
    ax.xaxis.label.set_color(TG); ax.yaxis.label.set_color(TG)
    ax.title.set_color(TW)
    for s in ax.spines.values(): s.set_edgecolor("#30363d")
    ax.grid(color="#21262d", lw=0.5, alpha=0.9)
    return ax

# ---------------------------------------------------------------------------
# Pull from BigQuery
# ---------------------------------------------------------------------------
print("Fetching from BigQuery: liberation_day.retail_prices ...")

# Brand-level summary
brand_summary = bq.query("""
    SELECT
        brand_name,
        product_type,
        COUNT(*)                              AS n_products,
        ROUND(AVG(price_increase_pct), 2)     AS avg_pct,
        ROUND(MIN(price_increase_pct), 2)     AS min_pct,
        ROUND(MAX(price_increase_pct), 2)     AS max_pct,
        ROUND(AVG(price_before), 2)           AS avg_before,
        ROUND(AVG(price_after),  2)           AS avg_after,
        ROUND(AVG(price_after - price_before), 2) AS avg_dollar_rise
    FROM `liberation-day-analysis.liberation_day.retail_prices`
    GROUP BY brand_name, product_type
    ORDER BY avg_pct DESC
""")

# Overall brand (across all product types)
brand_total = bq.query("""
    SELECT
        brand_name,
        COUNT(*)                              AS n_products,
        ROUND(AVG(price_increase_pct), 2)     AS avg_pct,
        ROUND(MIN(price_increase_pct), 2)     AS min_pct,
        ROUND(MAX(price_increase_pct), 2)     AS max_pct,
        ROUND(AVG(price_before), 2)           AS avg_before,
        ROUND(AVG(price_after - price_before), 2) AS avg_dollar_rise,
        ANY_VALUE(product_type)               AS product_type
    FROM `liberation-day-analysis.liberation_day.retail_prices`
    GROUP BY brand_name
    ORDER BY avg_pct DESC
""")

# Product-type summary
type_summary = bq.query("""
    SELECT
        product_type,
        COUNT(*)                              AS n_products,
        ROUND(AVG(price_increase_pct), 2)     AS avg_pct,
        ROUND(AVG(price_before), 2)           AS avg_before,
        ROUND(AVG(price_after - price_before), 2) AS avg_dollar_rise
    FROM `liberation-day-analysis.liberation_day.retail_prices`
    GROUP BY product_type
    ORDER BY avg_pct DESC
""")

# Distribution of price increases per brand (for box plots)
all_products = bq.query("""
    SELECT brand_name, product_type, price_before, price_after, price_increase_pct
    FROM `liberation-day-analysis.liberation_day.retail_prices`
    ORDER BY brand_name
""")

print(f"  Brands: {brand_total['brand_name'].nunique()}")
print(f"  Products: {len(all_products)}")

# ===========================================================================
# FIGURE
# ===========================================================================
fig = plt.figure(figsize=(22, 15), facecolor=DARK)
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.52, wspace=0.40,
                        left=0.05, right=0.97,
                        top=0.90, bottom=0.05)

fig.text(0.5, 0.956,
         "Hardest-Hit Brands — Liberation Day Tariff Price Impact",
         ha="center", fontsize=22, fontweight="bold", color=TW)
fig.text(0.5, 0.925,
         "Source: retail_prices table in BigQuery  |  "
         "retail_prices_illustrative.csv (project_data/)  |  "
         "1,000 products  |  19 brands  |  price_before vs price_after tariff",
         ha="center", fontsize=11, color=TG)

# ---- Panel A: Brand avg price increase — main ranked bar (top 2 rows) ----
ax_a = dark_ax(fig.add_subplot(gs[:2, :2]))

bt = brand_total.sort_values("avg_pct", ascending=True)
cols_a = [SECTOR_COL.get(pt, TG) for pt in bt["product_type"]]
bars = ax_a.barh(bt["brand_name"], bt["avg_pct"],
                 color=cols_a, edgecolor=DARK, linewidth=0.8,
                 height=0.65, zorder=3)

for bar, row in zip(bars, bt.itertuples()):
    ax_a.text(
        bar.get_width() + 0.3,
        bar.get_y() + bar.get_height()/2,
        f"  +{row.avg_pct:.1f}%   |   avg +${row.avg_dollar_rise:.0f} per item   "
        f"|   {row.n_products} products   |   {row.product_type}",
        va="center", fontsize=9, color=TW
    )

ax_a.axvline(brand_total["avg_pct"].mean(), color="white", lw=1.5,
             linestyle="--", alpha=0.5,
             label=f"Overall avg: {brand_total['avg_pct'].mean():.1f}%")
ax_a.set_xlabel("Average Price Increase Across All Products (%)", fontsize=11)
ax_a.set_title(
    "Average Price Increase by Brand — From retail_prices (BigQuery)\n"
    "Computed as: mean((price_after - price_before) / price_before × 100) per brand",
    fontsize=11, fontweight="bold", color=TW, pad=10
)
ax_a.tick_params(axis="y", colors=TW, labelsize=11)
ax_a.set_xlim(0, bt["avg_pct"].max() * 1.55)
ax_a.legend(fontsize=9, facecolor=CARD, labelcolor=TW, edgecolor="#30363d")

legend_patches = [mpatches.Patch(color=v, label=k) for k, v in SECTOR_COL.items()]
ax_a.legend(handles=legend_patches, loc="lower right", fontsize=9,
            facecolor=CARD, labelcolor=TW, edgecolor="#30363d")

# ---- Panel B: Product type avg increase (top right) ----
ax_b = dark_ax(fig.add_subplot(gs[0, 2]))

ts = type_summary.sort_values("avg_pct", ascending=True)
cols_b = [SECTOR_COL.get(pt, TG) for pt in ts["product_type"]]
bars_b = ax_b.barh(ts["product_type"], ts["avg_pct"],
                   color=cols_b, edgecolor=DARK, lw=0.8, height=0.55, zorder=3)
for bar, row in zip(bars_b, ts.itertuples()):
    ax_b.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
              f"+{row.avg_pct:.1f}%", va="center",
              fontsize=10, fontweight="bold", color=TW)
ax_b.set_xlabel("Avg Price Increase (%)", color=TG)
ax_b.set_title("By Product Category",
               fontsize=11, fontweight="bold", color=TW, pad=8)
ax_b.tick_params(axis="y", colors=TW, labelsize=10)
ax_b.set_xlim(0, ts["avg_pct"].max() * 1.35)

# ---- Panel C: Dollar rise per item by brand (mid right) ----
ax_c = dark_ax(fig.add_subplot(gs[1, 2]))

bt_d = brand_total.sort_values("avg_dollar_rise", ascending=True)
cols_c = [SECTOR_COL.get(pt, TG) for pt in bt_d["product_type"]]
bars_c = ax_c.barh(bt_d["brand_name"], bt_d["avg_dollar_rise"],
                   color=cols_c, edgecolor=DARK, lw=0.8, height=0.65, zorder=3)
for bar, val in zip(bars_c, bt_d["avg_dollar_rise"]):
    ax_c.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
              f"+${val:.0f}", va="center", fontsize=9,
              fontweight="bold", color=TW)
ax_c.set_xlabel("Avg Dollar Rise Per Item (USD)", color=TG)
ax_c.set_title("Avg Dollar Added\nPer Product at Checkout",
               fontsize=11, fontweight="bold", color=TW, pad=8)
ax_c.tick_params(axis="y", colors=TW, labelsize=9)

# ---- Panel D: Box plot — spread of price increases per brand (bottom full) ----
ax_d = dark_ax(fig.add_subplot(gs[2, :]))

brands_ordered = brand_total.sort_values("avg_pct", ascending=False)["brand_name"].tolist()
box_data = [
    all_products[all_products["brand_name"]==b]["price_increase_pct"].values
    for b in brands_ordered
]
bp = ax_d.boxplot(box_data, patch_artist=True, notch=False,
                  medianprops=dict(color="white", lw=2),
                  whiskerprops=dict(color=TG),
                  capprops=dict(color=TG),
                  flierprops=dict(marker="o", color=TG,
                                  markerfacecolor=TG, markersize=3, alpha=0.5))

sector_of = dict(zip(brand_total["brand_name"], brand_total["product_type"]))
for patch, brand in zip(bp["boxes"], brands_ordered):
    patch.set_facecolor(SECTOR_COL.get(sector_of.get(brand, ""), TG))
    patch.set_alpha(0.8)

ax_d.set_xticks(range(1, len(brands_ordered)+1))
ax_d.set_xticklabels(brands_ordered, rotation=30, ha="right",
                     fontsize=9.5, color=TW)
ax_d.set_ylabel("Price Increase % per Product", color=TG, fontsize=10)
ax_d.set_title(
    "Distribution of Price Increases Across All Products — By Brand\n"
    "(box = 25th–75th percentile  |  line = median  |  whiskers = min/max)",
    fontsize=11, fontweight="bold", color=TW, pad=8
)

fig.text(0.5, 0.022,
         "Data: liberation_day.retail_prices (BigQuery)  |  "
         "Original file: project_data/retail_prices_illustrative.csv  |  "
         "1,000 products across 19 brands and 5 product categories  |  "
         "Price increase = (price_after - price_before) / price_before x 100",
         ha="center", fontsize=8.5, color=TG, style="italic")

out = os.path.join(FIG_DIR, "retail_BRANDS_from_data.png")
fig.savefig(out, dpi=155, bbox_inches="tight", facecolor=DARK)
plt.close(fig)
print(f"\n[saved] retail_BRANDS_from_data.png")

print("\n=== BRAND SUMMARY FROM OUR DATA ===")
print(f"{'Brand':<15} {'Category':<18} {'Products':>8} "
      f"{'Avg Rise':>9} {'Min':>7} {'Max':>7} {'Avg $Rise':>9}")
print("-"*80)
for _, r in brand_total.iterrows():
    print(f"  {r['brand_name']:<13} {r['product_type']:<18} {r['n_products']:>8} "
          f"{r['avg_pct']:>8.1f}% {r['min_pct']:>6.1f}% "
          f"{r['max_pct']:>6.1f}%  +${r['avg_dollar_rise']:>7.0f}")
