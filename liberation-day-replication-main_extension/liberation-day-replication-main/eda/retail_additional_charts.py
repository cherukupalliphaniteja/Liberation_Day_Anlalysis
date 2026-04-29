"""
retail_additional_charts.py
===========================
6 additional retail charts from liberation_day.retail_prices (BigQuery).
All data: retail_prices_illustrative.csv → BigQuery.
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
TEAL = "#39d353"; PINK = "#ff7eb6"

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
    ax.xaxis.label.set_color(TG)
    ax.yaxis.label.set_color(TG)
    ax.title.set_color(TW)
    for s in ax.spines.values():
        s.set_edgecolor("#30363d")
    ax.grid(color="#21262d", lw=0.5, alpha=0.8)
    return ax

def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"[saved] {name}")

# ---------------------------------------------------------------------------
# Pull all data once
# ---------------------------------------------------------------------------
print("Fetching retail_prices from BigQuery...")
df = bq.query("SELECT * FROM `liberation-day-analysis.liberation_day.retail_prices`")
df["dollar_rise"] = df["price_after"] - df["price_before"]
df["price_band"]  = pd.cut(
    df["price_before"],
    bins=[0, 100, 300, 600, 1500],
    labels=["Budget\n<$100", "Mid\n$100-300", "Premium\n$300-600", "Luxury\n>$600"]
)
print(f"  {len(df)} products loaded")

# Pre-compute aggregates used across charts
brand_avg = df.groupby("brand_name")["price_increase_pct"].mean().sort_values(ascending=False)
type_avg  = df.groupby("product_type")["price_increase_pct"].mean().sort_values(ascending=False)
brand_dollar = df.groupby("brand_name")["dollar_rise"].mean().sort_values(ascending=False)

# ============================================================
# IMAGE 1 — 4 charts: scatter, price band, within-category,
#           cumulative distribution
# ============================================================
fig1 = plt.figure(figsize=(20, 13), facecolor=DARK)
gs1  = gridspec.GridSpec(2, 2, figure=fig1,
                         hspace=0.48, wspace=0.35,
                         left=0.06, right=0.97,
                         top=0.90, bottom=0.06)

fig1.text(0.5, 0.955, "Retail Tariff Impact — Deep Dive (Part 1 of 2)",
          ha="center", fontsize=20, fontweight="bold", color=TW)
fig1.text(0.5, 0.927,
          "Source: liberation_day.retail_prices  |  1,000 products  |  19 brands",
          ha="center", fontsize=11, color=TG)

# ---- Chart 1: Before vs After scatter (product-level) ----
ax1 = dark_ax(fig1.add_subplot(gs1[0, 0]))

for ptype, col in SECTOR_COL.items():
    sub = df[df["product_type"] == ptype]
    ax1.scatter(sub["price_before"], sub["price_after"],
                alpha=0.45, s=18, color=col, label=ptype, zorder=3)

# Perfect-pass-through line (price_after = price_before × 1.30)
x_range = np.linspace(df["price_before"].min(), df["price_before"].max(), 100)
ax1.plot(x_range, x_range,         color=TG,  lw=1.2, linestyle=":", label="No change")
ax1.plot(x_range, x_range * 1.30,  color=YEL, lw=1.5, linestyle="--", label="+30% line")
ax1.plot(x_range, x_range * 1.50,  color=RED, lw=1.5, linestyle="--", label="+50% line")

ax1.set_xlabel("Price Before Tariff (USD)")
ax1.set_ylabel("Price After Tariff (USD)")
ax1.set_title("Every Product: Before vs After Price\n(1,000 items plotted)",
              fontweight="bold", color=TW, pad=8)
ax1.legend(fontsize=7.5, facecolor=CARD, labelcolor=TW,
           edgecolor="#30363d", ncol=2)

# ---- Chart 2: Dollar impact by price band ----
ax2 = dark_ax(fig1.add_subplot(gs1[0, 1]))

band_stats = df.groupby("price_band", observed=True).agg(
    avg_pct    = ("price_increase_pct", "mean"),
    avg_dollar = ("dollar_rise",        "mean"),
    count      = ("row_id",             "count")
).reset_index()

x  = np.arange(len(band_stats))
w  = 0.38
b1 = ax2.bar(x - w/2, band_stats["avg_pct"],    w,
             color=BLU,  alpha=0.85, label="Avg % Increase",
             edgecolor=DARK, zorder=3)
ax2_r = ax2.twinx()
ax2_r.set_facecolor(CARD)
b2 = ax2_r.bar(x + w/2, band_stats["avg_dollar"], w,
               color=ORG, alpha=0.85, label="Avg $ Added",
               edgecolor=DARK, zorder=3)
ax2_r.tick_params(colors=TG)
ax2_r.yaxis.label.set_color(TG)
ax2_r.set_ylabel("Avg Dollar Added (USD)", color=TG)
ax2_r.spines["right"].set_edgecolor("#30363d")
ax2_r.spines["top"].set_edgecolor("#30363d")
ax2_r.spines["left"].set_edgecolor("#30363d")
ax2_r.spines["bottom"].set_edgecolor("#30363d")

for bar, row in zip(b1, band_stats.itertuples()):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f"{row.avg_pct:.1f}%", ha="center", fontsize=8.5,
             fontweight="bold", color=BLU)
for bar, row in zip(b2, band_stats.itertuples()):
    ax2_r.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
               f"+${row.avg_dollar:.0f}", ha="center", fontsize=8.5,
               fontweight="bold", color=ORG)

ax2.set_xticks(x)
ax2.set_xticklabels(band_stats["price_band"], fontsize=9, color=TW)
ax2.set_ylabel("Avg % Increase", color=TG)
ax2.set_title("Price Band Impact\n(% similar across bands — $ impact is what hurts)",
              fontweight="bold", color=TW, pad=8)
lines = [mpatches.Patch(color=BLU, label="Avg % Increase"),
         mpatches.Patch(color=ORG, label="Avg Dollar Added")]
ax2.legend(handles=lines, fontsize=8.5, facecolor=CARD,
           labelcolor=TW, edgecolor="#30363d")

# ---- Chart 3: Within-category brand comparison ----
ax3 = dark_ax(fig1.add_subplot(gs1[1, 0]))

# Footwear brands vs Electronics brands side by side
footwear = df[df["product_type"]=="Footwear"].groupby("brand_name")["price_increase_pct"].mean()
clothing  = df[df["product_type"]=="Clothing"].groupby("brand_name")["price_increase_pct"].mean()
electronics = df[df["product_type"]=="Electronics"].groupby("brand_name")["price_increase_pct"].mean()

groups = {
    "Footwear":    (footwear,    RED),
    "Clothing":    (clothing,    PUR),
    "Electronics": (electronics, BLU),
}

offset = 0
tick_positions, tick_labels = [], []
for gname, (series, col) in groups.items():
    s = series.sort_values(ascending=False)
    positions = [offset + i for i in range(len(s))]
    bars = ax3.bar(positions, s.values, color=col, alpha=0.85,
                   edgecolor=DARK, lw=0.7, label=gname, width=0.7, zorder=3)
    for pos, val, brand in zip(positions, s.values, s.index):
        ax3.text(pos, val+0.2, f"+{val:.1f}%", ha="center",
                 fontsize=8, color=TW, fontweight="bold")
    tick_positions.extend(positions)
    tick_labels.extend(s.index.tolist())
    offset += len(s) + 1.5

ax3.set_xticks(tick_positions)
ax3.set_xticklabels(tick_labels, rotation=35, ha="right",
                    fontsize=9, color=TW)
ax3.set_ylabel("Avg Price Increase (%)")
ax3.set_title("Brand vs Brand — Within Same Category\n"
              "(Footwear | Clothing | Electronics)",
              fontweight="bold", color=TW, pad=8)
ax3.legend(fontsize=9, facecolor=CARD, labelcolor=TW, edgecolor="#30363d")
ax3.axhline(df["price_increase_pct"].mean(), color=YEL, lw=1.5,
            linestyle="--", alpha=0.7, label="Overall avg")

# ---- Chart 4: Cumulative distribution ----
ax4 = dark_ax(fig1.add_subplot(gs1[1, 1]))

for ptype, col in SECTOR_COL.items():
    sub = df[df["product_type"]==ptype]["price_increase_pct"].sort_values()
    cdf = np.arange(1, len(sub)+1) / len(sub) * 100
    ax4.plot(sub, cdf, color=col, lw=2.5, label=f"{ptype} (n={len(sub)})", zorder=3)

ax4.axvline(30, color=YEL, lw=1.5, linestyle="--", alpha=0.8, label="+30% threshold")
ax4.axvline(40, color=RED, lw=1.5, linestyle="--", alpha=0.8, label="+40% threshold")
ax4.set_xlabel("Price Increase (%)")
ax4.set_ylabel("% of Products Below This Increase")
ax4.set_title("Cumulative Distribution — Price Increases\n"
              "(How many products face >30% or >40% rise?)",
              fontweight="bold", color=TW, pad=8)
ax4.legend(fontsize=8.5, facecolor=CARD, labelcolor=TW, edgecolor="#30363d")

pct_above_30 = (df["price_increase_pct"] > 30).mean() * 100
pct_above_40 = (df["price_increase_pct"] > 40).mean() * 100
ax4.text(30.5, 15, f"{pct_above_30:.0f}% of\nproducts >30%",
         color=YEL, fontsize=9, fontweight="bold")
ax4.text(40.5, 30, f"{pct_above_40:.0f}% of\nproducts >40%",
         color=RED, fontsize=9, fontweight="bold")

fig1.text(0.5, 0.025,
          "Data: liberation_day.retail_prices (BigQuery)  |  "
          "Source: project_data/retail_prices_illustrative.csv",
          ha="center", fontsize=8.5, color=TG, style="italic")
save(fig1, "retail_CHARTS_part1.png")

# ============================================================
# IMAGE 2 — 4 charts: heatmap, grocery focus,
#           high vs low tariff products, brand risk tiers
# ============================================================
fig2 = plt.figure(figsize=(20, 13), facecolor=DARK)
gs2  = gridspec.GridSpec(2, 2, figure=fig2,
                         hspace=0.48, wspace=0.38,
                         left=0.06, right=0.97,
                         top=0.90, bottom=0.06)

fig2.text(0.5, 0.955, "Retail Tariff Impact — Deep Dive (Part 2 of 2)",
          ha="center", fontsize=20, fontweight="bold", color=TW)
fig2.text(0.5, 0.927,
          "Source: liberation_day.retail_prices  |  1,000 products  |  19 brands",
          ha="center", fontsize=11, color=TG)

# ---- Chart 5: Heatmap — brand × product type avg increase ----
ax5 = dark_ax(fig2.add_subplot(gs2[0, :]))

import seaborn as sns
pivot = df.groupby(["brand_name", "product_type"])["price_increase_pct"].mean().unstack(fill_value=np.nan)
pivot = pivot.reindex(brand_avg.index)  # sort brands by overall avg

mask = pivot.isna()
sns.heatmap(pivot, ax=ax5, cmap="RdYlGn_r", vmin=10, vmax=50,
            annot=True, fmt=".1f", annot_kws={"size": 9, "color": "white"},
            linewidths=0.5, linecolor="#30363d",
            mask=mask,
            cbar_kws={"label": "Avg Price Increase (%)", "shrink": 0.7})
ax5.set_facecolor(CARD)
ax5.set_title("Brand × Product Category Heatmap — Avg Price Increase (%)\n"
              "(blank = brand does not sell in that category)",
              fontsize=12, fontweight="bold", color=TW, pad=10)
ax5.set_xlabel("Product Category", color=TG)
ax5.set_ylabel("Brand", color=TG)
ax5.tick_params(axis="x", colors=TW, labelsize=10)
ax5.tick_params(axis="y", colors=TW, labelsize=9)
ax5.collections[0].colorbar.ax.tick_params(colors=TG)
ax5.collections[0].colorbar.ax.yaxis.label.set_color(TG)

# ---- Chart 6: Grocery brand deep dive (everyday staples) ----
ax6 = dark_ax(fig2.add_subplot(gs2[1, 0]))

grocery = df[df["product_type"] == "Grocery"].copy()
g_brand = grocery.groupby("brand_name").agg(
    avg_pct    = ("price_increase_pct", "mean"),
    avg_before = ("price_before",        "mean"),
    avg_after  = ("price_after",         "mean"),
    avg_dollar = ("dollar_rise",         "mean"),
    count      = ("row_id",              "count")
).sort_values("avg_pct", ascending=False).reset_index()

g_cols = [GRN, TEAL, YEL, ORG]
x6 = np.arange(len(g_brand))
w6 = 0.38
b_bef = ax6.bar(x6 - w6/2, g_brand["avg_before"], w6,
                color=GRN, alpha=0.8, label="Avg Price Before", edgecolor=DARK)
b_aft = ax6.bar(x6 + w6/2, g_brand["avg_after"],  w6,
                color=RED, alpha=0.8, label="Avg Price After",  edgecolor=DARK)
for i, row in g_brand.iterrows():
    ax6.text(i+w6/2, row["avg_after"]+3,
             f"+{row['avg_pct']:.1f}%", ha="center",
             fontsize=9, color=RED, fontweight="bold")
ax6.set_xticks(x6)
ax6.set_xticklabels(g_brand["brand_name"], fontsize=10, color=TW)
ax6.set_ylabel("Average Product Price (USD)")
ax6.set_title("Grocery Brands — Before vs After\n"
              "(Everyday Staples: Cooking Oil, Rice — most regressive impact)",
              fontweight="bold", color=TW, pad=8)
ax6.legend(fontsize=9, facecolor=CARD, labelcolor=TW, edgecolor="#30363d")

# ---- Chart 7: Brand risk tiers ----
ax7 = dark_ax(fig2.add_subplot(gs2[1, 1]))

# Classify each brand by % products facing >30% increase
brand_risk = df.groupby("brand_name").apply(
    lambda x: pd.Series({
        "pct_above_30": (x["price_increase_pct"] > 30).mean() * 100,
        "pct_above_40": (x["price_increase_pct"] > 40).mean() * 100,
        "avg_pct":       x["price_increase_pct"].mean(),
        "product_type":  x["product_type"].iloc[0],
        "n":             len(x),
    })
).reset_index()

brand_risk = brand_risk.sort_values("pct_above_30", ascending=True)

bar_30 = ax7.barh(brand_risk["brand_name"], brand_risk["pct_above_30"],
                  color=[SECTOR_COL.get(pt, TG) for pt in brand_risk["product_type"]],
                  alpha=0.85, edgecolor=DARK, lw=0.7, height=0.65, zorder=3)
ax7.axvline(50, color=YEL, lw=1.5, linestyle="--",
            alpha=0.8, label="50% of products")
for bar, row in zip(bar_30, brand_risk.itertuples()):
    ax7.text(bar.get_width()+0.5,
             bar.get_y()+bar.get_height()/2,
             f"{row.pct_above_30:.0f}% of products >+30%",
             va="center", fontsize=8.5, color=TW)
ax7.set_xlabel("% of Brand's Products Facing >30% Price Rise")
ax7.set_title("Brand Risk Tier — Share of Products\nFacing >30% Price Increase",
              fontweight="bold", color=TW, pad=8)
ax7.tick_params(axis="y", colors=TW, labelsize=9.5)
ax7.set_xlim(0, 105)
ax7.legend(fontsize=9, facecolor=CARD, labelcolor=TW, edgecolor="#30363d")
legend_patches = [mpatches.Patch(color=v, label=k) for k, v in SECTOR_COL.items()]
ax7.legend(handles=legend_patches, fontsize=8, facecolor=CARD,
           labelcolor=TW, edgecolor="#30363d", loc="lower right")

fig2.text(0.5, 0.025,
          "Data: liberation_day.retail_prices (BigQuery)  |  "
          "Source: project_data/retail_prices_illustrative.csv",
          ha="center", fontsize=8.5, color=TG, style="italic")
save(fig2, "retail_CHARTS_part2.png")

# ---- Summary ----
print("\n=== KEY FINDINGS ===")
print(f"  Products facing >30% price rise: {pct_above_30:.0f}%")
print(f"  Products facing >40% price rise: {pct_above_40:.0f}%")
print(f"  Avg dollar added — Budget items  : +${df[df['price_band']=='Budget\n<$100']['dollar_rise'].mean():.0f}")
print(f"  Avg dollar added — Luxury items  : +${df[df['price_band']=='Luxury\n>$600']['dollar_rise'].mean():.0f}")
print(f"  Category with steepest rise: {type_avg.idxmax()} ({type_avg.max():.1f}%)")
print(f"  Category with mildest rise : {type_avg.idxmin()} ({type_avg.min():.1f}%)")
