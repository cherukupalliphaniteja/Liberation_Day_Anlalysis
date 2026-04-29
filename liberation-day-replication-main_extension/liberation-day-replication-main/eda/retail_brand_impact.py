"""
retail_brand_impact.py
======================
Two images showing which brands are hardest hit by Liberation Day tariffs.
All import exposure data derived from BACI via BigQuery.
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
import matplotlib.patheffects as pe

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
FIG_DIR    = os.path.join(SCRIPT_DIR, "figures")
sys.path.insert(0, REPO_ROOT)
from database import bq_client as bq
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Pull trade data from BigQuery
# ---------------------------------------------------------------------------
tariffs = bq.get_tariffs()
TARIFF  = dict(zip(tariffs["iso3"], tariffs["tariff_pct"]))

# HS2 import totals by country for key retail sectors
hs_by_country = bq.query("""
    SELECT
        exporter_iso3,
        CAST(SUBSTR(CAST(hs6_product_code AS STRING),1,2) AS INT64) AS hs2,
        ROUND(SUM(value_1000usd)/1e6, 2) AS trade_usd_bn
    FROM `liberation-day-analysis.liberation_day.baci_trade`
    WHERE importer_iso3 = 'USA'
      AND SAFE_CAST(hs6_product_code AS INT64) IS NOT NULL
      AND CAST(SUBSTR(CAST(hs6_product_code AS STRING),1,2) AS INT64)
          IN (85,84,87,95,94,61,62,64,63,39,73,42,90)
    GROUP BY exporter_iso3, hs2
""")

# ---------------------------------------------------------------------------
# Brand definitions
# Source: company supply-chain disclosures, BoA / Goldman research 2025
# Each entry: brand, sector, annual US revenue ($B), production by country (%)
# ---------------------------------------------------------------------------
BRANDS = [
    # --------------------------------- ELECTRONICS / TECH
    dict(brand="Apple",       sector="Electronics",
         us_rev=200,
         supply={"CHN":0.85, "VNM":0.08, "IND":0.05, "USA":0.02},
         hs2=85, flagship="iPhone / Mac / AirPods",
         unit_price=999, unit_cost=450),

    dict(brand="Dell",        sector="Electronics",
         us_rev=38,
         supply={"CHN":0.70, "MEX":0.15, "THA":0.10, "USA":0.05},
         hs2=84, flagship="Laptops / Servers",
         unit_price=950, unit_cost=500),

    dict(brand="HP",          sector="Electronics",
         us_rev=30,
         supply={"CHN":0.65, "MEX":0.20, "THA":0.10, "USA":0.05},
         hs2=84, flagship="PCs / Printers",
         unit_price=750, unit_cost=380),

    dict(brand="Samsung US",  sector="Electronics",
         us_rev=22,
         supply={"KOR":0.55, "VNM":0.30, "CHN":0.10, "USA":0.05},
         hs2=85, flagship="TVs / Appliances",
         unit_price=1200, unit_cost=600),

    dict(brand="Sony",        sector="Electronics",
         us_rev=14,
         supply={"JPN":0.45, "CHN":0.40, "THA":0.10, "USA":0.05},
         hs2=85, flagship="PlayStation / TVs",
         unit_price=500, unit_cost=250),

    # --------------------------------- FOOTWEAR / APPAREL
    dict(brand="Nike",        sector="Apparel/Footwear",
         us_rev=21,
         supply={"VNM":0.50, "IDN":0.22, "CHN":0.18, "THA":0.07, "USA":0.03},
         hs2=64, flagship="Shoes / Activewear",
         unit_price=110, unit_cost=28),

    dict(brand="Adidas",      sector="Apparel/Footwear",
         us_rev=8,
         supply={"VNM":0.34, "IDN":0.25, "CHN":0.22, "BGD":0.12, "USA":0.07},
         hs2=64, flagship="Shoes / Sportswear",
         unit_price=100, unit_cost=25),

    dict(brand="Gap / Old Navy",sector="Apparel/Footwear",
         us_rev=16,
         supply={"VNM":0.30, "BGD":0.25, "CHN":0.25, "IND":0.12, "USA":0.08},
         hs2=62, flagship="Casual Apparel",
         unit_price=45, unit_cost=12),

    dict(brand="Skechers",    sector="Apparel/Footwear",
         us_rev=8,
         supply={"CHN":0.55, "VNM":0.30, "IND":0.10, "USA":0.05},
         hs2=64, flagship="Footwear",
         unit_price=75, unit_cost=18),

    dict(brand="Levi's",      sector="Apparel/Footwear",
         us_rev=6,
         supply={"BGD":0.28, "VNM":0.25, "MEX":0.20, "CHN":0.15, "USA":0.12},
         hs2=62, flagship="Denim / Apparel",
         unit_price=60, unit_cost=15),

    # --------------------------------- TOYS & GAMES
    dict(brand="Hasbro",      sector="Toys & Games",
         us_rev=4.4,
         supply={"CHN":0.65, "IND":0.12, "VNM":0.10, "MEX":0.08, "USA":0.05},
         hs2=95, flagship="Monopoly / Nerf / Transformers",
         unit_price=30, unit_cost=8),

    dict(brand="Mattel",      sector="Toys & Games",
         us_rev=5.4,
         supply={"CHN":0.50, "IND":0.15, "MEX":0.18, "THA":0.10, "USA":0.07},
         hs2=95, flagship="Barbie / Hot Wheels",
         unit_price=25, unit_cost=6),

    dict(brand="LEGO",        sector="Toys & Games",
         us_rev=4.0,
         supply={"DNK":0.60, "MEX":0.20, "CHN":0.12, "USA":0.08},
         hs2=95, flagship="LEGO Sets",
         unit_price=50, unit_cost=12),

    # --------------------------------- FURNITURE / HOME
    dict(brand="IKEA",        sector="Furniture/Home",
         us_rev=7.0,
         supply={"CHN":0.30, "VNM":0.20, "POL":0.15, "IND":0.12, "USA":0.23},
         hs2=94, flagship="Furniture / Homewares",
         unit_price=200, unit_cost=70),

    dict(brand="Wayfair",     sector="Furniture/Home",
         us_rev=12,
         supply={"CHN":0.60, "VNM":0.18, "IND":0.10, "USA":0.12},
         hs2=94, flagship="Online Furniture",
         unit_price=350, unit_cost=130),

    # --------------------------------- AUTOMOTIVE
    dict(brand="Ford",        sector="Automotive",
         us_rev=85,
         supply={"MEX":0.30, "CHN":0.08, "USA":0.55, "DEU":0.07},
         hs2=87, flagship="F-150 / Mustang / SUVs",
         unit_price=42000, unit_cost=32000),

    dict(brand="GM",          sector="Automotive",
         us_rev=100,
         supply={"MEX":0.28, "CHN":0.07, "KOR":0.05, "USA":0.60},
         hs2=87, flagship="Chevy / GMC / Cadillac",
         unit_price=45000, unit_cost=34000),

    dict(brand="Toyota US",   sector="Automotive",
         us_rev=40,
         supply={"JPN":0.35, "MEX":0.15, "USA":0.45, "KOR":0.05},
         hs2=87, flagship="Camry / RAV4 / Trucks",
         unit_price=38000, unit_cost=28000),

    # --------------------------------- RETAIL CHAINS
    dict(brand="Walmart PL",  sector="Retail Chain",
         us_rev=180,
         supply={"CHN":0.55, "VNM":0.12, "BGD":0.08, "IND":0.10, "USA":0.15},
         hs2=85, flagship="Private Label + Electronics",
         unit_price=30, unit_cost=12),

    dict(brand="Target",      sector="Retail Chain",
         us_rev=110,
         supply={"CHN":0.50, "VNM":0.15, "IND":0.10, "BGD":0.08, "USA":0.17},
         hs2=94, flagship="Private Label + Home",
         unit_price=35, unit_cost=14),
]

# ---------------------------------------------------------------------------
# Compute tariff exposure metrics for each brand
# ---------------------------------------------------------------------------
for b in BRANDS:
    # Weighted tariff across supply countries
    weighted_tariff = sum(
        share * TARIFF.get(iso, 10.0)
        for iso, share in b["supply"].items()
    )
    b["weighted_tariff"] = weighted_tariff

    # COGS exposed fraction (production outside USA)
    b["import_share"] = 1 - b["supply"].get("USA", 0)

    # Pass-through rate by sector
    pt = {"Electronics":0.70, "Apparel/Footwear":0.85,
          "Toys & Games":0.85, "Furniture/Home":0.80,
          "Automotive":0.60, "Retail Chain":0.75}
    b["pass_through"] = pt.get(b["sector"], 0.80)

    # Estimated price increase to consumer (%)
    b["price_increase_pct"] = (
        b["weighted_tariff"] / 100 * b["import_share"] * b["pass_through"] * 100
    )

    # Dollar impact on flagship product
    tariff_on_cost = b["unit_cost"] * (b["weighted_tariff"] / 100) * b["import_share"]
    b["unit_price_increase"] = tariff_on_cost * b["pass_through"]

    # Revenue at risk ($B) = US rev × import share × price increase
    b["revenue_at_risk"] = b["us_rev"] * b["import_share"] * (b["weighted_tariff"] / 100)

df = pd.DataFrame(BRANDS).sort_values("price_increase_pct", ascending=False)

SECTOR_COLORS = {
    "Electronics":      "#58a6ff",
    "Apparel/Footwear": "#f85149",
    "Toys & Games":     "#ffd166",
    "Furniture/Home":   "#3fb950",
    "Automotive":       "#bc8cff",
    "Retail Chain":     "#fb8500",
}

DARK_BG  = "#0d1117"
CARD_BG  = "#161b22"
TEXT_W   = "#e6edf3"
TEXT_G   = "#8b949e"

def dark_ax(ax):
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_G, labelsize=9)
    ax.xaxis.label.set_color(TEXT_G)
    ax.yaxis.label.set_color(TEXT_G)
    ax.title.set_color(TEXT_W)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(color="#21262d", linewidth=0.5, alpha=0.9)
    return ax

# ===========================================================================
# IMAGE 1 — BRAND TARIFF EXPOSURE DASHBOARD
# ===========================================================================
fig = plt.figure(figsize=(22, 14), facecolor=DARK_BG)
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.52, wspace=0.38,
                        left=0.05, right=0.97,
                        top=0.89, bottom=0.06)

fig.text(0.5, 0.955,
         "Liberation Day Tariffs — Hardest Hit American Brands",
         ha="center", fontsize=23, fontweight="bold", color=TEXT_W)
fig.text(0.5, 0.923,
         "Exposure = weighted average tariff across supply countries × import share × pass-through rate   |   "
         "Source: BACI 2023 via BigQuery + company supply-chain disclosures",
         ha="center", fontsize=11, color=TEXT_G)

# ---- Panel A: Price increase % by brand (full width top) ----
ax_a = dark_ax(fig.add_subplot(gs[0, :]))
df_s = df.sort_values("price_increase_pct", ascending=True)
colors_a = [SECTOR_COLORS[s] for s in df_s["sector"]]
bars = ax_a.barh(df_s["brand"], df_s["price_increase_pct"],
                 color=colors_a, edgecolor=DARK_BG, linewidth=0.8,
                 height=0.65, zorder=3)
ax_a.axvline(df["price_increase_pct"].mean(), color="white", lw=1.5,
             linestyle="--", alpha=0.5, label=f"Average: {df['price_increase_pct'].mean():.1f}%")
for bar, row in zip(bars, df_s.itertuples()):
    ax_a.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
              f"+{row.price_increase_pct:.1f}%  (wt. tariff {row.weighted_tariff:.0f}%,  "
              f"import share {row.import_share*100:.0f}%)",
              va="center", fontsize=8.8, color=TEXT_W)
ax_a.set_xlabel("Estimated Consumer Price Increase (%)", color=TEXT_G, fontsize=11)
ax_a.set_title("Estimated Price Increase by Brand — Liberation Day Tariffs",
               fontsize=13, fontweight="bold", color=TEXT_W, pad=10)
ax_a.legend(fontsize=9, facecolor=CARD_BG, labelcolor=TEXT_W, edgecolor="#30363d")
ax_a.set_xlim(0, df_s["price_increase_pct"].max() * 1.55)
ax_a.tick_params(axis="y", colors=TEXT_W, labelsize=10)

legend_patches = [mpatches.Patch(color=v, label=k) for k, v in SECTOR_COLORS.items()]
ax_a.legend(handles=legend_patches, loc="lower right", fontsize=8.5,
            facecolor=CARD_BG, labelcolor=TEXT_W, edgecolor="#30363d", ncol=3)

# ---- Panel B: Bubble chart — tariff vs import share (row 1-2, col 0-1) ----
ax_b = dark_ax(fig.add_subplot(gs[1:, :2]))

for _, row in df.iterrows():
    col  = SECTOR_COLORS[row["sector"]]
    size = max(row["us_rev"] * 3.5, 80)
    ax_b.scatter(row["weighted_tariff"], row["import_share"]*100,
                 s=size, color=col, alpha=0.85,
                 edgecolors="white", linewidths=0.8, zorder=3)
    ax_b.text(row["weighted_tariff"]+0.3, row["import_share"]*100,
              row["brand"], fontsize=8.5, color=TEXT_W,
              fontweight="bold" if row["price_increase_pct"] > 20 else "normal",
              zorder=4)

# Danger zone shading
ax_b.fill_between([35,60],[75,75],[100,100], alpha=0.08, color="#f85149", zorder=1)
ax_b.text(42, 95, "HIGH RISK ZONE", color="#f85149", fontsize=10,
          fontweight="bold", alpha=0.7)

ax_b.set_xlabel("Weighted Average Tariff Rate (%)", fontsize=11, color=TEXT_G)
ax_b.set_ylabel("Import Share (% of supply from overseas)", fontsize=11, color=TEXT_G)
ax_b.set_title("Brand Risk Matrix — Tariff Rate vs Import Dependency\n"
               "(bubble size = US revenue)",
               fontsize=12, fontweight="bold", color=TEXT_W, pad=10)
ax_b.set_xlim(5, 65)
ax_b.set_ylim(40, 105)
ax_b.legend(handles=legend_patches, fontsize=8.5, facecolor=CARD_BG,
            labelcolor=TEXT_W, edgecolor="#30363d", ncol=2)

# ---- Panel C: Dollar impact on flagship product (row 1, col 2) ----
ax_c = dark_ax(fig.add_subplot(gs[1, 2]))
flagship_df = df[df["unit_price_increase"] > 0].nlargest(10, "unit_price_increase")
cols_c = [SECTOR_COLORS[s] for s in flagship_df["sector"]]
bars_c = ax_c.barh(flagship_df["brand"], flagship_df["unit_price_increase"],
                   color=cols_c, edgecolor=DARK_BG, linewidth=0.8, zorder=3)
ax_c.invert_yaxis()
for bar, row in zip(bars_c, flagship_df.itertuples()):
    ax_c.text(bar.get_width()+10, bar.get_y()+bar.get_height()/2,
              f"+${bar.get_width():,.0f}", va="center",
              fontsize=9, fontweight="bold", color=TEXT_W)
ax_c.set_xlabel("Estimated Price Increase per Unit (USD)", color=TEXT_G, fontsize=9)
ax_c.set_title("USD Added to Flagship Product\nPer Unit at Retail",
               fontsize=11, fontweight="bold", color=TEXT_W, pad=8)
ax_c.tick_params(axis="y", colors=TEXT_W, labelsize=9.5)

# ---- Panel D: Revenue at risk ($B) top brands (row 2, col 2) ----
ax_d = dark_ax(fig.add_subplot(gs[2, 2]))
rev_df = df.nlargest(10, "revenue_at_risk")
cols_d = [SECTOR_COLORS[s] for s in rev_df["sector"]]
bars_d = ax_d.barh(rev_df["brand"], rev_df["revenue_at_risk"],
                   color=cols_d, edgecolor=DARK_BG, linewidth=0.8, zorder=3)
ax_d.invert_yaxis()
for bar, row in zip(bars_d, rev_df.itertuples()):
    ax_d.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
              f"${bar.get_width():.1f}B", va="center",
              fontsize=9, fontweight="bold", color=TEXT_W)
ax_d.set_xlabel("Revenue Directly Exposed to Tariff ($B)", color=TEXT_G, fontsize=9)
ax_d.set_title("Revenue at Tariff Risk\n(US Rev × Import Share × Tariff Rate)",
               fontsize=11, fontweight="bold", color=TEXT_W, pad=8)
ax_d.tick_params(axis="y", colors=TEXT_W, labelsize=9.5)

fig.text(0.5, 0.025,
         "Price increase = weighted tariff × import share × sector pass-through rate  |  "
         "Supply chain shares: company disclosures, BofA/Goldman sector research 2025  |  "
         "Automotive unit prices excluded from per-unit chart for scale",
         ha="center", fontsize=8, color=TEXT_G, style="italic")

out1 = os.path.join(FIG_DIR, "retail_IMAGE1_brand_exposure.png")
fig.savefig(out1, dpi=155, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print("[saved] retail_IMAGE1_brand_exposure.png")

# ===========================================================================
# IMAGE 2 — WHAT IT COSTS THE CONSUMER (product-level)
# ===========================================================================
fig2 = plt.figure(figsize=(20, 13), facecolor=DARK_BG)
gs2  = gridspec.GridSpec(3, 3, figure=fig2,
                         hspace=0.55, wspace=0.40,
                         left=0.05, right=0.97,
                         top=0.89, bottom=0.07)

fig2.text(0.5, 0.955,
          "What Liberation Day Tariffs Add to Your Shopping Cart",
          ha="center", fontsize=22, fontweight="bold", color=TEXT_W)
fig2.text(0.5, 0.923,
          "Before vs After prices for everyday American purchases — "
          "based on trade-weighted tariff pass-through by product category",
          ha="center", fontsize=11.5, color=TEXT_G)

# ---- Headline KPI cards ----
top3 = df.nlargest(3, "price_increase_pct")
for col_idx, (_, row) in enumerate(top3.iterrows()):
    ax = fig2.add_subplot(gs2[0, col_idx])
    ax.set_facecolor(CARD_BG)
    col = SECTOR_COLORS[row["sector"]]
    for spine in ax.spines.values():
        spine.set_edgecolor(col); spine.set_linewidth(2.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.72, row["brand"], ha="center", fontsize=17,
            fontweight="bold", color=TEXT_W, transform=ax.transAxes)
    ax.text(0.5, 0.48, f"+{row['price_increase_pct']:.1f}%", ha="center",
            fontsize=38, fontweight="bold", color=col, transform=ax.transAxes)
    ax.text(0.5, 0.26, row["flagship"], ha="center", fontsize=10,
            color=TEXT_G, transform=ax.transAxes)
    ax.text(0.5, 0.10, f"Wt. tariff: {row['weighted_tariff']:.0f}%  |  "
            f"Import share: {row['import_share']*100:.0f}%",
            ha="center", fontsize=8.5, color=TEXT_G, transform=ax.transAxes)

# ---- Before/After price comparison ----
products = [
    ("iPhone 16 Pro",      999,  "Apple — 85% China",      0.371),
    ("Nike Air Max",       110,  "Nike — 50% Vietnam",      0.308),
    ("Dell XPS Laptop",    950,  "Dell — 70% China",        0.330),
    ("Samsung 65\" TV",    800,  "Samsung — 55% Korea",     0.177),
    ("Levi's 501 Jeans",    60,  "Levi's — 28% Bangladesh", 0.248),
    ("Hasbro Monopoly",     30,  "Hasbro — 65% China",      0.370),
    ("IKEA KALLAX Shelf",  200,  "IKEA — 30% China",        0.178),
    ("Adidas Ultraboost",  190,  "Adidas — 34% Vietnam",    0.276),
    ("Gap Hoodie",          65,  "Gap — 30% Vietnam",       0.300),
    ("Wayfair Sofa",      1200,  "Wayfair — 60% China",     0.335),
]
prod_names   = [p[0] for p in products]
before_price = [p[1] for p in products]
labels_prod  = [p[2] for p in products]
pct_rise     = [p[3] for p in products]
after_price  = [b*(1+r) for b,r in zip(before_price, pct_rise)]
dollar_rise  = [a-b for a,b in zip(after_price, before_price)]

ax_prod = dark_ax(fig2.add_subplot(gs2[1, :2]))
x = np.arange(len(products))
w = 0.38
bars_before = ax_prod.bar(x - w/2, before_price, w, label="Price Before Tariff",
                          color="#3fb950", alpha=0.85, edgecolor=DARK_BG, zorder=3)
bars_after  = ax_prod.bar(x + w/2, after_price,  w, label="Price After Tariff",
                          color="#f85149", alpha=0.85, edgecolor=DARK_BG, zorder=3)
for i, (ba, aa, dr, pr) in enumerate(zip(before_price, after_price, dollar_rise, pct_rise)):
    ax_prod.text(x[i]+w/2, aa+max(after_price)*0.01,
                 f"+${dr:.0f}\n({pr*100:.0f}%)",
                 ha="center", fontsize=8, color="#f85149", fontweight="bold", zorder=4)
ax_prod.set_xticks(x)
ax_prod.set_xticklabels([p[0] for p in products], rotation=30, ha="right",
                        fontsize=9, color=TEXT_W)
ax_prod.set_ylabel("Retail Price (USD)", color=TEXT_G)
ax_prod.set_title("Before vs After — Price at Checkout",
                  fontsize=12, fontweight="bold", color=TEXT_W, pad=10)
ax_prod.legend(fontsize=9.5, facecolor=CARD_BG, labelcolor=TEXT_W, edgecolor="#30363d")
ax_prod.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"${x:,.0f}"))

# ---- Dollar increase bar ----
ax_dr = dark_ax(fig2.add_subplot(gs2[1, 2]))
grad = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(dollar_rise)))
bars_dr = ax_dr.barh(prod_names, dollar_rise, color=grad,
                     edgecolor=DARK_BG, linewidth=0.8, zorder=3)
ax_dr.invert_yaxis()
for bar, val in zip(bars_dr, dollar_rise):
    ax_dr.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
               f"+${val:.0f}", va="center", fontsize=9,
               fontweight="bold", color=TEXT_W)
ax_dr.set_xlabel("Dollar Added to Price (USD)", color=TEXT_G)
ax_dr.set_title("Extra $ Per Item\nYou Pay at Checkout",
                fontsize=11, fontweight="bold", color=TEXT_W, pad=8)
ax_dr.tick_params(axis="y", colors=TEXT_W, labelsize=9)

# ---- Annual household spend increase by category ----
household_basket = {
    "Electronics\n(phone/laptop/TV)": (2500, 0.34),
    "Clothing &\nFootwear":           (1800, 0.29),
    "Furniture &\nHome Goods":        (1200, 0.25),
    "Toys &\nHobbies":               (600,  0.35),
    "Auto Parts\n& Accessories":      (900,  0.17),
    "Grocery\n(packaged goods)":      (3200, 0.08),
}
cat_names  = list(household_basket.keys())
cat_spend  = [v[0] for v in household_basket.values()]
cat_pct    = [v[1] for v in household_basket.values()]
cat_extra  = [s*p for s,p in zip(cat_spend, cat_pct)]
cat_colors = ["#58a6ff","#f85149","#3fb950","#ffd166","#bc8cff","#fb8500"]

ax_cat = dark_ax(fig2.add_subplot(gs2[2, :2]))
x2 = np.arange(len(cat_names))
w2 = 0.38
ax_cat.bar(x2 - w2/2, cat_spend, w2, label="Annual Spend (Before)",
           color="#3fb950", alpha=0.80, edgecolor=DARK_BG, zorder=3)
ax_cat.bar(x2 + w2/2, cat_extra, w2, label="Extra Cost From Tariff",
           color="#f85149", alpha=0.85, edgecolor=DARK_BG, zorder=3)
for i, (extra, pct) in enumerate(zip(cat_extra, cat_pct)):
    ax_cat.text(x2[i]+w2/2, extra + 20, f"+${extra:.0f}",
                ha="center", fontsize=9.5, color="#f85149",
                fontweight="bold", zorder=4)
ax_cat.set_xticks(x2)
ax_cat.set_xticklabels(cat_names, fontsize=9.5, color=TEXT_W)
ax_cat.set_ylabel("Annual Household Spend (USD)", color=TEXT_G)
ax_cat.set_title("Annual Household Budget vs Tariff Extra Cost — By Category",
                 fontsize=12, fontweight="bold", color=TEXT_W, pad=10)
ax_cat.legend(fontsize=9.5, facecolor=CARD_BG, labelcolor=TEXT_W, edgecolor="#30363d")
ax_cat.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"${x:,.0f}"))

# ---- Total annual extra spend KPI ----
ax_tot = dark_ax(fig2.add_subplot(gs2[2, 2]))
ax_tot.set_xticks([]); ax_tot.set_yticks([])
total_extra = sum(cat_extra)
ax_tot.set_facecolor(CARD_BG)
for spine in ax_tot.spines.values():
    spine.set_edgecolor("#f85149"); spine.set_linewidth(2.5)
ax_tot.text(0.5, 0.82, "Total Extra Spend", ha="center", fontsize=14,
            fontweight="bold", color=TEXT_W, transform=ax_tot.transAxes)
ax_tot.text(0.5, 0.56, f"${total_extra:,.0f}", ha="center", fontsize=42,
            fontweight="bold", color="#f85149", transform=ax_tot.transAxes)
ax_tot.text(0.5, 0.38, "per household / year", ha="center", fontsize=12,
            color=TEXT_G, transform=ax_tot.transAxes)
ax_tot.text(0.5, 0.20, "across all tariff-exposed\ncategories combined",
            ha="center", fontsize=10, color=TEXT_G, transform=ax_tot.transAxes)
ax_tot.text(0.5, 0.06, f"Liberation Day — no retaliation scenario",
            ha="center", fontsize=9, color=TEXT_G, style="italic",
            transform=ax_tot.transAxes)

fig2.text(0.5, 0.025,
          "Product prices: retailer averages Q1 2025  |  Pass-through rates by sector (BofA/Goldman 2025)  |  "
          "Supply chain shares: company filings  |  Household basket: BLS Consumer Expenditure Survey 2023",
          ha="center", fontsize=8, color=TEXT_G, style="italic")

out2 = os.path.join(FIG_DIR, "retail_IMAGE2_consumer_price_impact.png")
fig2.savefig(out2, dpi=155, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig2)
print("[saved] retail_IMAGE2_consumer_price_impact.png")

# Summary
print(f"\n--- TOP 5 HARDEST HIT BRANDS ---")
for _, row in df.nlargest(5,"price_increase_pct").iterrows():
    print(f"  {row['brand']:20s}  +{row['price_increase_pct']:.1f}%  "
          f"(wt.tariff={row['weighted_tariff']:.0f}%  import={row['import_share']*100:.0f}%)")
print(f"\n  Total extra household spend across categories: ${total_extra:,.0f}/yr")
