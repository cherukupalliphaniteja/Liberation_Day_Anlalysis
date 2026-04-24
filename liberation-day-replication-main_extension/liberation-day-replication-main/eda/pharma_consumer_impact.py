"""
pharma_consumer_impact.py
=========================
Two presentation-ready images:
  Image 1 — How Liberation Day tariffs hit consumers TODAY
  Image 2 — 10-year household cost projection (2025-2035)
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
# Pull data
# ---------------------------------------------------------------------------
tariffs = bq.get_tariffs()

pharma_total = bq.query("""
    SELECT exporter_iso3,
           ROUND(SUM(value_1000usd)/1e6, 3) AS trade_usd_bn
    FROM `liberation-day-analysis.liberation_day.baci_trade`
    WHERE importer_iso3 = 'USA'
      AND (CAST(hs6_product_code AS STRING) LIKE '29%'
           OR CAST(hs6_product_code AS STRING) LIKE '30%')
      AND SAFE_CAST(hs6_product_code AS INT64) IS NOT NULL
    GROUP BY exporter_iso3
    ORDER BY trade_usd_bn DESC
""")
pharma_total = pharma_total.merge(tariffs[["iso3","tariff_pct"]],
                                   left_on="exporter_iso3", right_on="iso3", how="left")
pharma_total["tariff_pct"] = pharma_total["tariff_pct"].fillna(10.0)
total_imports  = pharma_total["trade_usd_bn"].sum()
tw_tariff      = (pharma_total["tariff_pct"] * pharma_total["trade_usd_bn"] / total_imports).sum()

# Core economics
PASS_THROUGH   = 0.85
IMPORT_SHARE   = 0.54
IO_MULT        = 1.067
PRICE_RISE_PCT = (tw_tariff / 100) * PASS_THROUGH * IMPORT_SHARE * IO_MULT * 100  # ~10.56%

HH_SPEND_BASE  = 3623.0   # avg household pharma spend/yr (OECD 2023)
EXTRA_YR1      = HH_SPEND_BASE * (PRICE_RISE_PCT / 100)

# Income quintiles
QUINTILES = ["Q1\nLowest 20%\n~$17K/yr",
             "Q2\n~$43K/yr",
             "Q3 Middle\n~$72K/yr",
             "Q4\n~$112K/yr",
             "Q5\nTop 20%\n~$238K/yr"]
INCOMES   = [17_000, 43_000, 72_000, 112_000, 238_000]
PH_SHARES = [0.035, 0.022, 0.015, 0.009, 0.005]   # pharma as % income (KFF)
PH_SPEND  = [i*s for i,s in zip(INCOMES, PH_SHARES)]
EXTRA_PH  = [s * (PRICE_RISE_PCT/100) for s in PH_SPEND]
PCT_INC   = [e/i*100 for e,i in zip(EXTRA_PH, INCOMES)]
REGRESSIVITY = PCT_INC[0] / PCT_INC[-1]


# ============================================================
# IMAGE 1 — HOW TARIFFS HIT CONSUMERS TODAY
# ============================================================
fig = plt.figure(figsize=(20, 13), facecolor="#0d1117")
fig.patch.set_facecolor("#0d1117")

gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    hspace=0.55, wspace=0.40,
    left=0.06, right=0.97, top=0.88, bottom=0.06
)

DARK_BG  = "#0d1117"
CARD_BG  = "#161b22"
TEXT_W   = "#e6edf3"
TEXT_G   = "#8b949e"
RED      = "#f85149"
ORANGE   = "#fb8500"
YELLOW   = "#ffd166"
GREEN    = "#3fb950"
BLUE     = "#58a6ff"
PURPLE   = "#bc8cff"
TEAL     = "#39d353"

def dark_ax(ax):
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_G, labelsize=9)
    ax.xaxis.label.set_color(TEXT_G)
    ax.yaxis.label.set_color(TEXT_G)
    ax.title.set_color(TEXT_W)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(color="#21262d", linewidth=0.6, alpha=0.8)
    return ax

# ---- Title ----
fig.text(0.5, 0.95, "How Liberation Day Tariffs Hit American Healthcare Wallets",
         ha="center", va="top", fontsize=22, fontweight="bold",
         color=TEXT_W, fontfamily="DejaVu Sans")
fig.text(0.5, 0.915,
         f"US imports $220B in pharmaceuticals | Trade-weighted tariff: {tw_tariff:.1f}% "
         f"| Estimated price rise: +{PRICE_RISE_PCT:.1f}%",
         ha="center", va="top", fontsize=12, color=TEXT_G)

# ---- Panel A: Big number cards (row 0, col 0-2) ----
cards = [
    (f"+{PRICE_RISE_PCT:.1f}%",  "Drug Price Increase",      "For the average consumer",     RED),
    (f"+${EXTRA_YR1:.0f}/yr",    "Extra Cost Per Household",  "Family of 2.5, avg spend",     ORANGE),
    (f"{REGRESSIVITY:.1f}x",     "More Burden on Poor",       "Q1 vs Q5 as % of income",      PURPLE),
]
for col, (big, title, sub, color) in enumerate(cards):
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.62, big,  ha="center", va="center", fontsize=40,
            fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.5, 0.30, title, ha="center", va="center", fontsize=13,
            fontweight="bold", color=TEXT_W, transform=ax.transAxes)
    ax.text(0.5, 0.12, sub,  ha="center", va="center", fontsize=9.5,
            color=TEXT_G, transform=ax.transAxes)

# ---- Panel B: Income quintile dollar impact (row 1, col 0-1) ----
ax_b = dark_ax(fig.add_subplot(gs[1, :2]))
q_short = ["Q1\nLowest 20%", "Q2", "Q3\nMiddle", "Q4", "Q5\nTop 20%"]
bar_cols = [RED, ORANGE, YELLOW, TEAL, GREEN]
bars = ax_b.bar(q_short, EXTRA_PH, color=bar_cols, edgecolor=DARK_BG,
                linewidth=1.2, width=0.6, zorder=3)
ax_b.set_ylabel("Extra Annual Drug Cost (USD)", color=TEXT_G, fontsize=10)
ax_b.set_title("Extra Out-of-Pocket Drug Cost Per Year — By Income Group",
               fontsize=12, fontweight="bold", color=TEXT_W, pad=10)
for bar, val, spend, inc in zip(bars, EXTRA_PH, PH_SPEND, INCOMES):
    ax_b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
              f"+${val:.0f}/yr", ha="center", fontsize=11,
              fontweight="bold", color=TEXT_W, zorder=4)
    ax_b.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2,
              f"(base\n${spend:,.0f}/yr)", ha="center", fontsize=8,
              color=DARK_BG, zorder=4, fontweight="bold")
ax_b.set_ylim(0, max(EXTRA_PH)*1.35)
ax_b.tick_params(axis="x", colors=TEXT_W, labelsize=10)

# ---- Panel C: % of income bar (row 1, col 2) ----
ax_c = dark_ax(fig.add_subplot(gs[1, 2]))
h_bars = ax_c.barh(q_short[::-1], [p*1000 for p in PCT_INC[::-1]],
                   color=bar_cols[::-1], edgecolor=DARK_BG, linewidth=1, zorder=3)
ax_c.set_xlabel("Extra Cost as % of Income (x1000)", color=TEXT_G, fontsize=9)
ax_c.set_title("Regressive Burden\n(% of household income)",
               fontsize=11, fontweight="bold", color=TEXT_W, pad=10)
for bar, val in zip(h_bars, PCT_INC[::-1]):
    ax_c.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
              f"{val:.3f}%", va="center", fontsize=9.5,
              fontweight="bold", color=TEXT_W)
ax_c.tick_params(axis="y", colors=TEXT_W, labelsize=9)

# ---- Panel D: Supply chain — where drugs come from (row 2, col 0-1) ----
ax_d = dark_ax(fig.add_subplot(gs[2, :2]))
NAMES = {"IRL":"Ireland","DEU":"Germany","CHE":"Switzerland","SGP":"Singapore",
         "IND":"India","CHN":"China","GBR":"UK","CAN":"Canada","ITA":"Italy","BEL":"Belgium"}
top10 = pharma_total.head(10).copy()
top10["label"] = top10["exporter_iso3"].map(NAMES).fillna(top10["exporter_iso3"])

# Colour by tariff: low=green, mid=yellow, high=red
def tariff_color(t):
    if t <= 12:  return GREEN
    if t <= 22:  return YELLOW
    if t <= 30:  return ORANGE
    return RED

t_colors = [tariff_color(t) for t in top10["tariff_pct"]]
bars_d = ax_d.barh(top10["label"], top10["trade_usd_bn"],
                   color=t_colors, edgecolor=DARK_BG, linewidth=0.8, zorder=3)
ax_d.invert_yaxis()
ax_d.set_xlabel("US Pharma Imports (USD billions)", color=TEXT_G)
ax_d.set_title("Top 10 Drug Suppliers — Coloured by Tariff Rate",
               fontsize=12, fontweight="bold", color=TEXT_W, pad=10)
ax_d.tick_params(axis="y", colors=TEXT_W, labelsize=10)
for bar, row in zip(bars_d, top10.itertuples()):
    ax_d.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
              f"${bar.get_width():.0f}B  |  tariff: {row.tariff_pct:.0f}%",
              va="center", fontsize=9, color=TEXT_W)

legend_patches = [
    mpatches.Patch(color=GREEN,  label="10-12% tariff (EU core)"),
    mpatches.Patch(color=YELLOW, label="13-22% tariff (EU periphery)"),
    mpatches.Patch(color=ORANGE, label="23-30% tariff (India, others)"),
    mpatches.Patch(color=RED,    label="31-54% tariff (China)"),
]
ax_d.legend(handles=legend_patches, loc="lower right", fontsize=8,
            facecolor=CARD_BG, labelcolor=TEXT_W, edgecolor="#30363d")

# ---- Panel E: Tariff tier donut (row 2, col 2) ----
ax_e = fig.add_subplot(gs[2, 2])
ax_e.set_facecolor(CARD_BG)
for spine in ax_e.spines.values(): spine.set_edgecolor("#30363d")

tier_vals  = [38.9, 124.6, 24.9, 31.7]
tier_lbls  = ["10%\nFloor\n$38.9B", "11-20%\n$124.6B", "21-30%\n$24.9B", "31-54%\n$31.7B"]
tier_cols  = [GREEN, YELLOW, ORANGE, RED]
wedges, texts = ax_e.pie(
    tier_vals, labels=tier_lbls, colors=tier_cols,
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(edgecolor=DARK_BG, linewidth=2, width=0.55),
    textprops=dict(color=TEXT_W, fontsize=8.5)
)
ax_e.set_title("Import Share\nby Tariff Tier", fontsize=11,
               fontweight="bold", color=TEXT_W, pad=8)

# ---- Footnote ----
fig.text(0.5, 0.025,
         "Sources: BACI 2023 (BigQuery), OECD pharma spend data  |  "
         "Assumptions: 85% pass-through, 54% import dependency, 1.067 IO multiplier  |  "
         "Liberation Day tariff = April 2, 2025 schedule",
         ha="center", fontsize=8, color=TEXT_G, style="italic")

out1 = os.path.join(FIG_DIR, "pharma_IMAGE1_consumer_impact.png")
fig.savefig(out1, dpi=160, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print(f"[saved] pharma_IMAGE1_consumer_impact.png")


# ============================================================
# IMAGE 2 — 10-YEAR HOUSEHOLD PROJECTION (2025-2035)
# ============================================================

# Economic model for 10-year projection
# Baseline: 4.5% annual pharma inflation (historical US average)
# Liberation Day adds a shock + structural effects over time

YEARS = list(range(2025, 2036))
N = len(YEARS)
BASE_INFLATION = 0.045   # 4.5%/yr baseline pharma inflation

def project_spend(shock_yr1, supply_chain_drift, concentration_annual, label):
    """
    shock_yr1          : immediate price shock in year 1 (fraction, e.g. 0.106)
    supply_chain_drift : extra annual drift yrs 2-4 as supply chains restructure
    concentration_annual: annual price creep from HHI concentration yrs 4-10
    """
    spend = np.zeros(N)
    spend[0] = HH_SPEND_BASE  # 2025 base
    for i in range(1, N):
        yr = YEARS[i]
        inflation = BASE_INFLATION
        if i == 1:
            inflation += shock_yr1          # immediate tariff shock
        if 2 <= i <= 3:
            inflation += supply_chain_drift # restructuring friction
        if i >= 4:
            inflation += concentration_annual  # HHI concentration creep
        spend[i] = spend[i-1] * (1 + inflation)
    return spend

scenarios_10yr = {
    "No Tariff\n(Counterfactual)": project_spend(0,     0,      0,      "No Tariff"),
    "10% Floor Only":              project_spend(0.049, 0.005,  0.003,  "10% Floor"),
    "Liberation Day\n(Current)":   project_spend(PRICE_RISE_PCT/100, 0.012, 0.007, "Lib Day"),
    "Escalation\n(50% on China)":  project_spend(0.148, 0.018,  0.012,  "Escalation"),
    "Full Trade War":              project_spend(0.232, 0.028,  0.020,  "Trade War"),
}

# Cumulative EXTRA spend vs no-tariff baseline
base_proj = list(scenarios_10yr.values())[0]
cumulative_extra = {}
for name, proj in list(scenarios_10yr.items())[1:]:
    cumulative_extra[name] = np.cumsum(proj - base_proj)

# Build the figure
fig = plt.figure(figsize=(20, 13), facecolor=DARK_BG)
fig.patch.set_facecolor(DARK_BG)

gs2 = gridspec.GridSpec(
    3, 3,
    figure=fig,
    hspace=0.52, wspace=0.38,
    left=0.06, right=0.97, top=0.88, bottom=0.07
)

fig.text(0.5, 0.955,
         "10-Year Pharma Cost Projection for the Average American Household (2025-2035)",
         ha="center", va="top", fontsize=21, fontweight="bold", color=TEXT_W)
fig.text(0.5, 0.92,
         f"Base household pharma spend: ${HH_SPEND_BASE:,.0f}/yr  |  "
         f"Baseline inflation: {BASE_INFLATION*100:.1f}%/yr  |  "
         "Liberation Day adds: price shock + supply-chain friction + concentration drift",
         ha="center", va="top", fontsize=11.5, color=TEXT_G)

COLORS_10 = [GREEN, YELLOW, ORANGE, RED, "#8B0000"]
SC_NAMES  = list(scenarios_10yr.keys())
SC_PROJS  = list(scenarios_10yr.values())

# ---- Panel A: Annual spend lines (row 0-1, col 0-1) ----
ax_a = dark_ax(fig.add_subplot(gs2[:2, :2]))
for proj, name, col in zip(SC_PROJS, SC_NAMES, COLORS_10):
    lw = 3.5 if "Liberation" in name else 2
    ls = "-" if "Liberation" in name or "No Tariff" in name else "--"
    alpha = 1.0 if "Liberation" in name or "No Tariff" in name else 0.75
    ax_a.plot(YEARS, proj, color=col, lw=lw, linestyle=ls, alpha=alpha,
              label=name.replace("\n"," "), marker="o", markersize=4, zorder=3)
    ax_a.text(YEARS[-1]+0.15, proj[-1], f"  ${proj[-1]:,.0f}",
              color=col, fontsize=9.5, va="center", fontweight="bold")

ax_a.axvspan(2025, 2027, alpha=0.07, color=RED, zorder=1)
ax_a.axvspan(2027, 2029, alpha=0.05, color=ORANGE, zorder=1)
ax_a.axvspan(2029, 2035, alpha=0.03, color=PURPLE, zorder=1)

ax_a.text(2025.5, ax_a.get_ylim()[0] if ax_a.get_ylim()[0] > 0 else 3200,
          "Immediate\nShock", color=RED, fontsize=8.5, alpha=0.8, va="bottom")
ax_a.text(2027.1, ax_a.get_ylim()[0] if ax_a.get_ylim()[0] > 0 else 3200,
          "Supply Chain\nRestructuring", color=ORANGE, fontsize=8.5, alpha=0.8, va="bottom")
ax_a.text(2029.5, ax_a.get_ylim()[0] if ax_a.get_ylim()[0] > 0 else 3200,
          "Concentration Lock-In\n(HHI effect)", color=PURPLE, fontsize=8.5, alpha=0.8, va="bottom")

ax_a.set_ylabel("Annual Household Pharma Spend (USD)", color=TEXT_G, fontsize=11)
ax_a.set_title("Annual Drug Spending — Avg US Household (2025-2035)",
               fontsize=13, fontweight="bold", color=TEXT_W, pad=10)
ax_a.legend(loc="upper left", fontsize=9, facecolor=CARD_BG,
            labelcolor=TEXT_W, edgecolor="#30363d", framealpha=0.9)
ax_a.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax_a.set_xlim(2025, 2036.5)

# ---- Panel B: Cumulative extra spend (row 0-1, col 2) ----
ax_b2 = dark_ax(fig.add_subplot(gs2[:2, 2]))
cum_names  = list(cumulative_extra.keys())
cum_projs  = list(cumulative_extra.values())
cum_colors = COLORS_10[1:]

for proj, name, col in zip(cum_projs, cum_names, cum_colors):
    lw = 3 if "Liberation" in name else 2
    ax_b2.plot(YEARS, proj, color=col, lw=lw,
               label=name.replace("\n"," "), marker="o", markersize=3.5, zorder=3)
    ax_b2.text(YEARS[-1]+0.1, proj[-1], f"  ${proj[-1]:,.0f}",
               color=col, fontsize=9, va="center", fontweight="bold")

ax_b2.fill_between(YEARS, cum_projs[1], alpha=0.12, color=ORANGE)  # lib day fill
ax_b2.axhline(0, color=TEXT_G, lw=1, linestyle=":")
ax_b2.set_ylabel("Cumulative Extra Spend vs No Tariff (USD)", color=TEXT_G, fontsize=9.5)
ax_b2.set_title("Total Extra Cost\nvs No-Tariff World",
                fontsize=12, fontweight="bold", color=TEXT_W, pad=10)
ax_b2.legend(loc="upper left", fontsize=7.5, facecolor=CARD_BG,
             labelcolor=TEXT_W, edgecolor="#30363d")
ax_b2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax_b2.set_xlim(2025, 2036.5)

# ---- Panel C: 10-yr total extra cost bar (row 2, col 0) ----
ax_c2 = dark_ax(fig.add_subplot(gs2[2, 0]))
total_extra_10 = [proj[-1] for proj in cum_projs]
sc_short = ["10%\nFloor", "Liberation\nDay", "Escalation\n50% China", "Full\nTrade War"]
bars_c = ax_c2.bar(sc_short, total_extra_10, color=cum_colors,
                   edgecolor=DARK_BG, linewidth=1.2, width=0.6, zorder=3)
for bar, val in zip(bars_c, total_extra_10):
    ax_c2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
               f"${val:,.0f}", ha="center", fontsize=11,
               fontweight="bold", color=TEXT_W, zorder=4)
ax_c2.set_ylabel("Total Extra Spend 2025-2035 (USD)", color=TEXT_G)
ax_c2.set_title("Cumulative 10-Year Extra Cost\nPer Household", fontsize=11,
                fontweight="bold", color=TEXT_W, pad=8)
ax_c2.tick_params(axis="x", colors=TEXT_W, labelsize=9.5)
ax_c2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

# ---- Panel D: Year-by-year extra for Liberation Day (row 2, col 1) ----
ax_d2 = dark_ax(fig.add_subplot(gs2[2, 1]))
lib_day_proj = SC_PROJS[2]
annual_extra = lib_day_proj - base_proj
bar_colors_yr = []
for i, yr in enumerate(YEARS):
    if yr <= 2026: bar_colors_yr.append(RED)
    elif yr <= 2028: bar_colors_yr.append(ORANGE)
    else: bar_colors_yr.append(PURPLE)

bars_d2 = ax_d2.bar(YEARS, annual_extra, color=bar_colors_yr,
                    edgecolor=DARK_BG, linewidth=0.8, zorder=3)
for bar, val in zip(bars_d2, annual_extra):
    ax_d2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
               f"+${val:.0f}", ha="center", fontsize=8.5,
               fontweight="bold", color=TEXT_W, rotation=45, zorder=4)
ax_d2.set_ylabel("Extra Spend vs No Tariff (USD/yr)", color=TEXT_G)
ax_d2.set_title("Liberation Day: Year-by-Year\nExtra Cost Per Household",
                fontsize=11, fontweight="bold", color=TEXT_W, pad=8)
ax_d2.tick_params(axis="x", colors=TEXT_W, labelsize=9, rotation=45)
patches = [mpatches.Patch(color=RED, label="Price shock (2025-26)"),
           mpatches.Patch(color=ORANGE, label="Restructuring (2027-28)"),
           mpatches.Patch(color=PURPLE, label="Concentration drift (2029+)")]
ax_d2.legend(handles=patches, fontsize=8, facecolor=CARD_BG,
             labelcolor=TEXT_W, edgecolor="#30363d")

# ---- Panel E: Income quintile 10-yr total (row 2, col 2) ----
ax_e2 = dark_ax(fig.add_subplot(gs2[2, 2]))

# For each quintile, scale the Liberation Day extra by their pharma spend ratio
lib_total_10 = cumulative_extra[list(cumulative_extra.keys())[1]][-1]  # Liberation Day 10yr total

quintile_10yr = []
for spend, income in zip(PH_SPEND, INCOMES):
    # Scale by their spend relative to avg, then compound over 10 yrs
    scale = spend / HH_SPEND_BASE
    quintile_10yr.append(lib_total_10 * scale)

q_cols = [RED, ORANGE, YELLOW, TEAL, GREEN]
q_short2 = ["Q1\nLowest", "Q2", "Q3\nMiddle", "Q4", "Q5\nTop"]
bars_e2 = ax_e2.bar(q_short2, quintile_10yr, color=q_cols,
                    edgecolor=DARK_BG, linewidth=1, zorder=3)
for bar, val, inc in zip(bars_e2, quintile_10yr, INCOMES):
    pct = val / (inc * 10) * 100
    ax_e2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
               f"${val:,.0f}\n({pct:.2f}% of\n10yr income)",
               ha="center", fontsize=8.5, fontweight="bold",
               color=TEXT_W, zorder=4)
ax_e2.set_ylabel("Total 10-yr Extra Pharma Cost (USD)", color=TEXT_G)
ax_e2.set_title("10-Year Burden by\nIncome Quintile (Liberation Day)",
                fontsize=11, fontweight="bold", color=TEXT_W, pad=8)
ax_e2.tick_params(axis="x", colors=TEXT_W, labelsize=10)

# ---- Footnote ----
fig.text(0.5, 0.025,
         "Model assumptions: 4.5% baseline pharma inflation | Liberation Day: +10.6% yr1 shock, "
         "+1.2%/yr supply-chain friction (yrs 2-3), +0.7%/yr concentration drift (yr4+) | "
         "Escalation: shock +14.8% | Trade War: shock +23.2% | Sources: BACI 2023, OECD, KFF",
         ha="center", fontsize=8, color=TEXT_G, style="italic")

out2 = os.path.join(FIG_DIR, "pharma_IMAGE2_10year_projection.png")
fig.savefig(out2, dpi=160, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print(f"[saved] pharma_IMAGE2_10year_projection.png")
print(f"\nBoth images saved to: {FIG_DIR}")

# Print key numbers
print(f"\n--- KEY NUMBERS ---")
print(f"Liberation Day 10-yr cumulative extra cost per household: ${cumulative_extra[list(cumulative_extra.keys())[1]][-1]:,.0f}")
print(f"Full trade war 10-yr cumulative extra cost:               ${cumulative_extra[list(cumulative_extra.keys())[3]][-1]:,.0f}")
print(f"Liberation Day annual spend by 2035:                      ${SC_PROJS[2][-1]:,.0f}")
print(f"No-tariff baseline spend by 2035:                         ${SC_PROJS[0][-1]:,.0f}")
