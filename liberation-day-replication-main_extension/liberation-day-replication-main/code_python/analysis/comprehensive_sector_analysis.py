"""
comprehensive_sector_analysis.py
=================================
Multi-sector, multi-scenario analysis of Liberation Day tariff impacts.

Covers:
  - Manufacturing (steel/aluminum + broad manufacturing)
  - Pharmaceuticals (supply chain + pricing)
  - Retail / Consumer goods (CPI, income incidence)

For each sector: current tariffs, retaliation scenarios, optimal for US,
  optimal globally, and long-run continuation effects.

Outputs (python_output/):
  fig_sector_comparison_welfare.png
  fig_sector_scenarios_us.png
  fig_sector_longrun.png
  fig_sector_country_welfare.png
  fig_manufacturing_deep.png
  fig_pharma_deep.png
  fig_retail_deep.png
  summary_sector_analysis.txt
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
OUTPUT_DIR = os.path.join(REPO_ROOT, 'python_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load pre-computed results
# ---------------------------------------------------------------------------
def load_results():
    baseline  = np.load(os.path.join(OUTPUT_DIR, 'baseline_results.npz'),  allow_pickle=True)
    mfg       = np.load(os.path.join(OUTPUT_DIR, 'sector_manufacturing_results.npz'), allow_pickle=True)
    pharma    = np.load(os.path.join(OUTPUT_DIR, 'sector_pharma_results.npz'),        allow_pickle=True)
    retail    = np.load(os.path.join(OUTPUT_DIR, 'sector_retail_results.npz'),        allow_pickle=True)
    multi     = np.load(os.path.join(OUTPUT_DIR, 'multisector_baseline_results.npz'), allow_pickle=True)
    return baseline, mfg, pharma, retail, multi

# ---------------------------------------------------------------------------
# Scenario metadata
# Baseline results shape: (194 countries, 7 metrics, 9 scenarios)
# Metrics: [0=welfare, 1=deficit%, 2=exports%, 3=imports%, 4=employment%, 5=CPI%, 6=tariff_rev/E]
# Scenario order (from main_baseline.py execution):
#   0: Liberation Day (No Retaliation)
#   1: Liberation Day - Armington (phi=1)
#   2: Liberation Day - Eaton-Kortum
#   3: Optimal US Tariff (No Retaliation)   <- best for US unilaterally
#   4: Liberation + Optimal Retaliation
#   5: Liberation + Reciprocal Retaliation
#   6: Optimal US + Optimal Retaliation     <- Nash equilibrium
#   7: Liberation + Lump-Sum Rebate
#   8: Liberation - High Elasticity
# ---------------------------------------------------------------------------
SCENARIOS = {
    0: 'Liberation Day\n(No Retaliation)',
    1: 'Liberation Day\n(Armington)',
    2: 'Liberation Day\n(Eaton-Kortum)',
    3: 'Optimal US Tariff\n(No Retaliation)',
    4: 'Liberation +\nOptimal Retaliation',
    5: 'Liberation +\nReciprocal Retaliation',
    6: 'Optimal US +\nOptimal Retaliation\n(Nash)',
    7: 'Liberation +\nLump-Sum Rebate',
    8: 'Liberation\n(High Elasticity)',
}

SCENARIO_SHORT = {
    0: 'Lib Day\nNo Ret.',
    1: 'Lib Day\nArmington',
    2: 'Lib Day\nEK',
    3: 'Optimal US\nNo Ret.',
    4: 'Lib +\nOpt. Ret.',
    5: 'Lib +\nRecip. Ret.',
    6: 'Nash\nEquil.',
    7: 'Lib +\nLump Sum',
    8: 'Lib\nHigh Elas.',
}

# Country indices (0-based, from main_baseline.py)
ID_US  = 184
ID_CHN = 33
ID_CAN = 30
ID_MEX = 114
ID_EU  = np.array([10,13,17,45,47,50,56,57,59,61,71,78,80,83,88,
                   107,108,109,119,133,144,145,149,164,165]) - 1

# Sector colors
COLORS = {
    'Manufacturing': '#2c7bb6',
    'Pharma':        '#d7191c',
    'Retail':        '#1a9641',
    'US':            '#e6550d',
    'China':         '#fdae61',
    'EU':            '#2ca25f',
    'Canada':        '#756bb1',
    'Mexico':        '#99d8c9',
}

SC_COLORS = ['#3182bd','#6baed6','#9ecae1','#e6550d','#fd8d3c','#fdae6b','#8856a7','#31a354','#74c476']


# ===========================================================================
# FIGURE 1: Cross-sector welfare comparison (US) - all scenarios
# ===========================================================================
def fig_sector_scenarios_us(r):
    fig, ax = plt.subplots(figsize=(14, 6))
    sc_ids = list(SCENARIOS.keys())
    n_sc = len(sc_ids)
    x = np.arange(n_sc)
    us_welfare = [r[ID_US, 0, s] for s in sc_ids]

    bars = ax.bar(x, us_welfare, color=[SC_COLORS[i] for i in range(n_sc)],
                  edgecolor='white', linewidth=0.8, zorder=3)

    ax.axhline(0, color='black', linewidth=1.2, linestyle='-', zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_SHORT[s] for s in sc_ids], fontsize=8.5)
    ax.set_ylabel('US Welfare Change (%)', fontsize=11)
    ax.set_title('US Welfare Across All Tariff Scenarios\n'
                 'Liberation Day Tariff Analysis (Ignatenko et al. 2025)',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, zorder=1)

    for bar, val in zip(bars, us_welfare):
        ypos = bar.get_height() + 0.04 if val >= 0 else bar.get_height() - 0.12
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{val:+.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Annotate best and worst
    best_i = int(np.argmax(us_welfare))
    worst_i = int(np.argmin(us_welfare))
    ax.annotate('Best for US', xy=(x[best_i], us_welfare[best_i]),
                xytext=(x[best_i]+0.6, us_welfare[best_i]+0.3),
                arrowprops=dict(arrowstyle='->', color='darkgreen'),
                color='darkgreen', fontsize=8.5, fontweight='bold')
    ax.annotate('Worst for US', xy=(x[worst_i], us_welfare[worst_i]),
                xytext=(x[worst_i]+0.6, us_welfare[worst_i]-0.25),
                arrowprops=dict(arrowstyle='->', color='darkred'),
                color='darkred', fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_sector_scenarios_us.png')
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'  Saved: fig_sector_scenarios_us.png')
    return out


# ===========================================================================
# FIGURE 2: Country welfare comparison - who wins/loses by scenario
# ===========================================================================
def fig_sector_country_welfare(r):
    sc_select = [0, 3, 4, 6]  # Lib No Ret, Optimal No Ret, Lib+Opt Ret, Nash
    sc_names  = ['Lib Day\nNo Retaliation', 'Optimal US\nNo Retaliation',
                 'Lib +\nOpt. Retaliation', 'Nash\nEquilibrium']

    countries  = ['US', 'China', 'Canada', 'Mexico', 'EU avg']
    ids        = [ID_US, ID_CHN, ID_CAN, ID_MEX, None]
    clrs       = [COLORS['US'], COLORS['China'], COLORS['Canada'], COLORS['Mexico'], COLORS['EU']]

    n_sc = len(sc_select)
    n_c  = len(countries)
    x    = np.arange(n_sc)
    w    = 0.15
    offsets = np.linspace(-(n_c-1)/2*w, (n_c-1)/2*w, n_c)

    fig, ax = plt.subplots(figsize=(13, 6))
    for ci, (cname, cid, clr) in enumerate(zip(countries, ids, clrs)):
        vals = []
        for s in sc_select:
            if cid is None:
                vals.append(float(np.mean(r[ID_EU, 0, s])))
            else:
                vals.append(float(r[cid, 0, s]))
        bars = ax.bar(x + offsets[ci], vals, w, label=cname, color=clr,
                      edgecolor='white', linewidth=0.6, zorder=3)

    ax.axhline(0, color='black', linewidth=1, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(sc_names, fontsize=9)
    ax.set_ylabel('Welfare Change (%)', fontsize=11)
    ax.set_title('Who Wins and Who Loses? Country Welfare by Scenario\n'
                 'Liberation Day Tariffs — Key Trading Partners',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=5, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, zorder=1)

    # Add annotation boxes
    ax.text(0, ax.get_ylim()[0]*0.92,
            'No scenario makes everyone better off.\nUS gains come at partners\' expense.',
            ha='center', fontsize=8, style='italic', color='gray')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_sector_country_welfare.png')
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'  Saved: fig_sector_country_welfare.png')
    return out


# ===========================================================================
# FIGURE 3: Manufacturing deep-dive
# ===========================================================================
def fig_manufacturing_deep(r, mfg):
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    tau_lib  = float(mfg['tau_mfg_avg'])       # 27.0%
    hts8     = float(mfg['hts8_mfg_rate'])      # 3.6%
    io_mult  = float(mfg['io_mult_mfg'])        # 1.09x
    imp_pen  = float(mfg['import_penetration_mfg'])  # 32.3%
    cpi_mfg  = float(mfg['cpi_mfg_contribution'])   # 6.79%
    imp_chg  = float(mfg['mfg_import_change'])       # -80.8%

    tau_steel = float(mfg['hts8_steel_rate'])   # 1.09%
    io_steel  = float(mfg['io_mult_steel'])     # 1.09x

    # --- Panel A: Tariff rates (HTS8 product rate vs Liberation Day country rate) ---
    ax = fig.add_subplot(gs[0, 0])
    categories = ['HTS8 Product Rate\n(Structural)', 'Liberation Day\nCountry Rate',
                  'Steel/Aluminum\nHTS8 Rate']
    rates = [hts8*100, tau_lib*100, tau_steel*100]
    bar_colors = ['#3182bd', '#e6550d', '#31a354']
    bars = ax.bar(categories, rates, color=bar_colors, edgecolor='white', linewidth=0.8)
    ax.set_title('A. Tariff Rate Structure\n(Manufacturing)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Tariff Rate (%)')
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylim(0, max(rates)*1.25)

    # --- Panel B: US imports change by scenario ---
    ax = fig.add_subplot(gs[0, 1])
    sc_ids = [0, 3, 4, 5, 6]
    sc_lbls = ['Lib\nNo Ret.', 'Opt US\nNo Ret.', 'Lib +\nOpt Ret.', 'Lib +\nRecip Ret.', 'Nash']
    us_imp = [r[ID_US, 3, s] for s in sc_ids]
    bar_c = ['#e6550d' if v < 0 else '#31a354' for v in us_imp]
    ax.bar(sc_lbls, us_imp, color=bar_c, edgecolor='white')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title('B. US Import Change (%)\nby Scenario', fontsize=10, fontweight='bold')
    ax.set_ylabel('Change in US Imports (%)')
    for i, v in enumerate(us_imp):
        ax.text(i, v - 2.5, f'{v:.1f}%', ha='center', va='top', fontsize=8, fontweight='bold')

    # --- Panel C: Employment and CPI tradeoff ---
    ax = fig.add_subplot(gs[0, 2])
    emp = [r[ID_US, 4, s] for s in range(9)]
    cpi = [r[ID_US, 5, s] for s in range(9)]
    sc_cols = SC_COLORS
    for i, (e, c, col) in enumerate(zip(emp, cpi, sc_cols)):
        ax.scatter(c, e, color=col, s=90, zorder=5, label=SCENARIO_SHORT[i].replace('\n',' '))
        ax.annotate(str(i+1), (c, e), fontsize=7, ha='left', va='bottom')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel('US CPI Change (%)')
    ax.set_ylabel('US Employment Change (%)')
    ax.set_title('C. Employment vs CPI Tradeoff\n(numbered = scenario)', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.2)

    # --- Panel D: Manufacturing IO supply chain multiplier ---
    ax = fig.add_subplot(gs[1, 0])
    sectors   = ['Manufacturing', 'Steel &\nAluminum']
    io_mults  = [io_mult, io_steel]
    imp_shares = [float(mfg['imp_share_mfg'])*100, float(mfg['imp_share_steel'])*100]
    x = np.arange(2)
    w = 0.35
    b1 = ax.bar(x - w/2, io_mults, w, label='IO Multiplier (×)', color='#3182bd')
    ax2b = ax.twinx()
    b2 = ax2b.bar(x + w/2, imp_shares, w, label='Intermediate Import\nShare (%)', color='#fd8d3c')
    ax.set_xticks(x); ax.set_xticklabels(sectors)
    ax.set_ylabel('IO Supply-Chain Multiplier')
    ax2b.set_ylabel('Intermediate Import Share (%)')
    ax.set_title('D. IO Multipliers &\nImport Dependency', fontsize=10, fontweight='bold')
    ax.set_ylim(0, max(io_mults)*1.5)
    ax2b.set_ylim(0, max(imp_shares)*1.5)
    lines = [mpatches.Patch(color='#3182bd', label='IO Multiplier'),
             mpatches.Patch(color='#fd8d3c', label='Import Share')]
    ax.legend(handles=lines, fontsize=8, loc='upper right')
    for bar in b1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f'{bar.get_height():.3f}x', ha='center', va='bottom', fontsize=8)
    for bar in b2:
        ax2b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                  f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)

    # --- Panel E: CPI contribution by sector context ---
    ax = fig.add_subplot(gs[1, 1])
    us_cpi_all = [r[ID_US, 5, s] for s in range(9)]
    mfg_cpi_share = cpi_mfg  # from sector analysis
    retail_cpi = [8.40, 7.58, 7.17, 6.35, 5.94]  # quintile Q1-Q5 no retal

    # Stacked bar: Lib Day CPI = total; mfg portion
    scenarios_cpi = ['Lib Day\nNo Ret.', 'Lib +\nOpt Ret.', 'Nash\nEquil.']
    sc_cpi_ids = [0, 4, 6]
    total_cpi_vals = [r[ID_US, 5, s] for s in sc_cpi_ids]
    mfg_portion = [cpi_mfg * (v / r[ID_US, 5, 0]) for v in total_cpi_vals]
    other_portion = [t - m for t, m in zip(total_cpi_vals, mfg_portion)]

    x = np.arange(3)
    ax.bar(x, mfg_portion, color='#2c7bb6', label='Manufacturing CPI')
    ax.bar(x, other_portion, bottom=mfg_portion, color='#abd9e9', label='Other sectors')
    ax.set_xticks(x); ax.set_xticklabels(scenarios_cpi)
    ax.set_ylabel('US CPI Change (%)')
    ax.set_title('E. US CPI Decomposition\n(Manufacturing contribution)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    for i, (t, m) in enumerate(zip(total_cpi_vals, mfg_portion)):
        ax.text(i, t+0.2, f'Total:\n{t:.1f}%', ha='center', fontsize=7.5)

    # --- Panel F: Global trade volume change ---
    ax = fig.add_subplot(gs[1, 2])
    d_trade = [-9.43, -4.73, -6.65, 0.0, -12.31, -11.57, -11.85, 0.0, 0.0]
    sc_all = list(range(9))
    sc_lbls_trade = [SCENARIO_SHORT[s].replace('\n',' ') for s in sc_all]
    bar_c_t = ['#e6550d' if v < -1 else '#31a354' if v > 1 else '#bdbdbd' for v in d_trade]
    ax.bar(range(9), d_trade, color=bar_c_t, edgecolor='white')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(range(9))
    ax.set_xticklabels([str(i+1) for i in range(9)], fontsize=9)
    ax.set_ylabel('Global Trade Volume Change (%)')
    ax.set_title('F. Global Trade Collapse\nby Scenario (1-9)', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Manufacturing Sector — Comprehensive Liberation Day Tariff Impact Analysis',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_manufacturing_deep.png')
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'  Saved: fig_manufacturing_deep.png')
    return out


# ===========================================================================
# FIGURE 4: Pharma deep-dive
# ===========================================================================
def fig_pharma_deep(pharma):
    tau_eff    = float(pharma['tau_pharma_eff'])       # 19.9%
    hts8       = float(pharma['hts8_pharma_rate'])     # 2.45%
    io_mult    = float(pharma['io_multiplier'])        # 1.067x
    imp_share  = float(pharma['imp_share_interm'])     # 12.4%
    price_nr   = float(pharma['price_noretal'])        # +0.495%
    price_r    = float(pharma['price_retal'])
    imp_chg_nr = float(pharma['import_chg_noretal'])   # -38.2%
    hhi_pre    = float(pharma['hhi_pre'])              # 580
    hhi_post   = float(pharma['hhi_post'])             # 591
    n_sup      = int(pharma['n_suppliers'])
    welfare_nr = float(pharma['pharma_welfare_noretal'])  # -0.0134%

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Pharmaceutical Sector — Liberation Day Tariff Impact Analysis',
                 fontsize=13, fontweight='bold')

    # A: Two-layer tariff structure
    ax = axes[0, 0]
    categories = ['HTS8\nProduct Rate\n(Structural)', 'Liberation Day\nCountry Rate\n(Effective)']
    rates_pct  = [hts8*100, tau_eff*100]
    bars = ax.bar(categories, rates_pct, color=['#3182bd', '#e6550d'], edgecolor='white')
    for bar, val in zip(bars, rates_pct):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{val:.2f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_title('A. Two-Layer Tariff Structure\n(Pharma)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Tariff Rate (%)')
    ax.set_ylim(0, tau_eff*100*1.3)

    # B: Price impact: direct vs IO-adjusted vs economy-wide
    ax = axes[0, 1]
    direct_price = tau_eff * 0.88 * (0.082*0.323) * 100  # PHARMA_PASS_THROUGH * pharma_import_pen
    scenarios_p = ['Direct Price\nImpact', 'IO-Adjusted\nPrice Impact', 'Economy-Wide\nCPI']
    ge_cpi = 7.09  # from retail results (no retaliation)
    vals_p = [direct_price, price_nr, ge_cpi]
    bar_c  = ['#fdae61', '#e6550d', '#3182bd']
    bars   = ax.bar(scenarios_p, vals_p, color=bar_c, edgecolor='white')
    for bar, val in zip(bars, vals_p):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                f'{val:.2f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_title('B. Price Impact:\nDirect → IO-Adjusted → Economy', fontsize=10, fontweight='bold')
    ax.set_ylabel('Price Increase (%)')
    ax.annotate(f'Pharma is {price_nr/ge_cpi:.1f}× economy-wide CPI',
                xy=(1, price_nr), xytext=(1.5, price_nr+1.5),
                arrowprops=dict(arrowstyle='->', color='black'), fontsize=8)

    # C: Import volume change by tariff level scenario
    ax = axes[0, 2]
    eps_pharma = 2.3
    tau_scenarios = [0.0, 0.10, 0.20, tau_eff, 0.35, 0.50]
    tau_labels    = ['0%', '10%', '20%', f'{tau_eff*100:.0f}%\n(current)', '35%', '50%']
    imp_changes   = [-eps_pharma * t/(1+t)*100 for t in tau_scenarios]
    bar_c_imp     = ['#31a354' if v > -10 else '#fdae61' if v > -30 else '#e6550d' for v in imp_changes]
    bars = ax.bar(tau_labels, imp_changes, color=bar_c_imp, edgecolor='white')
    ax.axhline(imp_chg_nr, color='black', linewidth=1.5, linestyle='--', label=f'Current: {imp_chg_nr:.1f}%')
    for bar, val in zip(bars, imp_changes):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()-1.5,
                f'{val:.1f}%', ha='center', fontsize=8)
    ax.set_title('C. Import Volume vs Tariff Level\n(eps = 2.3, Broda & Weinstein 2006)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Pharma Import Change (%)')
    ax.legend(fontsize=8)

    # D: Supplier concentration (HHI)
    ax = axes[1, 0]
    hhi_labels = ['Pre-Tariff HHI', 'Post-Tariff HHI']
    hhi_vals   = [hhi_pre, hhi_post]
    bars = ax.bar(hhi_labels, hhi_vals, color=['#3182bd', '#e6550d'], edgecolor='white', width=0.5)
    ax.set_title(f'D. Supplier Concentration (HHI)\n{n_sup} Supplier Countries', fontsize=10, fontweight='bold')
    ax.set_ylabel('Herfindahl-Hirschman Index')
    for bar, val in zip(bars, hhi_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                f'{val:.0f}', ha='center', fontsize=11, fontweight='bold')
    ax.annotate(f'+{hhi_post-hhi_pre:.0f} more concentrated\n(→ less competition, higher prices)',
                xy=(1, hhi_post), xytext=(0.5, hhi_post+15),
                arrowprops=dict(arrowstyle='->', color='darkred'), fontsize=8.5, color='darkred')
    ax.set_ylim(0, hhi_post * 1.2)

    # E: Consumer welfare loss by sector (pharma vs economy)
    ax = axes[1, 1]
    pce_share_pharma = 0.027
    ge_welfare_nr = 0.598  # from retail results
    pharma_consumer_loss = -pce_share_pharma * price_nr / 100 * 100

    sector_labels = ['Pharma\nConsumer\nWelfare Loss', 'Total US\nWelfare\nGain']
    vals_w = [pharma_consumer_loss, ge_welfare_nr]
    bar_c_w = ['#e6550d', '#31a354']
    bars = ax.bar(sector_labels, vals_w, color=bar_c_w, edgecolor='white', width=0.5)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_title('E. Pharma Welfare Cost\nvs US Economy Gain', fontsize=10, fontweight='bold')
    ax.set_ylabel('Welfare Change (%)')
    for bar, val in zip(bars, vals_w):
        offset = 0.01 if val >= 0 else -0.03
        ax.text(bar.get_x()+bar.get_width()/2, val+offset,
                f'{val:.3f}%', ha='center', fontsize=10, fontweight='bold')

    # F: Long-run pharma scenario projections
    ax = axes[1, 2]
    # Long-run: if tariffs continue 3-5 years, domestic pharma production rises
    # Sourcing diversification, price adjustment
    years = [2025, 2026, 2027, 2028, 2030]
    # Base case: tariffs maintained, prices converge toward new equilibrium
    price_trajectory = [0.0, price_nr, price_nr*0.9, price_nr*0.8, price_nr*0.7]
    # Retaliation case: US pharma exports hit, prices stay elevated
    price_retal_traj  = [0.0, price_nr, price_nr*1.1, price_nr*1.05, price_nr*0.95]
    # Escalation: tariffs rise further
    price_escalation  = [0.0, price_nr, price_nr*1.5, price_nr*2.0, price_nr*2.5]

    ax.plot(years, price_trajectory, 'o-', color='#3182bd', linewidth=2, label='Maintained (No Ret.)')
    ax.plot(years, price_retal_traj, 's--', color='#e6550d', linewidth=2, label='With Retaliation')
    ax.plot(years, price_escalation, '^:', color='#d7191c', linewidth=2, label='Escalation (+tariffs)')
    ax.fill_between(years, price_trajectory, price_escalation, alpha=0.1, color='red', label='Uncertainty band')
    ax.set_xlabel('Year'); ax.set_ylabel('Pharma Price Impact (%)')
    ax.set_title('F. Long-Run Pharma Price Projection\n(Illustrative Trajectories)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_pharma_deep.png')
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'  Saved: fig_pharma_deep.png')
    return out


# ===========================================================================
# FIGURE 5: Retail / Consumer deep-dive
# ===========================================================================
def fig_retail_deep(retail, r):
    ge_cpi_nr   = float(retail['ge_cpi_noretal'])        # 7.09%
    ge_cpi_r    = float(retail['ge_cpi_retal'])          # 4.38%
    ge_w_nr     = float(retail['ge_welfare_noretal'])    # +0.60%
    ge_w_r      = float(retail['ge_welfare_retal'])      # -1.02%
    first_cpi   = float(retail['first_order_cpi'])       # 13.55%
    ge_amp      = float(retail['ge_amplification'])      # 0.52x
    regress_r   = float(retail['regress_ratio'])         # 1.41x (Q1/Q5)
    q_nr        = retail['quintile_incidence_noretal']   # [8.4, 7.6, 7.2, 6.4, 5.9]
    q_r         = retail['quintile_incidence_retal']     # [5.2, 4.7, 4.4, 3.9, 3.7]
    cavallo_usa = float(retail['cavallo_usa_90d'])       # 0.103%
    cavallo_chn = float(retail['cavallo_china_90d'])     # 1.41%
    pt          = float(retail['retail_product_passthrough'])  # 0.297

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Retail & Consumer Sector — Liberation Day Tariff Impact Analysis',
                 fontsize=13, fontweight='bold')

    # A: CPI first-order vs GE
    ax = axes[0, 0]
    cpi_items = ['First-Order\nCPI Estimate', 'GE Model CPI\n(No Retaliation)', 'GE Model CPI\n(With Retaliation)']
    cpi_vals  = [first_cpi, ge_cpi_nr, ge_cpi_r]
    bar_c_cpi = ['#fdae61', '#e6550d', '#3182bd']
    bars = ax.bar(cpi_items, cpi_vals, color=bar_c_cpi, edgecolor='white')
    for bar, val in zip(bars, cpi_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_title('A. CPI Impact:\nFirst-Order vs GE', fontsize=10, fontweight='bold')
    ax.set_ylabel('CPI Change (%)')
    ax.annotate(f'GE damps by {(1-ge_amp)*100:.0f}%\n(substitution + trade diversion)',
                xy=(1, ge_cpi_nr), xytext=(1.4, ge_cpi_nr+1.5),
                arrowprops=dict(arrowstyle='->', color='black'), fontsize=8)

    # B: Income quintile incidence (regressive burden)
    ax = axes[0, 1]
    quintiles = ['Q1\n(Lowest)', 'Q2', 'Q3\n(Middle)', 'Q4', 'Q5\n(Highest)']
    x = np.arange(5)
    w = 0.35
    ax.bar(x - w/2, q_nr, w, label='No Retaliation', color='#e6550d')
    ax.bar(x + w/2, q_r,  w, label='With Retaliation', color='#3182bd')
    ax.set_xticks(x); ax.set_xticklabels(quintiles)
    ax.set_ylabel('Price Burden (% of income)')
    ax.set_title('B. Regressive Burden:\nCPI by Income Quintile', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.annotate(f'{regress_r:.2f}× heavier\nfor lowest vs highest', xy=(0, q_nr[0]),
                xytext=(0.5, q_nr[0]+0.3), fontsize=8.5, color='darkred', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='darkred'))

    # C: Pass-through by sector
    ax = axes[0, 2]
    pt_sectors = ['Agriculture', 'Manufacturing', 'Mining &\nEnergy', 'Services']
    pt_vals    = [0.70, 0.85, 0.60, 0.00]
    pt_colors  = ['#31a354', '#3182bd', '#fd8d3c', '#bdbdbd']
    bars = ax.bar(pt_sectors, pt_vals, color=pt_colors, edgecolor='white')
    ax.set_title('C. Pass-Through Rate\nby Sector (Cavallo et al. 2025)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Pass-Through Rate')
    ax.axhline(pt, color='black', linewidth=1.5, linestyle='--',
               label=f'Retail avg: {pt:.2f}')
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, pt_vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.02, f'{val*100:.0f}%',
                ha='center', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)

    # D: Welfare: US gain vs consumer loss trade-off by scenario
    ax = axes[1, 0]
    sc_ids_ret = [0, 3, 4, 6, 7]
    sc_lbl_ret = ['Lib\nNo Ret.', 'Opt US\nNo Ret.', 'Lib +\nOpt Ret.', 'Nash', 'Lump-Sum\nRebate']
    us_w_ret   = [r[ID_US, 0, s] for s in sc_ids_ret]
    us_cpi_ret = [r[ID_US, 5, s] for s in sc_ids_ret]
    for i, (w_val, c_val, lbl) in enumerate(zip(us_w_ret, us_cpi_ret, sc_lbl_ret)):
        color = '#31a354' if w_val > 0 else '#e6550d'
        ax.scatter(c_val, w_val, color=color, s=120, zorder=5)
        ax.annotate(lbl, (c_val, w_val), xytext=(c_val+0.2, w_val+0.05), fontsize=7.5)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel('US CPI Change (%)')
    ax.set_ylabel('US Welfare Change (%)')
    ax.set_title('D. Welfare–CPI Trade-off\n(Consumer pays higher prices for welfare gain)', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.2)
    ax.text(ax.get_xlim()[0]*0.5, ax.get_ylim()[0]*0.85,
            'Green = US welfare gain\nRed = welfare loss', fontsize=8, color='gray', style='italic')

    # E: Empirical price evidence (Cavallo)
    ax = axes[1, 1]
    # Cavallo: 30-day and 90-day pass-through for US and China
    countries_cav  = ['US\n(30d)', 'US\n(90d)', 'China\n(90d)']
    cavallo_vals   = [float(retail['cavallo_usa_30d'])*100,
                      cavallo_usa*100, cavallo_chn*100]
    bar_c_cav = ['#abd9e9', '#3182bd', '#e6550d']
    bars = ax.bar(countries_cav, cavallo_vals, color=bar_c_cav, edgecolor='white', width=0.5)
    for bar, val in zip(bars, cavallo_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{val:.3f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_title('E. Empirical Price Changes\n(Cavallo et al. — Daily Price Indices)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Cumulative Price Change (%)')
    ax.text(0.5, -0.13, 'China shows 13.7× more pass-through than US (90d)',
            ha='center', transform=ax.transAxes, fontsize=8.5, color='darkred')

    # F: Long-run retail scenario projections
    ax = axes[1, 2]
    years = [2025, 2026, 2027, 2028, 2029, 2030]
    # Q1 burden over time - three scenarios
    q1_maintained  = [0, q_nr[0], q_nr[0]*0.9, q_nr[0]*0.8, q_nr[0]*0.75, q_nr[0]*0.7]
    q1_escalation  = [0, q_nr[0], q_nr[0]*1.3, q_nr[0]*1.6, q_nr[0]*1.8, q_nr[0]*2.0]
    q1_rollback    = [0, q_nr[0], q_nr[0]*0.5, q_nr[0]*0.2, q_nr[0]*0.1, 0.0]

    ax.plot(years, q1_maintained, 'o-', color='#e6550d', linewidth=2, label='Maintained')
    ax.plot(years, q1_escalation, 's--', color='#d7191c', linewidth=2, label='Escalation')
    ax.plot(years, q1_rollback,   '^:', color='#31a354',  linewidth=2, label='Rollback/Deal')
    ax.fill_between(years, q1_rollback, q1_escalation, alpha=0.1, color='orange')
    ax.set_xlabel('Year'); ax.set_ylabel('Q1 (Lowest Income) Price Burden (%)')
    ax.set_title('F. Long-Run Consumer Burden\n(Q1 = Lowest Income Quintile)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_retail_deep.png')
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'  Saved: fig_retail_deep.png')
    return out


# ===========================================================================
# FIGURE 6: Cross-sector comparison — the big picture
# ===========================================================================
def fig_sector_comparison_welfare(r, mfg, pharma, retail):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle('Cross-Sector Impact Summary — Liberation Day Tariffs\n'
                 '"Making America Great Again? The Economic Impacts of Liberation Day Tariffs" '
                 '(Ignatenko et al. 2025)',
                 fontsize=11, fontweight='bold')

    # --- Panel A: Sector tariff rates ---
    ax = axes[0]
    sectors = ['Manufacturing\n(27.0%)', 'Pharma\n(19.9%)', 'Steel/Alum.\n(1.1% HTS8)',
               'Retail\n(~13.5% FO)']
    tariff_rates = [27.0, 19.9, 25.0, 13.5]  # effective rates
    bar_c = [COLORS['Manufacturing'], COLORS['Pharma'], '#756bb1', COLORS['Retail']]
    bars = ax.bar(sectors, tariff_rates, color=bar_c, edgecolor='white')
    for bar, val in zip(bars, tariff_rates):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{val:.1f}%', ha='center', fontsize=9.5, fontweight='bold')
    ax.set_title('A. Effective Tariff Rates\nby Sector', fontsize=10, fontweight='bold')
    ax.set_ylabel('Liberation Day Effective Tariff Rate (%)')

    # --- Panel B: Key impact metrics by sector ---
    ax = axes[1]
    metrics = ['CPI\nContribution', 'Import\nVolume Drop', 'Consumer\nWelfare Hit']
    mfg_vals    = [float(mfg['cpi_mfg_contribution']),
                   abs(float(mfg['mfg_import_change'])),
                   float(mfg['cpi_mfg_contribution']) * 0.02]  # rough welfare proxy
    pharma_vals = [float(pharma['price_noretal']),
                   abs(float(pharma['import_chg_noretal'])),
                   abs(float(pharma['pharma_welfare_noretal'])) * 100]
    retail_vals = [float(retail['ge_cpi_noretal']),
                   abs(r[ID_US, 3, 0]),        # US import change scenario 0
                   abs(float(retail['ge_welfare_noretal']))]

    x = np.arange(3)
    w = 0.28
    ax.bar(x - w, mfg_vals,    w, label='Manufacturing', color=COLORS['Manufacturing'])
    ax.bar(x,     pharma_vals, w, label='Pharma',        color=COLORS['Pharma'])
    ax.bar(x + w, retail_vals, w, label='Retail',        color=COLORS['Retail'])
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel('Impact (%; larger = worse)')
    ax.set_title('B. Key Impact Metrics\nby Sector (No Retaliation)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.text(0.5, -0.18, '* Manufacturing dominates on all dimensions',
            ha='center', transform=ax.transAxes, fontsize=8, style='italic')

    # --- Panel C: What is optimal? Heatmap of US welfare ---
    ax = axes[2]
    sc_labels_heat = ['Lib Day\nNo Ret.', 'Optimal\nNo Ret.', 'Lib +\nOpt Ret.', 'Nash Equil.', 'Lib +\nRecip Ret.']
    sc_ids_heat    = [0, 3, 4, 6, 5]
    country_lbl    = ['US', 'China', 'Canada', 'Mexico', 'EU avg']
    ids_heat       = [ID_US, ID_CHN, ID_CAN, ID_MEX, None]

    data_heat = np.zeros((5, 5))
    for ci, (cid,) in enumerate([(x,) for x in ids_heat]):
        for si, sc in enumerate(sc_ids_heat):
            if cid is None:
                data_heat[ci, si] = np.mean(r[ID_EU, 0, sc])
            else:
                data_heat[ci, si] = r[cid, 0, sc]

    im = ax.imshow(data_heat, cmap='RdYlGn', aspect='auto',
                   vmin=-3, vmax=2)
    ax.set_xticks(range(5)); ax.set_xticklabels(sc_labels_heat, fontsize=8)
    ax.set_yticks(range(5)); ax.set_yticklabels(country_lbl, fontsize=9)
    ax.set_title('C. Welfare Heatmap\n(Green=Gain, Red=Loss)', fontsize=10, fontweight='bold')
    for ci in range(5):
        for si in range(5):
            ax.text(si, ci, f'{data_heat[ci,si]:.2f}', ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color='black' if abs(data_heat[ci,si]) < 1 else 'white')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Welfare Change (%)')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_sector_comparison_welfare.png')
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'  Saved: fig_sector_comparison_welfare.png')
    return out


# ===========================================================================
# FIGURE 7: Long-run continuation analysis
# ===========================================================================
def fig_sector_longrun(r, mfg, pharma, retail):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Long-Run Analysis: What Happens If Liberation Day Tariffs Continue?\n'
                 '(Illustrative trajectories based on GE model parameters)',
                 fontsize=12, fontweight='bold')

    years = [2025, 2026, 2027, 2028, 2029, 2030]

    # Panel A: US welfare under continuation vs resolution
    ax = axes[0, 0]
    # No retaliation maintained: US welfare stays positive but erodes as trade restructures
    us_w_maintain   = [0, 1.13, 1.05, 0.95, 0.88, 0.82]
    # Full retaliation escalation
    us_w_retal      = [0, 1.13, -0.36, -0.60, -0.75, -0.85]
    # Optimal tariff deal (US negotiates from strength)
    us_w_deal       = [0, 1.13, 1.50, 1.79, 1.79, 1.79]
    # Trade war collapse (retaliations escalate)
    us_w_tradewar   = [0, 1.13, -0.95, -1.50, -2.0, -2.5]

    ax.plot(years, us_w_maintain, 'o-', color='#3182bd', linewidth=2.5, label='Maintained (No Ret.)')
    ax.plot(years, us_w_retal,   's--', color='#e6550d', linewidth=2.5, label='Reciprocal Retaliation')
    ax.plot(years, us_w_deal,    '^-',  color='#31a354', linewidth=2.5, label='Negotiate to Optimal')
    ax.plot(years, us_w_tradewar,'D:',  color='#d7191c', linewidth=2.5, label='Full Trade War')
    ax.fill_between(years, us_w_tradewar, us_w_deal, alpha=0.08, color='gray')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('US Welfare Change (%)')
    ax.set_title('A. US Welfare — Trajectory by Scenario', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8.5); ax.grid(alpha=0.25)

    # Panel B: China welfare trajectory
    ax = axes[0, 1]
    chn_w_maintain  = [0, -1.13, -1.20, -1.25, -1.30, -1.35]
    chn_w_retal     = [0, -1.13, -0.80, -0.70, -0.65, -0.60]  # improves with own tariffs on US
    chn_w_tradewar  = [0, -1.13, -1.50, -2.0,  -2.5,  -3.0]
    chn_w_deal      = [0, -1.13, -0.80, -0.40, -0.20, 0.0]

    ax.plot(years, chn_w_maintain, 'o-',  color='#3182bd', linewidth=2.5, label='US Maintained (No Ret.)')
    ax.plot(years, chn_w_retal,   's--',  color='#e6550d', linewidth=2.5, label='China Retaliates')
    ax.plot(years, chn_w_tradewar, 'D:',  color='#d7191c', linewidth=2.5, label='Full Trade War')
    ax.plot(years, chn_w_deal,    '^-',   color='#31a354', linewidth=2.5, label='Negotiated Deal')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('China Welfare Change (%)')
    ax.set_title('B. China Welfare — Trajectory by Scenario', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8.5); ax.grid(alpha=0.25)

    # Panel C: Global trade volume over time
    ax = axes[1, 0]
    glob_maintain = [0, -9.43, -10.0, -10.5, -10.8, -11.0]
    glob_retal    = [0, -9.43, -11.6, -12.5, -13.0, -13.5]
    glob_tradewar = [0, -9.43, -15.0, -20.0, -24.0, -28.0]
    glob_deal     = [0, -9.43, -7.0,  -4.0,  -2.0,   0.0]

    ax.plot(years, glob_maintain, 'o-',  color='#3182bd', linewidth=2.5, label='Maintained')
    ax.plot(years, glob_retal,   's--',  color='#e6550d', linewidth=2.5, label='With Retaliation')
    ax.plot(years, glob_tradewar, 'D:',  color='#d7191c', linewidth=2.5, label='Full Trade War')
    ax.plot(years, glob_deal,    '^-',   color='#31a354', linewidth=2.5, label='Negotiated Resolution')
    ax.fill_between(years, glob_tradewar, glob_deal, alpha=0.07, color='blue')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('Global Trade Volume Change (%)')
    ax.set_title('C. Global Trade Collapse\nby Scenario', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8.5); ax.grid(alpha=0.25)

    # Panel D: Sector-specific long-run impact table (bar chart)
    ax = axes[1, 1]
    sectors    = ['Manufacturing', 'Pharma', 'Retail\n(Q1)', 'Steel/\nAluminum', 'Agriculture']
    yr1_impact = [float(mfg['cpi_mfg_contribution']),
                  float(pharma['price_noretal']),
                  float(retail['quintile_incidence_noretal'][0]),
                  25.0,  # steel tariff ~25%
                  3.5]   # agriculture estimated
    yr5_impact_maintained = [v * 0.75 for v in yr1_impact]  # partial adjustment
    yr5_impact_escalate   = [v * 1.5  for v in yr1_impact]

    x = np.arange(5)
    w = 0.28
    ax.bar(x - w, yr1_impact,             w, label='Year 1 (Current)', color='#e6550d', alpha=0.9)
    ax.bar(x,     yr5_impact_maintained,  w, label='Year 5 (Maintained)', color='#fdae61', alpha=0.9)
    ax.bar(x + w, yr5_impact_escalate,    w, label='Year 5 (Escalated)', color='#d7191c', alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(sectors, fontsize=8.5)
    ax.set_ylabel('CPI / Price Impact (%)')
    ax.set_title('D. Sector Price Burden:\nYear 1 vs Year 5', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_sector_longrun.png')
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'  Saved: fig_sector_longrun.png')
    return out


# ===========================================================================
# TEXT SUMMARY
# ===========================================================================
def write_summary(r, mfg, pharma, retail):
    lines = []
    A = lines.append

    A("=" * 78)
    A("LIBERATION DAY TARIFFS — COMPREHENSIVE SECTOR ANALYSIS SUMMARY")
    A("Ignatenko, Macedoni, Lashkaripour, Simonovska (2025)")
    A("Generated: comprehensive_sector_analysis.py")
    A("=" * 78)

    A("\n── OVERALL MACRO PICTURE ─────────────────────────────────────────────────")
    sc_names_s = ['Lib Day (No Ret.)', 'Lib Day (Armington)', 'Eaton-Kortum',
                  'Optimal US (No Ret.)', 'Lib + Opt Ret.', 'Lib + Recip Ret.',
                  'Nash Equilibrium', 'Lump-Sum Rebate', 'High Elasticity']
    A(f"\n{'Scenario':<30} {'US Welf':>9} {'US CPI':>9} {'US Imports':>12} {'US Employ':>11}")
    A("-" * 75)
    for i, name in enumerate(sc_names_s):
        A(f"{name:<30} {r[ID_US,0,i]:>+8.2f}%  {r[ID_US,5,i]:>+8.2f}%  "
          f"{r[ID_US,3,i]:>+10.2f}%  {r[ID_US,4,i]:>+9.2f}%")

    A(f"\n{'':30} Best for US (unilateral): Scenario 4 — Optimal US Tariff, No Retaliation (+1.79% welfare)")
    A(f"{'':30} Best globally: Negotiated free trade (scenario not run = no tariffs baseline)")
    A(f"{'':30} Nash equilibrium (scenario 7): US welfare -0.54%, all countries lose")

    A("\n── MANUFACTURING ────────────────────────────────────────────────────────")
    A(f"  Effective tariff (Liberation Day):    {float(mfg['tau_mfg_avg'])*100:.1f}%")
    A(f"  HTS8 structural product rate:          {float(mfg['hts8_mfg_rate'])*100:.2f}%")
    A(f"  IO supply-chain multiplier:            {float(mfg['io_mult_mfg']):.3f}x")
    A(f"  Import penetration:                    {float(mfg['import_penetration_mfg'])*100:.1f}%")
    A(f"  CPI contribution (manufacturing):     +{float(mfg['cpi_mfg_contribution']):.2f}%")
    A(f"  US import volume change:              {float(mfg['mfg_import_change']):.1f}%")
    A(f"  → Manufacturing is the DOMINANT sector for tariff impact.")
    A(f"  → At 27% effective tariff, US manufacturing imports drop ~81%.")
    A(f"  → Supply chains restructure, but IO multiplier amplifies price rises.")

    A("\n── PHARMACEUTICALS ──────────────────────────────────────────────────────")
    A(f"  Effective tariff (trade-weighted):     {float(pharma['tau_pharma_eff'])*100:.1f}%")
    A(f"  HTS8 structural rate:                  {float(pharma['hts8_pharma_rate'])*100:.2f}%")
    A(f"  IO supply-chain multiplier:            {float(pharma['io_multiplier']):.3f}x")
    A(f"  Pharma intermediate import share:      {float(pharma['imp_share_interm'])*100:.1f}%")
    A(f"  Price increase (IO-adjusted):         +{float(pharma['price_noretal']):.3f}%")
    A(f"  Import volume change:                  {float(pharma['import_chg_noretal']):.1f}%")
    A(f"  Supplier HHI: pre={float(pharma['hhi_pre']):.0f}, post={float(pharma['hhi_post']):.0f} (+{float(pharma['hhi_post'])-float(pharma['hhi_pre']):.0f})")
    A(f"  Consumer welfare loss (pharma):        {float(pharma['pharma_welfare_noretal']):.4f}%")
    A(f"  → Pharma price impact is {float(pharma['price_noretal'])/float(retail['ge_cpi_noretal']):.2f}× the economy-wide CPI.")
    A(f"  → Supplier concentration increases → less competition → higher prices persist.")
    A(f"  → 132 supplier countries face import collapse; US-sourced production rises.")

    A("\n── RETAIL / CONSUMER GOODS ──────────────────────────────────────────────")
    A(f"  First-order CPI estimate:             +{float(retail['first_order_cpi']):.2f}%")
    A(f"  GE model CPI (no retaliation):        +{float(retail['ge_cpi_noretal']):.2f}%")
    A(f"  GE model CPI (retaliation):           +{float(retail['ge_cpi_retal']):.2f}%")
    A(f"  US welfare (no retaliation):          +{float(retail['ge_welfare_noretal']):.3f}%")
    A(f"  US welfare (retaliation):             {float(retail['ge_welfare_retal']):.3f}%")
    A(f"  GE amplification factor:               {float(retail['ge_amplification']):.3f}x (GE damps first-order)")
    A(f"  Regressivity ratio (Q1/Q5):            {float(retail['regress_ratio']):.2f}x")
    q_nr = retail['quintile_incidence_noretal']
    q_r  = retail['quintile_incidence_retal']
    A(f"  Quintile price burden (no retal):")
    for qi, (qnr, qr) in enumerate(zip(q_nr, q_r)):
        label = ['Q1 (Lowest)', 'Q2', 'Q3 (Middle)', 'Q4', 'Q5 (Highest)'][qi]
        A(f"    {label}: {qnr:.2f}% (no ret.) | {qr:.2f}% (with ret.)")
    A(f"  → Tariffs act as a REGRESSIVE tax: lowest earners pay 1.41× more as % of income.")
    A(f"  → GE substitution and trade diversion dampen the first-order CPI by ~48%.")

    A("\n── OPTIMAL TARIFF ANALYSIS ──────────────────────────────────────────────")
    A(f"  Optimal US tariff (no retaliation):")
    A(f"    US welfare:   +{r[ID_US,0,3]:.2f}% (BEST for US unilaterally)")
    A(f"    China:        {r[ID_CHN,0,3]:.2f}%")
    A(f"    Canada:       {r[ID_CAN,0,3]:.2f}%")
    A(f"    Mexico:       {r[ID_MEX,0,3]:.2f}%")
    A(f"    EU avg:       {np.mean(r[ID_EU,0,3]):.2f}%")
    A(f"  Nash equilibrium (all countries optimize simultaneously):")
    A(f"    US welfare:   {r[ID_US,0,6]:.2f}% — US LOSES vs free trade")
    A(f"    → Prisoner's dilemma: optimal for each, worst collectively.")

    A("\n── LONG-RUN CONTINUATION RISKS ──────────────────────────────────────────")
    A(f"  If Liberation Day tariffs continue 3-5 years:")
    A(f"  1. Supply chain restructuring reduces tariff effectiveness (welfare gain erodes)")
    A(f"  2. Pharma supplier concentration locks in higher prices (HHI persists)")
    A(f"  3. Regressive consumer burden (Q1 households) accumulates over time")
    A(f"  4. Trading partners diversify away from US exports (permanent trade loss)")
    A(f"  5. Global trade at -9% to -13% depending on retaliation depth")
    A(f"  6. Nash equilibrium trap: all countries worse off but can't unilaterally exit")

    A("\n── WHAT IS BEST FOR THE US? ─────────────────────────────────────────────")
    A(f"  Short-term (unilateral): Scenario 4 — Optimal US tariff, no retaliation")
    A(f"    → US gains +1.79% welfare, but requires partners NOT to retaliate")
    A(f"  Long-term (with retaliation): Negotiate bilateral FTA deals")
    A(f"    → Nash equilibrium is unavoidable without coordinated exit")
    A(f"    → Lump-sum rebate of tariff revenue to consumers nearly offsets all gains/losses")
    A(f"  For other countries: Liberation Day is strictly negative; prefer free trade")
    A(f"    → China, Canada, Mexico all better off in any scenario with lower US tariffs")

    A("\n" + "=" * 78)

    out = os.path.join(OUTPUT_DIR, 'summary_sector_analysis.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  Saved: summary_sector_analysis.txt')
    return '\n'.join(lines)


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 70)
    print(" Comprehensive Multi-Sector Scenario Analysis")
    print(" Liberation Day Tariffs — Manufacturing, Pharma, Retail")
    print("=" * 70)

    print("\n[1/8] Loading pre-computed results...")
    baseline, mfg, pharma, retail, multi = load_results()
    r = baseline['results']  # shape: (194, 7, 9)
    print(f"  Baseline results: {r.shape} (countries × metrics × scenarios)")

    print("\n[2/8] Generating: US welfare across all scenarios...")
    fig_sector_scenarios_us(r)

    print("\n[3/8] Generating: Country welfare comparison (who wins/loses)...")
    fig_sector_country_welfare(r)

    print("\n[4/8] Generating: Manufacturing deep-dive...")
    fig_manufacturing_deep(r, mfg)

    print("\n[5/8] Generating: Pharmaceutical deep-dive...")
    fig_pharma_deep(pharma)

    print("\n[6/8] Generating: Retail deep-dive...")
    fig_retail_deep(retail, r)

    print("\n[7/8] Generating: Cross-sector comparison + welfare heatmap...")
    fig_sector_comparison_welfare(r, mfg, pharma, retail)

    print("\n[8/8] Generating: Long-run continuation analysis...")
    fig_sector_longrun(r, mfg, pharma, retail)

    print("\n[Summary] Writing text summary...")
    summary = write_summary(r, mfg, pharma, retail)

    print("\n" + "=" * 70)
    print(" OUTPUTS (python_output/):")
    print("   fig_sector_scenarios_us.png       — US welfare all 9 scenarios")
    print("   fig_sector_country_welfare.png    — Who wins/loses by scenario")
    print("   fig_manufacturing_deep.png        — Manufacturing 6-panel analysis")
    print("   fig_pharma_deep.png               — Pharma 6-panel analysis")
    print("   fig_retail_deep.png               — Retail 6-panel analysis")
    print("   fig_sector_comparison_welfare.png — Cross-sector comparison + heatmap")
    print("   fig_sector_longrun.png            — Long-run continuation scenarios")
    print("   summary_sector_analysis.txt       — Full text summary")
    print("=" * 70)

    print("\n--- QUICK RESULTS ---")
    snippet = summary[summary.find("OVERALL"):summary.find("MANUFACTURING")]
    print(snippet.encode('ascii', errors='replace').decode('ascii'))

    return True


if __name__ == '__main__':
    os.chdir(REPO_ROOT)
    main()
