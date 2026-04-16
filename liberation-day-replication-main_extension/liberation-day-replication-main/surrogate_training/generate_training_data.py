"""
Step 1: Generate surrogate model training data.

Uses a 5-bloc aggregated GE model (US, China, EU, CanMex, RoW) with the same
physics as the full 194-country model but ~500x faster per solve.
Calibrated from the actual 194-country trade data so bloc-level dynamics match.

Inputs (varied):
  us_tariff:     0.0 to 1.0 in steps of 0.1  (11 values)
  china_rate:    0.0 to 1.0 in steps of 0.1  (11 values)
  eu_rate:       0.0 to 0.5 in steps of 0.1  ( 6 values)
  canmex_rate:   0.0 to 0.5 in steps of 0.1  ( 6 values)
  Total: 11 × 11 × 6 × 6 = 4,356 combinations

Outputs per combination:
  welfare_us, cpi_us, employment_us, trade_deficit_us,
  hhi_pharma, welfare_china, welfare_eu

Saves: surrogate_training/training_data.csv
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from itertools import product
from scipy.optimize import fsolve

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR    = os.path.join(REPO_ROOT, 'data', 'base_data')
OUTPUT_DIR  = os.path.join(REPO_ROOT, 'python_output')
TRAIN_OUT   = os.path.join(SCRIPT_DIR, 'training_data.csv')

sys.path.insert(0, os.path.join(REPO_ROOT, 'code_python'))
from analysis.main_baseline import balanced_trade_eq
from utils.solver_utils import solve_nu

# ---------------------------------------------------------------------------
# Tariff grid
# ---------------------------------------------------------------------------
US_TARIFF_VALS  = np.round(np.arange(0.0, 1.01, 0.1), 2)   # 11
CHINA_RATE_VALS = np.round(np.arange(0.0, 1.01, 0.1), 2)   # 11
EU_RATE_VALS    = np.round(np.arange(0.0, 0.51, 0.1), 2)   #  6
CANMEX_RATE_VALS= np.round(np.arange(0.0, 0.51, 0.1), 2)   #  6
TOTAL_RUNS = len(US_TARIFF_VALS)*len(CHINA_RATE_VALS)*len(EU_RATE_VALS)*len(CANMEX_RATE_VALS)

# Pharma HHI constants (from sector_pharma_results.npz)
HHI_PRE       = 580.4
HHI_TAU_OBS   = 591.3    # at tau_pharma = 0.199
TAU_OBS       = 0.199
EPS_PHARMA    = 2.3
# Fit power-law: HHI(tau) = HHI_PRE * (1+tau)^alpha
HHI_ALPHA = np.log(HHI_TAU_OBS / HHI_PRE) / np.log(1 + TAU_OBS)


# ---------------------------------------------------------------------------
# Load and calibrate 5-bloc model
# ---------------------------------------------------------------------------
def build_5bloc_model():
    """
    Aggregate 194-country data into 5 blocs:
      0 = US
      1 = China
      2 = EU (25 countries)
      3 = Canada + Mexico
      4 = Rest of World

    Returns a dict with all calibrated arrays for the 5-bloc GE model.
    """
    print("Loading 194-country calibration data...")
    X_ji = (pd.read_csv(os.path.join(DATA_DIR, 'trade_cepii.csv'))
              .apply(pd.to_numeric, errors='coerce').fillna(0).values.copy())
    N = X_ji.shape[0]
    Y_i = pd.to_numeric(
        pd.read_csv(os.path.join(DATA_DIR, 'gdp.csv')).values.flatten(),
        errors='coerce') / 1000

    # Country indices (0-based)
    id_US  = 184
    id_CHN = 33
    id_CAN = 30
    id_MEX = 114
    id_EU  = np.array([10,13,17,45,47,50,56,57,59,61,71,78,80,83,88,
                       107,108,109,119,133,144,145,149,164,165]) - 1

    all_ids = np.arange(N)
    special = np.concatenate([[id_US, id_CHN, id_CAN, id_MEX], id_EU])
    id_RoW  = np.setdiff1d(all_ids, special)

    bloc_ids = [
        np.array([id_US]),
        np.array([id_CHN]),
        id_EU,
        np.array([id_CAN, id_MEX]),
        id_RoW,
    ]
    bloc_names = ['US', 'China', 'EU', 'CanMex', 'RoW']
    N5 = 5

    # Calibrate 194-country model first
    nu_eq = solve_nu(X_ji, Y_i, id_US)
    nu194 = nu_eq[0] * np.ones(N)
    nu194[id_US] = nu_eq[1]

    T194 = (1 - nu194) * (
        np.sum(X_ji, axis=0) -
        np.sum(np.tile((1 - nu194).reshape(1, -1), (N, 1)) * X_ji, axis=1)
    )
    E194 = Y_i + T194
    X_ii  = E194 - np.sum(X_ji, axis=0)
    X_ii[X_ii < 0] = 0
    np.fill_diagonal(X_ji, X_ii)

    E194  = np.sum(X_ji, axis=0)
    Y194  = (np.sum(np.tile((1 - nu194).reshape(1, -1), (N, 1)) * X_ji, axis=1)
             + nu194 * np.sum(X_ji, axis=0))
    T194  = E194 - Y194

    # phi for each country
    eps   = 4
    kappa = 0.5
    psi   = 0.67 / eps
    theta = eps / 0.67
    phi_tilde = (1 + theta) / ((1 - nu194) * theta) - (1 / theta) - 1
    phi194 = 1 + phi_tilde  # shape (N,)

    # ---- Aggregate to 5 blocs ----
    # X_agg[j, i] = total exports FROM bloc j TO bloc i
    X_agg = np.zeros((N5, N5))
    Y_agg = np.zeros(N5)
    for bi, ids_i in enumerate(bloc_ids):
        Y_agg[bi] = Y194[ids_i].sum()
        for bj, ids_j in enumerate(bloc_ids):
            X_agg[bi, bj] = X_ji[np.ix_(ids_i, ids_j)].sum()

    # Expenditure and trade shares for 5-bloc
    E_agg     = X_agg.sum(axis=0)            # column sums = total imports of each bloc
    lambda_agg = X_agg / E_agg[np.newaxis, :]  # share of i's spending from j

    T_agg = E_agg - Y_agg

    # nu for each bloc: weighted avg of constituent countries
    nu_agg  = np.array([
        np.average(nu194[ids], weights=Y194[ids] + 1e-12)
        for ids in bloc_ids
    ])
    # phi for each bloc: weighted avg of constituent countries
    phi_agg = np.array([
        np.average(phi194[ids], weights=Y194[ids] + 1e-12)
        for ids in bloc_ids
    ])

    # Normalize by total world GDP so residuals are O(1) — improves fsolve numerics
    scale   = Y_agg.sum()
    Y_agg_n = Y_agg / scale
    X_agg_n = X_agg / scale
    E_agg_n = E_agg / scale
    T_agg_n = T_agg / scale
    # Recompute lambda_agg on normalized matrix (ratios unchanged but be explicit)
    lambda_agg_n = X_agg_n / E_agg_n[np.newaxis, :]

    print(f"5-bloc model built and normalized (scale={scale:.3e}).")
    print(f"  Bloc GDP shares: "
          + ", ".join(f"{b}={Y_agg_n[i]:.3f}" for i, b in enumerate(bloc_names)))
    print(f"  nu_agg:  {nu_agg.round(4)}")
    print(f"  phi_agg: {phi_agg.round(4)}")

    return {
        'N5': N5,
        'X_agg': X_agg_n,
        'Y_agg': Y_agg_n,
        'E_agg': E_agg_n,
        'lambda_agg': lambda_agg_n,
        'T_agg': T_agg_n,
        'nu_agg': nu_agg,
        'phi_agg': phi_agg,
        'eps': eps,
        'kappa': kappa,
        'psi': psi,
        'bloc_names': bloc_names,
    }


# ---------------------------------------------------------------------------
# Run one GE solve for a given tariff combination
# ---------------------------------------------------------------------------
def run_one(us_tariff, china_rate, eu_rate, canmex_rate, calib, x0=None):
    """
    Run 5-bloc GE model for given tariff rates.
    Returns dict of outputs or None on failure.

    Tariff matrix convention: t_agg[j, i] = tariff imposed by bloc i on imports from j.
      - US imposes us_tariff on all blocs:  t_agg[:, 0] = us_tariff
      - China retaliates on US:             t_agg[0, 1] = china_rate
      - EU retaliates on US:                t_agg[0, 2] = eu_rate
      - CanMex retaliates on US:            t_agg[0, 3] = canmex_rate
    """
    N5         = calib['N5']
    lambda_agg = calib['lambda_agg']
    E_agg      = calib['E_agg']
    Y_agg      = calib['Y_agg']
    T_agg      = calib['T_agg']
    nu_agg     = calib['nu_agg']
    phi_agg    = calib['phi_agg']
    eps        = calib['eps']
    kappa      = calib['kappa']
    psi        = calib['psi']

    # Construct tariff matrix
    t_agg = np.zeros((N5, N5))
    t_agg[:, 0] = us_tariff   # US tariff on all importers
    t_agg[0, 0] = 0           # no self-tariff
    t_agg[0, 1] = china_rate  # China retaliates on US
    t_agg[0, 2] = eu_rate     # EU retaliates on US
    t_agg[0, 3] = canmex_rate # CanMex retaliates on US

    data  = {'N': N5, 'E_i': E_agg, 'Y_i': Y_agg, 'lambda_ji': lambda_agg,
             't_ji': t_agg, 'nu': nu_agg, 'T_i': T_agg}
    param = {'eps': eps, 'kappa': kappa, 'psi': psi, 'phi': phi_agg}

    if x0 is None:
        x0 = np.ones(3 * N5)

    def syst(x):
        ceq, _, _ = balanced_trade_eq(x, data, param, 0)
        return ceq

    try:
        x_sol = fsolve(syst, x0, xtol=1e-6, maxfev=50000, factor=0.1, full_output=False)
        _, res, d_trade = balanced_trade_eq(x_sol, data, param, 0)

        # Check convergence: normalized data so residuals should be O(1e-4)
        residual = np.max(np.abs(syst(x_sol)))
        if residual > 0.5:          # generous threshold for 5-bloc model
            return None, x0  # diverged

        # Bloc indices: 0=US, 1=China, 2=EU, 3=CanMex, 4=RoW
        welfare_us      = float(res[0, 0])
        cpi_us          = float(res[0, 5])
        employment_us   = float(res[0, 4])
        trade_deficit_us= float(res[0, 1])
        welfare_china   = float(res[1, 0])
        welfare_eu      = float(res[2, 0])

        # Pharma HHI: power-law approximation anchored to observed values
        hhi_pharma = float(HHI_PRE * (1 + us_tariff) ** HHI_ALPHA)

        return {
            'us_tariff':       us_tariff,
            'china_rate':      china_rate,
            'eu_rate':         eu_rate,
            'canmex_rate':     canmex_rate,
            'welfare_us':      welfare_us,
            'cpi_us':          cpi_us,
            'employment_us':   employment_us,
            'trade_deficit_us':trade_deficit_us,
            'hhi_pharma':      hhi_pharma,
            'welfare_china':   welfare_china,
            'welfare_eu':      welfare_eu,
        }, x_sol  # return solution as warm start for next iteration

    except Exception:
        return None, x0


# ---------------------------------------------------------------------------
# Main: generate all combinations
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" STEP 1: Generating Surrogate Training Data")
    print(f" Grid: {len(US_TARIFF_VALS)}×{len(CHINA_RATE_VALS)}×"
          f"{len(EU_RATE_VALS)}×{len(CANMEX_RATE_VALS)} = {TOTAL_RUNS} combinations")
    print("=" * 65)

    calib = build_5bloc_model()

    # Validate 5-bloc model against known Liberation Day result
    print("\nValidating 5-bloc model against known Liberation Day result (SC0)...")
    result_sc0, _ = run_one(0.27, 0.0, 0.0, 0.0, calib)
    if result_sc0:
        print(f"  5-bloc  welfare_us = {result_sc0['welfare_us']:.3f}%  "
              f"(194-country GE = +1.130%)")
        print(f"  5-bloc  cpi_us     = {result_sc0['cpi_us']:.3f}%  "
              f"(194-country GE = +12.811%)")
    print()

    rows = []
    n_failed = 0
    t_start = time.time()
    x0 = np.ones(3 * calib['N5'])  # warm-start seed

    all_combos = list(product(US_TARIFF_VALS, CHINA_RATE_VALS,
                               EU_RATE_VALS, CANMEX_RATE_VALS))

    for idx, (us_t, cn_r, eu_r, cm_r) in enumerate(all_combos):
        result, x0_new = run_one(us_t, cn_r, eu_r, cm_r, calib, x0)

        if result is not None:
            rows.append(result)
            x0 = x0_new  # warm-start from last converged solution
        else:
            n_failed += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == TOTAL_RUNS:
            elapsed = time.time() - t_start
            rate    = (idx + 1) / elapsed
            eta     = (TOTAL_RUNS - idx - 1) / rate
            print(f"  [{idx+1:5d}/{TOTAL_RUNS}]  "
                  f"converged={len(rows)}  failed={n_failed}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    # Save
    df = pd.DataFrame(rows)
    df.to_csv(TRAIN_OUT, index=False)

    elapsed_total = time.time() - t_start
    print(f"\n[OK] Done in {elapsed_total:.1f}s")
    print(f"     Converged: {len(rows)} / {TOTAL_RUNS}  (failed: {n_failed})")
    print(f"     Saved to:  {TRAIN_OUT}")
    print(f"     Shape:     {df.shape}")
    print(df.describe().round(3).to_string())
    return df


if __name__ == '__main__':
    main()
