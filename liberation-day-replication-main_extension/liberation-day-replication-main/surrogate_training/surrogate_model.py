"""
Step 3: Surrogate model predict function + bias-corrected validation.

Architecture
------------
  _predict_raw()        -> raw XGBoost prediction (5-bloc GE surrogate)
  _compute_calibration()-> OLS linear correction fit over 9 known anchor scenarios
  predict()             -> applies calibration, returns corrected values + CI

Public API:
    from surrogate_training.surrogate_model import predict

    result = predict(us_tariff=0.27, china_rate=0.0, eu_rate=0.0, canmex_rate=0.0)
    # Returns:
    # {
    #   "welfare_us":       {"value": X, "ci_low": X, "ci_high": X},
    #   "cpi_us":           {"value": X, "ci_low": X, "ci_high": X},
    #   "employment_us":    {...},
    #   "trade_deficit_us": {...},
    #   "hhi_pharma":       {...},
    #   "welfare_china":    {...},
    #   "welfare_eu":       {...},
    # }
"""

import os
import warnings
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("Install xgboost:  pip install xgboost")

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

FEATURES = ['us_tariff', 'china_rate', 'eu_rate', 'canmex_rate']
TARGETS  = ['welfare_us', 'cpi_us', 'employment_us', 'trade_deficit_us',
            'hhi_pharma', 'welfare_china', 'welfare_eu']

# HHI pharma is computed analytically; no bias-correct against the GE scenarios
TARGETS_CORRECTED = [t for t in TARGETS if t != 'hhi_pharma']

# ---------------------------------------------------------------------------
# Model cache (lazy-loaded once)
# ---------------------------------------------------------------------------
_models = {}
_calib  = {}   # {target: {'a': float, 'b': float, 'r2': float}}


def _load_models():
    """Load all XGBoost models from disk (lazy, cached)."""
    global _models
    if _models:
        return _models

    for target in TARGETS:
        for suffix in ['', '_q10', '_q90']:
            key  = f'{target}{suffix}'
            path = os.path.join(MODELS_DIR, f'model_{key}.json')
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model not found: {path}\n"
                    "Run train_surrogate.py first."
                )
            m = xgb.XGBRegressor()
            m.load_model(path)
            _models[key] = m

    return _models


# ---------------------------------------------------------------------------
# Known scenarios: tariff inputs + true 194-country GE values
# ---------------------------------------------------------------------------
# Scenario tariff parameterisation:
#   SC0-2, SC7-8: Liberation Day tariffs (us_tariff=0.27) with no partner retaliation.
#     These share identical tariff inputs but differ in model spec (phi, nu, fiscal rule).
#   SC3:  Optimal US tariff (us_tariff=0.50), no retaliation.
#   SC4:  Liberation Day + optimal retaliation  (china=0.70, eu/canmex=0.30).
#   SC5:  Liberation Day + reciprocal retaliation (all partners match US rate=0.27).
#   SC6:  Optimal US tariff + optimal retaliation (Nash equilibrium).

KNOWN_SCENARIOS = {
    0: {'name': 'Lib Day – No Retaliation',
        'us_tariff': 0.27, 'china_rate': 0.00, 'eu_rate': 0.00, 'canmex_rate': 0.00},
    1: {'name': 'Lib Day – Armington',
        'us_tariff': 0.27, 'china_rate': 0.00, 'eu_rate': 0.00, 'canmex_rate': 0.00},
    2: {'name': 'Lib Day – Eaton-Kortum',
        'us_tariff': 0.27, 'china_rate': 0.00, 'eu_rate': 0.00, 'canmex_rate': 0.00},
    3: {'name': 'Optimal US – No Retaliation',
        'us_tariff': 0.50, 'china_rate': 0.00, 'eu_rate': 0.00, 'canmex_rate': 0.00},
    4: {'name': 'Lib + Optimal Retaliation',
        'us_tariff': 0.27, 'china_rate': 0.70, 'eu_rate': 0.30, 'canmex_rate': 0.30},
    5: {'name': 'Lib + Reciprocal Retaliation',
        'us_tariff': 0.27, 'china_rate': 0.27, 'eu_rate': 0.27, 'canmex_rate': 0.27},
    6: {'name': 'Nash Equilibrium',
        'us_tariff': 0.50, 'china_rate': 0.70, 'eu_rate': 0.30, 'canmex_rate': 0.30},
    7: {'name': 'Lib + Lump-Sum Rebate',
        'us_tariff': 0.27, 'china_rate': 0.00, 'eu_rate': 0.00, 'canmex_rate': 0.00},
    8: {'name': 'Lib – High Elasticity',
        'us_tariff': 0.27, 'china_rate': 0.00, 'eu_rate': 0.00, 'canmex_rate': 0.00},
}

# True values from running main_baseline.py on the full 194-country model.
# Metrics: welfare(%), CPI(%), employment(%), trade_deficit(%), welfare_china(%), welfare_eu(%)
TRUE_VALUES = {
    0: {'welfare_us':  1.130, 'cpi_us': 12.811, 'employment_us':  0.318,
        'trade_deficit_us': -18.091, 'welfare_china': -1.132, 'welfare_eu':  0.071},
    1: {'welfare_us':  1.365, 'cpi_us':  7.933, 'employment_us':  0.296,
        'trade_deficit_us': -11.778, 'welfare_china': -0.398, 'welfare_eu': -0.016},
    2: {'welfare_us':  1.243, 'cpi_us': 10.909, 'employment_us':  0.415,
        'trade_deficit_us':  -0.426, 'welfare_china': -1.030, 'welfare_eu':  0.045},
    3: {'welfare_us':  1.789, 'cpi_us': 12.604, 'employment_us':  0.509,
        'trade_deficit_us': -19.089, 'welfare_china': -0.855, 'welfare_eu':  0.014},
    4: {'welfare_us': -0.949, 'cpi_us':  5.093, 'employment_us': -0.388,
        'trade_deficit_us': -30.186, 'welfare_china': -0.595, 'welfare_eu': -0.149},
    5: {'welfare_us': -0.359, 'cpi_us':  7.484, 'employment_us': -0.185,
        'trade_deficit_us': -26.736, 'welfare_china': -0.821, 'welfare_eu': -0.068},
    6: {'welfare_us': -0.544, 'cpi_us':  3.288, 'employment_us': -0.274,
        'trade_deficit_us': -29.661, 'welfare_china': -0.325, 'welfare_eu': -0.187},
    7: {'welfare_us': -0.006, 'cpi_us': 13.127, 'employment_us': -0.410,
        'trade_deficit_us': -18.411, 'welfare_china': -1.103, 'welfare_eu':  0.052},
    8: {'welfare_us':  0.325, 'cpi_us': 11.760, 'employment_us':  0.097,
        'trade_deficit_us': -26.350, 'welfare_china': -0.827, 'welfare_eu':  0.072},
}


# ---------------------------------------------------------------------------
# Internal: raw XGBoost prediction (no correction)
# ---------------------------------------------------------------------------
def _predict_raw(us_tariff: float, china_rate: float,
                 eu_rate: float, canmex_rate: float) -> dict:
    """
    Raw prediction straight from XGBoost (5-bloc GE surrogate, no bias correction).
    Returns {target: {'value', 'ci_low', 'ci_high'}}.
    """
    models = _load_models()
    X = np.array([[us_tariff, china_rate, eu_rate, canmex_rate]], dtype=float)

    result = {}
    for target in TARGETS:
        val    = float(models[target].predict(X)[0])
        ci_low = float(models[f'{target}_q10'].predict(X)[0])
        ci_high= float(models[f'{target}_q90'].predict(X)[0])
        if ci_low > ci_high:
            ci_low, ci_high = ci_high, ci_low
        result[target] = {'value': val, 'ci_low': ci_low, 'ci_high': ci_high}

    return result


# ---------------------------------------------------------------------------
# Calibration: OLS linear correction fit on 9 known anchor scenarios
# ---------------------------------------------------------------------------
def _compute_calibration(verbose: bool = False) -> dict:
    """
    Fit a per-target linear correction: true = a * raw + b.

    Uses all 9 known scenarios as anchor points.  For targets with no true
    GE value available (hhi_pharma), the identity transform (a=1, b=0) is used.

    Returns
    -------
    dict: {target: {'a': float, 'b': float, 'r2_raw': float, 'r2_corrected': float,
                    'mae_raw': float, 'mae_corrected': float}}
    """
    global _calib
    if _calib:
        return _calib

    # Collect raw predictions + true values across all 9 scenarios
    raw_by_target  = {t: [] for t in TARGETS}
    true_by_target = {t: [] for t in TARGETS}

    for sc_id, sc_info in KNOWN_SCENARIOS.items():
        raw = _predict_raw(sc_info['us_tariff'], sc_info['china_rate'],
                           sc_info['eu_rate'],   sc_info['canmex_rate'])
        true = TRUE_VALUES[sc_id]
        for t in TARGETS:
            raw_by_target[t].append(raw[t]['value'])
            true_by_target[t].append(true.get(t, np.nan))

    for t in TARGETS:
        raw_arr  = np.array(raw_by_target[t],  dtype=float)
        true_arr = np.array(true_by_target[t], dtype=float)
        mask     = np.isfinite(true_arr) & np.isfinite(raw_arr)

        if t == 'hhi_pharma' or mask.sum() < 2:
            # No GE ground truth → identity transform
            a, b = 1.0, 0.0
            r2_raw = r2_corrected = np.nan
            mae_raw = mae_corrected = np.nan
        else:
            x, y = raw_arr[mask], true_arr[mask]

            # OLS: y = a*x + b  (fit via normal equations)
            A      = np.column_stack([x, np.ones(len(x))])
            coefs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b   = float(coefs[0]), float(coefs[1])

            # Diagnostics
            y_pred_raw  = x                  # raw is the predictor
            y_pred_corr = a * x + b

            ss_tot = np.sum((y - y.mean()) ** 2)
            if ss_tot > 0:
                r2_raw        = float(1 - np.sum((y - y_pred_raw)  ** 2) / ss_tot)
                r2_corrected  = float(1 - np.sum((y - y_pred_corr) ** 2) / ss_tot)
            else:
                r2_raw = r2_corrected = np.nan

            mae_raw       = float(np.mean(np.abs(y - y_pred_raw)))
            mae_corrected = float(np.mean(np.abs(y - y_pred_corr)))

        _calib[t] = {
            'a': a, 'b': b,
            'r2_raw': r2_raw, 'r2_corrected': r2_corrected,
            'mae_raw': mae_raw, 'mae_corrected': mae_corrected,
        }

        if verbose:
            print(f"  {t:<22}: a={a:+.4f}  b={b:+.4f}  "
                  f"MAE_raw={mae_raw:.3f}  MAE_corr={mae_corrected:.3f}")

    return _calib


# ---------------------------------------------------------------------------
# Public: predict with bias correction applied
# ---------------------------------------------------------------------------
def predict(us_tariff: float, china_rate: float,
            eu_rate: float, canmex_rate: float) -> dict:
    """
    Predict GE outcomes with linear bias correction applied.

    Parameters
    ----------
    us_tariff    : US tariff rate on all imports (0.0 = free trade, 1.0 = 100%)
    china_rate   : China's retaliatory tariff on US exports (0.0 – 1.0)
    eu_rate      : EU's retaliatory tariff on US exports (0.0 – 0.5)
    canmex_rate  : Canada+Mexico's retaliatory tariff on US exports (0.0 – 0.5)

    Returns
    -------
    dict: {target: {"value": float, "ci_low": float, "ci_high": float}}
    CI is an approximate 80% prediction interval, bias-corrected.
    """
    raw   = _predict_raw(us_tariff, china_rate, eu_rate, canmex_rate)
    calib = _compute_calibration()

    result = {}
    for t in TARGETS:
        a, b = calib[t]['a'], calib[t]['b']

        val    = a * raw[t]['value']   + b
        ci_low = a * raw[t]['ci_low']  + b
        ci_hi  = a * raw[t]['ci_high'] + b

        # If slope is negative (flips direction), swap CI bounds
        if a < 0:
            ci_low, ci_hi = ci_hi, ci_low

        result[t] = {'value': val, 'ci_low': ci_low, 'ci_high': ci_hi}

    return result


# ---------------------------------------------------------------------------
# Validation: raw vs corrected vs true
# ---------------------------------------------------------------------------
def validate_against_known_scenarios(save_path: str = None) -> pd.DataFrame:
    """
    Compare raw and corrected surrogate predictions against the 9 known GE scenarios.
    Returns a DataFrame; saves to CSV if save_path is provided.
    """
    val_targets = ['welfare_us', 'cpi_us', 'employment_us',
                   'trade_deficit_us', 'welfare_china', 'welfare_eu']
    rows = []

    for sc_id, sc_info in KNOWN_SCENARIOS.items():
        kwargs = dict(us_tariff   = sc_info['us_tariff'],
                      china_rate  = sc_info['china_rate'],
                      eu_rate     = sc_info['eu_rate'],
                      canmex_rate = sc_info['canmex_rate'])

        raw_pred  = _predict_raw(**kwargs)
        corr_pred = predict(**kwargs)
        true      = TRUE_VALUES[sc_id]

        row = {
            'scenario_id':   sc_id,
            'scenario_name': sc_info['name'],
            'us_tariff':     sc_info['us_tariff'],
            'china_rate':    sc_info['china_rate'],
            'eu_rate':       sc_info['eu_rate'],
            'canmex_rate':   sc_info['canmex_rate'],
        }
        for t in val_targets:
            tv = true.get(t, np.nan)
            rv = raw_pred[t]['value']
            cv = corr_pred[t]['value']
            row[f'{t}_true']       = tv
            row[f'{t}_raw']        = rv
            row[f'{t}_corrected']  = cv
            row[f'{t}_ci_low']     = corr_pred[t]['ci_low']
            row[f'{t}_ci_high']    = corr_pred[t]['ci_high']
            row[f'{t}_err_raw']    = rv - tv
            row[f'{t}_err_corr']   = cv - tv
        rows.append(row)

    df = pd.DataFrame(rows)
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"[OK] Validation table saved to: {save_path}")
    return df


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------
def print_validation_summary(df: pd.DataFrame):
    """Print concise side-by-side table: True | Raw | Corrected."""
    val_targets = ['welfare_us', 'cpi_us', 'employment_us',
                   'trade_deficit_us', 'welfare_china', 'welfare_eu']

    # ---- welfare_us detail table ----
    print("\n" + "=" * 90)
    print(" VALIDATION: Surrogate vs True GE (194-country model)")
    print(" Format: True / Raw / Corrected")
    print("=" * 90)
    hdr = f"{'Scenario':<35}  {'welfare_us':^20}  {'cpi_us':^18}  {'china_W':^18}"
    print(hdr)
    sub = f"{'':35}  {'True  Raw  Corr':^20}  {'True  Raw Corr':^18}  {'True  Raw Corr':^18}"
    print(sub)
    print("-" * 90)

    for _, row in df.iterrows():
        sc = str(row['scenario_name'])[:34]
        print(
            f"{sc:<35}  "
            f"{row['welfare_us_true']:+5.2f} {row['welfare_us_raw']:+5.2f} {row['welfare_us_corrected']:+5.2f}  "
            f"{row['cpi_us_true']:+5.1f} {row['cpi_us_raw']:+5.1f} {row['cpi_us_corrected']:+5.1f}  "
            f"{row['welfare_china_true']:+5.2f} {row['welfare_china_raw']:+5.2f} {row['welfare_china_corrected']:+5.2f}"
        )

    print("-" * 90)

    # ---- MAE summary ----
    print("\nMAE summary (9 scenarios):")
    print(f"  {'Target':<22}  {'MAE_raw':>9}  {'MAE_corr':>9}  {'Improvement':>12}")
    print(f"  {'-'*22}  {'-'*9}  {'-'*9}  {'-'*12}")
    for t in val_targets:
        mae_raw  = df[f'{t}_err_raw'].abs().mean()
        mae_corr = df[f'{t}_err_corr'].abs().mean()
        improv   = (mae_raw - mae_corr) / mae_raw * 100 if mae_raw > 0 else 0
        print(f"  {t:<22}  {mae_raw:>9.4f}  {mae_corr:>9.4f}  {improv:>+11.1f}%")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Calibration report
# ---------------------------------------------------------------------------
def write_calibration_report(save_path: str) -> str:
    """
    Generate calibration_report.txt showing a, b coefficients and
    MAE improvement for each target variable.
    """
    calib = _compute_calibration()
    val_df = validate_against_known_scenarios()   # in-memory only

    val_targets = ['welfare_us', 'cpi_us', 'employment_us',
                   'trade_deficit_us', 'welfare_china', 'welfare_eu']

    lines = []
    lines.append("=" * 70)
    lines.append(" Surrogate Model Calibration Report")
    lines.append(" Linear bias correction: y_corrected = a * y_raw + b")
    lines.append(" Anchor points: 9 known 194-country GE scenarios")
    lines.append("=" * 70)
    lines.append("")

    lines.append(f"{'Target':<22}  {'a':>8}  {'b':>8}  "
                 f"{'MAE_raw':>9}  {'MAE_corr':>9}  {'Improvement':>12}")
    lines.append(f"{'-'*22}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*12}")

    for t in TARGETS:
        c = calib[t]
        if t in val_targets:
            mae_raw  = val_df[f'{t}_err_raw'].abs().mean()
            mae_corr = val_df[f'{t}_err_corr'].abs().mean()
            improv   = (mae_raw - mae_corr) / mae_raw * 100 if mae_raw > 0 else 0
            lines.append(
                f"{t:<22}  {c['a']:>+8.4f}  {c['b']:>+8.4f}  "
                f"{mae_raw:>9.4f}  {mae_corr:>9.4f}  {improv:>+11.1f}%"
            )
        else:
            lines.append(
                f"{t:<22}  {c['a']:>+8.4f}  {c['b']:>+8.4f}  "
                f"{'N/A (analytical)':>35}"
            )

    lines.append("")
    lines.append("=" * 70)
    lines.append(" Per-scenario welfare_us detail")
    lines.append("=" * 70)
    lines.append(f"{'Scenario':<35}  {'True':>6}  {'Raw':>6}  {'Corr':>6}  "
                 f"{'Err_raw':>8}  {'Err_corr':>9}  {'Sign OK?':>9}")
    lines.append("-" * 90)

    for _, row in val_df.iterrows():
        true = row['welfare_us_true']
        raw  = row['welfare_us_raw']
        corr = row['welfare_us_corrected']
        err_r = raw  - true
        err_c = corr - true
        sign_ok = 'YES' if (np.sign(corr) == np.sign(true) or abs(true) < 0.05) else 'NO'
        lines.append(
            f"{str(row['scenario_name'])[:34]:<35}  "
            f"{true:>+6.2f}  {raw:>+6.2f}  {corr:>+6.2f}  "
            f"{err_r:>+8.3f}  {err_c:>+9.3f}  {sign_ok:>9}"
        )

    lines.append("")
    lines.append("Notes:")
    lines.append("  * Calibration fits y_true = a*y_raw + b via OLS on all 9 scenarios.")
    lines.append("  * Scenarios sharing identical tariff inputs (SC0,1,2,7,8) produce the")
    lines.append("    same raw prediction; linear correction cannot distinguish them—")
    lines.append("    correction is averaged over those 5 true values.")
    lines.append("  * hhi_pharma uses identity (a=1, b=0): no GE ground truth available.")
    lines.append("  * CI bounds are transformed consistently: CI_corr = a*CI_raw + b.")

    report = "\n".join(lines)
    with open(save_path, 'w') as f:
        f.write(report)
    print(f"[OK] Calibration report saved to: {save_path}")
    return report


# ---------------------------------------------------------------------------
# Run as script
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Loading models and computing bias calibration...")
    calib = _compute_calibration(verbose=True)

    val_path    = os.path.join(SCRIPT_DIR, 'validation_table.csv')
    report_path = os.path.join(SCRIPT_DIR, 'calibration_report.txt')

    df = validate_against_known_scenarios(save_path=val_path)
    print_validation_summary(df)
    write_calibration_report(save_path=report_path)

    # Demo: arbitrary new scenario
    print("\n--- Demo: predict(us_tariff=0.5, china_rate=0.5, eu_rate=0.2, canmex_rate=0.2) ---")
    demo = predict(0.5, 0.5, 0.2, 0.2)
    units = {
        'welfare_us': '%', 'cpi_us': '%', 'employment_us': '%',
        'trade_deficit_us': '%', 'hhi_pharma': '(HHI)',
        'welfare_china': '%', 'welfare_eu': '%',
    }
    for k, v in demo.items():
        u = units.get(k, '')
        print(f"  {k:<22}: {v['value']:+8.3f} {u}  "
              f"80% CI [{v['ci_low']:+8.3f}, {v['ci_high']:+8.3f}]")
