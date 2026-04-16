"""
Step 2: Train XGBoost surrogate models on generated training data.

For each output variable, trains:
  - One main XGBoost regressor (median prediction)
  - Two quantile regressors (q=0.1 and q=0.9 for 80% CI)

Outputs:
  surrogate_training/models/model_<target>.json        (main models)
  surrogate_training/models/model_<target>_q10.json   (lower CI)
  surrogate_training/models/model_<target>_q90.json   (upper CI)
  surrogate_training/feature_importance.png
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("Install xgboost:  pip install xgboost")

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV  = os.path.join(SCRIPT_DIR, 'training_data.csv')
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
FIG_OUT    = os.path.join(SCRIPT_DIR, 'feature_importance.png')
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES = ['us_tariff', 'china_rate', 'eu_rate', 'canmex_rate']
TARGETS  = ['welfare_us', 'cpi_us', 'employment_us', 'trade_deficit_us',
            'hhi_pharma', 'welfare_china', 'welfare_eu']

# XGBoost hyperparameters (good defaults for small tabular datasets)
XGB_PARAMS = dict(
    n_estimators   = 400,
    max_depth      = 5,
    learning_rate  = 0.05,
    subsample      = 0.85,
    colsample_bytree = 0.85,
    min_child_weight = 3,
    reg_lambda     = 1.0,
    reg_alpha      = 0.1,
    random_state   = 42,
    n_jobs         = -1,
    tree_method    = 'hist',
)


# ---------------------------------------------------------------------------
# Train one model
# ---------------------------------------------------------------------------
def train_model(X_train, y_train, objective='reg:squarederror', quantile_alpha=None):
    params = dict(XGB_PARAMS)
    params['objective'] = objective
    if quantile_alpha is not None:
        params['quantile_alpha'] = quantile_alpha
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" STEP 2: Training XGBoost Surrogate Models")
    print("=" * 65)

    # Load data
    df = pd.read_csv(TRAIN_CSV)
    print(f"\nLoaded training data: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Targets: {TARGETS}")

    X = df[FEATURES].values
    results = {}

    # 80/20 split
    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(idx, test_size=0.20, random_state=42)
    X_train, X_test = X[idx_train], X[idx_test]

    importances = {}

    print(f"\n{'Target':<22} {'R²_train':>9} {'R²_test':>9} {'RMSE_test':>11}")
    print("-" * 55)

    for target in TARGETS:
        y       = df[target].values
        y_train = y[idx_train]
        y_test  = y[idx_test]

        # Main model (MSE)
        m_main = train_model(X_train, y_train)
        y_pred_tr = m_main.predict(X_train)
        y_pred_te = m_main.predict(X_test)
        r2_tr   = r2_score(y_train, y_pred_tr)
        r2_te   = r2_score(y_test,  y_pred_te)
        rmse_te = np.sqrt(mean_squared_error(y_test, y_pred_te))

        print(f"  {target:<20} {r2_tr:>9.4f} {r2_te:>9.4f} {rmse_te:>11.4f}")
        results[target] = {'r2_train': r2_tr, 'r2_test': r2_te, 'rmse_test': rmse_te}

        # Quantile models for confidence interval
        m_q10 = train_model(X_train, y_train, 'reg:quantileerror', quantile_alpha=0.10)
        m_q90 = train_model(X_train, y_train, 'reg:quantileerror', quantile_alpha=0.90)

        # Save models
        m_main.save_model(os.path.join(MODELS_DIR, f'model_{target}.json'))
        m_q10.save_model(os.path.join(MODELS_DIR,  f'model_{target}_q10.json'))
        m_q90.save_model(os.path.join(MODELS_DIR,  f'model_{target}_q90.json'))

        # Feature importances (gain)
        importances[target] = m_main.feature_importances_

    print("-" * 55)
    print(f"\n[OK] Models saved to: {MODELS_DIR}/")

    # Save R² summary
    r2_df = pd.DataFrame(results).T
    r2_path = os.path.join(SCRIPT_DIR, 'r2_scores.csv')
    r2_df.to_csv(r2_path)
    print(f"[OK] R² scores saved to: {r2_path}")

    # ---- Feature importance plot ----
    imp_arr = np.array([importances[t] for t in TARGETS])  # (n_targets, n_features)
    mean_imp = imp_arr.mean(axis=0)                          # average across targets

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    palette = ['#3182bd', '#e6550d', '#31a354', '#756bb1']

    for ti, target in enumerate(TARGETS):
        ax = axes[ti]
        imp = importances[target]
        order = np.argsort(imp)[::-1]
        bars = ax.barh([FEATURES[i] for i in order],
                       [imp[i] for i in order],
                       color=[palette[i] for i in order])
        ax.set_title(f'{target}\n(R²={results[target]["r2_test"]:.3f})',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Importance (gain)', fontsize=8)
        ax.tick_params(labelsize=8)
        for bar, val in zip(bars, [imp[i] for i in order]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=7)

    # Summary panel: mean importance across all targets
    ax = axes[len(TARGETS)]
    order = np.argsort(mean_imp)[::-1]
    ax.barh([FEATURES[i] for i in order], [mean_imp[i] for i in order],
            color=[palette[i] for i in order])
    ax.set_title('MEAN across all targets', fontsize=9, fontweight='bold')
    ax.set_xlabel('Mean Importance (gain)', fontsize=8)
    ax.tick_params(labelsize=8)

    # Hide unused subplots
    for ax in axes[len(TARGETS)+1:]:
        ax.set_visible(False)

    fig.suptitle('XGBoost Feature Importance\nSurrogate Model for Liberation Day Tariff GE',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Feature importance plot saved to: {FIG_OUT}")

    return results


if __name__ == '__main__':
    main()
