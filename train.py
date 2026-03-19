#!/usr/bin/env python3
"""
WEAR-ME AutoResearch: HOMA-IR Prediction from Wearables + Blood Biomarkers

This is the file the agent modifies. Everything is fair game:
- Feature engineering
- Model architecture (XGBoost, LightGBM, ElasticNet, etc.)
- Hyperparameters
- Blending strategies
- Target transforms
- Sample weighting
- Preprocessing

Current best R² = 0.5467 (V20 blend: LGB_QT 71% + ElasticNet 29%)
"""
import numpy as np
import pandas as pd
import time
import warnings
import sys
warnings.filterwarnings('ignore')

from prepare import load_data, get_feature_sets, get_cv_splits, engineer_all_features
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb

# ============================================================
# CONFIG
# ============================================================
SEED = 42
N_SPLITS = 5  # from get_cv_splits
LOG_TARGET = True  # log1p transform on y
SAMPLE_WEIGHT_EXP = 0.5  # sqrt(y) weighting

# ============================================================
# DATA
# ============================================================
t_start = time.time()
print("=" * 60)
print("  WEAR-ME AutoResearch Experiment")
print("=" * 60)
sys.stdout.flush()

X_df, y, feature_names = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)
splits = get_cv_splits(y)
n = len(y)

# Sample weights
w = np.power(y, SAMPLE_WEIGHT_EXP)
w = w / w.mean()

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def engineer_features(X_df, cols):
    """Engineer features from raw data. Agent can modify this freely."""
    X = X_df[cols].copy()
    g = X['glucose'].clip(lower=1)
    t = X['triglycerides'].clip(lower=1)
    h = X['hdl'].clip(lower=1)
    b = X['bmi']
    l = X['ldl']
    tc = X['total cholesterol']
    nh = X['non hdl']
    ch = X['chol/hdl']
    a = X['age']

    # Metabolic indices
    X['tyg'] = np.log(t * g / 2)
    X['tyg_bmi'] = np.log(t * g / 2) * b
    X['mets_ir'] = np.log(2 * g + t) * b / np.log(h)
    X['trig_hdl'] = t / h
    X['trig_hdl_log'] = np.log1p(t / h)
    X['vat_proxy'] = b * t / h
    X['ir_proxy'] = g * b * t / (h * 100)

    # Glucose interactions
    X['glucose_bmi'] = g * b
    X['glucose_sq'] = g ** 2
    X['glucose_log'] = np.log(g)
    X['glucose_hdl'] = g / h
    X['glucose_trig'] = g * t / 1000
    X['glucose_non_hdl'] = g * nh / 100
    X['glucose_chol_hdl'] = g * ch

    # BMI interactions
    X['bmi_sq'] = b ** 2
    X['bmi_log'] = np.log(b.clip(lower=1))
    X['bmi_trig'] = b * t / 100
    X['bmi_hdl_inv'] = b / h
    X['bmi_age'] = b * a

    # Lipid ratios
    X['ldl_hdl'] = l / h
    X['non_hdl_ratio'] = nh / h
    X['tc_hdl_bmi'] = tc / h * b
    X['trig_tc'] = t / tc.clip(lower=1)

    # Squared terms
    X['tyg_sq'] = X['tyg'] ** 2
    X['mets_ir_sq'] = X['mets_ir'] ** 2
    X['trig_hdl_sq'] = X['trig_hdl'] ** 2
    X['vat_sq'] = X['vat_proxy'] ** 2
    X['ir_proxy_sq'] = X['ir_proxy'] ** 2
    X['ir_proxy_log'] = np.log1p(X['ir_proxy'])

    # Wearable interactions (if available)
    rhr = 'Resting Heart Rate (mean)'
    hrv = 'HRV (mean)'
    stp = 'STEPS (mean)'
    if rhr in X.columns:
        X['bmi_rhr'] = b * X[rhr]
        X['glucose_rhr'] = g * X[rhr]
        X['trig_hdl_rhr'] = X['trig_hdl'] * X[rhr]
        X['ir_proxy_rhr'] = X['ir_proxy'] * X[rhr] / 100
        X['tyg_rhr'] = X['tyg'] * X[rhr]
        X['mets_rhr'] = X['mets_ir'] * X[rhr]
        X['bmi_hrv_inv'] = b / X[hrv].clip(lower=1)
        X['cardio_fitness'] = X[hrv] * X[stp] / X[rhr].clip(lower=1)
        X['met_load'] = b * X[rhr] / X[stp].clip(lower=1) * 1000

    # Log transforms
    X['log_glucose'] = np.log(g)
    X['log_trig'] = np.log(t)
    X['log_bmi'] = np.log(b.clip(lower=1))
    X['log_hdl'] = np.log(h)
    X['log_homa_proxy'] = np.log(g) + np.log(b.clip(lower=1)) + np.log(t) - np.log(h)

    return X.fillna(0)

X_eng = engineer_features(X_df[all_cols], all_cols).values
eng_cols = engineer_features(X_df[all_cols], all_cols).columns.tolist()

# Target transform
log_fn = np.log1p
inv_log = np.expm1

# ============================================================
# MODELS
# ============================================================
def make_xgb_params():
    return dict(
        n_estimators=800, max_depth=3, learning_rate=0.015,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=2.0, min_child_weight=5,
        random_state=SEED, n_jobs=-1
    )

def make_lgb_params():
    return dict(
        n_estimators=600, max_depth=4, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=2.0, min_child_weight=5,
        random_state=SEED, n_jobs=-1, verbose=-1
    )

def make_elasticnet_params():
    return dict(alpha=0.01, l1_ratio=0.5, max_iter=10000)

# ============================================================
# CROSS-VALIDATION
# ============================================================
oof_xgb = np.zeros(n)
oof_lgb = np.zeros(n)
oof_enet = np.zeros(n)

for fold_idx, (tr_idx, va_idx) in enumerate(splits):
    X_tr, X_va = X_eng[tr_idx], X_eng[va_idx]
    y_tr_raw, y_va_raw = y[tr_idx], y[va_idx]
    w_tr = w[tr_idx]

    # Target transform
    if LOG_TARGET:
        y_tr = log_fn(y_tr_raw)
        y_va = log_fn(y_va_raw)
    else:
        y_tr = y_tr_raw
        y_va = y_va_raw

    # QuantileTransformer on inputs for LGB
    qt = QuantileTransformer(n_quantiles=200, output_distribution='normal', random_state=SEED)
    X_tr_qt = qt.fit_transform(X_tr)
    X_va_qt = qt.transform(X_va)

    # StandardScaler for ElasticNet
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_va_sc = sc.transform(X_va)

    # XGBoost
    xgb_model = xgb.XGBRegressor(**make_xgb_params())
    xgb_model.fit(X_tr, y_tr, sample_weight=w_tr)
    pred_xgb = xgb_model.predict(X_va)
    if LOG_TARGET:
        pred_xgb = inv_log(pred_xgb)
    oof_xgb[va_idx] = pred_xgb

    # LightGBM with QuantileTransformer inputs
    lgb_model = lgb.LGBMRegressor(**make_lgb_params())
    lgb_model.fit(X_tr_qt, y_tr, sample_weight=w_tr)
    pred_lgb = lgb_model.predict(X_va_qt)
    if LOG_TARGET:
        pred_lgb = inv_log(pred_lgb)
    oof_lgb[va_idx] = pred_lgb

    # ElasticNet
    enet_model = ElasticNet(**make_elasticnet_params())
    enet_model.fit(X_tr_sc, y_tr)
    pred_enet = enet_model.predict(X_va_sc)
    if LOG_TARGET:
        pred_enet = inv_log(pred_enet)
    oof_enet[va_idx] = pred_enet

    fold_r2_xgb = r2_score(y[va_idx], oof_xgb[va_idx])
    fold_r2_lgb = r2_score(y[va_idx], oof_lgb[va_idx])
    print(f"  Fold {fold_idx}: XGB={fold_r2_xgb:.4f} LGB={fold_r2_lgb:.4f}")
    sys.stdout.flush()

# ============================================================
# RESULTS
# ============================================================
r2_xgb = r2_score(y, oof_xgb)
r2_lgb = r2_score(y, oof_lgb)
r2_enet = r2_score(y, oof_enet)

print(f"\nSingle model R²: XGB={r2_xgb:.4f} LGB={r2_lgb:.4f} ElasticNet={r2_enet:.4f}")

# Blend optimization (grid search)
best_r2 = -1
best_w = None
for w_lgb in np.arange(0, 1.01, 0.01):
    for w_enet in np.arange(0, 1.01 - w_lgb, 0.01):
        w_xgb_b = 1 - w_lgb - w_enet
        blend = w_xgb_b * oof_xgb + w_lgb * oof_lgb + w_enet * oof_enet
        r2 = r2_score(y, blend)
        if r2 > best_r2:
            best_r2 = r2
            best_w = (w_xgb_b, w_lgb, w_enet)

elapsed = time.time() - t_start

print(f"\nBest blend: XGB={best_w[0]:.2f} LGB={best_w[1]:.2f} ElasticNet={best_w[2]:.2f}")
print()
print("---")
print(f"val_r2:           {best_r2:.6f}")
print(f"val_r2_xgb:       {r2_xgb:.6f}")
print(f"val_r2_lgb:       {r2_lgb:.6f}")
print(f"val_r2_enet:      {r2_enet:.6f}")
print(f"blend_weights:    XGB={best_w[0]:.2f} LGB={best_w[1]:.2f} EN={best_w[2]:.2f}")
print(f"n_features:       {X_eng.shape[1]}")
print(f"n_samples:        {n}")
print(f"total_seconds:    {elapsed:.1f}")
print(f"n_splits:         {N_SPLITS}")
