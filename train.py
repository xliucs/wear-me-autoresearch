#!/usr/bin/env python3
"""
WEAR-ME AutoResearch: HOMA-IR Prediction from WEARABLES + DEMOGRAPHICS ONLY (Model B)
No blood biomarkers allowed!
"""
import numpy as np
import pandas as pd
import time
import warnings
import sys
warnings.filterwarnings('ignore')

from prepare import load_data, get_feature_sets, get_cv_splits
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

# ============================================================
# CONFIG
# ============================================================
SEED = 42
LOG_TARGET = True
SAMPLE_WEIGHT_EXP = 0.5

# ============================================================
# DATA
# ============================================================
t_start = time.time()
print("=" * 60)
print("  WEAR-ME AutoResearch: WEARABLES-ONLY (Model B)")
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
# FEATURE ENGINEERING (wearables + demographics only!)
# ============================================================
def engineer_features(X_df, cols):
    """Engineer features from demographics + wearables only. No blood!"""
    X = X_df[cols].copy()
    b = X['bmi']
    a = X['age']
    s = X['sex_num'] if 'sex_num' in X.columns else pd.Series(0, index=X.index)

    rhr_m = X['Resting Heart Rate (mean)']
    rhr_md = X['Resting Heart Rate (median)']
    rhr_s = X['Resting Heart Rate (std)']
    hrv_m = X['HRV (mean)']
    hrv_md = X['HRV (median)']
    hrv_s = X['HRV (std)']
    stp_m = X['STEPS (mean)']
    stp_md = X['STEPS (median)']
    stp_s = X['STEPS (std)']
    slp_m = X['SLEEP Duration (mean)']
    slp_md = X['SLEEP Duration (median)']
    slp_s = X['SLEEP Duration (std)']
    azm_m = X['AZM Weekly (mean)']
    azm_md = X['AZM Weekly (median)']
    azm_s = X['AZM Weekly (std)']

    # === BMI transforms (strongest single predictor of HOMA-IR) ===
    X['bmi_sq'] = b ** 2
    X['bmi_cubed'] = b ** 3
    X['bmi_log'] = np.log(b.clip(lower=1))
    X['bmi_inv'] = 1.0 / b.clip(lower=10)
    X['bmi_sqrt'] = np.sqrt(b)

    # === Age transforms ===
    X['age_sq'] = a ** 2
    X['age_log'] = np.log(a.clip(lower=1))
    X['bmi_age'] = b * a
    X['bmi_sq_age'] = b ** 2 * a
    X['age_sex'] = a * s
    X['bmi_sex'] = b * s

    # === BMI x wearable interactions (metabolic load) ===
    X['bmi_rhr'] = b * rhr_m
    X['bmi_sq_rhr'] = b ** 2 * rhr_m
    X['bmi_hrv'] = b * hrv_m
    X['bmi_hrv_inv'] = b / hrv_m.clip(lower=1)
    X['bmi_steps'] = b * stp_m
    X['bmi_steps_inv'] = b / stp_m.clip(lower=1) * 1000
    X['bmi_sleep'] = b * slp_m
    X['bmi_azm'] = b * azm_m
    X['bmi_azm_inv'] = b / azm_m.clip(lower=1)

    # === Cardio fitness proxies ===
    X['cardio_fitness'] = hrv_m * stp_m / rhr_m.clip(lower=1)
    X['cardio_fitness_log'] = np.log1p(X['cardio_fitness'].clip(lower=0))
    X['met_load'] = b * rhr_m / stp_m.clip(lower=1) * 1000
    X['met_load_log'] = np.log1p(X['met_load'].clip(lower=0))
    X['rhr_hrv_ratio'] = rhr_m / hrv_m.clip(lower=1)
    X['autonomic_balance'] = hrv_m / rhr_m.clip(lower=1)

    # === Insulin resistance proxy indices ===
    # Sedentary risk: high BMI * high RHR / (low steps * low HRV) = bad
    X['sed_risk'] = b ** 2 * rhr_m / (stp_m.clip(lower=1) * hrv_m.clip(lower=1))
    X['sed_risk_log'] = np.log1p(X['sed_risk'].clip(lower=0))
    # Metabolic syndrome proxy
    X['met_syn_proxy'] = b * rhr_m / (hrv_m.clip(lower=1) * azm_m.clip(lower=1)) * 1e4
    # Fitness-adjusted BMI
    X['bmi_fitness_adj'] = b / (stp_m.clip(lower=1) / 5000 + azm_m.clip(lower=1) / 200)
    # BMI × resting heart load
    X['bmi_rhr_hrv'] = b * rhr_m / hrv_m.clip(lower=1)
    X['bmi_rhr_stp'] = b * rhr_m / stp_m.clip(lower=1) * 1000

    # === Variability (CV = std/mean) - instability signals ===
    X['rhr_cv'] = rhr_s / rhr_m.clip(lower=0.01)
    X['hrv_cv'] = hrv_s / hrv_m.clip(lower=0.01)
    X['steps_cv'] = stp_s / stp_m.clip(lower=0.01)
    X['sleep_cv'] = slp_s / slp_m.clip(lower=0.01)
    X['azm_cv'] = azm_s / azm_m.clip(lower=0.01)

    # === Skewness (mean - median) / std ===
    X['rhr_skew'] = (rhr_m - rhr_md) / rhr_s.clip(lower=0.01)
    X['hrv_skew'] = (hrv_m - hrv_md) / hrv_s.clip(lower=0.01)
    X['stp_skew'] = (stp_m - stp_md) / stp_s.clip(lower=0.01)
    X['slp_skew'] = (slp_m - slp_md) / slp_s.clip(lower=0.01)
    X['azm_skew'] = (azm_m - azm_md) / azm_s.clip(lower=0.01)

    # === CV × BMI interactions ===
    X['rhr_cv_bmi'] = X['rhr_cv'] * b
    X['hrv_cv_bmi'] = X['hrv_cv'] * b
    X['steps_cv_bmi'] = X['steps_cv'] * b

    # === Activity level composites ===
    X['active_score'] = stp_m * azm_m / 1000
    X['active_score_log'] = np.log1p(X['active_score'])
    X['sedentary_proxy'] = b * rhr_m / (stp_m.clip(lower=1) * azm_m.clip(lower=1)) * 1e6
    X['activity_bmi'] = (stp_m + azm_m) / b

    # === Recovery & sleep quality ===
    X['recovery'] = hrv_m / rhr_m.clip(lower=1) * slp_m
    X['sleep_efficiency'] = slp_md / slp_m.clip(lower=0.01)  # median/mean ~ 1 if stable
    X['hr_reserve'] = (220 - a - rhr_m) / b
    X['fitness_age'] = a * rhr_m / hrv_m.clip(lower=1)

    # === Age interactions ===
    X['age_rhr'] = a * rhr_m
    X['age_hrv'] = a * hrv_m
    X['age_hrv_inv'] = a / hrv_m.clip(lower=1)
    X['age_bmi_rhr'] = a * b * rhr_m / 1000
    X['age_bmi_sex'] = a * b * s

    # === Wearable cross features ===
    X['rhr_stp'] = rhr_m * stp_m
    X['hrv_stp'] = hrv_m * stp_m
    X['slp_hrv'] = slp_m * hrv_m
    X['slp_rhr'] = slp_m / rhr_m.clip(lower=1)
    X['azm_stp'] = azm_m / stp_m.clip(lower=1)
    X['rhr_slp_inv'] = rhr_m / slp_m.clip(lower=0.01)

    # === Conditional/threshold features ===
    X['obese'] = (b >= 30).astype(float)
    X['overweight'] = (b >= 25).astype(float)
    X['older'] = (a >= 50).astype(float)
    X['obese_rhr'] = X['obese'] * rhr_m
    X['obese_low_hrv'] = X['obese'] * (hrv_m < hrv_m.median()).astype(float)
    X['older_bmi'] = X['older'] * b
    X['older_rhr'] = X['older'] * rhr_m
    X['obese_low_steps'] = X['obese'] * (stp_m < stp_m.median()).astype(float)
    X['obese_poor_sleep'] = X['obese'] * (slp_m < slp_m.median()).astype(float)

    # === Log transforms of key wearables ===
    X['log_rhr'] = np.log(rhr_m.clip(lower=1))
    X['log_hrv'] = np.log(hrv_m.clip(lower=1))
    X['log_steps'] = np.log(stp_m.clip(lower=1))
    X['log_sleep'] = np.log(slp_m.clip(lower=1))
    X['log_azm'] = np.log(azm_m.clip(lower=1))

    # === Rank features (percentile-based) ===
    for col_name, col_data in [('bmi', b), ('age', a), ('rhr', rhr_m), ('hrv', hrv_m), ('stp', stp_m)]:
        X[f'rank_{col_name}'] = col_data.rank(pct=True)

    return X.fillna(0)

X_eng = engineer_features(X_df[dw_cols], dw_cols).values
eng_cols = engineer_features(X_df[dw_cols], dw_cols).columns.tolist()
print(f"Features: {len(eng_cols)} (wearables + demographics only)")

# Target transform
log_fn = np.log1p
inv_log = np.expm1

# ============================================================
# MODELS
# ============================================================
def make_xgb_params():
    return dict(
        n_estimators=1200, max_depth=3, learning_rate=0.01,
        subsample=0.75, colsample_bytree=0.6,
        reg_alpha=3.0, reg_lambda=15.0, min_child_weight=10,
        gamma=12.0,
        random_state=SEED, n_jobs=-1
    )

def make_lgb_params():
    return dict(
        n_estimators=1200, num_leaves=15, learning_rate=0.01,
        subsample=0.75, colsample_bytree=0.6,
        reg_alpha=3.0, reg_lambda=15.0, min_child_weight=10,
        random_state=SEED, n_jobs=-1, verbose=-1
    )

def make_ridge_params():
    return dict(alpha=10.0)

def make_enet_params():
    return dict(alpha=0.1, l1_ratio=0.3, max_iter=10000)

# ============================================================
# CROSS-VALIDATION
# ============================================================
oof_xgb = np.zeros(n)
oof_lgb = np.zeros(n)
oof_ridge = np.zeros(n)
oof_enet = np.zeros(n)
counts = np.zeros(n)

for fold_idx, (tr_idx, va_idx) in enumerate(splits):
    X_tr, X_va = X_eng[tr_idx], X_eng[va_idx]
    y_tr_raw, y_va_raw = y[tr_idx], y[va_idx]
    w_tr = w[tr_idx]

    if LOG_TARGET:
        y_tr = log_fn(y_tr_raw)
    else:
        y_tr = y_tr_raw

    # Scalers for linear models
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_va_sc = sc.transform(X_va)

    # PowerTransformer for linear models (better Gaussian approximation)
    pt = PowerTransformer(method='yeo-johnson')
    X_tr_pt = pt.fit_transform(X_tr)
    X_va_pt = pt.transform(X_va)

    # XGBoost
    xgb_model = xgb.XGBRegressor(**make_xgb_params())
    xgb_model.fit(X_tr, y_tr, sample_weight=w_tr)
    pred_xgb = xgb_model.predict(X_va)
    if LOG_TARGET:
        pred_xgb = inv_log(pred_xgb)
    oof_xgb[va_idx] += pred_xgb

    # LightGBM
    lgb_model = lgb.LGBMRegressor(**make_lgb_params())
    lgb_model.fit(X_tr, y_tr, sample_weight=w_tr)
    pred_lgb = lgb_model.predict(X_va)
    if LOG_TARGET:
        pred_lgb = inv_log(pred_lgb)
    oof_lgb[va_idx] += pred_lgb

    # Ridge on PowerTransformed features
    ridge_model = Ridge(**make_ridge_params())
    ridge_model.fit(X_tr_pt, y_tr, sample_weight=w_tr)
    pred_ridge = ridge_model.predict(X_va_pt)
    if LOG_TARGET:
        pred_ridge = inv_log(pred_ridge)
    oof_ridge[va_idx] += pred_ridge

    # ElasticNet on PowerTransformed features
    enet_model = ElasticNet(**make_enet_params())
    enet_model.fit(X_tr_pt, y_tr, sample_weight=w_tr)
    pred_enet = enet_model.predict(X_va_pt)
    if LOG_TARGET:
        pred_enet = inv_log(pred_enet)
    oof_enet[va_idx] += pred_enet

    counts[va_idx] += 1

    if fold_idx % 5 == 4:
        fold_r2_xgb = r2_score(y[va_idx], pred_xgb)
        fold_r2_lgb = r2_score(y[va_idx], pred_lgb)
        print(f"  Fold {fold_idx}: XGB={fold_r2_xgb:.4f} LGB={fold_r2_lgb:.4f}")
        sys.stdout.flush()

# Average OOF predictions across repeats
oof_xgb /= np.clip(counts, 1, None)
oof_lgb /= np.clip(counts, 1, None)
oof_ridge /= np.clip(counts, 1, None)
oof_enet /= np.clip(counts, 1, None)

# ============================================================
# RESULTS
# ============================================================
r2_xgb = r2_score(y, oof_xgb)
r2_lgb = r2_score(y, oof_lgb)
r2_ridge = r2_score(y, oof_ridge)
r2_enet = r2_score(y, oof_enet)

print(f"\nSingle model R²: XGB={r2_xgb:.4f} LGB={r2_lgb:.4f} Ridge={r2_ridge:.4f} EN={r2_enet:.4f}")

# Clip predictions
y_lo, y_hi = np.percentile(y, 0.5), np.percentile(y, 99.5)
oof_xgb_c = np.clip(oof_xgb, y_lo, y_hi)
oof_lgb_c = np.clip(oof_lgb, y_lo, y_hi)
oof_ridge_c = np.clip(oof_ridge, y_lo, y_hi)
oof_enet_c = np.clip(oof_enet, y_lo, y_hi)

# 4-model blend optimization
best_r2 = -1
best_w = None
# Grid search over 4 models
for w1 in np.arange(0, 1.01, 0.05):
    for w2 in np.arange(0, 1.01 - w1, 0.05):
        for w3 in np.arange(0, 1.01 - w1 - w2, 0.05):
            w4 = 1 - w1 - w2 - w3
            blend = w1 * oof_xgb_c + w2 * oof_lgb_c + w3 * oof_ridge_c + w4 * oof_enet_c
            r2 = r2_score(y, blend)
            if r2 > best_r2:
                best_r2 = r2
                best_w = (w1, w2, w3, w4)

# Fine-tune around best weights
w1c, w2c, w3c, w4c = best_w
for dw1 in np.arange(-0.05, 0.06, 0.01):
    for dw2 in np.arange(-0.05, 0.06, 0.01):
        for dw3 in np.arange(-0.05, 0.06, 0.01):
            ww1 = max(0, w1c + dw1)
            ww2 = max(0, w2c + dw2)
            ww3 = max(0, w3c + dw3)
            ww4 = max(0, 1 - ww1 - ww2 - ww3)
            if ww1 + ww2 + ww3 > 1:
                continue
            blend = ww1 * oof_xgb_c + ww2 * oof_lgb_c + ww3 * oof_ridge_c + ww4 * oof_enet_c
            r2 = r2_score(y, blend)
            if r2 > best_r2:
                best_r2 = r2
                best_w = (ww1, ww2, ww3, ww4)

elapsed = time.time() - t_start

print(f"\nBest blend: XGB={best_w[0]:.2f} LGB={best_w[1]:.2f} Ridge={best_w[2]:.2f} EN={best_w[3]:.2f}")
print()
print("---")
print(f"val_r2:           {best_r2:.6f}")
print(f"val_r2_xgb:       {r2_xgb:.6f}")
print(f"val_r2_lgb:       {r2_lgb:.6f}")
print(f"val_r2_ridge:     {r2_ridge:.6f}")
print(f"val_r2_enet:      {r2_enet:.6f}")
print(f"blend_weights:    XGB={best_w[0]:.2f} LGB={best_w[1]:.2f} Ridge={best_w[2]:.2f} EN={best_w[3]:.2f}")
print(f"n_features:       {X_eng.shape[1]}")
print(f"n_samples:        {n}")
print(f"total_seconds:    {elapsed:.1f}")
