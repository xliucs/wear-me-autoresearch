# WEAR-ME AutoResearch

Autonomous hill-climbing for HOMA-IR prediction from wearables + blood biomarkers.
Adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for tabular ML.

## Setup

To set up a new experiment run:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar18`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — repository context
   - `prepare.py` — fixed: data loading, feature sets, CV splits, feature engineering utilities. **Do not modify.**
   - `train.py` — the file you modify. Models, features, hyperparameters, blending.
4. **Verify data exists**: Check that `data.csv` exists in the repo root. If not, tell the human.
5. **Initialize results.tsv**: Create with header row. Baseline is first run.
6. **Confirm and go**.

## Experimentation

Each experiment runs on CPU (no GPU needed). Typical runtime is 30-120 seconds.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game:
  - Feature engineering (add/remove/transform features)
  - Model types (XGBoost, LightGBM, CatBoost, ElasticNet, Ridge, KNN, GBR, neural nets, etc.)
  - Hyperparameters (learning rate, depth, regularization, etc.)
  - Target transforms (log, Box-Cox, power, etc.)
  - Sample weighting strategies
  - Blending/stacking strategies
  - Preprocessing (QuantileTransformer, StandardScaler, etc.)
  - New model combinations

**What you CANNOT do:**
- Modify `prepare.py`. It's read-only. It contains data loading, CV splits, and base feature engineering.
- Change the evaluation metric (R² on 5-fold CV out-of-fold predictions).
- Add data leakage (no using validation data for training, no target encoding without nesting).
- Install new packages beyond what's available (numpy, pandas, sklearn, xgboost, lightgbm, scipy).

**The goal: get the highest val_r2 (R²).** Current best is 0.5467.

**Known constraints from 28 prior experiments (V1-V28):**
- Theoretical max R² ≈ 0.614 (estimated from k-NN neighbor variance analysis)
- Log1p target transform is optimal (+0.015 R²)
- sqrt(y) sample weighting is optimal exponent (+0.008 R²)
- V7-style engineered features (72 features) are optimal for trees
- XGBoost depth=3-4 with low LR is optimal tree config
- Model diversity matters: XGB + ElasticNet blend >> multiple XGB seeds
- Error correlations between tree models are 0.99+ — only linear models add diversity
- L2/nested stacking ≈ simple blending (no stacking gain)
- SVR/KNN have wrong inductive bias (R² = 0.36-0.45)
- Residual-feature correlations ≈ 0 (signal is largely extracted)
- CatBoost disappoints (0.5259)
- Neural nets crash on some Python versions

**Ideas the agent could try:**
- Novel feature interactions not yet explored
- Different target transforms (quantile, rank, etc.)
- Huber loss or custom objectives
- Feature selection via importance thresholds
- Non-linear blending
- Model-specific feature subsets
- Bayesian optimization of blend weights
- Novel preprocessing pipelines
- Multi-output prediction (predict components separately)
- Ensemble of different random seeds with architectural variation
- Gradient boosting with different loss functions

## Output format

The script prints a summary:

```
---
val_r2:           0.546700
val_r2_xgb:       0.540800
val_r2_lgb:       0.539800
val_r2_enet:      0.510000
blend_weights:    XGB=0.00 LGB=0.71 EN=0.29
n_features:       72
n_samples:        1078
total_seconds:    45.2
n_splits:         5
```

Extract the key metric: `grep "^val_r2:" run.log`

## Logging results

Log to `results.tsv` (tab-separated):

```
commit	val_r2	runtime_s	status	description
a1b2c3d	0.546700	45.2	keep	baseline (V20 best)
b2c3d4e	0.548100	52.1	keep	added cubic glucose interaction
c3d4e5f	0.545200	48.3	discard	removed sample weighting
```

## The experiment loop

LOOP FOREVER:

1. Look at git state
2. Modify `train.py` with an experimental idea
3. `git commit`
4. Run: `python train.py > run.log 2>&1`
5. Read results: `grep "^val_r2:" run.log`
6. If grep is empty → crash. `tail -n 50 run.log` for stack trace, attempt fix.
7. Record in results.tsv
8. If val_r2 improved (higher), keep the commit
9. If val_r2 is equal or worse, `git reset` back
10. Repeat

**NEVER STOP.** The human may be asleep. Keep running experiments indefinitely. If stuck, think harder — try radical changes, re-read the code, try combining near-misses. Each experiment takes ~1-2 minutes, so you can run 30-60 per hour.
