# WEAR-ME AutoResearch

Autonomous ML experimentation for HOMA-IR (insulin resistance) prediction from wearable and clinical data. Adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for tabular ML hill-climbing.

## Concept

Instead of manually iterating through experiments, an AI coding agent (Claude Code) autonomously:
1. Reads the current training code and experiment history
2. Forms a hypothesis for improvement
3. Modifies `train.py` (the only mutable file)
4. Runs the experiment (~1-2 min on CPU)
5. Keeps improvements, reverts failures
6. Repeats indefinitely (30-60 experiments/hour)

## Results

### All Features (demographics + wearables + blood biomarkers)

| Run | Experiments | Best R² | Key Discovery |
|-----|------------|---------|---------------|
| Manual (V1-V28) | 28 | 0.5467 | LGB+ElasticNet blend |
| Autonomous Run 1 | 99 | **0.5536** | PowerTransformer on ElasticNet, heavy XGB regularization |

Theoretical maximum: **R² ≈ 0.614** (estimated via k-NN neighbor variance analysis)

### Wearables Only (demographics + wearable features, no blood)

| Run | Experiments | Best R² | Key Discovery |
|-----|------------|---------|---------------|
| Manual (V20 Model B) | 1 | 0.2592 | Baseline dw_cols |
| Autonomous Run | In progress | — | — |

## Dataset

- **1,078 participants** from the WEAR-ME study
- **25 raw features**: 3 demographics (age, BMI, sex) + 15 wearable stats (RHR, HRV, steps, sleep, AZM — each with mean/median/std) + 7 blood biomarkers
- **Target**: True_HOMA_IR (insulin resistance index)
- Data file: `data.csv` (not included — private dataset)

## Structure

```
prepare.py      — READ-ONLY: data loading, CV splits, feature engineering utilities
train.py        — AGENT-MODIFIABLE: models, features, hyperparams, blending (wearables-only version)
train_all.py    — All-features version (best R²=0.5536)
program.md      — Agent instructions and experiment loop
results.tsv     — Experiment log from autonomous runs
report.html     — Analysis report
data.csv        — Dataset (not in repo — private)
```

## Quick Start

```bash
# 1. Place data.csv in repo root (private dataset required)
# 2. Install dependencies
pip install numpy pandas scikit-learn xgboost lightgbm scipy

# 3. Run all-features best model
python train_all.py

# 4. Run wearables-only model
python train.py

# 5. Start autonomous research with Claude Code
claude --dangerously-skip-permissions -p "Read program.md and follow the experiment loop."
```

## Key Findings

### From 99 autonomous experiments:
- **Model diversity in blending** matters more than tuning one model (XGB + ElasticNet >> multiple XGB seeds)
- **PowerTransformer on linear models** was the key breakthrough for the all-features version
- **Heavy XGB regularization** added +0.02 R²
- **Prediction clipping** (0.5-99.75 percentile) added +0.004 R²
- **Log1p target transform** is optimal for right-skewed HOMA-IR (+0.015 R²)
- **sqrt(y) sample weighting** optimal exponent (+0.008 R²)
- Error correlations between tree models are 0.99+ — only linear models add diversity
- CatBoost underperformed (R²=0.5259), SVR/KNN have wrong inductive bias (R²=0.36-0.45)

## Evaluation

- 5×5 Repeated Stratified KFold (25 splits)
- R² computed on out-of-fold predictions only (no leakage)
- Deterministic splits (fixed seed) for reproducibility

## Adapting for Your Own Task

See [program.md](program.md) for the full agent instruction template. The core idea:

1. Write a read-only `prepare.py` (your evaluation framework)
2. Write a baseline `train.py` (agent modifies this)
3. Write `program.md` (agent instructions)
4. Launch Claude Code and let it iterate

## License

MIT
