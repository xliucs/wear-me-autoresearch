# WEAR-ME AutoResearch

Autonomous ML research for HOMA-IR prediction from wearable + clinical data.

Adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for tabular ML hill-climbing.

## Concept

Instead of manually iterating through V1-V28 experiments (as in [wear-me-dl-v2](https://github.com/xliucs/wear-me-dl-v2)), an AI coding agent autonomously:
1. Reads the current training code
2. Forms a hypothesis for improvement
3. Modifies `train.py`
4. Runs the experiment (~1-2 min on CPU)
5. Keeps improvements, reverts failures
6. Repeats indefinitely

## Current Best

**R² = 0.5467** (from 28 manual experiments in wear-me-dl-v2)

Theoretical maximum: **R² ≈ 0.614**

## Dataset

- **1,078 participants** from the WEAR-ME study
- **25 raw features**: 3 demographics + 15 wearable stats + 7 blood biomarkers
- **Target**: True_HOMA_IR (insulin resistance index)
- Data file: `data.csv` (not included — private dataset)

## Structure

```
prepare.py      — data loading, CV splits, base feature engineering (DO NOT MODIFY)
train.py        — models, features, blending (AGENT MODIFIES THIS)
program.md      — agent instructions
data.csv        — dataset (not in repo)
results.tsv     — experiment log (untracked)
```

## Quick Start

```bash
# 1. Place data.csv in repo root
# 2. Install dependencies
pip install numpy pandas scikit-learn xgboost lightgbm

# 3. Run baseline
python train.py

# 4. Start autonomous research
# Point Claude Code / Codex at program.md and let it go
claude --dangerously-skip-permissions -p "Read program.md and let's kick off a new experiment!"
```

## Key Findings from Manual Research (V1-V28)

- Log1p target transform: +0.015 R²
- sqrt(y) sample weighting: +0.008 R²
- 72 engineered features optimal for trees
- Blend diversity matters: XGB + ElasticNet >> multiple XGB
- Theoretical ceiling ~0.614 (k-NN neighbor variance)

## License

MIT
