---
name: autoresearch
description: Autonomous ML/DS experimentation loop adapted from Karpathy's autoresearch. Use when asked to optimize a predictive model, beat a baseline score, run autonomous experiments, or iterate on ML pipelines. Supports any tabular/time-series prediction task with hill-climbing over feature engineering, models, hyperparameters, and blending. Works with Claude Code CLI as the coding agent.
---

# AutoResearch — Autonomous ML Experimentation

Autonomous hill-climbing loop for ML tasks. An AI coding agent iterates on a training script, running experiments, keeping improvements, reverting failures — indefinitely until stopped.

## Architecture

```
project/
├── prepare.py      — READ-ONLY: data loading, CV splits, evaluation (human writes this)
├── train.py        — AGENT-MODIFIABLE: models, features, hyperparams, blending
├── program.md      — Agent instructions (what to try, constraints, output format)
├── results.tsv     — Experiment log (auto-generated)
├── data.csv        — Dataset (never committed)
└── .gitignore      — Must exclude data files
```

## Setup Workflow

### 1. Prepare the dataset and evaluation framework

Create `prepare.py` with:
- Data loading function
- Feature set definitions (which columns are available)
- CV split strategy (e.g., 5x5 repeated stratified KFold for tabular)
- Metric computation (R², RMSE, AUC — whatever fits the task)
- Any base feature engineering utilities the agent can call

Key principle: `prepare.py` is the trusted evaluation layer. The agent cannot modify it, preventing metric gaming or data leakage.

### 2. Create the baseline train.py

Write an initial `train.py` that:
- Imports from `prepare.py`
- Implements a reasonable baseline (e.g., XGBoost with default params)
- Prints results in a parseable format:

```
---
val_r2:           0.5467
val_r2_xgb:       0.5408
blend_weights:    XGB=0.50 LGB=0.50
n_features:       72
total_seconds:    45.2
n_splits:         25
```

### 3. Write program.md

This is the agent's instruction manual. Include:

- **What files to read** (prepare.py, train.py, README.md)
- **What the agent CAN modify** (only train.py)
- **What it CANNOT do** (modify prepare.py, add leakage, install new packages)
- **The target metric** and current best score
- **Known constraints** from prior experiments (what worked, what failed)
- **Ideas to try** (seed the agent's exploration)
- **Output format** for results parsing
- **The experiment loop** (modify → commit → run → evaluate → keep/revert)

See references/program-template.md for a complete template.

### 4. Initialize the repo

```bash
git init
echo "data.csv\n__pycache__/\n*.pyc\nrun.log\nresults.tsv" > .gitignore
git add prepare.py train.py program.md .gitignore README.md
git commit -m "initial setup"
```

If pushing to GitHub, make repo **private** if dataset is sensitive.

### 5. Launch the agent

```bash
cd /path/to/project
claude --dangerously-skip-permissions -p \
  "Read program.md and follow the experiment loop. Keep running experiments indefinitely."
```

Or spawn via OpenClaw:

```
sessions_spawn(
  task="cd /path/to/project && claude --dangerously-skip-permissions -p 'Read program.md and follow the instructions. Run experiments indefinitely.'",
  mode="run",
  label="autoresearch-run"
)
```

Monitor progress: `tail -f results.tsv` or check git log.

### 6. Analyze results

After the run, `results.tsv` contains all experiments. The git history has every attempted change (kept commits = improvements).

## Key Design Decisions

### Evaluation integrity
- `prepare.py` is read-only — prevents the agent from gaming metrics
- CV splits are deterministic (fixed seed) — ensures comparability
- Out-of-fold predictions only — no train-set leakage

### Hill-climbing strategy
- Agent commits before each run
- If score improves → keep commit
- If score stays same or drops → `git reset --hard` to revert
- Git history becomes a clean chain of improvements

### What the agent typically explores
- Feature engineering (interactions, transforms, ratios)
- Model types (XGBoost, LightGBM, ElasticNet, Ridge, CatBoost, etc.)
- Hyperparameter tuning (learning rate, depth, regularization)
- Target transforms (log1p, Box-Cox, quantile)
- Sample weighting strategies
- Blending/stacking (weighted average of diverse models)
- Preprocessing (StandardScaler, QuantileTransformer, PowerTransformer)
- Feature selection (importance thresholds, Lasso-based)

### Typical throughput
- ~1-2 min per experiment on CPU (tabular data)
- ~30-60 experiments per hour
- 4-hour run = ~100 experiments

## Adapting to New Tasks

To use autoresearch for a different ML/DS problem:

1. **Replace `prepare.py`** with your data loading, splits, and evaluation
2. **Replace `train.py`** with a baseline for your task
3. **Update `program.md`** with:
   - Your metric (R², AUC, RMSE, F1, etc.)
   - Direction (higher-is-better or lower-is-better)
   - Your baseline score and theoretical ceiling (if known)
   - Domain-specific constraints and ideas
4. **Flip keep/discard logic** if your metric is lower-is-better

Works for: regression, classification, ranking, time-series forecasting — any task with a single scalar metric and a training script.

## Lessons Learned (HOMA-IR case study)

From 99 autonomous experiments on WEAR-ME HOMA-IR prediction:

- **Model diversity in blending matters more than tuning one model** — XGB + ElasticNet >> multiple XGB seeds (error correlation 0.99+ between tree models)
- **PowerTransformer on linear models** was the breakthrough (ElasticNet on power-transformed features)
- **Heavy regularization on trees** helped (+0.02 R²)
- **Prediction clipping** to 0.5-99.75 percentile added +0.004 R²
- **Log1p target transform** is almost always worth trying for right-skewed targets
- **The agent plateaus** around experiment 60-70, then occasionally finds breakthroughs via radical changes
- **Seeding the agent with known failures** saves wasted experiments
