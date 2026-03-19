# program.md Template

Adapt this for your ML task. Replace bracketed items.

---

```markdown
# [PROJECT NAME] AutoResearch

Autonomous hill-climbing for [TASK DESCRIPTION].

## Setup

1. Propose a run tag (e.g. `mar18`). Branch: `autoresearch/<tag>`.
2. Read: `README.md`, `prepare.py` (DO NOT MODIFY), `train.py` (you modify this).
3. Verify `data.csv` exists.
4. Initialize `results.tsv` with header. Baseline is first run.

## What you CAN do

Modify `train.py` only:
- Feature engineering (add/remove/transform)
- Model types ([LIST ALLOWED MODELS])
- Hyperparameters
- Target transforms
- Sample weighting
- Blending/stacking
- Preprocessing

## What you CANNOT do

- Modify `prepare.py`
- Change evaluation metric ([METRIC] on [CV SCHEME])
- Add data leakage
- Install packages beyond: [LIST AVAILABLE PACKAGES]

## Goal

Maximize [METRIC]. Current best: [SCORE]. Theoretical max: [MAX] (if known).

## Known constraints from prior work

- [FINDING 1]
- [FINDING 2]
- [WHAT FAILED]

## Ideas to try

- [IDEA 1]
- [IDEA 2]

## Output format

The script prints:

```
---
[metric_name]:    [value]
n_features:       [N]
total_seconds:    [T]
```

Extract: `grep "^[metric_name]:" run.log`

## Logging

Log to `results.tsv` (tab-separated):

```
commit	[metric_name]	runtime_s	status	description
```

## The experiment loop

LOOP FOREVER:
1. Look at git state
2. Modify `train.py` with an idea
3. `git commit`
4. Run: `python train.py > run.log 2>&1`
5. Read results: `grep "^[metric_name]:" run.log`
6. If empty → crash. `tail -n 50 run.log`, fix.
7. Record in results.tsv
8. If [METRIC] improved → keep
9. If equal or worse → `git reset --hard HEAD~1`
10. Repeat

NEVER STOP.
```
