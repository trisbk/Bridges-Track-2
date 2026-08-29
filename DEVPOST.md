# Written Project Description (Devpost) — Track 2

## How our solution addresses the problem statement

Track 2 asks for an autonomous ML research agent that improves a recommender
model on the KuaiRand-Pure within-user ranking task (label `long_view`,
primary metric = mean of GAUC and nDCG@5, baseline 0.5946).

We built exactly that: an agent that runs the research loop a TikTok ML
engineer runs daily — propose a hypothesis, implement it, train, evaluate,
reflect, decide what to try next — and we let it iterate until the official
convergence criterion (ε = 0.002 over N = 3 iterations) said stop.

**Result: test primary 0.6116 (GAUC 0.6825, nDCG@5 0.5408) — +0.0170 over
the published baseline**, from 29 logged runs covering ~60 configurations.
The two structural discoveries, in the order the agent found them:

1. **Match the objective to the metric.** The baseline trains pointwise
   ("will this video be watched?") but is graded on within-user ranking. The
   agent replaced logloss with a listwise InfoNCE objective (softmax over one
   positive and four within-user negatives): +0.0028, with a capacity-matched
   control proving the gain came from the objective alone.
2. **Causal sequence features.** The kit's five static fields ignore what the
   user *just did*. The agent added seven features computed strictly from
   each impression's past (previous-impression outcome, rolling watch rates,
   per-author and per-tag history, session gap) — never a row's own label,
   never anything later in time. This was the largest single gain (+0.013)
   and explains why bigger architectures had plateaued: the models were
   information-starved, not under-powered.

The final model is deliberately simple: a five-seed committee of
Factorization Machines (k = 16), reproducible from raw data in one command
(`python3 src/final_model.py`, ~5 minutes, CPU-only).

Three properties of the *process* matter as much as the score:

- **Enforced discipline.** Every experiment runs through a harness that
  mechanically requires ≥3 seeds, a pre-committed 0.002 significance bar
  (seed noise σ ≈ 0.0008), and intent-logged-before-training so failed runs
  cannot be hidden. All selection decisions use validation only; the test
  split is measured, never chosen by. Our log documents two occasions where
  configurations with better-looking *test* numbers were refused because
  validation did not justify them — our reported score is an unbiased
  estimate, and we can prove it.
- **Honest negatives.** The log contains more rejected ideas than accepted
  ones — including a target-encoding leakage trap the agent caught and
  diagnosed, and a legality analysis that retired the random-exposure log
  unused (its rows overlap the evaluation window; training on it would be
  temporal leakage).
- **Autonomy with a defined boundary.** A driver loops fresh agent sessions,
  each executing one experiment under standing instructions; crashes are
  restarted automatically (per the organizers' ruling, restarts are not
  human intervention). Humans set direction and constraints; the agent did
  the research. The full trail is in `experiments/RESULTS.md` and
  `experiments/LOG.jsonl`.

## Development tools used

- **Claude Code (Anthropic)** — the autonomous agent itself: it wrote the
  experiment code, ran the research loop, and authored the run logs and
  experimental commits (visible in the git history)
- **Git + GitHub** — version control; the commit trail doubles as a
  timestamped record of the agent's iterations
- **macOS terminal / zsh, `caffeinate`** — unattended runs on a laptop
- **Python 3.12 (CPython)** — all model and evaluation code

## APIs used

- **Anthropic Claude API** (via the Claude Code CLI) — the LLM powering the
  research agent's reasoning and code generation. No other external APIs:
  training and evaluation are fully local and offline.

## Libraries and frameworks used

- **NumPy** — the only numerical dependency; all models (Factorization
  Machine, MLPs, FFM, attention pooling) and training loops are implemented
  from scratch in NumPy, no ML framework
- **Python standard library** — `csv`, `json`, `collections`, `subprocess`,
  `os`, `time` (data loading, logging, the autonomy driver)

## Datasets and assets used

- **KuaiRand-Pure** (official Track 2 dataset; Kuaishou research release) —
  1.14M train / 125K validation / 171K test logged impressions, used
  exclusively via the official date split
- **Official Track 2 starter kit** — `baseline.py`, `data.py`, `evaluate.py`
  (the scoring code is used byte-identical; the split dates are unmodified)
- The dataset's random-exposure file was deliberately **not** used (temporal
  overlap with the evaluation window — see the legality note in the log)
- No other datasets, no pretrained models, no manually labelled data
