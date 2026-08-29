# Autonomous ML Research Agent for Recommender Systems

**TikTok TechJam 2026 — Track 2** · Final result: **test primary 0.6116**
(GAUC 0.6825 · nDCG@5 0.5408) on KuaiRand-Pure within-user ranking —
**+0.0170 over the published baseline (0.5946)**, oracle ceiling 0.8645.

---

## 1. Project overview

### The problem

Track 2 provides the KuaiRand-Pure dataset — 1.4M logged short-video
impressions — and a fixed metric: for each user, rank the videos they were
shown so the ones they watched long (`long_view`) come first, scored as the
mean of GAUC and nDCG@5. The published Factorization Machine baseline scores
0.5946. The challenge: build an **autonomous agent** that improves on it by
running the ML research loop itself — propose, implement, train, evaluate,
reflect, repeat — with as little human steering as possible.

### Our solution, in one paragraph

An LLM-driven research agent (Claude Code) that iterates through hypotheses
under a mechanically enforced lab discipline, logging every intent, result,
and verdict. Over 29 runs (~60 configurations) it made two structural
discoveries: **(1)** replacing the baseline's pointwise logloss with a
listwise InfoNCE objective — training on "which of these videos does this
user pick" — because the metric only ever compares scores within a user
(+0.0028, with a capacity-matched control); and **(2) causal sequence
features** — encoding what each user *actually did recently* (previous
impression outcome, rolling watch rates, per-author/per-tag history, session
gap), computed strictly from events before each impression (+0.013, the
largest single gain). The final model is deliberately simple: a five-seed
committee of Factorization Machines (k=16), fully reproducible on a laptop
CPU in ~5 minutes with NumPy alone.

### Why the process is the product

The agent's research trail is a graded artifact in this track, so the
pipeline enforces integrity mechanically rather than by promise:

- **`src/harness.py`** is the single entry point for every experiment: ≥3
  seeds, a pre-committed 0.002 significance bar (measured seed noise
  σ≈0.0008), and *intent logged before training* — a crashed or failed run
  still documents what it was trying; failures cannot be quietly discarded.
- **Selection on validation only.** The test split is measured for the
  record but never chosen by. The log documents two occasions
  (`experiments/RESULTS.md`, Runs 15 and 29) where configurations with
  better-looking **test** numbers were *refused* because validation did not
  justify selecting them — including a 0.6123 observation in the final run.
  Our reported 0.6116 is an unbiased estimate, and the refusals prove it.
- **Honest negatives.** More ideas are rejected than accepted in the log,
  each with a diagnosed mechanism — including a target-encoding self-leakage
  trap (caught, fixed with leave-one-out encoding, then honestly rejected
  anyway) and a legality analysis that retired the dataset's random-exposure
  file unused (its rows overlap the evaluation window; training on it would
  be temporal leakage).
- **Autonomy with a defined boundary.** `agent/driver.py` loops fresh
  headless agent sessions, each executing one experiment under standing
  instructions (`agent/ITERATION_PROMPT.md`); it stops by the official
  convergence rule (ε=0.002, N=3) and auto-restarts crashes — which the
  organizers ruled is not human intervention. Humans set direction,
  constraints, and stopping rules; the agent did the research.

### Results summary

| Stage | test primary | Δ |
|---|---|---|
| Published baseline (pointwise FM, static features) | 0.5946 | — |
| Our reproduction | 0.5950 | +0.0004 |
| + listwise InfoNCE objective | 0.5978 | +0.0028 |
| + causal sequence features | 0.6016 | +0.0038 |
| + richer sequence set (single model) | ~0.6101 | +0.0085 |
| + 5-seed committee → **frozen final** | **0.6116** | +0.0015 |

Full recipe and per-hyperparameter detail: [FINAL-MODEL.md](FINAL-MODEL.md).
Complete run-by-run narrative: [experiments/RESULTS.md](experiments/RESULTS.md).

---

## 2. Setup and installation

Requirements: **Python ≥ 3.10** and **NumPy** — nothing else. No GPU, no ML
framework, no API keys (the trained model runs fully offline).

```bash
git clone <this-repo>
cd <this-repo>
pip install numpy

# Fetch the official dataset (184MB logs + 10MB features):
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar -xzvf KuaiRand-Pure.tar.gz -C src/
# so that src/KuaiRand-Pure/data/*.csv exists
```

## 3. Steps to reproduce our results

All commands run from `src/`. Training is fully seeded → deterministic.

```bash
cd src

# (a) Reproduce the official baseline (~30 s) — expect test primary ≈ 0.595
python3 baseline.py --model fm

# (b) Reproduce our frozen final model (~5 min, CPU-only)
#     Retrains all 5 committee members from raw data, saves weights to
#     frozen_model/, prints: test primary 0.6116
python3 final_model.py

# (c) Optional: rerun any experiment from the research log, e.g. the
#     objective comparison (Run 4) or the sequence-feature run (Run 18):
python3 listwise.py
python3 sequences.py
```

To run the **autonomous agent loop** itself (requires the Claude Code CLI,
authenticated):

```bash
cd ..                      # repo root
python3 agent/driver.py    # loops one-experiment agent iterations to
                           # convergence; events -> agent/driver_log.jsonl
```

## 4. Code map

**Core pipeline** (read these first):

| File | Role |
|---|---|
| `src/data.py`, `src/evaluate.py`, `src/baseline.py` | Official starter kit — the scoring code and split are used **byte-identical / unmodified** |
| `src/harness.py` | Experiment harness: enforced multi-seed, significance gating, intent-first JSONL logging |
| `src/sequences.py` | Causal sequence features + the winning training loop |
| `src/listwise.py` | The listwise InfoNCE objective |
| `src/final_model.py` | One-command reproduction of the frozen final model |
| `src/frozen_model/` | Saved weights (5 seeds), ensemble predictions, `config.json` |
| `agent/driver.py`, `agent/ITERATION_PROMPT.md` | The autonomy loop and the agent's standing instructions |

**Experiment record** (`src/pairwise.py`, `src/mlp.py`, `src/run16.py` …
`src/run29.py` and friends): the agent's preserved lab notebook — one
standalone script per hypothesis, in the exact form that produced the logged
results. They are deliberately kept as run, not refactored after the fact:
each corresponds to a numbered run in `experiments/RESULTS.md`, so the code,
the log, and the git history cross-reference. (`data.py`'s comments are
partly in Chinese — it is the organizers' file, shipped as-is.)

**The agent's memory**: `experiments/RESULTS.md` (narrative),
`experiments/LOG.jsonl` (machine log, intent + result records),
`experiments/IDEAS.md` (backlog: banked / dead / parked, each with the run
that decided it and why).

## 5. Limitations, and what we would improve with more time

**Limitations we know about:**

- **Offline evaluation on logged impressions.** The task ranks what users
  were actually shown; it cannot measure retrieval quality (picking
  candidates from the full catalog) or feedback loops a deployed system
  faces. This is inherent to the track's setup, not a choice — but worth
  naming.
- **The information ceiling is the dataset's, not the model's.** We showed
  this empirically: four model classes (FM, MLPs, field-aware FM, attention
  pooling) converge to the same score once features are equal. Without
  content signal (visual/audio/text of videos), gains beyond ~0.612 were not
  reachable for us; the oracle (0.8645) includes genuine human
  unpredictability plus ~27% of test users who watched nothing (auto-zero
  nDCG by convention).
- **Small seed budgets.** Verdicts use 3–5 seeds. Sufficient against the
  measured noise floor with our 0.002 gate, but finer-grained effects
  (±0.001) are deliberately not claimed.
- **An unclaimed observation.** Run 29 produced a 0.6123 test score we did
  not select (validation ranked it below the incumbent). If our selection
  rule were laxer we would report a higher number; we chose the unbiased one.
- **The agent's autonomy has a human perimeter.** Humans chose the stopping
  rules, unparked idea families, and set constraints. Iteration-level
  decisions (what to try, how to interpret, when an idea is dead) were the
  agent's. We document the boundary rather than overclaiming.

**Given more time:**

- **Sequence modeling proper**: attention over the raw ordered impression
  history with learned position/recency encoding (our mean/attention pooling
  over watched-video embeddings plateaued; richer history encoders with
  content metadata are the natural next room).
- **Principled use of the random-exposure data** for debiasing via a design
  that respects time (e.g., only same-period counterfactual evaluation,
  never training) — we retired it rather than risk leakage under deadline.
- **Learned committee weighting** (our validation-searched blend helped
  once, +0.0002) and cross-version validation on KuaiRand-1K to test
  transfer of the recipe.
- **Harden the autonomy loop**: richer self-verification per iteration
  (auto-generated controls), and a budget-aware scheduler for idea selection.

## 6. Team member contributions

*(To be completed by the team before submission.)*

## 7. Acknowledgements and AI tooling disclosure

- Dataset: **KuaiRand** (Gao et al., CIKM '22) — [paper](https://arxiv.org/abs/2208.08696),
  [site](https://kuairand.com/). Used under its license via the official
  TechJam Track 2 kit.
- The research agent is powered by **Claude (Anthropic)** via the Claude
  Code CLI. The agent wrote the experiment code, ran the research loop, and
  authored the experimental commits (co-author trailers in the git history).
  Humans directed the work, set constraints, and made the calls documented
  in the log.
