# Final Model — frozen 29 Aug 2026

## Scores (official evaluator, untouched)

| Split | GAUC | nDCG@5 | **primary** |
|---|---|---|---|
| validation | 0.6926 | 0.5455 | 0.6191 |
| **test** | **0.6825** | **0.5408** | **0.6116** |

Published FM baseline: 0.5946 → **delta +0.0170**. Oracle ceiling 0.8645.

## Reproduce (one command)

```bash
cd kuairand-kit/kuairand-starter-kit
python3 final_model.py
```

~5 minutes, CPU-only, numpy-only. Retrains all five models from raw data,
saves weights + ensemble predictions to `frozen_model/`, prints the scores
above. Fully seeded → deterministic.

## The recipe

| Component | Setting |
|---|---|
| Model | Factorization Machine, embedding dim k=16 |
| Committee | seeds 0–4; per-model z-scored predictions, averaged |
| Objective | listwise InfoNCE: softmax over 1 positive + K=4 within-user sampled negatives |
| Optimizer | Adam, lr 0.001, L2 1e-6, batch 8192 |
| Early stopping | patience 4 on validation primary, max 40 epochs |
| Fields (12) | user_id, video_id, author_id, tab, dur_bucket (kit base) + prev1, hist10, hist_n, auth_hist, hist30, tag_hist, gap (causal sequence features) |

### The causal sequence features (the biggest single win, +0.013)

All computed from events **strictly earlier in time** than the impression
being scored (rows sorted by user, date, time_ms); a row's own label is never
part of its own features:

- `prev1` — outcome of the user's previous impression (0/1/none)
- `hist10` — long_view count over the user's last ≤10 impressions
- `hist_n` — log-bucketed depth of the user's history
- `auth_hist` — user's prior long_views with this author (0/1/2/3+)
- `hist30` — long_view rate over last ≤30, decile-bucketed
- `tag_hist` — user's prior long_views on this content tag (0/1/2/3+)
- `gap` — time since the user's previous impression (<1m/<1h/<1d/1d+)

## Gain attribution (each step controlled; details in experiments/RESULTS.md)

| Change | test primary | delta |
|---|---|---|
| Published baseline (pointwise FM) | 0.5946 | — |
| Our baseline reproduction | 0.5950 | +0.0004 |
| + listwise InfoNCE objective | 0.5978 | +0.0028 |
| + causal sequence features | 0.6016 | +0.0038 |
| + richer sequence set | ~0.6101 (singles) | +0.0085 |
| + 5-seed committee | **0.6116** | **+0.0015** |

## Integrity notes

- All selection decisions made on validation; test measured, never chosen by.
  Two documented refusals of better-looking test numbers (Runs 15, 29).
- Official `evaluate.py` / `data.py` / split dates byte-identical to the kit.
- Random-exposure log retired unused (rows overlap the eval window —
  temporal leakage if trained on).
- Every claim: ≥3 seeds, pre-committed 0.002 significance bar (noise σ≈0.0008).
- 29 runs, ~60 configurations; full narrative in `experiments/RESULTS.md`,
  machine log in `experiments/LOG.jsonl`.
