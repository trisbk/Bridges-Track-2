# Autonomous ML Research Agent — TikTok TechJam 2026, Track 2

**Final result: test primary 0.6116** (GAUC 0.6825, nDCG@5 0.5408) on the
KuaiRand-Pure within-user ranking task — **+0.0170 over the published
baseline (0.5946)**.

> Full README with architecture, methodology, and agent documentation is
> being finalized. Meanwhile: [FINAL-MODEL.md](FINAL-MODEL.md) has the frozen
> recipe and reproduction command; [experiments/RESULTS.md](experiments/RESULTS.md)
> holds the complete 29-run research narrative.

## Quick start

1. Download KuaiRand-Pure (from the official TechJam kit / Zenodo) and place
   it at `src/KuaiRand-Pure/` (so `src/KuaiRand-Pure/data/*.csv` exists).
2. Reproduce the final model:

```bash
cd src
python3 final_model.py     # ~5 min, CPU-only, numpy-only
```

3. Reproduce the official baseline for comparison:

```bash
python3 baseline.py --model fm
```

## Repo map

| Path | What |
|---|---|
| `src/` | All model, training, and experiment code (numpy only) |
| `src/final_model.py` | One-command reproduction of the frozen final model |
| `src/frozen_model/` | Saved weights, ensemble predictions, config.json |
| `src/harness.py` | Experiment harness: enforced multi-seed, significance gating, intent-first logging |
| `experiments/RESULTS.md` | The research narrative — every hypothesis, result, and verdict |
| `experiments/LOG.jsonl` | Machine-readable log of every run |
| `experiments/IDEAS.md` | The agent's idea backlog: banked / dead / open, with reasons |
| `agent/driver.py` | Autonomy driver — loops unattended agent iterations to convergence |

Research history (all-tracks workspace, full commit trail):
https://github.com/SteveWilsonK/techjam-2026-workspace
