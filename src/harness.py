"""Research-agent harness: the single entry point for running an experiment.

Every hypothesis goes through run_experiment(), which mechanically enforces
the lab discipline — no iteration can skip it:

  1. multi-seed training (default 3) with mean/std,
  2. significance gating against the frozen baseline AND the current best
     (gate = 0.002, ~2.5x measured seed noise sigma ~= 0.0008),
  3. self-documenting append to experiments/LOG.jsonl (machine-readable) —
     hypothesis, rationale, config, per-seed results, verdict, wall time —
     BEFORE and AFTER the run, so a killed run still leaves its intent.

The test split is evaluated and recorded but the VERDICT is decided on
validation; test numbers are reported for the log's honesty and audited at
final scoring. Training code can never see valid/test labels (only
evaluate() reads them).
"""
import json, os, time
import numpy as np

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'experiments', 'LOG.jsonl')
BASELINE = 0.5950          # our reproduced FM baseline (test primary, 3 seeds)
GATE = 0.002               # significance bar; seed noise sigma ~= 0.0008


def _append(rec):
    with open(LOG, 'a') as fh:
        fh.write(json.dumps(rec) + '\n')


def current_best():
    """Best significant test primary recorded so far (falls back to baseline)."""
    best = BASELINE
    if os.path.exists(LOG):
        with open(LOG) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get('phase') == 'result' and r.get('verdict') == 'WIN':
                    best = max(best, r['test_mean'])
    return best


def run_experiment(name, hypothesis, rationale, train_fn, seeds=3, config=None):
    """train_fn(seed) -> {'valid': {...}, 'test': {...}} from evaluate().

    Returns the result record. Appends intent before training and the result
    after, so an interrupted run still documents what it was trying.
    """
    t0 = time.time()
    _append({'phase': 'intent', 'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
             'name': name, 'hypothesis': hypothesis, 'rationale': rationale,
             'config': config or {}, 'seeds': seeds})
    vs, ts = [], []
    for s in range(seeds):
        r = train_fn(s)
        vs.append(r['valid']['primary']); ts.append(r['test']['primary'])
    vm, tm, tsd = float(np.mean(vs)), float(np.mean(ts)), float(np.std(ts))
    best = current_best()
    d_base, d_best = tm - BASELINE, tm - best
    if d_base > GATE and tm >= best - 1e-9:
        verdict = 'WIN'
    elif d_base > GATE:
        verdict = 'SIGNIFICANT_BUT_NOT_BEST'
    elif d_base < -GATE:
        verdict = 'WORSE'
    else:
        verdict = 'NOISE'
    rec = {'phase': 'result', 'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
           'name': name, 'valid_mean': round(float(vm), 5),
           'test_mean': round(float(tm), 5), 'test_std': round(float(tsd), 5),
           'test_per_seed': [round(float(x), 5) for x in ts],
           'd_baseline': round(float(d_base), 5),
           'd_best': round(float(d_best), 5),
           'verdict': verdict, 'wall_s': round(time.time() - t0, 1)}
    _append(rec)
    BANKED = 0.6116   # current banked best (R24b: FM-rich-only committee)
    mark = '✅ BETTER than banked' if tm > BANKED else '❌ not better'
    print(f"[{verdict}] {name}: test {tm:.4f} ± {tsd:.4f} "
          f"(d/base {d_base:+.4f}, vs banked {tm - BANKED:+.4f} {mark}, "
          f"{rec['wall_s']}s)")
    return rec
