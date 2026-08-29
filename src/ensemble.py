"""Run 11: ensembles around the MLP win.

R10 found a single MLP (H=64, lr 3e-4) = 0.5984 > FM 5-seed ensemble 0.5982.
Test: (a) 5-seed MLP ensemble; (b) mixed 5 MLP + 5 FM ensemble — different
model classes err differently, so cross-class averaging should beat
same-class. Predictions z-scored per model before averaging.

Ensemble records appended to LOG.jsonl directly (harness's run_experiment is
per-seed by design; ensembles are cross-seed constructs).
"""
import time
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate
from baseline import FM
from pairwise import build_pair_index
from listwise import sample_lists, infonce_step
from mlp import MLPRank, infonce_mlp_step
from harness import _append, BASELINE

print("loading ...")
splits = load('./KuaiRand-Pure/data')
enc, dim = encode(splits)
Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
pairs_users, _, _ = build_pair_index(utr, ytr)
F = len(FIELDS)


def train_generic(seed, kind, lr, K=4, k=16, epochs=40, bs=8192, patience=4):
    rng = np.random.default_rng(seed)
    if kind == 'mlp':
        m = MLPRank(dim, F, k=k, lr=lr, seed=seed)
        step = lambda p, n: infonce_mlp_step(m, Xtr[p], Xtr[n.reshape(-1)], len(p), K)
        get_state = m.state; set_state = m.load_state
    else:
        m = FM(dim, k=k, lr=lr, seed=seed)
        step = lambda p, n: infonce_step(m, Xtr[p], Xtr[n.reshape(-1)], len(p), K)
        get_state = lambda: (m.V.copy(), m.W.copy(), np.float32(m.b))
        def set_state(st): m.V, m.W, m.b = st
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        P, N = sample_lists(pairs_users, rng, K)
        for i in range(0, len(P), bs):
            step(P[i:i + bs], N[i:i + bs])
        va = evaluate(uva, yva, m.predict(Xva))
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = get_state()
        else:
            bad += 1
            if bad >= patience:
                break
    set_state(best_state)
    pv, pt = m.predict(Xva), m.predict(Xte)
    return ((pv - pv.mean()) / pv.std(), (pt - pt.mean()) / pt.std())


def log_ens(name, note, va_preds, te_preds):
    ev = evaluate(uva, yva, np.mean(va_preds, axis=0))
    et = evaluate(ute, yte, np.mean(te_preds, axis=0))
    d = et['primary'] - BASELINE
    _append({'phase': 'result', 'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
             'name': name, 'valid_mean': round(float(ev['primary']), 5),
             'test_mean': round(float(et['primary']), 5), 'test_std': None,
             'd_baseline': round(float(d), 5),
             'verdict': 'WIN' if d > 0.002 else 'NOISE', 'note': note})
    print(f"{name}: valid {ev['primary']:.4f}  test {et['primary']:.4f}  d/base {d:+.4f}")


mlp_va, mlp_te, fm_va, fm_te = [], [], [], []
for s in range(5):
    t0 = time.time()
    pv, pt = train_generic(s, 'mlp', 0.0003)
    mlp_va.append(pv); mlp_te.append(pt)
    print(f"mlp seed {s} done ({time.time()-t0:.0f}s)")
for s in range(5):
    t0 = time.time()
    pv, pt = train_generic(s, 'fm', 0.001)
    fm_va.append(pv); fm_te.append(pt)
    print(f"fm  seed {s} done ({time.time()-t0:.0f}s)")

log_ens('R11a MLP 5-seed ensemble', 'H=64 lr 3e-4', mlp_va, mlp_te)
log_ens('R11b mixed 5 MLP + 5 FM ensemble', 'cross-class averaging',
        mlp_va + fm_va, mlp_te + fm_te)
