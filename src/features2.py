"""Run 6: feature stacking + personalization affinities under L1.

Run 5 found each feature group individually worth ~+0.0003 (noise-level) but
all four consistently positive on valid and test. Test whether they stack
(F5), and add a stronger family: user-conditional history rates (F6/F7).

  F5 combo    : base + hour + content + uactive + vpop
  F6 affinity : base + user->author and user->tag smoothed long_view rates,
                decile-bucketed. Computed from TRAIN rows ONLY — causally
                clean: the train period predates valid/test.
  F7 all      : F5 + affinity fields

Leakage guards: affinity tables come from train rows; a (user, author) pair
unseen in train falls back to the user's own train mean, then to global mean;
bucket edges from train distribution. Nothing from valid/test behaviour.
"""
import time, collections
import numpy as np
from features import (load_rows, add_derived, encode_fields, run_l1,
                      BASE_FIELDS)


def add_affinity(splits, prior=5.0):
    tr = splits['train']
    gmean = np.mean([x['y'] for x in tr])
    un = collections.Counter(); up = collections.Counter()
    uan = collections.Counter(); uap = collections.Counter()
    utn = collections.Counter(); utp = collections.Counter()
    for x in tr:
        u = x['user_id']
        un[u] += 1; up[u] += x['y']
        ka = (u, x['author_id']); uan[ka] += 1; uap[ka] += x['y']
        kt = (u, x['tag']);       utn[kt] += 1; utp[kt] += x['y']

    def umean(u):
        return (up[u] + prior * gmean) / (un[u] + prior) if un[u] else gmean

    def rate(kp, kn, key, fallback):
        n = kn[key]
        return (kp[key] + prior * fallback) / (n + prior) if n else fallback

    tr_ua = [rate(uap, uan, (x['user_id'], x['author_id']), umean(x['user_id'])) for x in tr]
    tr_ut = [rate(utp, utn, (x['user_id'], x['tag']), umean(x['user_id'])) for x in tr]
    ea = np.quantile(np.array(tr_ua), np.linspace(0, 1, 11)[1:-1])
    et = np.quantile(np.array(tr_ut), np.linspace(0, 1, 11)[1:-1])

    for rws in splits.values():
        for x in rws:
            um = umean(x['user_id'])
            ra = rate(uap, uan, (x['user_id'], x['author_id']), um)
            rt = rate(utp, utn, (x['user_id'], x['tag']), um)
            x['ua_aff'] = str(int(np.searchsorted(ea, ra)))
            x['ut_aff'] = str(int(np.searchsorted(et, rt)))
    return splits


if __name__ == '__main__':
    print("loading ...")
    splits = add_affinity(add_derived(load_rows()))
    print({k_: len(v) for k_, v in splits.items()})

    EXTRA = ['hour', 'video_type', 'music_type', 'tag', 'uactive', 'vpop']
    AFF = ['ua_aff', 'ut_aff']
    groups = [
        ("F5 combo",        BASE_FIELDS + EXTRA),
        ("F6 base+affinity", BASE_FIELDS + AFF),
        ("F7 all",          BASE_FIELDS + EXTRA + AFF),
    ]
    SEEDS = 3
    BASE, L1 = 0.5950, 0.5978
    print(f"\n{SEEDS} seeds each, objective = InfoNCE K=4."
          f" Baseline {BASE} | L1 base fields {L1} (+0.0028)\n")
    print(f"{'config':<20} {'valid':>10}   {'test primary':>20}   {'d/base':>8} {'d/L1':>8}")
    for name, fields in groups:
        enc, dim = encode_fields(splits, fields)
        vs, ts = [], []
        t0 = time.time()
        for s in range(SEEDS):
            r = run_l1(enc, dim, seed=s)
            vs.append(r['valid']['primary']); ts.append(r['test']['primary'])
        vm, tm, tsd = np.mean(vs), np.mean(ts), np.std(ts)
        print(f"{name:<20} {vm:>10.4f}   {tm:>12.4f} ± {tsd:.4f}   {tm-BASE:+.4f} {tm-L1:+.4f}  ({time.time()-t0:.0f}s)")
