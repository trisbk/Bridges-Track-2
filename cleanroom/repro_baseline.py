"""Iteration 1, step 1: reproduce the official FM baseline through the harness.

Nothing new here — this exists so the clean-room log starts with a
multi-seed, harness-gated measurement of the thing we have to beat, using
exactly the official `baseline.run_fm` code path.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import load
from baseline import run_fm
from harness import run_experiment

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'KuaiRand-Pure', 'data')

if __name__ == '__main__':
    splits = load(DATA)
    run_experiment(
        name='baseline_fm_repro',
        hypothesis='The published FM baseline (test primary 0.5946) reproduces '
                   'on this machine; establish its 3-seed mean/std as the '
                   'reference point for every later claim.',
        rationale='Clean-room iteration 1 has no prior runs. Before proposing '
                  'anything we need the baseline measured under the same '
                  'multi-seed protocol we will judge improvements by, so that '
                  'the 0.002 gate is applied to like-for-like numbers.',
        train_fn=lambda s: run_fm(splits, seed=s, verbose=False),
        seeds=3,
        config={'model': 'FM', 'k': 16, 'lr': 0.001, 'epochs': 40,
                'bs': 8192, 'patience': 4, 'fields': 'official 5',
                'loss': 'pointwise BCE'})
