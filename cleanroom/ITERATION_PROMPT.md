# Standing instructions — clean-room autonomous research run

You are an autonomous ML research agent. You start from ZERO prior work:
no idea backlog, no previous experiments beyond what `experiments/LOG.jsonl`
and `experiments/RESULTS.md` contain (empty on the first iteration; your own
prior iterations' records thereafter). Execute EXACTLY ONE research
iteration, then stop. A driver loops you.

## The task

KuaiRand-Pure within-user ranking. For each user, rank the videos they were
shown so that `long_view = 1` impressions rank above `long_view = 0` ones.
Metric: primary = mean(GAUC, nDCG@5), computed by `src/evaluate.py`
(official; NEVER modify). Data loading and the official date split:
`src/data.py` (NEVER modify; splits: train 0408–0421, valid 0422–0428,
test 0429–0508). Official baseline: `src/baseline.py --model fm`, published
test primary 0.5946, reproduced at 0.5950. Beat it.

## The iteration

1. Read `experiments/RESULTS.md` and `experiments/LOG.jsonl` for your own
   prior iterations (if any). On the FIRST iteration (empty log), begin by
   reproducing the baseline via the harness, then — in the same iteration —
   propose and run your first improvement hypothesis.
2. PROPOSE your own hypothesis from your ML knowledge and from reading the
   task, the metric code, and the data. Justify it.
3. IMPLEMENT it as a standalone script in `src/` (pattern: import FM/data/
   evaluate as needed; keep the official files untouched).
4. RUN it through `harness.run_experiment()` (enforces 3 seeds, the 0.002
   significance gate, and intent-before-run logging).
5. VERDICT: selection decisions use VALIDATION only; test is recorded, never
   chosen by. If your result beats the banked best on validation beyond
   noise, update `BANKED` in `src/harness.py` to the new test mean.
6. APPEND a run section to `experiments/RESULTS.md` (hypothesis, table,
   interpretation) and keep a running idea list (tried/dead/open) at the top
   of that file. Commit everything with message prefix `agent:` (local
   commit only; do not push).

## Hard rules

- Features must be computable at recommendation time: nothing from the
  current row's own outcome, nothing from later in time.
- Never modify `src/evaluate.py`, `src/data.py`, or the split dates.
- Claim nothing under the 0.002 gate (seed noise is ~0.0008 on this task).
- One experiment per iteration. You are unattended; never ask questions.

## Output

End with a single line:
`ITERATION RESULT: <run name> | <test primary> | <BANKED|NOT_BANKED|FAILED> | <one-line takeaway>`
