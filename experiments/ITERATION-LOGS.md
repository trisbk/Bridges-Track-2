# Run & Iteration Logs — compiled index

This document indexes every research iteration against the Track 2 run-log
requirements. Three layers of record back it:

- **`LOG.jsonl`** — machine log: intent (hypothesis, rationale, config) is
  appended **before** training; results (per-seed scores, GAUC/nDCG
  components for later runs, verdict, wall time) after. A crashed run
  therefore still documents what it was attempting.
- **`RESULTS.md`** — the narrative: full tables and mechanism analysis per run.
- **Git history** — the code diff applied per iteration. Runs 30–32 were
  committed by the agent itself in this repo (hashes below); the manual-phase
  history lives in the research workspace repo
  (github.com/SteveWilsonK/techjam-2026-workspace), whose commits are cited
  per run. Each run's experiment code is also preserved verbatim as a
  standalone script in `src/`.

Metrics context: primary = mean(GAUC, nDCG@5); published baseline 0.5946;
our reproduced baseline 0.5950; seed noise σ≈0.0008; significance gate
0.002; every entry ≥3 seeds. Final component scores: GAUC 0.6825,
nDCG@5 0.5408 (valid: 0.6926 / 0.5455).

## Manual interventions summary (Task Requirement 2)

**Autonomous demonstration run (Runs 30–32, 30 Aug 00:47–01:39): zero
manual interventions, zero crash-restarts.** The driver
(`agent/driver.py`) looped fresh headless agent sessions to convergence
(ε=0.002, N=3) with no human input after launch. The launch itself was
human-initiated by design. Event log: `agent/driver_log.jsonl`.

**Interactive research phase (Runs 1–29):** the agent (Claude Code, in an
interactive session) made all iteration-level research decisions — what to
try, how to implement, how to interpret, when an idea was dead. Human
(owner) interventions were strategic and are enumerated exhaustively:
choice of track; the decision to defer, then later permit, three idea
families (sequence features, multi-task labels, random-exposure log);
setting the stopping rule (stop at 0.65 or after 5 unbanked runs); and a
reporting-format request (✅/❌ vs banked). No human proposed, implemented,
tuned, or interpreted any experiment. Restart-after-interruption occurred
once (Run 6, lid-close SIGTERM) — per the organizers' ruling, restarts are
not interventions.

## Error & recovery events

| When | Event | Handling |
|---|---|---|
| Run 6 (29 Aug) | External SIGTERM (laptop lid closed) killed the process after config 1 of 3 | Completed config's result was already logged; remaining configs relaunched; nothing lost |
| Run 9, attempt 1 (29 Aug) | Harness crash writing results (numpy float32 not JSON-serializable) **after** training completed | Intent record survived as designed; harness fixed (float casts); run repeated |
| Driver false start (30 Aug 00:40) | All iterations failed in seconds: CLI not authenticated ("Not logged in") | Detected immediately; log archived as `agent/driver_log.failed-start.jsonl` (kept for the record); authenticated; relaunched cleanly |
| Overnight run (30 Aug 00:47–01:39) | — | No errors, no restarts |

## Per-iteration index

**Phase 1 — interactive research (agent-driven, human-strategic).**
Full detail per run in RESULTS.md; code in `src/`; workspace-repo commit
in brackets.

| Run | Hypothesis (short) | Code | test primary | Δ base | Verdict |
|---|---|---|---|---|---|
| 1-B1 | Metric weights users equally → reweight rows 1/impressions | experiments.py [2a1056f] | 0.5888 ± 0.0008 | −0.0062 | ❌ GAUC is positive-count-weighted; helping nDCG hurts GAUC more |
| 1-B2 | Train on continuous play-ratio instead of binary label | experiments.py [2a1056f] | 0.5598 ± 0.0002 | −0.0352 | ❌ duration-mediated → learns "prefer short videos" |
| 2-P1 | Pairwise BPR matches the ranking metric | pairwise.py [900a9b9] | 0.5967 ± 0.0005 | +0.0017 | ⚠️ real direction, under gate |
| 2-P2 | BPR + pointwise blend | pairwise.py [900a9b9] | 0.5949 ± 0.0005 | −0.0001 | ❌ pointwise erases the gain |
| 3 | BPR gain scales with lr / pair count (4 configs) | pairwise_sweep.py [900a9b9] | best 0.5965 | +0.0015 | ❌ capped; lr ≥ 2e-3 destructive |
| 4-L1 | Listwise InfoNCE upweights hard negatives | listwise.py [06e2a95] | **0.5978 ± 0.0002** | **+0.0028** | ✅ **first banked win** |
| 4-L2 | Bigger K (8) helps | listwise.py [06e2a95] | 0.5977 ± 0.0002 | +0.0027 | ❌ K saturated at 4 |
| 4-C0/C1 | Capacity k=32 (with pointwise control) | listwise.py [06e2a95] | 0.5965 / 0.5950 | — | ❌ capacity does nothing; gain attributed to objective |
| 5 | Side features stack with the objective (4 groups) | features.py [4c8e5ce] | ~0.5981 each | +0.0031 | ❌ +0.0003 each, noise-level |
| 6-F5 | The four groups stack combined | features2.py [4c8e5ce] | 0.5981 ± 0.0001 | +0.0031 | ❌ redundant, no stacking |
| 6-F6/F7 | User×author/tag affinity rates | features2.py [4c8e5ce] | 0.5858 ± 0.0010 | −0.0092 | ❌ self-inclusion leakage diagnosed |
| 7 | Leave-one-out fixes the affinity leakage | (inline) [5057b90] | 0.5970 ± 0.0003 | +0.0020 | ❌ leak confirmed as mechanism; honest version redundant with FM |
| 8 | Seed ensembling cancels noise | ensemble seeds [5057b90] | 0.5982 (5 seeds) | +0.0032 | ✅ banked; +0.0004 only — FM seeds converge |
| 9 | FwFM: field-pair weights matter | fwfm.py [e7ead3e] | 0.5978 ± 0.0002 | +0.0028 | ❌ = plain FM; interactions already balanced |
| 10 | Nonlinear MLP head over embeddings (2 lrs) | mlp.py [513a4f9] | **0.5984 ± 0.0002** | +0.0034 | ✅ best single model at the time |
| 11 | Cross-class committee beats same-class | ensemble.py [513a4f9] | **0.5986** | +0.0036 | ✅ banked; same-class adds ~nothing |
| 12 | MLP capacity (k=32, H=128) | (inline) [bfcb046] | ≤0.5983 | — | ❌ capacity flat for MLP too |
| 13a | Pointwise warm-start anneal | refinements.py [bfcb046] | 0.5982 ± 0.0003 | +0.0032 | ❌ pointwise adds nothing in any mixture |
| 13b | Hard-negative mining curriculum | refinements.py [bfcb046] | 0.5849 ± 0.0015 | −0.0101 | ❌ top-scored negatives are near-positives |
| 14 | Depth (MLP 64→32) and FFM field-aware embeddings | architectures.py [fc697fe] | 0.5982 / 0.5976 | — | ❌ same ceiling from both directions |
| 15 | 4-class committees | ensemble2.py [2d8997d] | best-test 0.5988 | +0.0038 | ❌ **test-peek refusal #1**: validation tied; incumbent kept |
| 16 | user×tab cross; InfoNCE temperature | run16.py [4545622] | ≤0.5982 | — | ❌ sparse cross overfits; τ=1 optimal |
| 17 | Validation-weighted / diverse-config / lr-decay | run17.py [86faf64] | 0.5988 / 0.5986 / 0.5981 | — | ✅ α-blend banked +0.0002 (validation-selected); others ❌ |
| 18 | **Causal sequence features** (strictly-prior behavior) | sequences.py [08cae0d] | **0.6016 ± 0.0004** | **+0.0066** | ✅ **breakthrough**; largest single gain |
| 19 | Multi-task aux heads (click/like) | multitask.py [67f6c7e] | 0.6014 ± 0.0005 | +0.0064 | ❌ recency already carries the signal |
| 20 | Committee on seq features; richer sequence set | run20.py [fdfae5b] | **0.6043** / 0.6040 | +0.0093 | ✅ banked; richer set promising, high variance |
| 21 | Committee on richer sequences | run21.py [825f08d] | **0.6104** | +0.0154 | ✅ banked; FM-rich singles revelation (avg 0.6101) |
| 22 | Interest vector (pooled watched-video embeddings) | interest.py [825f08d] | 0.6036 ± 0.0003 | +0.0086 | ✅ best single on base+seq; superseded by rich line |
| 23 | Grand committee incl. interest models | run23.py [86faf64] | 0.6045 | +0.0095 | ❌ superseded by rich line; diversity lesson kept |
| 24 | FM-rich k=32; FM-only committee; deeper history | run24.py [86faf64] | 0.6099 / **0.6116** / 0.6079 | +0.0166 | ✅ **0.6116 banked — final**; capacity dead 3rd time; long windows dilute |
| 25 | Auth-cap isolate; cross-view blend (α on valid) | run25.py [a777d1e] | 0.6105 / 0.6116 | — | ❌ flat; validation put zero weight on the interest view |
| 26 | Attention pooling over history (+ control) | run26.py [86faf64] | 0.6020 both | — | ❌ attention = mean pool exactly (clean null) |
| 27 | Hybrid FM (candidate·history dot term) | run27.py [86faf64] | 0.6097 ± 0.0010 | +0.0147 | ❌ FM interactions already encode it |
| 28 | Recipe retune on new premises (4 configs) | run28.py [86faf64] | best singles 0.6113 | — | ❌ vs committee; lr 5e-4 lifts singles → fed Run 29 |
| 29 | Committees on improved singles | run29.py [d53cdf1] | 0.6123 / 0.6117 | — | ❌ **test-peek refusal #2**: best-test arm has worst validation; incumbent kept. Convergence → **freeze at 0.6116** |

**Phase 2 — autonomous unattended run** (driver-launched, zero
interventions; agent commits in THIS repo; GAUC/nDCG components in
LOG.jsonl per the upgraded harness):

| Run | Hypothesis | Commit | test primary | Verdict |
|---|---|---|---|---|
| 30 | hour/content fields under the FM-rich premise (with same-session control) | aaa5ef3 | best arm 0.6103 | ❌ +0.0003 — same margin as Run 5 across two model classes and feature regimes → the fields, not the architecture, are the limit. **Test-peek refusal #3** recorded |
| 31 | Duration-normalised play-ratio as auxiliary head (3 arms; gradients finite-difference-verified; control bit-identical to banked step) | 49a2192 | 0.6095 | ❌ all arms below control; overturns Run 1's explanation — duration mediation was symptom, not cause |
| 32 | Per-field embedding sizes as interaction rank (2×3 grid) | b2f6007 | best arm 0.6098 (=control) | ❌ hypothesis inverted: narrow-field rank is the binding constraint (−0.005 at rank 8); wide rank buys nothing |

**Driver convergence:** 3 consecutive iterations < ε=0.002 → stopped at
01:39, banked best unchanged at **0.6116**. Full event stream:
`agent/driver_log.jsonl`.

## Tally

32 runs · ~70 configurations · 3 banked structural wins (objective,
sequence features, committee) · 3 documented test-peek refusals ·
2 diagnosed leakage traps · 1 legality retirement (random-exposure log) ·
2 error-recovery events, both with zero data loss · final: **0.6116
(+0.0170 vs published), twice-converged and independently reproduced.**

## Addendum (30 Aug): clean-room autonomous run + spec compliance

**Clean-room run** (records in `cleanroom/`): the same agent relaunched with
ZERO prior knowledge — empty logs, no backlog, bare official baseline. Six
iterations, 1 h 48 m, zero interventions: reproduced the baseline (0.59497),
banked a feature-engineering win at iteration 3 (**0.59744, +0.0028 over the
published baseline**), refused two better-looking test scores on validation
grounds entirely on its own (iterations 4–5), and converged by the official
rule. Its research path diverged from the main campaign's (feature fields
paid under its BCE regime; pairwise loss refuted with a mechanism), which
makes it a genuine independent trajectory, not a replay.

**Official-limits compliance, all campaigns** (per the Primary-metric
clause: validation-ε convergence, 50-iteration cap, 6 h ceiling —
whichever first):

| Campaign | Iterations (≤50) | Wall-clock (≤6 h) | Terminated by |
|---|---|---|---|
| Interactive (Runs 1–29) | 29 | supervised session (~1 day, spec's semi-automated mode) | validation-ε convergence → freeze |
| Demo A (Runs 30–32) | 3 | 52 min | validation-ε convergence |
| Clean-room (6 runs) | 6 | 1 h 48 m | validation-ε convergence |

Scored checkpoint = validation-best at convergence (R24b; the R29
validation tie resolved to the incumbent), evaluated once on the test split.
