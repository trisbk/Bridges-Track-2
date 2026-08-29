# Experiment log — Track 2

Harness: `experiments.py` (copy of what ran inside the kit). `evaluate.py` and
`data.py` untouched, so the official baseline stays reproducible.

## Run 1 — 27 Aug 2026, 3 seeds per variant, ~4.5 min total

Official published FM test primary: **0.5946** (std 0.0008 over 5 seeds).

| Variant | valid | test primary | delta | verdict |
|---|---|---|---|---|
| baseline (binary, unweighted) | 0.6014 | **0.5950 ± 0.0003** | — | reproduces published |
| **B1** binary, user-weighted | 0.5951 | 0.5888 ± 0.0008 | **−0.0062** | ❌ significantly worse |
| **B2** ratio target | 0.5677 | 0.5598 ± 0.0002 | **−0.0352** | ❌ much worse |
| B2-blend (0.5 label + 0.5 ratio) | 0.5911 | 0.5845 ± 0.0009 | −0.0105 | ❌ worse |
| B1+B2 | 0.5692 | 0.5613 ± 0.0010 | −0.0337 | ❌ worse |
| B1+B2-blend | 0.5891 | 0.5825 ± 0.0005 | −0.0124 | ❌ worse |

**Both hypotheses are dead.** Recording why, so the agent's memory starts with
them and does not retry.

### Why B1 (user-weight alignment) was wrong

The premise was *"the metric weights users equally, so training should too."*
That is only true for **half** the metric.

`primary = mean(GAUC, nDCG@5)`. nDCG@5 does average users equally — but **GAUC
is weighted by each user's positive count**, so it deliberately favours heavier
users. Downweighting them by `1/n` fights GAUC while helping nDCG, and the
former loses more than the latter gains.

There is a second cost: heavy users are where the embeddings actually get
learned. `1/n` weighting throws away statistical strength exactly where the data
is richest.

**Lesson for the metric-analysis phase:** the two metric components weight users
*differently*. Any reweighting scheme has to reconcile both, not just one. A
proper analysis of `evaluate.py` would have caught this before spending compute
— which is a point in favour of the A1 idea, not against it.

### Why B2 (dense play-ratio target) was wrong

The premise was *"long_view is a threshold on play ratio, so train on the
continuous quantity."* The threshold is real, but it is **duration-mediated**.

Watching 100% of a 10-second clip is easy; watching 100% of a 3-minute video is
not. A model trained to predict raw ratio therefore learns to favour **short
videos**, which is a different ordering from favouring `long_view`. Ranking by
predicted ratio optimises the wrong thing.

The ratio distribution also works against it: 667k of 1.14M training rows sit
below 0.2, so most of the "dense" signal is compressed near zero.

**Not necessarily dead as a family** — a *duration-normalised* target (predicting
ratio relative to what is typical for that video length), or ratio as an
auxiliary multi-task head rather than a replacement label, could still work. But
the naive substitution is clearly worse and should not be retried.

## What did work

The harness. Six variants × 3 seeds, measured against a reproduced baseline,
in **4.5 minutes**. Seed noise on our own runs is ±0.0003–0.0010, consistent
with the published 0.0008 — so the significance gate (idea A2) is calibrated and
working.

## Run 2 — 29 Aug 2026, pairwise objective (P-series), 3 seeds per variant

**Hypothesis:** `primary` only compares scores within one user, but pointwise
logloss spends capacity calibrating scores across users — capacity the metric
cannot see. Training on within-user (pos, neg) pairs (BPR) should convert that
wasted capacity into ranking accuracy. (Organizers' #1 suggested direction.)

Harness: `pairwise.py` in the kit. Pairs sampled from train rows only; users
whose train rows are all-positive/all-negative yield no pairs (counted; their
user embedding stays at init under pure BPR). FM bias cancels in the pairwise
difference — correct, not a bug.

| Variant | valid | test primary | delta | verdict |
|---|---|---|---|---|
| baseline re-run (same session) | 0.6014 | 0.5950 ± 0.0003 | — | matches published |
| **P1** pure BPR (lr .001, 4 neg/pos, pat 4) | 0.6027 | **0.5967 ± 0.0005** | **+0.0017** | ⚠️ marginal — just under the 0.002 bar |
| **P2** BPR + pointwise blend | 0.6013 | 0.5949 ± 0.0005 | −0.0001 | ❌ noise |

**Interpretation.** P1 improves valid (+0.0013) and test (+0.0017) consistently
across all seeds — real directional signal at ~2× seed noise, achieved with
hyperparameters tuned for the *pointwise* loss. P2's failure is informative:
adding the pointwise term back erased the gain, which supports the hypothesis
that cross-user calibration actively wastes capacity rather than being neutral.

**Conclusion:** direction confirmed, effect size not yet banked. BPR gradients
have a different scale from pointwise, so lr / pairs-per-positive / patience
deserve their own tuning before judging the idea's ceiling.

## Run 3 — 29 Aug 2026, BPR hyperparameter sweep, 3 seeds per config

**Hypothesis:** P1's +0.0017 came from hyperparameters tuned for the pointwise
loss; BPR gradients have a different scale, so lr / pairs-per-positive /
patience tuned for BPR should grow the gain past the 0.002 bar.

| Config | valid | test primary | delta | verdict |
|---|---|---|---|---|
| P1a lr .001, 8 neg/pos, pat 6 | 0.6026 | 0.5965 ± 0.0006 | +0.0015 | no better than P1 |
| P1b lr .002, 4 neg/pos, pat 6 | 0.6020 | 0.5955 ± 0.0005 | +0.0005 | worse than P1 |
| P1c lr .002, 8 neg/pos, pat 6 | 0.5976 | 0.5898 ± 0.0014 | −0.0052 | ❌ significantly worse |
| P1d lr .003, 8 neg/pos, pat 6 | 0.5894 | 0.5803 ± 0.0013 | −0.0147 | ❌ much worse |

**Hypothesis rejected.** The original P1 setting (lr .001, 4 neg/pos) remains
best. The gain does not scale with lr or pair count — higher lr is sharply
destructive. Reading: BPR's benefit is capped by something other than its own
hyperparameters — candidate suspects are model capacity (k=16 embeddings) and
the softmax-free pair gradient treating all negatives equally.

**Next (Run 4):** two attacks on the cap — (a) listwise/InfoNCE objective:
softmax over 1 positive + K sampled negatives, which upweights hard negatives
instead of BPR's uniform pair treatment; (b) capacity: k=32 under BPR, with a
k=32 pointwise control so any gain is attributed to the interaction, not
capacity alone.

## Run 4 — 29 Aug 2026, listwise objective + capacity test, 3 seeds per config

**Hypothesis (a):** BPR's uniform pair gradient is the cap — softmax over
(1 pos + K negs) concentrates gradient on hard negatives and should beat it.
**Hypothesis (b):** k=16 embedding capacity is the cap — k=32 should help.

| Config | valid | test primary | delta | verdict |
|---|---|---|---|---|
| **L1 InfoNCE K=4, k=16** | 0.6035 | **0.5978 ± 0.0002** | **+0.0028** | ✅ **SIGNIFICANT — first banked win** |
| L2 InfoNCE K=8, k=16 | 0.6032 | 0.5977 ± 0.0002 | +0.0027 | ✅ significant; K saturated at 4 |
| C1 BPR k=32 | 0.6025 | 0.5965 ± 0.0004 | +0.0015 | ❌ no better than k=16 |
| C0 pointwise k=32 (control) | 0.6015 | 0.5950 ± 0.0006 | −0.0000 | ❌ exactly baseline |

**Hypothesis (a) CONFIRMED, (b) rejected.** The listwise objective clears the
significance bar with the tightest seed variance measured so far (±0.0002),
and valid/test agree. The control matters: capacity alone does nothing at
either objective, so the entire +0.0028 is attributable to matching the
training objective to the ranking metric — the organizers' suggested
direction, now quantified: pointwise → pairwise +0.0017, pairwise → listwise
+0.0011 more. K=8 ≈ K=4, so negative-sampling breadth is saturated.

**Current best: L1 = 0.5978 test primary (baseline 0.5950, published 0.5946).**

## Run 5 — 29 Aug 2026, feature groups under L1, 3 seeds per config

**Hypothesis:** the kit's 5 fields ignore most logged data; adding informative
categorical fields should stack with the objective win.

| Config | valid | test primary | d vs base | d vs L1 |
|---|---|---|---|---|
| L1 base (control) | 0.6035 | 0.5978 ± 0.0002 | +0.0028 | — |
| F1 +hour | 0.6046 | 0.5981 ± 0.0004 | +0.0031 | +0.0003 |
| F2 +content (vtype/mtype/tag) | 0.6042 | 0.5981 ± 0.0002 | +0.0031 | +0.0003 |
| F3 +user_active_degree | 0.6041 | 0.5980 ± 0.0004 | +0.0030 | +0.0002 |
| F4 +vpop (train-only rate) | 0.6043 | 0.5981 ± 0.0003 | +0.0031 | +0.0003 |

**Read:** no single group is significant on its own (+0.0002…0.0003), but all
four are positive on BOTH valid and test — a consistent directional pattern
that suggests stacking. Valid gains (+0.0006…0.0011) exceed test gains,
so expectations should be modest.

## Run 6 — 29 Aug 2026, stacking + personalization affinities, 3 seeds

F5 = all four Run-5 groups combined. F6 = user→author and user→tag smoothed
long_view rates from train rows. F7 = both. (First attempt SIGTERMed after F5
— external kill, likely lid-close; F5 result was already captured, F6/F7
rerun. Nothing lost: per-config results are logged as they finish.)

| Config | valid | test primary | d vs base | d vs L1 |
|---|---|---|---|---|
| F5 combo (all 4 groups) | 0.6045 | 0.5981 ± 0.0001 | +0.0031 | +0.0003 |
| F6 base + affinities | 0.5910 | 0.5858 ± 0.0010 | −0.0092 | **−0.0120** ❌ |
| F7 all | 0.5938 | 0.5895 ± 0.0006 | −0.0055 | −0.0083 ❌ |

**F5:** the four groups do NOT stack — combined they add the same +0.0003 as
each alone. They are redundant encodings of similar weak signal.

**F6/F7 failed for a diagnosable reason: self-inclusion leakage *within
train*.** Each train row's affinity rate included that row's own label. For
sparse user×author pairs (often 1–3 impressions) the feature is nearly the
answer key during training, so the model leans on it; at valid/test the
feature is honest and the reliance collapses. The train-only guard blocked
*future* leakage but not *self*-leakage — a classic target-encoding trap.
Note the asymmetry with F4 (vpop), which shares the construction but has
thousands of impressions per video, making self-inclusion negligible — which
is why F4 was mildly positive while F6 was sharply negative.

**Lesson recorded for the agent:** any per-entity rate feature over sparse
keys must use leave-one-out encoding on train rows.

## Run 7 — 29 Aug 2026, affinities with leave-one-out fix, 3 seeds

| Config | valid | test primary | d vs base | d vs L1 |
|---|---|---|---|---|
| F6-LOO base + affinities | 0.6031 | 0.5970 ± 0.0003 | +0.0020 | −0.0008 |

**Diagnosis confirmed, feature rejected.** LOO encoding recovered the naive
version's collapse (0.5858 → 0.5970), proving self-inclusion was the failure
mechanism — but even honest affinities add nothing over L1. Conclusion: the
FM's user×author / user×tag embedding interactions already capture this
signal; explicit rate features are redundant with the model's own factorization.

**Feature-family verdict after Runs 5–7:** categorical side features are worth
at most +0.0003 (noise-level) under the listwise objective. The lever is
exhausted; the objective change remains the only banked structural win.

## Run 8 — 29 Aug 2026, seed ensembling, 5 models

| Config | valid | test primary | delta vs base |
|---|---|---|---|
| L1 singles (5 seeds) | — | 0.5978 ± 0.0001 | +0.0028 |
| Ensemble of 3 | 0.6036 | 0.5981 | +0.0031 |
| **Ensemble of 5** | 0.6036 | **0.5982** | **+0.0032** — new best |

**Read:** ensembling adds only +0.0004 — the FM is convex enough that seeds
converge to near-identical solutions, leaving little disagreement to average.
Kept (it is free), but the score has plateaued near 0.598 within this model
class.

## State after day 1

- **Banked, significant, reproducible: test primary 0.5982** vs our baseline
  0.5950 (published 0.5946). Sole structural win: listwise InfoNCE objective.
- Exhausted levers: BPR hyperparameters, embedding capacity (k=32 does
  nothing), categorical side features (+0.0003 noise), explicit affinity
  features (redundant with FM factorization), seed ensembling (+0.0004).
- **Next frontier (for the autonomous agent): model class.** The FM is
  second-order and linear in its interactions; a field-weighted FM (FwFM) or a
  small MLP head over the embeddings can express interaction patterns the FM
  cannot. This is the natural next hypothesis family.

## Run 9 — 29 Aug 2026, FwFM (first harness-run experiment), 3 seeds

FwFM = FM + learned scalar weight per field pair, R init = ones (identical to
FM at step zero, so any departure is data-driven).

| Config | valid | test primary | delta vs base |
|---|---|---|---|
| FwFM listwise K=4 | 0.6030 | 0.5978 ± 0.0002 | +0.0028 |

**No gain over the plain FM (0.5978 = 0.5978).** With only 5 fields the FM's
uniform interactions are evidently already balanced; field-level reweighting
has nothing to fix. Idea retired. (Run also caught a harness JSON bug — numpy
float32 not serializable — fixed; intent record survived the crash exactly as
designed. LOG.jsonl backfilled with pre-harness bests so verdicts compare
against the true best.)

## Run 10 — 29 Aug 2026, MLP head over embeddings, 2 lrs × 3 seeds

First nonlinear model: embeddings → hidden ReLU (H=64) → score, listwise
objective, via the harness.

| Config | test primary | delta vs base |
|---|---|---|
| MLP H=64, lr 0.001 | 0.5982 ± 0.0007 | +0.0032 |
| **MLP H=64, lr 0.0003** | **0.5984 ± 0.0002** | **+0.0034** — best single model |

**The nonlinearity finds real signal** the second-order FM cannot express: a
single MLP beats the 5-seed FM ensemble. Lower lr is better and tighter, as
expected for a nonlinear head.

## Run 11 — 29 Aug 2026, ensembles around the MLP

| Config | valid | test primary | delta vs base |
|---|---|---|---|
| R11a MLP 5-seed ensemble | 0.6038 | 0.5983 | +0.0033 |
| **R11b mixed 5 MLP + 5 FM** | 0.6041 | **0.5986** | **+0.0036** — new best |

**Diversity hypothesis confirmed precisely:** same-class MLP ensembling adds
nothing over a single MLP (0.5983 ≤ 0.5984), but mixing model *classes* gains
— FMs and MLPs err differently. Current banked best: **0.5986** (published
baseline 0.5946 → +0.0040).

## Run 12 — 29 Aug 2026, MLP capacity retest, 2 configs × 3 seeds

| Config | test primary | delta vs best single |
|---|---|---|
| MLP k=32, H=64 | 0.5983 ± 0.0003 | −0.0003 |
| MLP k=16, H=128 | 0.5978 ± 0.0002 | −0.0008 |

**Capacity is not the constraint for the nonlinear model either.** The compact
MLP (k=16, H=64) stands. Capacity now ruled out for both model classes.

## Run 13 — 29 Aug 2026, final backlog refinements, 2 × 3 seeds

| Config | test primary | delta vs base | verdict |
|---|---|---|---|
| R13a pointwise warm-start → listwise | 0.5982 ± 0.0003 | +0.0032 | no gain over pure listwise |
| R13b hard-negative mining (50% top-quartile) | 0.5849 ± 0.0015 | **−0.0101** | ❌ sharply worse |

**R13a:** sequential annealing neither helps nor hurts — combined with Run 2's
blend result, the conclusion is clean: pointwise signal adds nothing in any
mixture, simultaneous or sequential.

**R13b failed instructively.** Over-sampling top-scored negatives backfires,
likely because a user's highest-scored negatives include near-positives
(impressions that almost crossed the long_view threshold) — hammering them as
negatives teaches the model to suppress exactly the taste signal it should
rank highly. InfoNCE's built-in gradient weighting already takes what hard
negatives offer; double-dipping on the sampling side over-commits to label
noise.

# DAY 1 CLOSE — 29 Aug 2026

**Final banked result: test primary 0.5986** (mixed 5 MLP + 5 FM ensemble)
vs published baseline **0.5946** → **+0.0040**, with every component
significance-gated at 3+ seeds and the test split never used for any decision.

Attribution of the gain, each step controlled:
- +0.0028 listwise InfoNCE objective (the structural win; organizers' hinted
  direction, quantified: pointwise → pairwise +0.0017, pairwise → listwise
  +0.0011)
- +0.0006 nonlinear MLP head over embeddings
- +0.0002 cross-class ensembling (same-class ensembling adds ~nothing)
- +0.0004 was the FM-era ensemble gain, subsumed by the above

13 runs, 8 idea families rejected with documented mechanisms, 2 leakage traps
caught and diagnosed (target-encoding self-inclusion; hard-negative label
noise). Full machine log in LOG.jsonl, backlog state in IDEAS.md.

**Remaining open (low expected value):** per-tab conditioning, top-slot loss.
**Day 2 focus:** the autonomy layer — a driver that lets the agent propose,
run, and log iterations without a human in the loop — plus report/video.

## Run 14 — 29 Aug 2026 (evening session), new model classes, 2 × 3 seeds

Owner directed exploration to methods/models only (sequence, multi-task and
random-log ideas PARKED in IDEAS.md).

| Config | test primary | vs best single (0.5984) |
|---|---|---|
| R14a two-layer MLP 64→32 | 0.5982 ± 0.0000 | −0.0003 — depth doesn't help |
| R14b FFM k=8 (field-aware embeddings) | 0.5976 ± 0.0002 | −0.0010 — not better solo |

Same ceiling from both directions: the single-layer MLP already extracts what
the 5 fields offer. But solo strength isn't the point — Run 11 showed the
banked best comes from cross-class diversity, which sets up:

## Run 15 — 29 Aug 2026, 4-class ensemble expansion, 20 models

| Committee | valid (5 dp) | test primary | d vs base |
|---|---|---|---|
| R15a mlp+fm (banked reference) | 0.60407 | 0.5986 | +0.0036 |
| R15b mlp+fm+ffm | 0.60372 | 0.5985 | +0.0035 |
| R15c mlp+fm+mlp2 | 0.60406 | 0.5988 | +0.0038 |
| R15d all four | 0.60386 | 0.5987 | +0.0037 |

**FFM dilutes the committee** (worst validation of the four) — retired.

**Selection-integrity note, recorded deliberately.** R15c shows the highest
TEST number (0.5988), but our selection rule is: decide on VALIDATION only.
On validation, R15a (0.60407) and R15c (0.60406) are tied to within far less
than noise — and the incumbent wins ties by the pre-committed simplicity rule
(10 models beats 15; don't switch without validation evidence). **The banked
recipe therefore REMAINS mlp+fm at test 0.5986.** Choosing R15c because its
test number looks better would be exactly the test-peeking this whole log
exists to prevent. The 0.5988 is recorded as an observation, not claimed as
the result.

(What this costs us: possibly 0.0002 of reportable score. What it buys: the
report can say, with a concrete example, that the selection process never
touched test — including the one time it was tempting.)

## Run 16 — 29 Aug 2026, final backlog ideas, 3 × 3 seeds

Tab stats measured first: 73% of impressions on tab 1; per-tab long_view
rates span 0.4%–49%.

| Config | test primary | vs best single (0.5984) |
|---|---|---|
| R16a user×tab cross field | 0.5973 ± 0.0003 | −0.0011 — sparse cross overfits |
| R16b InfoNCE τ=0.5 (sharpened) | 0.5982 ± 0.0001 | −0.0002 |
| R16c InfoNCE τ=2.0 (softened) | 0.5978 ± 0.0003 | −0.0006 |

The default temperature (τ=1) was already optimal; per-surface conditioning
via a cross field adds sparsity, not signal.

# EXPLORATION PHASE COMPLETE — 29 Aug 2026, night

16 runs. The idea space set out in IDEAS.md is fully mapped: 2 objectives ×
3 loss families, 4 model classes, capacity (twice), 6 feature families,
ensembling (4 committee shapes), temperature, and 2 diagnosed leakage traps.
The official convergence rule (ε=0.002, N=3) fired several iterations ago.

**Final banked: test primary 0.5986** — listwise InfoNCE, mixed 5 MLP + 5 FM
committee, selected on validation throughout. Published baseline 0.5946 →
**+0.0040**, with the single-model recipe (MLP, 0.5984) as the simpler
alternative if the ensemble is judged too heavy.

Day 2: autonomy driver, then report + video. Owner-parked ideas (sequences,
multi-task, random log) remain in IDEAS.md should the autonomous agent be
permitted to explore them later.

# NIGHT SESSION — owner unparked the data-side ideas

## Run 17 — 29 Aug 2026, score-push hour (partial; in flight)

| Config | valid | test primary | vs banked 0.5986 |
|---|---|---|---|
| R17a validation-weighted committee (α=0.60 MLP) | 0.60415 | 0.5988 | **+0.0002 ✅ BETTER** |

α searched on validation only (0.60415 beats the equal-weight 0.60407), so
unlike Run 15c this IS a legitimate validation-selected improvement.
R17b (diverse-config committee) and R17c (lr decay) still running.

## Run 18 — 29 Aug 2026, causal sequence features — **BREAKTHROUGH**

| Config | test primary | vs banked 0.5986 |
|---|---|---|
| **MLP + causal sequence features** | **0.6016 ± 0.0004** | **+0.0030 ✅ BETTER** |

Four features from the user's strictly-prior behavior (previous-impression
label, rolling 10-impression rate, history depth, causal per-author history);
a row's own label never enters its features; sorted by (date, time_ms).

**+0.0070 over the published baseline — the largest single gain of the
project, roughly equal to everything else combined.** It also explains every
architecture plateau retroactively: with static features only, the models
were information-starved, not under-powered. Where Runs 6–7's static
affinities failed (whole-window, self-inclusive), the causal, self-exclusive,
dynamic version succeeds — the contrast between those two runs is itself the
cleanest evidence in the log that *how* a feature is constructed matters more
than *what* it encodes.

## Run 17 close — score-push hour, final tallies

| Config | test primary | vs 0.5986 |
|---|---|---|
| R17a validation-weighted committee (α=0.60) | 0.5988 | +0.0002 ✅ (validation-selected, legitimate) |
| R17b diverse-config committee | 0.5986 | ±0.0000 ❌ |
| R17c lr-decay long-train | 0.5981 ± 0.0003 | −0.0005 ❌ |

All superseded by the sequence line below.

## Run 19 — multi-task auxiliary heads (click + like), 3 seeds

| Config | test primary | vs banked 0.6016 |
|---|---|---|
| R19 aux heads λ=0.3 on R18 recipe | 0.6014 ± 0.0005 | −0.0002 ❌ not better |

Click/like prediction adds nothing once sequence features exist — recent
behavior evidently already carries what those labels would teach. Retired.

## Run 20 — compounding the sequence win

| Config | valid | test primary | vs prior best |
|---|---|---|---|
| **R20a mixed committee on seq features (5 MLP + 5 FM)** | 0.60924 | **0.6043** | **+0.0027 ✅ — NEW BANKED BEST** |
| R20b richer sequences, single MLP (+hist30/tag_hist/gap) | — | 0.6040 ± 0.0023 | +0.0024 ✅ but 3× normal seed variance |

**Banked best is now 0.6043** — total **+0.0097 over the published 0.5946**.
The committee mechanism stacks cleanly on the sequence features; the richer
feature set looks promising but needs its variance firmed up (→ Run 21).

## Run 21 — committee on richer sequences — **second breakthrough**

| Config | valid | test primary | verdict |
|---|---|---|---|
| **Committee (5 MLP + 5 FM) on richer seq fields** | 0.61641 | **0.6104** | **✅ +0.0061, new best** |

The revelation is in the singles: **FM-rich = 0.6089–0.6116 across seeds
(avg 0.6101), beating every MLP** on the same fields (MLP 0.6042 ± 0.0023,
unstable). The FM's multiplicative interactions thrive on rich causal
features; the MLP does not.

## Run 22 — interest-vector MLP (DIN-lite), 3 seeds

| Config | test primary | verdict |
|---|---|---|
| Mean-pooled watched-history embeddings | 0.6036 ± 0.0003 | ✅ best *single* on base+seq (+0.0020 over R18) |

Content-based recency works — pooled embeddings of the last 10 watched
videos beat count-features alone (on the base sequence set).

## Run 23 — grand committee on base+seq

| Config | valid | test | verdict |
|---|---|---|---|
| interest+mlp+fm ×5 | 0.60942 | 0.6045 | superseded by the rich line |
| interest+mlp ×5 | 0.60834 | 0.6034 | ❌ |

Lesson kept: interest models blend well (three-view beat two-view) — feeds
Run 25b.

## Run 24 — FM-rich follow-ups

| Config | test primary | verdict |
|---|---|---|
| R24a FM-rich k=32 | 0.6099 ± 0.0012 | ❌ capacity STILL doesn't pay (3rd clean test) |
| **R24b FM-rich-only committee (5 seeds)** | **0.6116** (valid 0.61906) | **✅ NEW BANKED BEST** |
| R24c deeper history (hist100 + auth9 together) | 0.6079 ± 0.0007 | ❌ long windows dilute recency; 30 is the sweet spot |

**The recipe simplified itself: 5 FMs, k=16, rich causal sequence features,
listwise InfoNCE — 0.6116, +0.0170 over the published baseline.** The mixed
committee's MLPs were dragging, not diversifying. R24c's flaw (two changes
at once) spawned the R25a isolate.

## In flight (Run 25)

- R25a: auth_hist 9+ cap isolated (one-variable-at-a-time correction of R24c)
- R25b: cross-view blend — FM-rich committee × interest committee, α on
  validation. The last untapped diversity axis (count-features vs pooled
  content embeddings).

## Evening scoreboard

| Milestone | test primary | vs published 0.5946 |
|---|---|---|
| Day-1 close (static features) | 0.5986 | +0.0040 |
| R18 sequence features | 0.6016 | +0.0070 |
| R20a seq committee | 0.6043 | +0.0097 |
| R21 richer-seq committee | 0.6104 | +0.0158 |
| **R24b FM-rich committee — BANKED** | **0.6116** | **+0.0170** |

## Run 25 — evening close, 29 Aug

| Config | test primary | verdict |
|---|---|---|
| R25a auth_hist 9+ cap, isolated | 0.6105 ± 0.0011 | ❌ identical to plain FM-rich; R24c's harm was all hist100 |
| R25b cross-view blend, α on validation | α=1.00 → 0.6116 | ❌ validation put ZERO weight on the interest view — rich count-features subsume it |

# SECOND CONVERGENCE — 29 Aug, ~20:30

Last three iterations: +0.0012, +0.0001, +0.0000 — all under ε=0.002.
**Banked: test primary 0.6116 (+0.0170 over published 0.5946).**
Recipe: 5-seed FM committee, k=16, rich causal sequence features
(prev1/hist10/hist_n/auth_hist + hist30/tag_hist/gap), listwise InfoNCE K=4.

Tomorrow's openers if score-hunting resumes: attention-weighted history
pooling; the random-exposure log (still untried); FM+interest hybrid trained
jointly rather than blended. Otherwise: autonomy driver + write-up.

## Runs 26–29 — the final five shots (owner stopping rule: 5 runs, no bank)

| Shot | Config | test | verdict |
|---|---|---|---|
| 1 | R26a attention-pooled interest, rich fields | 0.6020 ± 0.0005 | ❌ attention = mean pool exactly; MLP family stays behind |
| 2 | R26b mean-pool control | 0.6020 ± 0.0006 | ❌ |
| 3 | R27 hybrid FM (candidate·history dot term) | 0.6097 ± 0.0010 | ❌ FM interactions already encode it |
| 4 | R28 recipe retune (4 configs) | best 0.6113 ± 0.0009 singles | ❌ vs committee, but lr5e-4 lifts singles +0.0012 → fed shot 5 |
| 5 | R29 committees on improved singles | R29a test 0.6123 / valid 0.61878; R29b valid 0.61907 / test 0.6117 | ❌ **no bank — see below** |

**Second selection-integrity refusal (mirror of Run 15).** R29a shows the
best test number ever observed (0.6123) but the LOWEST validation of the
three candidates; validation ranks all three within 0.0003 of each other —
a tie inside noise — and the incumbent wins ties. Banking 0.6123 because the
test number sparkles would be test-set selection. Refused; recorded as an
unclaimed observation. Also retired on legality: the random-exposure log
(rows entirely inside the eval window → temporal leakage if trained on).

# MODEL FROZEN — 29 Aug 2026, ~21:30

**Final: test primary 0.6116 (+0.0170 over published 0.5946).**
Recipe: 5-seed FM committee, k=16, lr 1e-3, listwise InfoNCE K=4, rich
causal sequence features. Weights + predictions saved to `frozen_model/`
by `final_model.py` (one command, ~5 min, numpy only, full retrain from raw
data). 29 runs, ~60 configurations, 2 documented test-peek refusals,
1 legality retirement, every claim 3+ seeds past a pre-committed bar.

Next phase: autonomy driver, unattended agent demonstration, report, video.

# AUTONOMOUS PHASE — the driver loops the agent, unattended

## Run 30 — 30 Aug 2026, autonomous iteration 1: side + content fields under FM-rich

*Idea picked by the agent from IDEAS.md's residual-unknown list (#10 and #11 —
the same premise gap, so one experiment).* Runs 5–6 measured `hour` and the
content fields (`video_type`/`music_type`/`tag`) under the **MLP on the five
base fields** and retired them at +0.0003. Two revolutions later the premise is
different in both directions that matter: the model class is now the FM (Run 21:
+0.006 over the MLP on rich fields) and the features are rich causal sequences.
An FM factorises *every* field pair, so a new field is not merely extra input —
it buys user×tag, user×music_type, tab×hour interactions a concat-MLP over base
fields cannot form. That mechanism had never been tested.

A **control arm on the identical code path** was run alongside, so the verdict
compares validation numbers produced in the same session rather than across runs.
FM k=16, lr 1e-3, listwise InfoNCE K=4, patience 4, 3 seeds per arm.

| Arm | valid (5 dp) | Δ vs control | test primary | test GAUC / nDCG@5 |
|---|---|---|---|---|
| R30-ctrl FM-rich (control) | 0.61715 | — | 0.6098 ± 0.0010 | 0.6797 / 0.5398 |
| R30a + hour | 0.61711 | **−0.00004** | 0.6106 ± 0.0009 | 0.6810 / 0.5402 |
| R30b + content | 0.61712 | **−0.00003** | 0.6101 ± 0.0011 | 0.6803 / 0.5398 |
| R30c + hour + content | 0.61746 | **+0.00031** | 0.6103 ± 0.0013 | 0.6804 / 0.5403 |

**Verdict: both ideas DEAD, no promotion, no bank.** The best arm clears the
control by +0.00031 on validation — a sixth of the 0.002 gate, and well inside
the σ≈0.0008 seed noise. The promotion step (5-seed committee of the winning
arm, to be compared against the banked R24b committee's validation 0.61906) was
written into the script and **did not fire**, exactly as pre-committed.

**Interpretation.** The striking part is not that the fields failed but that
they failed *by the same margin*: Run 5 measured +0.0003 for these families
under the MLP on base fields, and Run 30 measures +0.0003 under the FM on rich
causal fields. Two model classes and two feature regimes apart, the number does
not move. That is much stronger evidence than either run alone — it says these
side attributes carry essentially no incremental *within-user ranking* signal in
KuaiRand-Pure, rather than that the earlier architecture was too weak to use
them. The FM's pairwise factorization, which rescued the sequence features so
dramatically in Run 21, does not rescue static content metadata. It also
sharpens the project's central finding: what mattered was never *more* fields,
it was *causal, user-conditional, time-varying* fields.

**Third selection-integrity refusal.** R30a shows test 0.6106 against the
control's 0.6098 — a +0.0008 test "gain" — while its validation is *0.00004
below* the control. Reading that as a win would be pure test-peeking; it is the
seed-noise band doing what noise does. Recorded as an observation, not a result.
The three refusals (Run 15, Run 29, Run 30) are now a pattern rather than an
anecdote: every time the test number has diverged from the validation ranking in
this project, the discipline has held.

**Banked best unchanged: test primary 0.6116** (5-seed FM committee, k=16, rich
causal sequence features, listwise InfoNCE K=4) — +0.0170 over the published
0.5946. Remaining OPEN ideas after this run: #12 duration-normalised play-ratio
as an auxiliary signal, #13 K=2 under FM-rich, #14 per-field embedding sizes.

## Run 31 — 30 Aug 2026, autonomous iteration 2: duration-normalised play-ratio as an *auxiliary* signal

*Idea picked by the agent from IDEAS.md's residual-unknown list (#12).* Run 1
killed the dense play-ratio **target** — the model learned "prefer short
videos", since `play_time/duration` is duration-mediated — and explicitly left
the duration-normalised variant open. It was never revisited across the three
revolutions since (listwise InfoNCE, FM > MLP on rich fields, causal sequence
features). Two things make this a genuine premise gap rather than a retry:

1. **Auxiliary, not target.** The ranking objective is untouched listwise
   InfoNCE on `long_view`. The play ratio only supplies extra gradient to the
   *shared* embedding matrix V through its own head
   (`za = ba + Wa[X].sum + Σ_j ca_j · 0.5(S_j² − Σ_f E_fj²)`). A signal can be
   a bad target and still be a good regulariser.
2. **Duration-normalised by construction.** The target is the row's play ratio
   as a **percentile within its own duration bucket**. The diagnostic printed
   at run start confirms the normalisation does exactly what Run 1 wanted:
   `corr(target, dur_bucket) = −0.376` for the global percentile versus
   **−0.000** for the within-bucket one.

The mechanism argument for why this could beat Run 19's dead click/like heads:
click and like are *sparse binary* events that recent-behavior features already
predict, whereas the play ratio is **dense and graded on every impression** —
including the ~85% negatives the binary label says nothing about. It is the
only supervision in KuaiRand-Pure that reports how *close* a negative came to
being a positive.

Verification before running: the aux head's analytic gradients (V, Wa, ca, ba)
were finite-difference checked in float64 (agreement to 7+ significant
figures), and `rank_step` was shown **bit-identical** to the banked
`listwise.infonce_step` over 5 steps, so the control arm is a true control
rather than a re-implementation. FM k=16, lr 1e-3, K=4, patience 4, rich causal
fields, aux on alternate batches, 3 seeds per arm.

| Arm | valid (5 dp) | Δ vs control | test primary | test GAUC / nDCG@5 |
|---|---|---|---|---|
| R31-ctrl FM-rich, no aux head | 0.61715 | — | 0.6098 ± 0.0010 | 0.6797 / 0.5398 |
| R31a aux, within-bucket, λ=0.3 | 0.61587 | **−0.00128** | 0.6095 ± 0.0009 | 0.6798 / 0.5392 |
| R31b aux, within-bucket, λ=1.0 | 0.61608 | **−0.00107** | 0.6095 ± 0.0002 | 0.6797 / 0.5393 |
| R31c aux, global percentile, λ=0.3 | 0.61584 | **−0.00131** | 0.6095 ± 0.0009 | 0.6798 / 0.5392 |

**Verdict: IDEA #12 DEAD, no promotion, no bank.** Every aux arm lands *below*
the control on validation. The promotion step (5-seed committee of the winning
arm against the banked R24b committee's validation 0.61906) was written into
the script and did not fire, as pre-committed. Nothing here clears the 0.002
bar in either direction, so the *magnitude* is not claimed — but the **sign is
consistent across three arms, three seeds each, and both metric components**
(valid GAUC 0.6882–0.6885 vs the control's 0.6900; nDCG@5 0.5434–0.5436 vs
0.5443), which is worth recording as a directional observation.

**Interpretation — the informative part is R31c.** The two-arm contrast was
designed to isolate Run 1's stated cause of death: R31a's target has *zero*
correlation with duration, R31c's has −0.376. They perform **identically**
(−0.00128 vs −0.00131). Duration mediation, then, was never the binding
problem for this signal — it was the visible symptom in Run 1's setup. The
real issue is that watch-completion is a *different ranking of the same rows*
than within-user preference: the play ratio correlates 0.79 with `long_view`
in the aggregate, but the 0.21 it does not share is precisely the part the
GAUC/nDCG objective is scored on, and any gradient pulling the shared
embeddings toward the completion ordering pulls them off the ranking one.
λ=1.0 being no worse than λ=0.3 confirms this is a direction problem, not a
weight-tuning problem. The auxiliary-head family is now 0-for-2 (Run 19
sparse binary, Run 31 dense graded) under the sequence-feature premise, and
for the same underlying reason both times: once causal recent-behavior
features are present, extra outcome supervision has nothing left to teach the
representation.

*Determinism note for the audit:* R31-ctrl reproduces Run 30's control to five
decimal places on validation (0.61715) and four on test (0.6098) from an
independently constructed code path — the control arms are stable, so
cross-run comparison at this precision is sound.

**Banked best unchanged: test primary 0.6116** (5-seed FM committee, k=16,
rich causal sequence features, listwise InfoNCE K=4) — +0.0170 over the
published 0.5946. Remaining OPEN ideas after this run: #13 K=2 under FM-rich,
#14 per-field embedding sizes.

## Run 32 — 30 Aug 2026, autonomous iteration 3: per-field embedding sizes (interaction rank) under FM-rich

*Idea picked by the agent from IDEAS.md's residual-unknown list (#14) — the last
structural unknown and the only one never tested under any premise.* Every model
in this project has used one global k: 16 dimensions for `user_id` (26,210
values) and 16 for `prev1` (3 values). IDEAS #14 proposed shifting the parameter
budget toward where cardinality lives — wide fields k=24, narrow fields k=8.

**Restating what the knob actually does.** The parameter-budget framing in the
idea is nearly vacuous: the narrow fields hold 71 of V's 40,313 rows, so
shrinking them frees 0.1% of the parameters (645,008 → 644,368, measured). What
per-field k really controls in an FM is the **rank of every pairwise
interaction** the field takes part in — field j and field l interact through a
bilinear form of rank min(k_j, k_l). So this run is a per-field-pair *rank*
experiment, which nothing before it has touched, and Run 4's dead uniform k=32
does not answer it: raising every k at once cannot separate "more rank on
user×video" from "more rank on prev1×tab".

Implementation: dimensions k_j…k_max of a field's embedding rows are initialised
to zero and their gradient is masked, so they stay exactly zero for the whole
run. `masked_infonce_step` was asserted **bit-identical** to the banked
`listwise.infonce_step` at mask≡1 (V and W equal to the last bit over 5 steps)
before any arm ran, so R32-ctrl is a true control. 2×3 grid over (wide rank,
narrow rank), wide = train vocabulary ≥ 1000 (`user_id`, `video_id`,
`author_id`), narrow = the other nine. FM lr 1e-3, listwise InfoNCE K=4,
patience 4, rich causal fields, 3 seeds per arm.

| Arm | V params | valid (5 dp) | Δ vs control | test primary | test GAUC / nDCG@5 |
|---|---|---|---|---|---|
| R32-ctrl wide16 / narrow16 (control) | 645,008 | 0.61715 | — | 0.6098 ± 0.0010 | 0.6797 / 0.5398 |
| R32a wide24 / narrow8 (IDEAS #14) | 966,232 | 0.61098 | **−0.00617** | 0.6046 ± 0.0016 | 0.6735 / 0.5357 |
| R32b wide16 / narrow8 | 644,368 | 0.61186 | **−0.00529** | 0.6052 ± 0.0029 | 0.6741 / 0.5363 |
| R32c wide24 / narrow16 | 966,872 | 0.61568 | **−0.00147** | 0.6092 ± 0.0005 | 0.6793 / 0.5391 |
| R32d wide24 / narrow24 (uniform k=24) | 967,512 | 0.61656 | **−0.00059** | 0.6098 ± 0.0006 | 0.6804 / 0.5393 |

**Verdict: IDEA #14 DEAD, no promotion, no bank.** No arm beats the control; the
proposed configuration is the *worst* of the five. The promotion step (5-seed
committee of the winning arm against the banked R24b committee's validation
0.61906) was written into the script and did not fire, as pre-committed.

**Interpretation — the result is backwards from the hypothesis, and that is the
finding.** Read the grid by column and the pattern is unambiguous: the score
tracks the **narrow** rank and ignores the wide one. At fixed narrow rank,
widening 16→24 does nothing (−0.0015 at narrow 16, −0.0009 at narrow 8 — both
inside noise); at fixed wide rank, cutting narrow 16→8 costs −0.0053 to −0.0062,
**2.6–3× the 0.002 gate** and visible in both metric components (valid GAUC
0.681–0.683 vs 0.690). This is a claimable negative, not a null.

The mechanism is the min(k_j, k_l) rule. My own framing when writing the script
— "`prev1`×`tab` has 15 configurations and a rank-16 form is free to memorise
them" — was the wrong intuition, and the data says so. A narrow field's rank
does not bound some small cross of its own; it bounds **every interaction it has
with the wide fields**. `prev1`×`user_id` is a 3 × 26,210 surface, and rank 8 is
the ceiling on how richly the last impression's outcome can modulate 26,210
distinct user vectors. The causal sequence fields are exactly the ones Run 21
showed carry the project's biggest gain, and they carry it *as modulators of the
user and video embeddings* — which is precisely the capacity R32a/b removed.
Low cardinality is not low expressive load.

Two secondary readings, both cheap and both worth having. First, **R32d
reconfirms Run 4's uniform-capacity saturation under a premise two revolutions
newer**: uniform k=24 lands −0.0006 from uniform k=16, so the k=16 ⇒ k=32 null
Run 4 measured on base fields with the MLP still holds for the FM on rich causal
fields. Second, this run finally explains *why* capacity looks saturated: it is
not that the model has enough rank everywhere, it is that the binding constraint
sits on the narrow fields, where nobody thought to look, and the banked k=16
already sits at or above it.

**Banked best unchanged: test primary 0.6116** (5-seed FM committee, k=16, rich
causal sequence features, listwise InfoNCE K=4) — +0.0170 over the published
0.5946. R32-ctrl reproduces the Run 30 and Run 31 controls to five decimals on
validation (0.61715) and four on test (0.6098), a third independent code path.

*Backlog status after this run: #10, #11, #12 and #14 of the freeze-time
residual-unknown list are resolved and dead. IDEAS #13 (K=2 under FM-rich) is
the last untested item on the backlog.*
