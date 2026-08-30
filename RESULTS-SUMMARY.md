# Final Submission & Results Summary (Deliverable 4)

## Final model output

`src/submission.csv` — the frozen model's scores for every test-split row, in
the Starter Kit schema (`row_id,user_id,video_id,score`), generated from
`src/frozen_model/ensemble_predictions.npz` and validated with the kit's own
checker: `python3 submit.py --check --split test submission.csv` →
✓ 170,588 rows, aligned. Model checkpoint: `src/frozen_model/` (5 seed
weight files + config.json); regenerate end-to-end with
`python3 final_model.py`.

## Results table (required benchmark: KuaiRand-Pure)

Scored submission = the validation-best checkpoint at convergence
(per the Primary-metric rule), evaluated once on the local test split
(proxy for the hidden test set).

| Metric | Official baseline | Ours (validation) | Ours (test) | **Δ vs baseline (test)** |
|---|---|---|---|---|
| GAUC | 0.6610 (valid 0.6674) | 0.6926 | 0.6825 | **+0.0215** |
| nDCG@5 | 0.5282 (valid 0.5357) | 0.5455 | 0.5408 | **+0.0126** |
| **primary** | **0.5946** (valid 0.6016) | **0.6191** | **0.6116** | **+0.0170** |

Per the scoring formula (mean over metrics of absolute delta):
**score = (0.0215 + 0.0126) / 2 = +0.0170.** Context: the attainable range
runs from random 0.4753 to the oracle ceiling 0.8645; the baseline captures
~31% of it, our submission ~37%. (We independently derived the ceiling
arithmetic — 27.1% zero-positive users, 9.2% all-positive — before reading
the organizers' numbers; they match exactly.)

Bonus benchmarks (KuaiRand-1k / 27k): not attempted.

## Resource usage (Feasibility & Practicality)

Measured, not estimated — token counts summed from the recorded usage fields
of every headless agent session transcript; wall-clock from driver event
timestamps. No GPU was used at any point (GPU-hours = 0); all training is
single-machine CPU (Apple Silicon laptop), numpy only.

| Run | Iterations (of 50) | Agent wall-clock | LLM tokens in+out | (incl. cache reads/writes) |
|---|---|---|---|---|
| Autonomous verification run ("Demo A", Runs 30–32) | 3 | 52 min 32 s | 149,658 | 6,188,547 |
| Clean-room autonomous run (from bare baseline) | 6 | 1 h 47 min 41 s | 325,476 | 17,978,927 |

Both runs terminated by the official convergence rule (ε = 0.002, N = 3 on
validation), well inside the 50-iteration cap and 6 h wall-clock ceiling.
Model training within iterations is a negligible share of wall-clock
(~40–90 s per 3-seed experiment); the cost is agent reasoning.

The interactive research campaign (Runs 1–29, which produced the 0.6116
checkpoint) ran as a supervised session over ~1 day; its interventions are
enumerated in `experiments/ITERATION-LOGS.md` (a "handful": five strategic
decisions, zero iteration-level ones). Token usage for that phase was not
separately metered (it shared a general-purpose session); the two fully
instrumented autonomous runs above are the metered reference points.

## The three runs, side by side (autonomy spectrum)

| Run | Start state | Manual interventions | Converged at |
|---|---|---|---|
| Interactive campaign (29 runs) | official baseline | 5 strategic (0 iteration-level) | **0.6116** — the scored submission |
| Demo A (3 runs, overnight) | finished research state | **0** | 0.6116 confirmed |
| Clean-room (6 runs) | bare baseline, empty memory | **0** | **0.59744** (+0.0028 over baseline, its own discovery) |

The clean-room run also independently exercised the selection discipline:
its iterations 4–5 posted higher *test* scores (0.59801, 0.59756) that it
refused to bank because validation did not justify them — the project's
fourth and fifth documented test-peek refusals, this time by the agent alone.
