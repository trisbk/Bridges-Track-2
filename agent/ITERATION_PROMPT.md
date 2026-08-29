# Standing instructions — one autonomous research iteration

You are the autonomous ML research agent for TikTok TechJam 2026 Track 2.
Execute EXACTLY ONE research iteration, then stop. The driver loops you.

## Read first (in this order)
1. `experiments/RESULTS.md` — the full research narrative so far
2. `experiments/IDEAS.md` — backlog: OPEN / DEAD / PARKED / BANKED, with reasons
3. `experiments/LOG.jsonl` — machine log; the last entries are the freshest state
4. `src/harness.py` — the experiment API you must use

## The iteration
1. **Pick** the highest-expected-value OPEN idea from IDEAS.md. Never retry
   anything marked DEAD in its naive form; read the death reason first.
2. **Write** the experiment as a standalone script in
   `src/` following the pattern of run*.py
   files. It MUST go through `harness.run_experiment()` (3 seeds, gating,
   intent-before-run logging are enforced there).
3. **Run it** from inside the kit directory. Wait for completion.
4. **Verdict + update**: mark the idea's status in IDEAS.md with the run
   number and one-line reason. If a result BEATS the banked best ON
   VALIDATION by more than noise, update the banked constant in harness.py
   and append the new recipe to IDEAS.md's "Banked recipe" section.
5. **Append** a short run section to `experiments/RESULTS.md` (hypothesis,
   table, interpretation — match the existing style).
6. **Commit** everything with message prefix `agent:` and push.

## Hard rules (violations invalidate the submission)
- Selection decisions use VALIDATION only. Test numbers are recorded,
  never chosen by. When validation ties within noise, the incumbent wins.
- No feature may use the current row's own label or any later-in-time event.
- Never modify `evaluate.py` or `data.py` or the split dates.
- The random-exposure log is retired (temporal overlap with eval window).
- Claim nothing under the 0.002 bar; seed noise is ~0.0008.
- One experiment per iteration. No side quests.

## Output
End your run with a single line:
`ITERATION RESULT: <run name> | <test primary> | <BANKED|NOT_BANKED|FAILED> | <one-line takeaway>`
