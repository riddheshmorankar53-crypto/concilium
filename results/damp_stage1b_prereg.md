# Damping stage 1b: pre-registration (written 2026-10-11, before any stage 1b run)

Question: does server momentum 0.5 (V1) reduce the filter's own dips (CNN, no attack) vs the plain filter (V0)?
Stage 1 (`damp_stage1.log`, draws 5-44): dips < 0.7 were V0 4 / V1 1 of 200, p = 0.25. Not significant; this is
a fresh, larger sample. Stage 1 data is NOT pooled in.

Design
- `VARIANTS=V0,V1 gap_e.py damp <seed> 45 200 0` for seeds 0-4: draws 45-244, 1000 pairs, paired by torch seed.
  GPU (`~/ppfl-gpu`), OMP_NUM_THREADS=1, one job at a time. Log: `results/damp_stage1b.log`.
- Analysis on all 1000 pairs, run to completion (no peeking-based stopping).

Pass (all required)
1. Fewer V1 dips (acc < 0.7) than V0, exact two-sided McNemar on discordant pairs, p < 0.05.
2. Median accuracy of V1 >= median of V0 - 0.01.

Reported, not gating: mean, 5th percentile, worst, honest-rejection rate, skipped rounds.
Change from stage 1: the "honest rejection below V0" criterion is dropped. Momentum changes only the server step,
not the filter, so it isn't what's being tested. This change is made before seeing stage 1b data.

If it passes -> stage 2 (attacks), defined now: attacks 1-6 of CNN_ATTACKS, draws 45-84 per seed (1200 pairs).
Pass: V1 dips <= V0 dips, and V1's worst accuracy per attack is not more than 0.02 below V0's.
If it fails -> damping is dropped; README gap E records it.
