# OPDI
PRC Data Challenge 2026

Taxi-out time prediction for 10 European airports (EUROCONTROL PRC Data Challenge 2026).

## Baseline
`taxiout.py` trains LightGBM models on 2025 departures.

- Features: airport, runway, stand, aircraft type, operator, market segment, wake category,
  time of day / weekday / season, NM flight-plan times as offsets from takeoff, and traffic
  counts (departures/arrivals within ±5–60 min, same-runway departures in the previous 10/20 min).
- Two-stage mixture: some off-block stamps are a copy of the scheduled time (`|BLOCK-SCHED| <= 60 s`),
  which makes taxi time equal `MVT - SCHED`. A classifier gives P(copy); a regressor trained on the
  other flights gives the normal value; the prediction is the expectation of the two.
- NM-missing stage: departures with no NM flight record (~1%, mostly LIRF) hold ~90% of the
  multi-hour outliers, so they get their own small mixture, applied where `MVT - SCHED > 1 h`.
- Cell statistics: median/P10/P90/mean/count of clean taxi times per (airport, stand, runway) and coarser
  cells, out-of-fold by month for training rows.
- Queue features: departures/arrivals on the surface and same-runway queue at the estimated push-back
  (`AOBT_3`, fallback takeoff - 15 min), takeoffs ahead on the runway, gaps to neighbouring takeoffs.
- Neighbour taxi level: mean of (takeoff - AOBT_3) over departures that took off in the previous 20/60 min
  (airport) and 30 min (runway); a live read of how slow the airport is right now.
- Copy detection: explicit SCHED minus AOBT_3/EOBT/LOBT/IOBT differences, SCHED minute-mod-5 and seconds, and
  out-of-fold copy rates per (airport, stand), (airport, operator), (airport, runway). Improves both months
  (Jan 364.4 -> 361.1, Jul 385.8 -> 357.4); normal flights 275.7 -> 271.4. Much of the July gain is on tail rows.
- LOBT window: in 2025 all 2.06M matched departures have |BLOCK - LOBT| <= 3606 s, so predictions are clipped to
  taxi in [MVT - LOBT - 3606, MVT - LOBT + 3606] (rule from the training data, not tuned on the holdout).
  Jan 361.1 -> 360.7, Jul 357.4 -> 350.2.
- Weather: IEM ASOS METAR (https://mesonet.agron.iastate.edu/request/download.phtml, 10 airports, 2025-01-01 to
  2026-07-31, ~31 MB, saved to `opdi/metar/<ICAO>.csv`, not in the repo). Latest observation >=10 min before takeoff, rolling
  freezing/snow/precip/fog flags, temperature trend. Jan 360.7 -> 359.6, Jul 350.2 -> 349.0 (1 LightGBM member).
- Copy-impossible rule: a schedule copy (taxi = MVT - SCHED) is impossible when that value lies outside the LOBT
  window, so P(copy) is set to 0 there. Neutral on the 2025 holdout; touches ~9 Rome rows by >600 s on the ranking set.
- Per-airport blend (`--per-airport`): one LightGBM regressor per airport on non-copy flights, averaged 50/50 with the
  global regressor. 1 member: Jan 359.6 -> 357.6, Jul 349.0 -> 347.8.
- Ensemble option: `--seeds N --cat` averages N LightGBM members plus a CatBoost regressor.
- Validation: train on 2025 months other than Jan and Jul, score RMSE on Jan + Jul 2025,
  outliers kept (dropping them makes the score look much better than it is).

| Model | RMSE (s) |
|---|---|
| Mean prediction | 687 |
| Single regressor | 464 |
| Two-stage mixture | 420 |
| + NM-missing stage | 384 |
| + stand/runway cell statistics | 381 |
| + congestion window fix, surface-queue features | 378 |
| + neighbour taxi-level features (recent takeoffs' takeoff - AOBT_3) | 376 |
| + schedule-copy features (SCHED vs AOBT_3/EOBT/LOBT/IOBT, round-time flags, OOF copy rates per stand/operator/runway) | 359 |
| + LOBT-window clip of predictions | 355 |
| + IEM METAR weather features | 354 |
| + per-airport regressors blended 50/50 with the global one (3 members) | 352 |
| + cross-fitted stacking corrector (`stack.py`, base 352.2 -> 348.0) | 348 |
| + neighbour future/relative delay, delayed-flight copy rate, EUROCONTROL daily series in the corrector | 346.6 |
| + corrector ensemble (global LightGBM, per-airport LightGBM, CatBoost, averaged) | 344.9 |

```bash
pip install pandas pyarrow lightgbm scikit-learn
python taxiout.py --data /path/to/data   # needs training_2025-*.parquet, ranking.parquet, submitting.parquet
```

Data is not included in this repo; see https://prc-data-challenge-2026.netlify.app/data.html.

## Official leaderboard history (RMSE, s)

| File | Official RMSE | Rank |
|---|---|---|
| quirky-honey_v2 | 347.24 | 159 / 233 |
| quirky-honey_v3 | 314.14 | 140 / 233 |
| quirky-honey_v5 | 311.85 | 138 / 233 |
| quirky-honey_v6 | 302.01 | 130 / 233 |
| quirky-honey_v9 (v6 + LIRF no-NM specialist) | 277.91 | 78 / 237 |
| quirky-honey_v10 (v7 + LIRF no-NM specialist) | 277.94 | not best |

(`quirky-honey_v8` is the same file as v9; it was rejected by the scorer for the daily upload limit and has no score.
v7 itself, the v6 pipeline with 5 LightGBM base members, was not uploaded without the specialist.)

The leaderboard was used as a sanity check only, never to choose features or tune rules.

## What worked / what did not
Honest validation: train on 2025 except Jan+Jul, score Jan+Jul 2025, every departure kept (outliers included).

Worked (held-out RMSE, s): single regressor 464 -> copy-of-schedule mixture 420 -> dedicated model for
departures without an NM record 384 -> stand/runway cell stats 381 -> queue features 378 -> neighbour
taxi level 376. The mixture and the NM-missing stage did most of the work.

Did not work / not worth it:
- Dropping long taxi times from training and validation made the score look like 252 s. That was an artifact.
- Ensembling (3 LightGBM + CatBoost): normal flights -1.4 s, total unchanged (379.2 vs 378.4), July slightly worse.
- A per-bucket rule for the one-day-shift outliers (off-block date taken from the schedule): RMSE 378 -> 455.
  The buckets hold ~60 training rows, so the probabilities are noise.
- OPDI flight list (flight_list_202501.parquet, matched by callsign + airport; 98.7% of January departures
  match, 77% get ground time since the same aircraft's previous landing there): on a January-only check
  (train days 1-21, test 22-31) RMSE 197.2 s without vs 197.3 s with the feature. The flight-plan and queue
  features already carry that information, so the other months were not downloaded.
- Regressor variants on normal (non-copy) flights, Jan+Jul 2025, RMSE 234.4 s baseline: residual target around the
  cell median 234.3, lr 0.03 / 255 leaves / 1400 rounds 234.4, arrival taxi-in times near push-back plus
  operator-level recent lateness 234.5 (Jan slightly worse). The model is saturated on the current inputs.
- Other label regimes: ~21% of departures have |BLOCK - AOBT_3| <= 60 s (EOBT/LOBT/IOBT copies: 11-12%). A
  separate AOBT_3-copy mixture is hard to learn (classifier AUC 0.72): normal-flight RMSE 235.0 vs 234.5; a
  50/50 blend with the baseline reached 233.8, which is just an ensemble effect. No repeated default taxi times.
- CatBoost copy classifier averaged into P(copy): AUC 0.887 -> 0.895, but holdout RMSE 359.1 -> 362.5
  (Jan 361.1 -> 363.0, Jul 357.4 -> 362.2), so not kept. Better AUC did not mean a better mixture.
- ADS-B ground traces (adsb.lol, ~2.1 TB for 2025) and OPDI flight events (221 MB per 10 days, airborne milestones
  only as far as documented) were judged infeasible and not downloaded.
- Ideas reviewed from another AI run's code (per-airport LightGBM x10, runway-heading wind components, stand-to-runway
  distance, METAR nearest-obs join): their holdout raw RMSE was 545 s (winter) / 470 s (summer) against our 359 / 348 s on
  Jan/Jul, mainly because they exclude MVT_TIME and AOBT_3 and have no copy-regime or LOBT handling. Only per-airport models
  helped (kept). Wind along the runway heading was neutral (353.8 -> 353.8; off by default via PRC_WIND=1). Stand-to-runway
  distance needs geometry files not available here.
- Shrinking or capping extreme predictions: worse. Scaling them up 1.25x looked better but only through
  luck on a handful of July rows, so it was not kept.

The ~60 departures over 3 h in Jan+Jul (mostly LIRF flights with no NM record) account for ~46% of the
squared error, i.e. ~256 s of RMSE by themselves. Whatever is done to the other flights, the total
cannot go much below that without a way to predict those rows. (The LIRF no-NM specialist below predicts part of them
better; see that section.)

## Submissions
- `quirky-honey_v2.parquet` uploaded 2026-10-07 (PDT). 3 LightGBM members, two-stage copy mixture, NM-missing stage,
  stand/runway cell stats, queue and neighbour features, schedule-copy features. Local holdout RMSE (Jan+Jul 2025,
  outliers kept): 359.6 s. The official score is not used to tune the model.
- `quirky-honey_v2.parquet` official leaderboard RMSE: 347.24 s (rank 159/233 when checked on 2026-10-07; used as a sanity check only).
- `quirky-honey_v3.parquet`: v2 plus the LOBT clip and METAR weather features; local holdout RMSE 353.3 s (3 LightGBM members, Jan+Jul 2025 held out, outliers kept). Official leaderboard RMSE: 314.14 s (rank 140/233 on 2026-10-07).
- `quirky-honey_v5.parquet`: v3 plus the copy-impossible rule and per-airport blend; local holdout RMSE 352.1 s (Jan 357.4, Jul 347.8). Official leaderboard RMSE: 311.85 s (rank 138/233 on 2026-10-07). (v4, the same without the per-airport blend, was not uploaded.)

## Stacking corrector (`stack.py`)
`python stack.py validate --data DIR [--variants] [--ensemble]` and `python stack.py submit --data DIR` implement a
second stage on top of the `taxiout.py` base model (copy mixture, per-airport blend, copy-impossible rule, NM-missing stage).

- Out-of-block base predictions: the training months are split into 2-month blocks and the base is retrained without the
  block it predicts, so the corrector never sees the base's own training error. Validation uses 5 blocks over the 10
  non-Jan/Jul months, held-out Jan/Jul predicted by a base trained on those 10 months; the submission uses 6 blocks over
  all 12 months and a base trained on all of 2025 for the ranking rows.
- The corrector learns the clipped residual (`y - base`) on normal flights (taxi time up to 3 h) whose base prediction is not a
  tail bet (<= 7200 s). Tail rows keep the base prediction. Inputs: all base features, the base prediction, P(copy), the normal
  regressor, distances to both LOBT window edges, plus the extra groups below. The result is projected back onto the LOBT window.
- Extra inputs, each measured on both months: neighbour delay in the next 20/60 min (future windows) and the flight's own
  proxy minus its neighbours'; copy rate among delayed flights (takeoff > 1 h after SCHED) per airport/operator/NM-or-not,
  smoothed toward the airport rate and computed without the row's own month; EUROCONTROL daily airport series
  (regulated and off-slot share, pre-departure delay per flight; see `docs/external_data.md`).
- Held-out Jan+Jul 2025, outliers kept: base 352.2 (Jan 357.3, Jul 348.1); corrector 348.0 (354.2, 342.8); + extra inputs
  346.6 (353.6, 340.8); average of global LightGBM, per-airport LightGBM and CatBoost correctors 344.9 (352.1, 339.0).
- Ideas credited to the public repository EnioAguiar/prc-taxiout-2026 (GPLv3), re-implemented here from their written
  description and not copied: cross-fitted corrector, delayed-flight copy rate, future/relative neighbour windows, averaging
  global/per-airport/CatBoost models.
- Not adopted: a hand-fitted Rome rule for no-NM LIRF flights 15-30 h late (looks large on the 2025 holdout, 356.6 -> 333.2,
  but rests on 7 holdout and 8 training rows and would touch 4 ranking flights); ADS-B (about 2.1 TB for 2025, no ground
  coverage at the airports where error is largest).
- `quirky-honey_v6.parquet`: v5 plus the cross-fitted stacking corrector (`stack.py submit`), local holdout 344.9 s (Jan 352.1, Jul 339.0). Official leaderboard RMSE: 302.01 s (rank 130/233 on 2026-10-07).

## LIRF no-NM specialist (`specialist.py`)
`python specialist.py validate --data DIR [--scope lirf|other|all]` and
`python specialist.py apply --data DIR --base quirky-honey_v6.parquet --out OUT.parquet`.

- Scope: departures at LIRF without a Network Manager flight record (`FLIGHT_ID_mvt` missing): about 1,490 rows in 2025,
  397 in the Jan+Jul holdout, 383 in the ranking set. They hold most multi-hour and one-day-shift labels, and our base
  RMSE on them was 5640 s (29.6% of the holdout squared error).
- Model: CatBoost (depth 5, 1000 trees, learning rate 0.04, L2 10, mean of 5 seeds) fitted on those rows only, with the
  target `y - max(0, MVT - SCHED)`; prediction is `max(0, MVT - SCHED)` plus the learned residual, floored at 30 s. The
  hyperparameters were taken from the public description and not tuned here.
- Sanity cap at `max(gap, 88,500 s)`: in 2025 training a no-NM LIRF label is either close to the schedule gap or in the
  one-day-shift cluster (largest label 88,392 s), so a larger prediction is not supported by the data. It touches a few rows
  and was not part of the leave-one-month-out run below.
- Validation, leave-one-month-out over all 12 months of 2025 (the model for a month never sees that month): better than the
  base in 12 of 12 months; scope RMSE on Jan+Jul 5640 -> 3555 (all 12 months 5572 -> 4001). On the whole Jan+Jul holdout
  the base-only RMSE goes 352.2 -> 319.3 (Jan 357.3 -> 346.2, Jul 348.1 -> 295.9). These are local numbers; they are not
  an official score.
- Not adopted: the same specialist on no-NM rows at the other airports is worse (RMSE 1153 -> 1748 over 12 months; whole
  holdout 352.2 -> 373.1), so it is limited to LIRF. Applying it to every no-NM row gave 335.1, worse than LIRF only.
- The idea (a separate specialist for unmatched LIRF rows on a MVT-SCHED baseline) comes from the public repository
  Phoenix-Ops-LTD/prc2026-taxiout (GPLv3, team zestful-fountain), found by comparing their reported holdout error scopes with
  ours. It is re-implemented here independently on our features, not copied.
- `quirky-honey_v8.parquet` (v6 with the 383 LIRF no-NM ranking rows replaced) was rejected by the scorer with
  `DAILY_LIMIT_REACHED` (5 of 5 uploads used that day), so it has no official score. The identical file was uploaded as
  `quirky-honey_v9.parquet` after the quota reset.
- Official results (2026-10-08): `quirky-honey_v9.parquet` 277.91 s (rank 78/237), 24.1 s better than v6's 302.01 s. The
  official gain is close to the top of what the local numbers suggested (352.2 -> 319.3 on the base-only holdout, about 10%
  lower officially than locally).
- `quirky-honey_v10.parquet` is the same specialist on top of v7 (the v6 pipeline with 5 LightGBM base members instead of 1,
  `stack.py submit --seeds 5`): 277.94 s, 0.03 s worse than v9. More base members are not worth the extra 2.5 h of compute.
- The leaderboard was used as a sanity check only; the specialist was chosen from the leave-one-month-out results above.
