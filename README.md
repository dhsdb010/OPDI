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

```bash
pip install pandas pyarrow lightgbm scikit-learn
python taxiout.py --data /path/to/data   # needs training_2025-*.parquet, ranking.parquet, submitting.parquet
```

Data is not included in this repo; see https://prc-data-challenge-2026.netlify.app/data.html.

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
cannot go much below that without a way to predict those rows.

## Submissions
- `quirky-honey_v2.parquet` uploaded 2026-10-07 (PDT). 3 LightGBM members, two-stage copy mixture, NM-missing stage,
  stand/runway cell stats, queue and neighbour features, schedule-copy features. Local holdout RMSE (Jan+Jul 2025,
  outliers kept): 359.6 s. The official score is not used to tune the model.
- `quirky-honey_v2.parquet` official leaderboard RMSE: 347.24 s (rank 159/233 when checked on 2026-10-07; used as a sanity check only).
- `quirky-honey_v3.parquet`: v2 plus the LOBT clip and METAR weather features; local holdout RMSE 353.3 s (3 LightGBM members, Jan+Jul 2025 held out, outliers kept). Official leaderboard RMSE: 314.14 s (rank 140/233 on 2026-10-07).
- `quirky-honey_v5.parquet`: v3 plus the copy-impossible rule and per-airport blend; local holdout RMSE 352.1 s (Jan 357.4, Jul 347.8). Official leaderboard RMSE: 311.85 s (rank 138/233 on 2026-10-07). (v4, the same without the per-airport blend, was not uploaded.)
