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
- Shrinking or capping extreme predictions: worse. Scaling them up 1.25x looked better but only through
  luck on a handful of July rows, so it was not kept.

The ~60 departures over 3 h in Jan+Jul (mostly LIRF flights with no NM record) account for ~46% of the
squared error, i.e. ~256 s of RMSE by themselves. Whatever is done to the other flights, the total
cannot go much below that without a way to predict those rows.
