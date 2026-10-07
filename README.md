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
- Validation: train on 2025 months other than Jan and Jul, score RMSE on Jan + Jul 2025,
  outliers kept (dropping them makes the score look much better than it is).

| Model | RMSE (s) |
|---|---|
| Mean prediction | 687 |
| Single regressor | 464 |
| Two-stage mixture | 420 |
| + NM-missing stage | 384 |
| + stand/runway cell statistics | 381 |

```bash
pip install pandas pyarrow lightgbm scikit-learn
python taxiout.py --data /path/to/data   # needs training_2025-*.parquet, ranking.parquet, submitting.parquet
```

Data is not included in this repo; see https://prc-data-challenge-2026.netlify.app/data.html.
