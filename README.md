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
pip install -r requirements.txt
python taxiout.py --data /path/to/data   # base model only; needs training_2025-*.parquet, ranking.parquet, submitting.parquet
```

The submitted files use more than the base model: see "Reproducing the submitted file (v12)" below.

Data is not included in this repo; see https://prc-data-challenge-2026.netlify.app/data.html.

## Reproducing the submitted file (v12)
`quirky-honey_v12.parquet` is built in stages; every stage is a command in this repository. `DIR` holds the organisers' files
(`training_2025-*.parquet`, `ranking.parquet`, `submitting.parquet`); `PRC_CACHE` is a folder for intermediate arrays (default `/tmp`).
Timings are for an 8-core laptop. No 2026 label is used anywhere. Not bit-exact between runs: multi-threaded gradient boosting
does not reproduce to the last digit, so expect differences of a fraction of a second in RMSE.

| Step | Command | What it makes | Time |
|---|---|---|---|
| 0 | download METAR and the EUROCONTROL daily series as described in `docs/external_data.md` | `opdi/metar/`, `opdi/eurocontrol/` | minutes |
| 1 | `export PRC_CACHE=/some/folder; python stack.py validate --data DIR` | out-of-block base predictions for Jan+Jul 2025 (`stack_val_base.npz`) | ~40 min |
| 2 | `python stack.py submit --data DIR --out v11_corrected.parquet` | base model + cross-fitted corrector for the ranking rows (`stack_submit_base.npz`) | ~45 min |
| 3 | `python specialist.py apply --data DIR --base v11_corrected.parquet --out quirky-honey_v11.parquet --seeds 5` | v11: LIRF no-NM specialist on top (official 276.30) | ~5 min |
| 4 | `python holdout.py --data DIR --out holdout.parquet` | out-of-sample v11-equivalent predictions for Jan+Jul 2025 | ~20 min |
| 5 | `python adsb.py fetch --days 2025-01,2025-07,2026-01,2026-07 --procs 8` | ADS-B cuts, `opdi/adsb/cut/` (about 1.5 GB kept, 124 days streamed) | ~3 h |
| 6 | `python adsb_stage.py events --data DIR` | per-day off-block events, `opdi/adsb/feat/` | ~2 min |
| 7 | `python adsb_stage.py validate --data DIR --holdout holdout.parquet` | the test that decides whether the stage is kept | ~10 min |
| 8 | `python adsb_stage.py apply --data DIR --holdout holdout.parquet --base quirky-honey_v11.parquet --out quirky-honey_v12.parquet` | v12 (official 269.33) | ~10 min |

`specialist.py validate` and `stack.py validate --variants --ensemble` print the validation tables quoted in this README.

## Final phase (Jan, Feb, Jun, Jul 2026)
The organisers added a single blind final submission: `final_ranking.parquet` (670,790 departures of Jan, Feb, Jun and Jul 2026, same columns and
airports as `ranking.parquet`; take-off and arrival times carry a deliberate +-12 s dither, so the old predictions are not reused) and the template
`final_submitting.parquet`. It is ranked on Jan/Jul, on Feb/Jun and on all four months. The model design is unchanged; only the file names are
settings. The organisers stated on their Discord (26 Sep) that open data sources may be used to devise a better model, and that feature
engineering with the provided data is permitted.

Run on the final files (`PRC_RANKING` and `PRC_TEMPLATE` replace `ranking.parquet` and `submitting.parquet` everywhere):

```bash
export PRC_CACHE=/some/folder PRC_RANKING=final_ranking.parquet PRC_TEMPLATE=final_submitting.parquet
python stack.py submit --data DIR --out final_corrected.parquet    # reuses the out-of-block training predictions of an earlier default run if present
python specialist.py apply --data DIR --base final_corrected.parquet --out quirky-honey_final_v11.parquet --seeds 5   # fallback file, no ADS-B
python adsb.py fetch --days 2026-01,2026-02,2026-06,2026-07,2025-01,2025-02,2025-03,2025-04,2025-05,2025-06,2025-07,2025-08,2025-09,2025-10,2025-11,2025-12 --procs 8   # ADS-B cuts
PRC_RANKING=final_ranking.parquet PRC_FEAT_DIR=opdi/adsb/feat_final python adsb_stage.py events --data DIR
python holdout.py --data DIR --all-months --out oos_all.parquet      # out-of-sample v11-style predictions for all twelve months of 2025
PRC_FEAT_DIR=opdi/adsb/feat_final python adsb_stage.py validate-all --data DIR --holdout oos_all.parquet   # prints KEEP or DO NOT KEEP
PRC_FEAT_DIR=opdi/adsb/feat_final python adsb_stage.py apply --data DIR --holdout oos_all.parquet --base quirky-honey_final_v11.parquet --out quirky-honey_final_v12_allmonths.parquet
```

`python holdout.py --all-months` writes out-of-sample predictions for all twelve months of 2025 (the stage's training set), and
`python adsb_stage.py validate-all` is the check that decides whether the stage may be trained on all months instead of Jan+Jul only (kept only if it
beats the Jan+Jul stage in both Jan/Jul and Feb/Jun).

- `quirky-honey_final_v11.parquet` (fallback): the v11 pipeline run on the final file plus the LIRF no-NM specialist. Template checks pass (670,790 rows,
  IDs, order, dtypes, no NaN or infinity).
- `quirky-honey_final_v12.parquet` (primary candidate): the same plus the ADS-B stage trained on the Jan+Jul 2025 out-of-sample predictions. It changes the
  fallback by 63 s RMS (mean +0.4 s). Coverage of matched ADS-B flights is steady at EDDM, EHAM, LEBL and LSZH in all four months; EDDF had a gap in
  February 2026 (8% matched against 50-61% in the other months), where the stage simply has no ADS-B input.
- Does the stage transfer to the new seasons? A stage trained on Jan+Jul 2025 only was tested on the days of Feb to May 2025 that were downloaded at the
  time. It beat both the v11-style prediction and a no-ADS-B control in every month: Feb 224.3 -> 219.8 (control) -> 214.5, Mar 192.3 -> 195.1 -> 188.2,
  Apr 201.6 -> 203.3 -> 196.3, May 242.4 -> 244.2 -> 237.4, Jun 193.9 -> 194.4 -> 193.1 (only about a week of June at that time).
- Local proxy for the final (out-of-sample, every flight kept, all four months of 2025): v11-style 302.8 (Jan 344.5, Feb 261.0, Jun 306.1, Jul 292.9);
  the stage is worth about 5 s on top. Official scores of the Jan/Jul months were about 0.87 times the local figures. In January 2025 five rows alone move
  the RMSE from 217 to 344 (one-day-shift labels), so the final ranking will depend heavily on how those rows fall in the new months.
- `quirky-honey_final_v12_allmonths.parquet`: the same ADS-B stage trained on out-of-sample predictions for all twelve months of 2025 (2,083,124 flights; ADS-B
  cuts for all 365 days of 2025 and the four 2026 ranking months) instead of Jan+Jul only. `adsb_stage.py validate-all` compares, on the four months
  Jan, Feb, Jun, Jul 2025 with every flight kept and the stage never seeing the month it is scored on (leave-one-week-out):

  | | v11-style | Jan+Jul stage | all-months stage |
  |---|---|---|---|
  | Jan + Jul (344,419 flights) | 317.0 | 312.2 | 312.0 |
  | Feb + Jun (326,873 flights) | 287.1 | 279.4 | 278.0 |
  | all four months (671,292 flights) | 302.8 | 296.7 | 295.9 |

  It beats the Jan+Jul stage in both month groups, so it is kept (the rule set in advance), but by 0.2 s and 1.4 s only. Template checks pass (670,790 rows,
  IDs, order, dtypes, no NaN or infinity). It differs from `quirky-honey_final_v12.parquet` by 48 s RMS (both differ from the fallback by 63-64 s).

- `quirky-honey_final_stage600.parquet`: the same all-months stage with more capacity (600 rounds, 63 leaves instead of 300 and 31), since the training set is now
  6 times larger than the one the old size was chosen on. Same leave-one-week-out protocol, four variants tried (all flights kept, Jan+Jul / Feb+Jun / all four):
  300 rounds, 31 leaves 311.98 / 277.99 / 295.92; 600 rounds, 63 leaves 311.87 / 276.30 / 295.08 (kept: better in both groups, but nearly all of the gain is
  February, 245.9 -> 241.6, and Jan+Jul moves by 0.1 s); 800 rounds, learning rate 0.03, 127 leaves 311.78 / 276.55 / 295.15 (also better in both, not chosen, no
  simpler); adding hour of day and weekday 312.05 / 278.90 / 296.37 (worse in both groups, not kept; with the larger model 311.98 / 278.14 / 295.99, not kept either).
  Template checks pass; it differs from `quirky-honey_final_v12_allmonths.parquet` by 16 s RMS.
- Submitted as `quirky-honey_final.parquet`: `quirky-honey_final_stage600.parquet`. `quirky-honey_final_v12_allmonths.parquet`, `quirky-honey_final_v12.parquet`
  (Jan+Jul stage) and `quirky-honey_final_v11.parquet` (no ADS-B) are the fallbacks and were not submitted.


## Official leaderboard history (RMSE, s)

| File | Official RMSE | Rank |
|---|---|---|
| quirky-honey_v2 | 347.24 | 159 / 233 |
| quirky-honey_v3 | 314.14 | 140 / 233 |
| quirky-honey_v5 | 311.85 | 138 / 233 |
| quirky-honey_v6 | 302.01 | 130 / 233 |
| quirky-honey_v9 (v6 + LIRF no-NM specialist) | 277.91 | 78 / 237 |
| quirky-honey_v10 (v7 + LIRF no-NM specialist) | 277.94 | not best |
| quirky-honey_v11 (v9 + neighbour-delay inputs in the corrector) | 276.30 | 74 / 237 |
| quirky-honey_v12 (v11 + ADS-B stage) | 269.33 | 62 / 238 |
| quirky-honey_v13 (alternative pipeline, not used) | 303.14 | not best |

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
- OPDI flight events (221 MB per 10 days, airborne milestones only as far as documented) were judged not worth downloading.
  (ADS-B ground traces were first judged infeasible too; a streamed, cut download of only Jan+Jul turned out to be possible
  and useful, see the ADS-B stage section.)
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
  but rests on 7 holdout and 8 training rows and would touch 4 ranking flights).
- `quirky-honey_v6.parquet`: v5 plus the cross-fitted stacking corrector (`stack.py submit`), local holdout 344.9 s (Jan 352.1, Jul 339.0). Official leaderboard RMSE: 302.01 s (rank 130/233 on 2026-10-07).
- Neighbour-delay inputs (`_neighbour_delay`, added for v11): lateness (takeoff - SCHED, clipped to [-1 h, 4 h]) of the
  other departures at the airport taking off within +-15/30/60 min, the +-30 min spread, and the flight's own lateness
  minus the +-60 min mean. No labels are used. Global corrector 346.59 -> 345.51 (Jan 353.65 -> 353.24, Jul 340.79 -> 339.16);
  3-model average 344.9 -> 343.7 (Jan 352.1 -> 351.4, Jul 339.0 -> 337.4). Tested and dropped: airport-day means of the
  taxi proxy and lateness (346.47, no real change) and the +-30 min spread of the taxi proxy (346.56, Jan worse).
- `quirky-honey_v11.parquet`: v6 pipeline with the neighbour-delay inputs plus the LIRF no-NM specialist. Official
  leaderboard RMSE 276.30 s (rank 74/237 on 2026-10-08), 1.6 s better than v9.

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

## ADS-B stage (`adsb.py`, `adsb_stage.py`)
`python adsb.py fetch --procs 8` streams the adsb.lol `globe_history` daily archives (ODbL 1.0) for every day of January and July of
2025 and 2026 and keeps only the points within about 11 km of the ten airports that are on the ground or below 3,000 ft
(about 1.5 GB for 124 days; the ~3 GB daily archives are never stored). `python adsb_stage.py events|validate|apply` derives
off-block events and applies the stage; see `docs/external_data.md` for the source and licence.

- Events: a takeoff is a ground point followed by an airborne point of the same aircraft within 120 s; the off-block event is
  the first point of the ground segment that ends there (segments break at gaps over 600 s). Flights are matched to movement
  records by callsign within 300 s of takeoff, otherwise by the nearest unused takeoff within 90 s. Inputs: time from first
  ground point and from first movement to takeoff, speed at the first point, "seen parked", takeoff error, number of points,
  largest gap, and differences to the v11 prediction and to the NM off-block proxy.
- Signal: at EDDM, LSZH, LEBL and EHAM an aircraft seen parked before push-back gives an off-block time within 60 s of the
  official one 50-80% of the time, against 18-29% for the NM off-block time (AOBT_3). Only 7-25% of flights there are seen
  parked, and ADS-B is almost absent at LTFM, LFPG, LEMD and EGLL. RMSE of the raw ADS-B estimate is sometimes worse than NM
  because of a few bad matches, so it is only used as an input, never as a replacement.
- Stage: a LightGBM correction of the v11 prediction (clipped residual, 300 rounds, 3 seeds in the final fit), trained on the
  out-of-sample v11 predictions for Jan+Jul 2025 (344k flights), applied to predictions up to 7,200 s outside the LIRF no-NM
  specialist scope. A control stage with the same inputs minus the ADS-B ones is trained alongside.
- Coverage guard: the share of flights with a matched event differs a lot between the 2025 training months and the 2026
  ranking months (EGLL 14% -> 80%, LEMD 2% -> 64%, LIRF 44% -> 24%, EDDF 71% -> 55%, EDDM 58% -> 92%). The stage could not know how
  far to trust a trace where the training months hardly contain any, so the ADS-B inputs are used only at EDDF, EDDM, EHAM, LEBL
  and LSZH, and set to missing at the other airports. This was chosen from coverage statistics, before looking at scores, and
  cost 0.6 s of validated gain.
- Validation (all of Jan+Jul 2025, outliers kept, decision rule fixed in advance: keep only if better than both the v11
  prediction and the control in both months and overall). Leave-one-week-out: v11 312.9 (Jan 341.8, Jul 287.5), control 312.5
  (340.2, 288.3), with ADS-B 308.0 (335.7, 283.7); normal flights at the five airports 187.3 -> 170.7. Cross-month (train
  on the other month, stricter): 312.9 -> 311.4 (Jan 341.8 -> 340.7, Jul 287.5 -> 285.7). An earlier 10-day test with a
  leave-one-day-out design had made the control look useful (-3.3 s); with whole weeks held out the control does nothing.
- `quirky-honey_v12.parquet`: v11 plus the stage. Official leaderboard RMSE 269.33 s (rank 62/238 on 2026-10-08), 7.0 s better
  than v11, larger than the 4.9 s local gain.
- Idea credited to the public repository EnioAguiar/prc-taxiout-2026 (GPLv3): adsb.lol traces cut to airport boxes, off-block
  events per takeoff, and a stacking stage on top. Re-implemented here from their written description; no code copied.
- Coverage probe for the other months: three sample days of late 2025 (15 Oct, 15 Nov, 15 Dec) were checked to see whether training
  on more months could unlock EGLL, LEMD or LFPG. It cannot: at EGLL and LEMD the matched share swings from day to day (EGLL 65% on
  15 Nov, 11% on 15 Dec) but few flights are seen parked and almost none of those are within 60 s of the official off-block time
  (EGLL 0%, LEMD 5-25%), because the traces mostly start after push-back. LFPG and LTFM have essentially no coverage.
- Robustness: a corrupted compressed trace file inside a daily archive (seen on 15 Oct 2025) used to crash the whole day; the cutter
  now skips that trace and loses only that aircraft. None of the 124 days used for v12 hit this.
- Stage variants (leave-one-week-out on all of Jan+Jul 2025, against the stage used in v12: 308.08, Jan 335.75, Jul 283.81; keep only
  if both months improve): neighbour ADS-B inputs (mean ADS-B minus NM off-block estimate and share seen parked among the other
  flights within +-30/60 min at the airport) 307.89 but January 335.79, not kept; 150 rounds 308.17 and 600 rounds 308.30, not kept
  (300 was the right size for the 343,917-flight training set of v12; with all twelve months the larger setting wins, see "Final phase"); 5 seeds instead of 3 in the final fit 307.98, both months better but only by 0.1 s, not worth a new file.
- Log-target blend for the stage-2 regressor on non-copy flights (same features and rows, Jan+Jul 2025 holdout, normal flights):
  raw target 233.72, log target with smearing 238.28, best blend (25% log) 233.51 with January better and July worse. The
  predictions correlate at 0.988, so the target change adds little diversity; not kept, and not carried into the full base model.
- More ADS-B months: downloading the other ten months of 2025 (about 300 days, ~3 GB streamed per day) was first started and abandoned
  on a shared network (inbound speed fell from about 30 MB/s to 1-2 MB/s). It was resumed after the final-phase announcement, with
  Feb to Jun fetched locally and Aug to Dec on a server owned by the team, and is now complete (all 365 days of 2025); see "Final phase".
  Three days needed special handling: the 2025-10-15 "prod" release is truncated (its second part does not continue the archive), so the
  "staging" copy of the same day is used; 2025-12-31 was published in the `globe_history_2026` repository; and 2025-09-26 stalled once on a
  dead connection. `adsb.py fetch` now tries every published copy of a day in turn and drains the stream so a damaged archive cannot hang a worker.
- `quirky-honey_v13.parquet`: an alternative, independently developed pipeline (two-stage copy mixture with a 50/50 blend of a raw-target
  and a log-target regressor, no ADS-B, no LIRF specialist), uploaded once as a test candidate. Official leaderboard RMSE 303.14 s,
  33.8 s worse than v12 and about equal to our v6. Its own report had estimated about 274 s from a local validation whose gain was
  concentrated in one month (February, -32.5 s of a -9.7 s four-month average); our own test of the same log-target blend inside the
  stage-2 regressor gave 0.2 s at best. Not used for the final phase.
- Callsign-recurrence features for the LIRF no-NM specialist (tested 2026-10-09). The rows with unexplained huge labels (15 in 2025, all without an
  NM record) are not copies of the previous day's block of the same flight number (the previous record of that flight number is usually days or weeks
  earlier and its block time does not match). They are mostly one-off flight numbers: among no-NM flights at LIRF the rate is 1.24% (10 of 810) when the
  flight number does not also leave the airport about a day before or after, against 0.30% (2 of 677) when it does; at the other airports 3 of 7,736
  one-off no-NM rows and 0 of 13,201 recurring ones (about 0.04%), too rare for a hedge to matter (about 0.002 s). Adding "same flight number 18-30 h
  before / after" and the hours since the previous same-flight-number departure to the specialist, leave-one-month-out with 6 seeds: scope RMSE over the
  twelve months 3968 -> 3883, better in 11 of 12 months (June worse, 6767 -> 6838). On the four final months with every flight kept: Jan 344.5 -> 343.5,
  Feb 260.9 -> 260.9, Jun 306.3 -> 307.9, Jul 292.2 -> 289.7, i.e. Jan+Jul 316.6 -> 314.8, Feb+Jun 287.3 -> 288.2, four months 302.7 -> 302.2. Not kept:
  the gain is 0.5 s because the error sits in a few rows no flag can single out, and Feb+Jun does not improve.
- The January 2025 RMSE of 344.5 (217.0 without its five worst rows) is dominated by two LFPG flights without an NM record whose recorded off-block time is
  about 23 h and 16 h before takeoff (labels 84,240 s and 58,206 s) while their schedule gap is under 35 min; nothing observable at prediction time marks them.
- Tried and not kept on the corrector (all on top of the v11 corrector, both months required to improve): arrival taxi-in near
  push-back (-0.14 s, within noise), NM time-consistency differences, within-hour rank of the taxi proxy, stand-prefix groups,
  an ML-fitted unimpeded taxi time and an ATFM-style airport x 2-hour x weekday cell (January better, July worse).
