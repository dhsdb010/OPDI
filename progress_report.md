# quirky-honey: progress report (PRC Data Challenge 2026)

Status on 2026-10-08 (updated after v12). Deadline: 11 Oct 2026, 23:59:59 CET. Metric: RMSE of departure taxi-out time over the
January and July 2026 ranking set (344,841 departures).

## 1. Where we stand

| | Value |
|---|---|
| Best official score | **269.33 s** (`quirky-honey_v12`) |
| Rank | **62 / 238** (was 130 with v6) |
| Leader / top-10 cutoff | 213.3 s / 222.9 s (checked 2026-10-08 16:20 UTC) |
| Goal | at most 270 s officially: met by v12; 250 s was a stretch and is not expected |

## 2. Official leaderboard history

| File | What changed | Official RMSE | Rank |
|---|---|---|---|
| v2 | Copy-of-schedule mixture, NM-missing stage, queue / stand / neighbour features | 347.24 | 159 / 233 |
| v3 | + LOBT window clip, METAR weather | 314.14 | 140 / 233 |
| v5 | + per-airport blend, copy-impossible rule | 311.85 | 138 / 233 |
| v6 | + cross-fitted stacking corrector | 302.01 | 130 / 233 |
| v8 | v6 + LIRF no-NM specialist | rejected: daily upload limit (5/5) | – |
| v9 | same file as v8, uploaded after the reset | 277.91 | 78 / 237 |
| v10 | v7 (5 base members) + specialist | 277.94 | – |
| v11 | v9 + neighbour-delay inputs in the corrector | 276.30 | 74 / 237 |
| **v12** | v11 + ADS-B stage (adsb.lol off-block events at 5 airports) | **269.33** | **62 / 238** |

The leaderboard was only used as a sanity check, never to choose features or tune rules.

## 3. Local validation (train on 2025 without Jan/Jul, score Jan + Jul 2025, every departure kept)

| Stage | All | Jan | Jul |
|---|---|---|---|
| Base model (out-of-block run) | 352.2 | 357.3 | 348.1 |
| + stacking corrector, 3-model average (v6) | 344.9 | 352.1 | 339.0 |
| Base + LIRF no-NM specialist (without corrector) | 319.3 | 346.2 | 295.9 |
| Corrector + neighbour-delay features, 3-model average (v11 corrector) | 343.7 | 351.4 | 337.4 |

Local and official scores relate fairly consistently: about 0.87 × local (v6: 344.9 → 302.0, specialist base 319.3 → v9 277.9).

## 4. Key analysis

### 4.1 Where the error comes from
- **A tiny number of rows dominate.** Squared error is driven by departures without a Network Manager (NM) flight
  record. On the Jan + Jul 2025 holdout, before the specialist, 5,366 no-NM rows (1.6%) held 58.9% of the squared error.
- **LIRF no-NM rows** (397 in the holdout, 383 in the ranking set): RMSE 5,640 s, 29.6% of all squared error. They mix
  schedule-copy stamps (taxi = MVT − SCHED, up to 30 h) with one-day-shift labels near 87,000 s.
- **LFPG, January:** two rows with labels of 84,240 s and 58,206 s while takeoff was only ~30 min after schedule.
  They are one-day-shift label errors. Together they make up ~29% of the holdout squared error after the specialist.
  Without them the local RMSE would be about 268 instead of 319.
- **Such unexplained huge labels are very rare and not predictable:** in all of 2025 (2.08 M departures) only 15 rows
  have a label above 30,000 s that a schedule copy does not explain: 12 at LIRF, 2 at LFPG, 1 at LSZH, all without an
  NM record. At LFPG that is 0.05% of no-NM rows, too few to learn. We do not try to fit them.

### 4.2 What the competitor review found
- Public GPLv3 repositories were read through the GitHub API (READMEs and method notes only, nothing cloned).
- **Phoenix-Ops-LTD/prc2026-taxiout** (team zestful-fountain, 273.6 official) reports local numbers on the same
  344,419 holdout rows. Their gap to us was almost entirely the LIRF no-NM scope (RMSE ~3,470 vs our 5,640). That led to
  the specialist.
- sergiuv11's 286.7 was scored on 215,876 rows, an earlier ranking set, so it is not comparable.

### 4.3 The LIRF no-NM specialist (`specialist.py`, credited to Phoenix-Ops-LTD's idea)
- CatBoost (depth 5, 1000 trees, lr 0.04, L2 10, mean of 5 seeds) trained only on LIRF no-NM rows. Target: residual
  over `max(0, MVT − SCHED)`. Cap at `max(gap, 88,500 s)` (largest one-day-shift label in training is 88,392 s).
- Leave-one-month-out over all 12 months of 2025: better than the base in **12 of 12 months**; scope RMSE on Jan + Jul
  5,640 → 3,555.
- Official effect: **302.01 → 277.91 (−24.1 s)**.
- Tuning check (7 predeclared variants, keep only if better in all 12 months and in Jan and Jul): none passed. Depth 3 / 7,
  500 trees, a 50/50 blend with the base and training on all airports were all worse or mixed. Removing the cap was worse
  (4,001 vs 3,972), which confirms the cap. The specialist on no-NM rows at other airports made things worse (1,153 → 1,748),
  so it stays LIRF-only.

### 4.4 Normal flights (step 2)
Three new groups were tested on top of the v6 corrector; each uses only fields present in the ranking file:

| Group | All | Jan | Jul | Normal flights | Decision |
|---|---|---|---|---|---|
| Reference (v6 corrector, global LightGBM) | 346.59 | 353.65 | 340.79 | 259.40 | – |
| G1 neighbour delay (lateness of flights within ±15/30/60 min) | 345.51 | 353.24 | 339.16 | 257.93 | **kept** |
| G2 airport-day state | 346.47 | 353.53 | 340.68 | 259.23 | dropped (no effect) |
| G3 neighbour spread | 346.56 | 353.73 | 340.67 | 259.34 | dropped (Jan worse) |

With the full 3-model corrector: 344.9 → 343.7 (Jan 352.1 → 351.4, Jul 339.0 → 337.4). Both months improve.

### 4.5 Things that did not work (this round)
- 5 base members instead of 1 (v10): 0.03 s worse than v9 officially; not worth the 2.5 h of compute.
- The other AI's v7–v15 report: its "summer" window (24–31 Jul 2025) lies inside its training period (`train < 2025-12-24`),
  so those numbers are in-sample. On its only out-of-sample window (24–31 Dec) every change after its v7 was flat or
  worse. Their finding that LIRF no-NM rows cause the summer error agrees with ours. The neighbour-feature builder was
  not in the shared code, so possible leakage there was not checked.

## 5. ADS-B stage (v12)

- adsb.lol daily archives for all of January and July of 2025 and 2026 were streamed and cut to airport boxes (124 days, 1.5 GB kept).
  Off-block events per departure are matched to movement records and fed to a LightGBM correction of the v11 prediction.
- Signal: at EDDM, LSZH, LEBL and EHAM an aircraft seen parked gives an off-block time within 60 s of the official one 50-80% of
  the time (NM off-block: 18-29%). Only 7-25% of flights are seen parked; LTFM, LFPG, LEMD and EGLL have almost no coverage.
- Validation on all of Jan+Jul 2025 (leave-one-week-out, rule fixed in advance: must beat v11 and a no-ADS-B control in both months):
  v11 312.9 -> control 312.5 -> with ADS-B 308.0 (Jan 341.8 -> 335.7, Jul 287.5 -> 283.7). Cross-month: 312.9 -> 311.4.
- Coverage guard: the share of flights with an event differs strongly between 2025 and 2026 at EGLL (14% -> 80%), LEMD (2% -> 64%),
  LIRF and others, so ADS-B inputs are used only at the five airports with stable coverage. Cost: 0.6 s of local gain.
- Official: **269.33** (-7.0 s vs v11, more than the 4.9 s local gain).

## 6. Tests that did not pay off (this round)

- Specialist tuning (7 variants, leave-one-month-out): none better in all 12 months.
- Corrector inputs (arrival taxi-in, NM time consistency, within-hour rank, stand prefix, ML unimpeded taxi time, ATFM-style cell):
  at most -0.14 s, or one month better and the other worse. The corrector is saturated on this kind of input.
- Normal-flight error is spread evenly over airports and well calibrated; a perfect copy classifier would be worth at most ~8 s locally.
- Two label-noise rows (one-day-shift labels at LFPG, January 2025) make up ~29% of the local squared error and cannot be predicted.

- Stage variants (neighbour ADS-B inputs, 150/600 rounds, 5 seeds) and a log-target blend for the stage-2 regressor: none worth a new
  file (best case 0.1-0.2 s, or one month worse). An attempt to add the other ten months of ADS-B was abandoned because the
  connection fell to 1-2 MB/s.

## 7. Next steps

1. Review the other AI agent's v17 file for ideas only (it is not uploaded): compare tail handling and normal-flight differences with ours.
2. Possible follow-ups: ADS-B events for the remaining months of 2025 would add training data for the stage but need roughly 6-7 more hours of download
   and out-of-sample v11 predictions for those months, which we do not have; not planned.
3. Final submission is v12 unless something better passes the same both-months rule before 11 Oct 2026 23:59:59 CET.
