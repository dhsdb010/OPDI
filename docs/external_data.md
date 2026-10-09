# External datasets

The PRC Data Challenge 2026 prize rules require that all external datasets are openly accessible and documented.
This file lists every dataset used by `taxiout.py` or evaluated while developing it, what it is used for, where it
comes from, and what its licence or attribution statement says. The organisers' own files (`training_*.parquet`,
`ranking.parquet`, `submitting.parquet`) are distributed through the OpenSky MinIO bucket with per-team credentials and are
not redistributed here. No 2026 ground-truth label is used by any model.

## Used by the model

| Dataset | Used for | Source | Licence / attribution as stated by the source | Local path (git-ignored) |
|---|---|---|---|---|
| METAR / SPECI observations, 10 airports (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LTFM, LSZH), 2025-01-01 to 2026-07-31 | Weather features: wind, visibility, ceiling, temperature trend, freezing / snow / precipitation / fog flags. The latest observation at least 10 minutes before takeoff is joined to each movement. | Iowa Environmental Mesonet (Iowa State University) ASOS archive, `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py` | Public observations (NOAA/ASOS network) served by IEM. IEM asks for attribution to the Iowa Environmental Mesonet. Cite: Iowa State University, Iowa Environmental Mesonet. | `opdi/metar/<ICAO>.csv` (~31 MB) |

Download used (one request per airport):

```
curl -o opdi/metar/EDDF.csv "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=EDDF&data=tmpf&data=dwpf&data=relh&data=drct&data=sknt&data=gust&data=vsby&data=wxcodes&data=skyl1&data=skyc1&data=skyl2&data=skyc2&data=alti&year1=2025&month1=1&day1=1&year2=2026&month2=8&day2=1&tz=Etc/UTC&format=onlycomma&latlon=no&missing=null&trace=null&direct=no&report_type=3&report_type=4"
```

(repeat for each of the ten ICAO codes).

## Added for v6 (EUROCONTROL daily airport series)

| Dataset | Used for | Source | Licence / attribution as stated by the source | Local path (git-ignored) |
|---|---|---|---|---|
| `atfm_slot_adherence_{2025,2026}.csv`, `all_pre_departure_delays_{2025,2026}.csv`, `atc_pre_departure_delays_{2025,2026}.csv` | Day-level airport state per airport: regulated departures, off-slot departures, pre-departure delay per flight. | EUROCONTROL Performance Review Unit, `https://www.eurocontrol.int/performance/data/download/csv/` (listing at `https://ansperformance.eu/csv/`) | The download page shows no explicit licence text. The data is published openly by EUROCONTROL / PRU; we attribute it to EUROCONTROL PRU. If the organisers or EUROCONTROL state different terms, those apply. | `opdi/eurocontrol/*.csv` (~31 MB) |

Downloaded 2026-10-07. The 2026 files cover 2026-01-01 to 2026-08-31 (slot adherence to 2026-09-18), which includes the
January and July ranking months.

## Added for v12 (adsb.lol ADS-B ground traces)

| Dataset | Used for | Source | Licence / attribution as stated by the source | Local path (git-ignored) |
|---|---|---|---|---|
| adsb.lol `globe_history_2025` and `globe_history_2026` daily releases: every day of January, February, June and July 2026 (the four final-phase months) and the 2025 days used to train the stage (January and July in full, more months added when available) | Off-block events per departure: first point on the ground and first movement before an observed takeoff, matched to the movement records by callsign and takeoff time (`adsb.py`). Used as inputs of a second stage at EDDF, EDDM, EHAM, LEBL and LSZH (`adsb_stage.py`). | `https://github.com/adsblol/globe_history_2025` and `https://github.com/adsblol/globe_history_2026`, release assets `v<YYYY.MM.DD>-planes-readsb-prod-0.tar.*` (the staging release is used when no prod release exists) | Open Database Licence (ODbL 1.0); the archives carry `LICENSE-ODbL.txt` and `LICENSE-cc0.txt`. Attribution: adsb.lol contributors. | `opdi/adsb/cut/<date>.parquet` (~1.5 GB after cutting; the archives themselves, ~3 GB per day, are streamed and not stored) and `opdi/adsb/feat/<date>.parquet` |

Download and cut (resumable; finished days are skipped):

```
python adsb.py fetch --days 2025-01,2025-07,2026-01,2026-02,2026-06,2026-07 --procs 8
python adsb_stage.py events --data DIR
```

Only trace points within about 11 km of the airport reference point (latitude half-width 0.10 degrees) that are on the ground or
below 3,000 ft are kept. The 2026 days are the ranking months, so no 2026 ground-truth label is involved; the stage is trained
on 2025 labels only. At the other five airports the ADS-B inputs are set to missing, because their share of matched flights
differs a lot between 2025 and 2026 (see README). The approach (cutting the daily archives to airport boxes and deriving
off-block events) is credited to EnioAguiar/prc-taxiout-2026 (GPLv3) and was implemented independently here.

## Evaluated, not used by the model

| Dataset | What was tried | Result |
|---|---|---|
| OPDI v0.0.2 flight list (EUROCONTROL / OpenSky), `flight_list_202501.parquet`, `https://www.opdi.aero/flight-list-data.html` | Matched departures by callsign and airport to get the aircraft's ground time since its previous landing. | No gain on a January-only check (RMSE 197.2 s without vs 197.3 s with), so it is not used. OPDI states its data may be freely used provided the source is attributed. |
| adsb.lol global history, whole year 2025 (ODbL 1.0) | Surveyed only. About 2.1 TB for 2025. Only January and July of 2025 and 2026 were downloaded and cut (see above); coverage turned out to be useful at five airports only. | Months other than January and July not downloaded. |
| OPDI flight events and measurements | Checked file sizes only (about 221 MB and 153 MB per 10-day window). | Not downloaded. |

## Code

All source code in this repository is released under the GNU GPL v3 (see `LICENSE`). Dependencies: pandas, pyarrow,
LightGBM, scikit-learn and, optionally, CatBoost.
