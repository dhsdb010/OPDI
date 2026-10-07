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

## Evaluated, not used by the model

| Dataset | What was tried | Result |
|---|---|---|
| OPDI v0.0.2 flight list (EUROCONTROL / OpenSky), `flight_list_202501.parquet`, `https://www.opdi.aero/flight-list-data.html` | Matched departures by callsign and airport to get the aircraft's ground time since its previous landing. | No gain on a January-only check (RMSE 197.2 s without vs 197.3 s with), so it is not used. OPDI states its data may be freely used provided the source is attributed. |
| adsb.lol global history, `globe_history_2025` (ODbL 1.0) | Surveyed only. About 2.1 TB for 2025, and ground coverage is near zero at LFPG, EGLL, LIRF, EDDM and LTFM according to the PRU's published coverage study. | Not downloaded. |
| OPDI flight events and measurements | Checked file sizes only (about 221 MB and 153 MB per 10-day window). | Not downloaded. |

## Code

All source code in this repository is released under the GNU GPL v3 (see `LICENSE`). Dependencies: pandas, pyarrow,
LightGBM, scikit-learn and, optionally, CatBoost.
