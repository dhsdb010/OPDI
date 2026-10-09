"""ADS-B ground traces from adsb.lol near the 10 airports (globe_history, ODbL 1.0).

Each day of https://github.com/adsblol/globe_history_2025 (and _2026) is a tar split into release parts.
The parts are streamed straight into this script; only trace points within ~11 km of an airport that are on the
ground or below 3,000 ft are kept, in opdi/adsb/cut/YYYY-MM-DD.parquet. Nothing else is stored.

  python adsb.py fetch --days 2025-01,2025-07,2026-01,2026-07 --procs 6     (resumable: finished days are skipped)
  python adsb.py cut --date 2025-01-15 < day.tar                            (one day from a tar stream)

Approach (cutting the daily trace archives to airport boxes, then deriving off-block events) credited to
EnioAguiar/prc-taxiout-2026 (GPLv3); implemented here independently.
"""
import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
import zlib
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opdi", "adsb")
ALT_MAX_FT = 3000
BOX_DEG = 0.10  # half-width in latitude (~11 km); longitude scaled by cos(lat)
AIRPORTS = {  # aerodrome reference points (deg)
    "EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4700, -0.4543),
    "EHAM": (52.3086, 4.7639), "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389), "LTFM": (41.2753, 28.7519),
    "LSZH": (47.4647, 8.5492),
}
_BOXES = [(k, la - BOX_DEG, la + BOX_DEG, lo - BOX_DEG / math.cos(math.radians(la)), lo + BOX_DEG / math.cos(math.radians(la)))
          for k, (la, lo) in AIRPORTS.items()]


def _airport(lat, lon):
    for k, a, b, c, d in _BOXES:
        if a <= lat <= b and c <= lon <= d:
            return k
    return None


def cut_stream(fobj):
    """Read a globe_history tar stream; return the kept points as a DataFrame."""
    rows = []
    with tarfile.open(fileobj=fobj, mode="r|") as tf:
        for m in tf:
            if not m.isfile() or "/traces/" not in "/" + m.name.lstrip("./"):
                continue
            raw = tf.extractfile(m).read()
            try:
                if raw[:2] == b"\x1f\x8b":
                    raw = gzip.decompress(raw)
                tr = json.loads(raw)
            except (ValueError, OSError, EOFError, zlib.error):  # a corrupted trace file loses that aircraft only
                continue
            t0 = float(tr.get("timestamp", 0.0))
            icao = tr.get("icao", "")
            reg, typ = tr.get("r", ""), tr.get("t", "")
            cs = ""
            for p in tr.get("trace", ()):
                det = p[8] if len(p) > 8 else None
                if isinstance(det, dict) and det.get("flight"):
                    cs = det["flight"].strip()
                lat, lon = p[1], p[2]
                if lat is None or lon is None:
                    continue
                alt = p[3]
                ground = alt == "ground"
                if not ground and (alt is None or alt > ALT_MAX_FT):
                    continue
                apt = _airport(lat, lon)
                if apt is None:
                    continue
                rows.append((icao, reg, typ, cs, t0 + p[0], apt, lat, lon, ground, -1 if ground else alt, p[4]))
    df = pd.DataFrame(rows, columns=["icao", "reg", "type", "callsign", "t", "apt", "lat", "lon", "ground", "alt", "gs"])
    for c in ("icao", "reg", "type", "callsign", "apt"):
        df[c] = df[c].astype("category")
    return df


def release_parts(d):
    """Download URL lists for day d, one per published release (prod first, then staging, then the -0tmp tags)."""
    # a day's release normally sits in the repo of its own year; 2025-12-31 was published in globe_history_2026
    out = []
    for repo in (f"globe_history_{d.year}", f"globe_history_{d.year + 1}"):
        for kind, suffix in (("prod", "0"), ("staging", "0"), ("prod", "0tmp"), ("staging", "0tmp")):  # some days only exist as "-0tmp"
            tag = f"v{d:%Y.%m.%d}-planes-readsb-{kind}-{suffix}"
            try:
                with urllib.request.urlopen(f"https://api.github.com/repos/adsblol/{repo}/releases/tags/{tag}", timeout=60) as r:
                    rel = json.load(r)
            except Exception:
                continue
            urls = sorted(a["browser_download_url"] for a in rel.get("assets", []) if ".tar" in a["name"])
            if urls:
                out.append(urls)
    return out


def _fetch_release(urls):
    """Stream one release through cut_stream; returns (DataFrame, None) or (None, error text)."""
    cmd = "(" + "; ".join(f"curl -sSfL --retry 8 --retry-all-errors --connect-timeout 30 --speed-time 180 --speed-limit 5000 '{u}'" for u in urls) + ")"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    try:
        df = cut_stream(p.stdout)
    except Exception as e:  # truncated stream etc.
        p.kill()
        return None, f"{type(e).__name__}: {e}"
    left = 0
    while True:  # the tar reader can stop before the end of the stream; drain it so curl cannot block on a full pipe
        chunk = p.stdout.read(1 << 20)
        if not chunk:
            break
        left += len(chunk)
    if left > (1 << 20):  # e.g. the 2025-10-15 prod release: the second part does not continue the archive
        p.wait()
        return None, f"tar ended {left / 1e6:.0f} MB before the end of the stream"
    if p.wait() != 0:
        return None, f"curl exit {p.returncode}"
    return df, None


def fetch_day(d):
    os.makedirs(os.path.join(ROOT, "cut"), exist_ok=True)
    out = os.path.join(ROOT, "cut", f"{d:%Y-%m-%d}.parquet")
    if os.path.exists(out):
        return f"{d} skip"
    releases = release_parts(d)
    if not releases:
        return f"{d} ERROR no release"
    t0 = time.time()
    tmp = out + ".part"
    errs = []
    for urls in releases:  # a damaged release falls through to the next published copy of the same day
        df, err = _fetch_release(urls)
        if err is None:
            break
        errs.append(err)
    else:
        return f"{d} ERROR " + "; ".join(errs)
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, out)
    return f"{d} ok {len(df):,} points, {time.time()-t0:.0f}s"


TAKEOFF_GAP_S = 120   # last ground point followed by an airborne point within this many seconds = takeoff
SEG_GAP_S = 600       # a ground segment ends at a gap longer than this
CS_TOL_S, NEAR_TOL_S = 300, 90
EVENT_FEATURES = ["adsb_taxi", "adsb_taxi_move", "adsb_gs0", "adsb_parked", "adsb_takeoff_err", "adsb_n", "adsb_gap_max"]


def takeoffs(cut):
    """One row per observed takeoff: airport, callsign, takeoff time and the ground segment that led to it."""
    c = cut.sort_values(["icao", "t"], kind="stable")
    ic = c["icao"].astype(str).values; t = c["t"].values; g = c["ground"].values
    gs = c["gs"].astype(float).values; apt = c["apt"].astype(str).values; cs = c["callsign"].astype(str).values
    nxt_same = np.r_[ic[1:] == ic[:-1], False]
    dt_next = np.r_[t[1:] - t[:-1], np.inf]
    is_to = g & nxt_same & np.r_[~g[1:], False] & (dt_next < TAKEOFF_GAP_S)
    out = []
    for i in np.where(is_to)[0]:
        j = i
        while j > 0 and ic[j - 1] == ic[i] and g[j - 1] and t[j] - t[j - 1] < SEG_GAP_S and apt[j - 1] == apt[i]:
            j -= 1
        seg = slice(j, i + 1)
        tt, gg = t[seg], gs[seg]
        mv = np.where(gg > 1.0)[0]
        out.append((apt[i], cs[i + 1] or cs[i], t[i + 1], t[j], tt[mv[0]] if len(mv) else np.nan, gs[j],
                    i - j + 1, float(np.max(np.diff(tt))) if i > j else 0.0))
    return pd.DataFrame(out, columns=["apt", "callsign", "takeoff", "first_ground", "first_move", "gs0", "n", "gap_max"])


def match(dep, ev):
    """MVT_ID_mvt -> event: same airport and callsign within CS_TOL_S, else the nearest takeoff within NEAR_TOL_S."""
    dep = dep.assign(m=secs_utc(dep["MVT_TIME_UTC_mvt"]), cs=dep["FLIGHT_mvt"].astype(str).str.strip().str.upper())
    ev = ev.reset_index(drop=True).assign(cs=ev["callsign"].astype(str).str.strip().str.upper())
    used, res = set(), {}
    for a, D in dep.groupby("ADEP_mvt"):
        E = ev[ev["apt"] == a]
        if E.empty:
            continue
        by_cs = {k: g for k, g in E.groupby("cs")}
        et, ei = E["takeoff"].values, E.index.values
        o = np.argsort(et); et, ei = et[o], ei[o]
        for mid, m, cs in zip(D["MVT_ID_mvt"].values, D["m"].values, D["cs"].values):
            best = None
            g = by_cs.get(cs) if cs not in ("", "NAN", "NONE") else None
            if g is not None:
                dd = np.abs(g["takeoff"].values - m)
                k = int(np.argmin(dd))
                if dd[k] <= CS_TOL_S and g.index[k] not in used:
                    best = g.index[k]
            if best is None:
                p = np.searchsorted(et, m)
                cand = [q for q in (p - 1, p) if 0 <= q < len(et) and abs(et[q] - m) <= NEAR_TOL_S and ei[q] not in used]
                if cand:
                    best = ei[min(cand, key=lambda q: abs(et[q] - m))]
            if best is not None:
                used.add(best)
                res[mid] = best
    if not res:
        return pd.DataFrame(columns=["MVT_ID_mvt"] + EVENT_FEATURES)
    ids = np.array(list(res.keys())); e = ev.loc[list(res.values())].reset_index(drop=True)
    m = dep.set_index("MVT_ID_mvt").loc[ids, "m"].values
    return pd.DataFrame({"MVT_ID_mvt": ids, "adsb_taxi": m - e["first_ground"].values, "adsb_taxi_move": m - e["first_move"].values,
                         "adsb_gs0": e["gs0"].values, "adsb_parked": (e["gs0"].values <= 1.0).astype(float),
                         "adsb_takeoff_err": e["takeoff"].values - m, "adsb_n": e["n"].values.astype(float),
                         "adsb_gap_max": e["gap_max"].values})


def secs_utc(s):
    return (s - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().values


def day_events(d):
    """Takeoffs on day d, using the last 3 h of day d-1 so that taxis starting before midnight are complete."""
    f = lambda x: os.path.join(ROOT, "cut", f"{x:%Y-%m-%d}.parquet")
    if not os.path.exists(f(d)):
        return None
    cut = pd.read_parquet(f(d))
    if os.path.exists(f(d - timedelta(days=1))):
        prev = pd.read_parquet(f(d - timedelta(days=1)))
        start = pd.Timestamp(d, tz="UTC").timestamp()
        cut = pd.concat([prev[prev["t"] >= start - 3 * 3600], cut], ignore_index=True)
    ev = takeoffs(cut)
    lo = pd.Timestamp(d, tz="UTC").timestamp()
    return ev[(ev["takeoff"] >= lo - NEAR_TOL_S) & (ev["takeoff"] < lo + 86400 + NEAR_TOL_S)]


def build_events(dep):
    """Event features for every departure in `dep` whose day has been cut (dep needs MVT_ID_mvt, ADEP_mvt, FLIGHT_mvt, MVT_TIME_UTC_mvt)."""
    parts = []
    days = dep["MVT_TIME_UTC_mvt"].dt.tz_convert("UTC").dt.date if dep["MVT_TIME_UTC_mvt"].dt.tz is not None else dep["MVT_TIME_UTC_mvt"].dt.date
    for dd, D in dep.groupby(days):
        ev = day_events(dd)
        if ev is None or ev.empty:
            continue
        parts.append(match(D, ev))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["MVT_ID_mvt"] + EVENT_FEATURES)


def days_of(spec):
    out = []
    for ym in spec.split(","):
        y, m = map(int, ym.split("-"))
        d = date(y, m, 1)
        while d.month == m:
            out.append(d)
            d += timedelta(days=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "cut"])
    ap.add_argument("--days", default="2025-01,2025-07,2026-01,2026-07")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--date")
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "cut"), exist_ok=True)
    if a.cmd == "cut":
        df = cut_stream(sys.stdin.buffer)
        df.to_parquet(os.path.join(ROOT, "cut", f"{a.date}.parquet"), index=False, compression="zstd")
        print(a.date, len(df))
        return
    # interleave the months so training (2025) and ranking (2026) days arrive together
    per_month = [days_of(m) for m in a.days.split(",")]
    order = [d for i in range(31) for ms in per_month for d in ms[i:i + 1]]
    with ProcessPoolExecutor(a.procs) as ex:
        for msg in ex.map(fetch_day, order):
            print(time.strftime("%H:%M:%S"), msg, flush=True)


if __name__ == "__main__":
    main()
