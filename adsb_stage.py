"""ADS-B stage on top of the v11 predictions.

  python adsb_stage.py events --data DIR                       per-day event features (cached in opdi/adsb/feat/)
  python adsb_stage.py validate --data DIR --holdout H.parquet  leave-one-week-out + cross-month on Jan+Jul 2025
  python adsb_stage.py apply --data DIR --holdout H.parquet --base quirky-honey_v11.parquet --out quirky-honey_v12.parquet

The stage is a LightGBM correction of the v11 prediction (target: clipped residual y - pred) that sees the ADS-B off-block
events (see adsb.py) next to the prediction, P(copy), the NM off-block proxy and the airport. It is applied to normal-looking
predictions (<= 7200 s) outside the LIRF no-NM specialist scope. The control stage has the same inputs without ADS-B.
Idea (ADS-B off-block events as inputs to a second stage) credited to EnioAguiar/prc-taxiout-2026 (GPLv3); implemented independently.
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd
import lightgbm as lgb

import adsb

FEAT_DIR = os.environ.get("PRC_FEAT_DIR", os.path.join(adsb.ROOT, "feat"))
BASE_COLS = ["pred", "p_copy", "ml_aobt", "ap"]
ADSB_COLS = ["adsb_taxi", "adsb_taxi_move", "adsb_gs0", "adsb_parked", "adsb_takeoff_err", "adsb_n", "adsb_gap_max",
             "ad_minus_pred", "ad_minus_aobt"]
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, lambda_l2=10, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
ROUNDS = 300
AIRPORTS = sorted(adsb.AIRPORTS)
# ADS-B inputs are used only where the 2025 training months have stable, meaningful coverage. At the other airports the
# 2026 ranking months have a very different share of matched flights (EGLL 14% -> 80%, LEMD 2% -> 64%, LIRF 44% -> 24%),
# so the stage could not know how far to trust a trace there. Chosen from coverage statistics, not from scores.
ADSB_AIRPORTS = ["EDDF", "EDDM", "EHAM", "LEBL", "LSZH"]
RAW_ADSB = ["adsb_taxi", "adsb_taxi_move", "adsb_gs0", "adsb_parked", "adsb_takeoff_err", "adsb_n", "adsb_gap_max"]


def _day_job(args):
    data, d = args
    out = os.path.join(FEAT_DIR, f"{d}.parquet")
    if os.path.exists(out):
        return d, "cached"
    y, m = d[:4], d[5:7]
    src = os.path.join(data, os.environ.get("PRC_RANKING", "ranking.parquet")) if y == "2026" else glob.glob(os.path.join(data, f"training_2025-{m}-01_*.parquet"))[0]
    x = pd.read_parquet(src, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"])
    x = x[(x.PHASE_mvt == "DEP") & (x.MVT_TIME_UTC_mvt.dt.date.astype(str) == d)]
    ev = adsb.day_events(date.fromisoformat(d))
    if ev is None:
        return d, "no cut"
    F = adsb.match(x, ev) if len(ev) else pd.DataFrame(columns=["MVT_ID_mvt"] + adsb.EVENT_FEATURES)
    r = x[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]].merge(F, on="MVT_ID_mvt", how="left")
    os.makedirs(FEAT_DIR, exist_ok=True)
    r.to_parquet(out, index=False)
    return d, f"{len(r)} deps, {int(r.adsb_taxi.notna().sum())} matched"


def events(a):
    days = sorted(os.path.basename(f)[:10] for f in glob.glob(os.path.join(adsb.ROOT, "cut", "*.parquet")))
    print(f"{len(days)} days with ADS-B cuts")
    with ProcessPoolExecutor(a.procs) as ex:
        for d, msg in ex.map(_day_job, [(a.data, d) for d in days]):
            print(d, msg, flush=True)


def load_features(prefix):
    fs = sorted(glob.glob(os.path.join(FEAT_DIR, f"{prefix}-*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else None


def add_derived(df):
    off = ~df["ADEP_mvt"].isin(ADSB_AIRPORTS).values
    df.loc[off, RAW_ADSB] = np.nan
    df["ml_aobt"] = (df["MVT_TIME_UTC_mvt"] - df["AOBT_3_flt"]).dt.total_seconds()
    df["ad_minus_pred"] = df["adsb_taxi_move"] - df["pred"]
    df["ad_minus_aobt"] = df["adsb_taxi_move"] - df["ml_aobt"]
    df["ap"] = pd.Categorical(df["ADEP_mvt"], categories=AIRPORTS)
    return df


def holdout_frame(path):
    h = pd.read_parquet(path)
    f = load_features("2025")
    df = h.merge(f, on="MVT_ID_mvt", how="inner")
    df["day"] = df["MVT_TIME_UTC_mvt"].dt.date.astype(str)
    df["week"] = df["MVT_TIME_UTC_mvt"].dt.isocalendar().week.astype(int) + 100 * df["month"]
    return add_derived(df)


def fit_stage(df, tr, cols):
    m = lgb.train(PARAMS, lgb.Dataset(df.loc[tr, cols], np.clip(df.y[tr] - df.pred[tr], -3600, 3600)), ROUNDS)
    return m


def cross_validate(df, cols, groups, train_ok):
    out = df["pred"].values.copy()
    for g in sorted(df[groups].unique()):
        te = (df[groups] == g).values
        tr = ~te & train_ok
        sel = te & (df.pred <= 7200).values & ~df.spec.values
        out[sel] = df.pred.values[sel] + fit_stage(df, tr, cols).predict(df.loc[sel, cols])
    return out


def report(df, preds, title):
    rm = lambda p, m: float(np.sqrt(np.mean((p[m] - df.y.values[m]) ** 2)))
    allm = np.ones(len(df), bool)
    norm = ((df.y >= 0) & (df.y <= 10800)).values
    cov = df.ADEP_mvt.isin(["EDDF", "EDDM", "EHAM", "LEBL", "LSZH"]).values
    print(f"\n{title}: {len(df):,} flights")
    print(f"{'':32s} {'all':>7s} {'normal':>7s} {'covered':>8s} {'Jan':>7s} {'Jul':>7s}")
    res = {}
    for lab, p in preds.items():
        res[lab] = (rm(p, allm), rm(p, (df.month == 1).values), rm(p, (df.month == 7).values))
        print(f"{lab:32s} {rm(p, allm):7.1f} {rm(p, norm):7.1f} {rm(p, norm & cov):8.1f} {res[lab][1]:7.1f} {res[lab][2]:7.1f}")
    return res


def validate(a):
    df = holdout_frame(a.holdout)
    train_ok = ((df.pred <= 7200) & (df.y >= 0) & (df.y <= 10800) & ~df.spec).values
    cols0, cols1 = BASE_COLS, BASE_COLS + ADSB_COLS
    p0, p1 = cross_validate(df, cols0, "week", train_ok), cross_validate(df, cols1, "week", train_ok)
    r = report(df, {"v11 prediction": df.pred.values, "control stage (no ADS-B)": p0, "stage with ADS-B": p1},
               "leave-one-week-out (primary)")
    # cross-month: train on one month, test on the other
    q0, q1 = df.pred.values.copy(), df.pred.values.copy()
    for tm in (1, 7):
        te = (df.month != tm).values
        sel = te & (df.pred <= 7200).values & ~df.spec.values
        tr = (df.month == tm).values & train_ok
        for q, cols in ((q0, cols0), (q1, cols1)):
            q[sel] = df.pred.values[sel] + fit_stage(df, tr, cols).predict(df.loc[sel, cols])
    report(df, {"v11 prediction": df.pred.values, "control stage (no ADS-B)": q0, "stage with ADS-B": q1},
           "cross-month (train on the other month; stricter)")
    base, ctl, ad = r["v11 prediction"], r["control stage (no ADS-B)"], r["stage with ADS-B"]
    keep = all(ad[i] < base[i] and ad[i] < ctl[i] for i in (1, 2)) and ad[0] < base[0]
    print("\nDECISION (primary test): ADS-B stage beats v11 and the control in both months and overall:", "KEEP" if keep else "DO NOT KEEP")


def validate_all(a):
    """Milestone check for training the stage on all months: for every week of Jan, Feb, Jun and Jul 2025 compare
    (a) the v11-style prediction, (b) the stage trained on Jan+Jul only (the v12 recipe; for Jan/Jul weeks the held-out week is left out) and
    (c) the stage trained on all other weeks of every month. (c) is kept only if it beats (b) in BOTH Jan/Jul and Feb/Jun."""
    df = holdout_frame(a.holdout)
    train_ok = ((df.pred <= 7200) & (df.y >= 0) & (df.y <= 10800) & ~df.spec).values
    cols = BASE_COLS + ADSB_COLS
    sel_all = (df.pred <= 7200).values & ~df.spec.values
    jj = df.month.isin([1, 7]).values
    test_months = [1, 2, 6, 7]
    p_jj, p_all = df.pred.values.copy(), df.pred.values.copy()
    m_jj_full = fit_stage(df, jj & train_ok, cols)  # Jan+Jul model, used for the Feb/Jun weeks
    for g in sorted(df.loc[df.month.isin(test_months), "week"].unique()):
        te = (df["week"] == g).values
        sel = te & sel_all
        mo = int(df.loc[te, "month"].iloc[0])
        m_a = fit_stage(df, ~te & train_ok, cols)
        p_all[sel] = df.pred.values[sel] + m_a.predict(df.loc[sel, cols])
        m_b = m_jj_full if mo in (2, 6) else fit_stage(df, jj & ~te & train_ok, cols)
        p_jj[sel] = df.pred.values[sel] + m_b.predict(df.loc[sel, cols])
        print("week", g, "done", flush=True)
    rm = lambda p, m: float(np.sqrt(np.mean((p[m] - df.y.values[m]) ** 2)))
    groups = {"Jan+Jul": df.month.isin([1, 7]).values, "Feb+Jun": df.month.isin([2, 6]).values, "all four months": df.month.isin(test_months).values}
    print(f"\n{'':22s} {'v11-style':>10s} {'Jan+Jul stage':>14s} {'all-months stage':>17s}")
    res = {}
    for k, m in groups.items():
        res[k] = (rm(df.pred.values, m), rm(p_jj, m), rm(p_all, m))
        print(f"{k:22s} {res[k][0]:10.1f} {res[k][1]:14.1f} {res[k][2]:17.1f}   ({int(m.sum()):,} flights)")
    keep = res["Jan+Jul"][2] < res["Jan+Jul"][1] and res["Feb+Jun"][2] < res["Feb+Jun"][1]
    print("\nDECISION: all-months stage beats the Jan+Jul stage in both groups:", "KEEP" if keep else "DO NOT KEEP")
    print("four-month proxy (all flights kept): v11-style %.1f | Jan+Jul stage %.1f | all-months stage %.1f" % res["all four months"])


def apply(a):
    import stack as S
    df = holdout_frame(a.holdout)
    train_ok = ((df.pred <= 7200) & (df.y >= 0) & (df.y <= 10800) & ~df.spec).values
    cols = BASE_COLS + ADSB_COLS
    models = []
    for s in range(a.seeds):
        m = lgb.train({**PARAMS, "seed": s, "bagging_seed": s}, lgb.Dataset(df.loc[train_ok, cols], np.clip(df.y[train_ok] - df.pred[train_ok], -3600, 3600)), ROUNDS)
        models.append(m)
    # ranking rows: v11 predictions, base P(copy), event features
    P = S.prepare_all(a.data)
    rk = P["is_rank"].values
    z = np.load(S.submit_cache())
    rid = P["d"]["MVT_ID_mvt"].values[rk]
    pc = pd.Series(z["p"][rk], index=rid)
    nm_lirf = pd.Series((P["nm"].values & (P["K"]["ap"] == "LIRF").values)[rk], index=rid)
    sub = pd.read_parquet(a.base)
    f = load_features("2026")
    r = sub.merge(f, on="MVT_ID_mvt", how="left").rename(columns={"TAXITIME_SEC_mvt": "pred"})
    r["p_copy"] = r["MVT_ID_mvt"].map(pc).values
    r["spec"] = r["MVT_ID_mvt"].map(nm_lirf).fillna(False).values
    assert r["ADEP_mvt"].notna().all(), "ranking departures without event rows: run `events` after the download finished"
    r = add_derived(r)
    sel = ((r.pred <= 7200) & ~r.spec).values
    corr = np.mean([m.predict(r.loc[sel, cols]) for m in models], axis=0)
    new = r["pred"].values.copy()
    new[sel] = new[sel] + corr
    new = np.clip(new, 30, None)
    out = sub.copy()
    out["TAXITIME_SEC_mvt"] = new
    out.to_parquet(a.out, index=False)
    print(f"stage trained on {int(train_ok.sum()):,} flights ({a.seeds} seeds); applied to {int(sel.sum()):,} of {len(r):,} ranking rows; "
          f"mean change {np.mean(corr):.1f} s, mean abs {np.mean(np.abs(corr)):.1f} s; matched events {int(r.adsb_taxi.notna().sum()):,}; wrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["events", "validate", "validate-all", "apply"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--holdout")
    ap.add_argument("--base")
    ap.add_argument("--out")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    {"events": events, "validate": validate, "validate-all": validate_all, "apply": apply}[a.cmd](a)
