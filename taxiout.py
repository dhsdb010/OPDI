"""PRC Data Challenge 2026 - taxi-out time baseline (LightGBM).

Usage:
    python taxiout.py --data DIR            # validate on Jan+Jul 2025, then write submission
    python taxiout.py --data DIR --no-submit

DIR must contain training_2025-*.parquet, ranking.parquet, submitting.parquet.
"""
import argparse, glob, os
import numpy as np, pandas as pd, lightgbm as lgb

TIME_COLS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt",
             "IOBT_flt", "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt"]
CATS = ["AIRPORT", "RUNWAY", "STAND", "AC", "OPERATOR", "MARKET", "WAKE", "OTHER_AP",
        "FLIGHT_TYPE", "FLIGHT_RULE", "RWY_STAND"]


def load(path_or_glob):
    files = sorted(glob.glob(path_or_glob))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for c in TIME_COLS:
        if c in df:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce").dt.tz_localize(None)
    return df


def add_congestion(df):
    """Traffic counts around each movement at its airport. Uses only times that are
    present in ranking data (MVT_TIME for all rows, BLOCK_TIME only for ARR)."""
    df = df.copy()
    df["AIRPORT"] = np.where(df["PHASE_mvt"] == "DEP", df["ADEP_mvt"], df["ADES_mvt"])
    df["_t"] = df["MVT_TIME_UTC_mvt"].astype("int64") // 10**9
    for ph, name in [("DEP", "dep"), ("ARR", "arr")]:
        for w in (5, 15, 30, 60):
            df[f"n_{name}_{w}m"] = np.nan
    for ap, g in df.groupby("AIRPORT"):
        t = g["_t"].values
        for ph, name in [("DEP", "dep"), ("ARR", "arr")]:
            ref = np.sort(t[g["PHASE_mvt"].values == ph])
            for w in (5, 15, 30, 60):
                lo = np.searchsorted(ref, t - w * 60, "left")
                hi = np.searchsorted(ref, t + w * 60, "right")
                df.loc[g.index, f"n_{name}_{w}m"] = hi - lo
        # departures taking off in the preceding window = queue proxy (same runway)
        dep = g[g["PHASE_mvt"] == "DEP"]
        for rwy, gr in dep.groupby("RUNWAY_mvt", dropna=False):
            ref = np.sort(gr["_t"].values)
            tt = gr["_t"].values
            for w in (10, 20):
                df.loc[gr.index, f"rwy_dep_{w}m"] = (
                    np.searchsorted(ref, tt, "left") - np.searchsorted(ref, tt - w * 60, "left"))
    return df.drop(columns="_t")


def features(df):
    d = add_congestion(df)
    t = d["MVT_TIME_UTC_mvt"]
    X = pd.DataFrame(index=d.index)
    X["hour"] = t.dt.hour + t.dt.minute / 60
    X["dow"] = t.dt.dayofweek
    X["month"] = t.dt.month
    X["doy"] = t.dt.dayofyear
    X["AIRPORT"] = d["AIRPORT"]
    X["RUNWAY"] = d["RUNWAY_mvt"]
    X["STAND"] = d["STAND_mvt"]
    X["RWY_STAND"] = d["RUNWAY_mvt"].astype(str) + "|" + d["STAND_mvt"].astype(str)
    X["AC"] = d["AIRCRAFT_TYPE_mvt"].fillna(d.get("AIRCRAFT_TYPE_flt"))
    X["OPERATOR"] = d.get("AIRCRAFT_OPERATOR_flt")
    X["MARKET"] = d.get("MARKET_SEGMENT_flt")
    X["WAKE"] = d.get("WK_TBL_CAT_flt")
    X["FLIGHT_TYPE"] = d.get("FLIGHT_TYPE_flt")
    X["FLIGHT_RULE"] = d.get("FLIGHT_RULE_mvt")
    X["OTHER_AP"] = d["ADES_mvt"]
    for c in [c for c in d if c.startswith(("n_dep", "n_arr", "rwy_dep"))]:
        X[c] = d[c]
    # schedule / flight-plan offsets relative to takeoff (minutes). Some of these may
    # leak the block time (AOBT_3, LOBT, IOBT); the model decides how useful they are.
    for c in ["SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt"]:
        if c in d:
            X["mvt_minus_" + c] = (t - d[c]).dt.total_seconds() / 60
    if "ARVT_3_flt" in d:
        X["flight_min"] = (d["ARVT_3_flt"] - d["AOBT_3_flt"]).dt.total_seconds() / 60
    for c in CATS:
        X[c] = X[c].astype("string").fillna("NA").astype("category")
    return X


def fit_predict(Xtr, ytr, Xte, rounds=2000, yva=None, Xva=None):
    # align categories across frames
    for c in CATS:
        cats = pd.api.types.union_categoricals([Xtr[c], Xte[c]]).categories
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats)
        Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
    p = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=50,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5,
             cat_smooth=20, cat_l2=10, max_cat_to_onehot=8, verbose=-1, num_threads=8)
    dtr = lgb.Dataset(Xtr, ytr)
    cb = []
    valid = []
    if yva is not None:
        valid = [lgb.Dataset(Xte, yva, reference=dtr)]
        cb = [lgb.early_stopping(100), lgb.log_evaluation(100)]
    m = lgb.train({**p, "metric": "rmse"}, dtr, rounds, valid_sets=valid, callbacks=cb)
    return m, m.predict(Xte, num_iteration=m.best_iteration or None)


def clean_train(df):
    d = df[df["PHASE_mvt"] == "DEP"].copy()
    d = d[d["TAXITIME_SEC_mvt"].between(60, 7200)]  # drop obviously broken labels
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--no-submit", action="store_true")
    ap.add_argument("--out", default="submission.parquet")
    a = ap.parse_args()

    train = load(os.path.join(a.data, "training_2025-*.parquet"))
    # congestion needs ALL movements (arrivals too), so build features before filtering
    Xall = features(train)
    mask = (train["PHASE_mvt"] == "DEP") & train["TAXITIME_SEC_mvt"].between(60, 7200)
    y = train.loc[mask, "TAXITIME_SEC_mvt"].astype(float)
    X = Xall.loc[mask]
    m = train.loc[mask, "MVT_TIME_UTC_mvt"].dt.month

    # Validation mimics the ranking setup: hold out Jan and Jul, train on the rest
    va = m.isin([1, 7])
    print(f"train {int((~va).sum()):,}  valid {int(va.sum()):,}  baseline RMSE (mean) "
          f"{np.sqrt(((y[va] - y[~va].mean())**2).mean()):.1f}s")
    model, pred = fit_predict(X[~va].copy(), y[~va], X[va].copy(), yva=y[va])
    rmse = np.sqrt(((pred - y[va]) ** 2).mean())
    print(f"VALID RMSE (Jan+Jul 2025): {rmse:.1f}s")
    imp = pd.Series(model.feature_importance("gain"), index=model.feature_name()).sort_values(ascending=False)
    print(imp.head(15))
    best = model.best_iteration or 500

    if a.no_submit:
        return
    rank = load(os.path.join(a.data, "ranking.parquet"))
    Xr = features(rank)
    dep = (rank["PHASE_mvt"] == "DEP").values
    # final model on all 2025 data with the iteration count found above
    cats_X = X.copy()
    mdl, pr = fit_predict(cats_X, y, Xr[dep].copy(), rounds=int(best * 1.1))
    sub = pd.read_parquet(os.path.join(a.data, "submitting.parquet"))
    pm = pd.Series(np.clip(pr, 30, None), index=rank.loc[dep, "MVT_ID_mvt"].values)
    sub["TAXITIME_SEC_mvt"] = sub["MVT_ID_mvt"].map(pm).fillna(y.median()).astype(float)
    sub.to_parquet(a.out, index=False)
    print("wrote", a.out, len(sub), "rows; missing mapped:", int(sub["MVT_ID_mvt"].isin(pm.index).sum()))


if __name__ == "__main__":
    main()
