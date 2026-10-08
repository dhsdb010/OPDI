"""Cross-fitted stacking corrector on top of the taxiout.py base model.

Protocol (mirrors how the ranking set is built: Jan and Jul held out of a year of training):
  1. Out-of-block base predictions for every training month: the training months are split into 2-month
     blocks and the base is retrained without the block it predicts.
  2. The held-out months (Jan, Jul 2025) are predicted by the base trained on all training months.
  3. A second-stage LightGBM learns the base's residual from the original features plus the base outputs and the
     distances to both edges of the LOBT window. It is trained only on the out-of-block predictions.
  4. The corrected prediction is projected back onto the LOBT window.

Usage:  python stack.py validate --data DIR      (writes the out-of-block cache, then scores the corrector)
"""
import argparse, os, time
import numpy as np, pandas as pd, lightgbm as lgb
import taxiout as T

CACHE = os.environ.get("PRC_CACHE", "/tmp")
BLOCKS = [[2, 3], [4, 5], [6, 8], [9, 10], [11, 12]]


def prepare(data):
    train = T.load(os.path.join(data, "training_2025-*.parquet"))
    Xall = T.features(train)
    mask, X = T.dep_frame(train, Xall)
    d = train[mask].reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = d["TAXITIME_SEC_mvt"].astype(float)
    gap = (d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    eq = (d["BLOCK_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).abs().dt.total_seconds() <= 60
    month = d["MVT_TIME_UTC_mvt"].dt.month
    K = pd.DataFrame({"ap": d["ADEP_mvt"].values, "stand": d["STAND_mvt"].astype(str).values,
                      "rwy": d["RUNWAY_mvt"].astype(str).values, "op": d["AIRCRAFT_OPERATOR_flt"].astype(str).values})
    return dict(d=d, X=X, y=y, gap=gap, eq=eq, month=month, K=K, ok=y.notna() & gap.notna(),
                nm=d["FLIGHT_ID_mvt"].isna(), nmf=T.nm_frame(d))


def prepare_all(data):
    """Training and ranking departures in one frame (ranking rows have no label), so every helper works with masks."""
    train = T.load(os.path.join(data, "training_2025-*.parquet"))
    rank = T.load(os.path.join(data, "ranking.parquet"))
    parts = []
    for df in (train, rank):
        Xall = T.features(df)
        mask, X = T.dep_frame(df, Xall)
        parts.append((df[mask].reset_index(drop=True), X.reset_index(drop=True)))
    for c in T.CATS:  # one category vocabulary across both frames
        cats = pd.api.types.union_categoricals([parts[0][1][c].astype(str).astype("category"),
                                                parts[1][1][c].astype(str).astype("category")]).categories
        for _, X in parts:
            X[c] = pd.Categorical(X[c].astype(str), categories=cats)
    d = pd.concat([parts[0][0], parts[1][0]], ignore_index=True)
    X = pd.concat([parts[0][1], parts[1][1]], ignore_index=True)
    is_rank = np.r_[np.zeros(len(parts[0][0]), bool), np.ones(len(parts[1][0]), bool)]
    y = d["TAXITIME_SEC_mvt"].astype(float)
    gap = (d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    eq = ((d["BLOCK_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).abs().dt.total_seconds() <= 60).fillna(False)
    month = d["MVT_TIME_UTC_mvt"].dt.month
    K = pd.DataFrame({"ap": d["ADEP_mvt"].values, "stand": d["STAND_mvt"].astype(str).values,
                      "rwy": d["RUNWAY_mvt"].astype(str).values, "op": d["AIRCRAFT_OPERATOR_flt"].astype(str).values})
    return dict(d=d, X=X, y=y, gap=gap, eq=eq, month=month, K=K, ok=y.notna() & gap.notna() & ~pd.Series(is_rank),
                nm=d["FLIGHT_ID_mvt"].isna(), nmf=T.nm_frame(d), is_rank=pd.Series(is_rank))


def with_cells(P, tr, ap):
    """Stand/runway cell statistics and copy rates: out-of-month for the training rows, training-only for the rest."""
    K, y, eq, month = P["K"], P["y"], P["eq"], P["month"]
    yc = y.clip(30, 7200)
    usable = P["ok"] & ~eq
    Ftr = T.cell_oof(K[tr], yc[tr].values, usable[tr].values, month[tr].values)
    Fap = T.cell_stats(K[tr][usable[tr].values], yc[tr][usable[tr]].values, K[ap])
    Ctr = T.copy_oof(K[tr], eq[tr].values, month[tr].values)
    Cap = T.copy_rates(K[tr], eq[tr].values, K[ap])
    Xtr = pd.concat([P["X"][tr], Ftr, Ctr], axis=1)
    Xap = pd.concat([P["X"][ap], Fap, Cap], axis=1)
    return Xtr, Xap


def base_run(P, tr, ap, seeds=1):
    """The taxiout.py base model: copy mixture, per-airport blend, copy-impossible rule, NM-missing stage.
    Returns unclipped predictions for the `ap` rows plus the classifier probability and the normal-flight regressor."""
    d, y, eq, gap, nm, nmf = P["d"], P["y"], P["eq"], P["gap"], P["nm"], P["nmf"]
    Xtr, Xap = with_cells(P, tr, ap)
    mix, p, r = T.two_stage(Xtr.copy(), y[tr].reset_index(drop=True), eq[tr].reset_index(drop=True), Xap.copy(),
                            gap[ap].values, seeds=seeds, per_airport=True, copy_ok=T.copy_feasible(d[ap]))
    sel = nm[ap].values
    tn = tr & nm
    hyb = mix.copy()
    if sel.any():
        nmp = T.nm_stage(nmf[tn].reset_index(drop=True), y[tn].reset_index(drop=True),
                         eq[tn].reset_index(drop=True), nmf[ap & nm].reset_index(drop=True))
        use = sel[sel] & (gap[ap & nm].values > 3600)
        idx = np.where(sel)[0]
        hyb[idx[use]] = nmp[use]
    return hyb, p, r


def corrector_frame(P, X_global, base, rows):
    """Original features + base outputs + distances to the LOBT window edges."""
    d = P["d"][rows]
    ml = (d["MVT_TIME_UTC_mvt"] - d["LOBT_flt"]).dt.total_seconds().values
    pred = base["pred"][rows]
    F = X_global[rows].copy()
    F["b_pred"] = pred
    F["b_p"] = base["p"][rows]
    F["b_reg"] = base["r"][rows]
    F["b_minus_reg"] = pred - base["r"][rows]
    F["b_dist_lo"] = pred - (ml - T.LOBT_WINDOW)
    F["b_dist_hi"] = (ml + T.LOBT_WINDOW) - pred
    return F


def _past_future_means(P):
    """Mean taxi proxy (takeoff - push proxy) of OTHER departures that take off in the next 20/60 min (airport) and
    30 min (runway), plus the flight's own proxy minus those means. Past windows live in taxiout.add_queue."""
    d = P["d"]
    m = T.secs(d["MVT_TIME_UTC_mvt"]).values
    a = T.secs(d["AOBT_3_flt"]).values
    p = np.minimum(np.where(np.isfinite(a), a, m - 900.0), m)
    v = np.clip(m - p, 0, 7200.0)
    n = len(d)
    out = {k: np.full(n, np.nan) for k in ["fut_20m", "fut_60m", "rwy_fut_30m"]}
    ap = d["ADEP_mvt"].values
    rw = d["RUNWAY_mvt"].astype(str).values

    def fut(idx, w, name):
        o = np.argsort(m[idx]); ii = idx[o]; ms = m[ii]
        C = np.concatenate([[0.0], np.cumsum(v[ii])])
        lo = np.searchsorted(ms, ms, "right"); hi = np.searchsorted(ms, ms + w, "right")
        cnt = hi - lo
        out[name][ii] = np.where(cnt > 0, (C[hi] - C[lo]) / np.maximum(cnt, 1), np.nan)

    for a_ in np.unique(ap):
        idx = np.where(ap == a_)[0]
        fut(idx, 1200, "fut_20m"); fut(idx, 3600, "fut_60m")
        for r_ in np.unique(rw[idx]):
            fut(idx[rw[idx] == r_], 1800, "rwy_fut_30m")
    F = pd.DataFrame({"x_" + k: v_ for k, v_ in out.items()})
    F["x_own_proxy"] = v
    F["x_own_minus_fut20"] = v - F["x_fut_20m"]
    F["x_own_minus_fut60"] = v - F["x_fut_60m"]
    F["x_own_minus_past20"] = v - P["X"]["q_nbr_taxi_20m"].values
    F["x_own_minus_past60"] = v - P["X"]["q_nbr_taxi_60m"].values
    return F


def _delayed_copy_rate(P, tr, va, k=20.0):
    """Copy rate among delayed flights (takeoff > 1 h after SCHED) per (airport, operator, NM-or-not), smoothed toward
    the airport rate. Training rows use the other training months; the rest use the training months only."""
    d, eq, gap, month, nm = P["d"], P["eq"], P["gap"], P["month"], P["nm"]
    df = pd.DataFrame({"ap": d["ADEP_mvt"].values, "op": d["AIRCRAFT_OPERATOR_flt"].astype(str).values, "nm": nm.values,
                       "mo": month.values, "dl": ((gap > 3600) & P["ok"]).values, "cp": eq.values.astype(float)})
    df["cp"] = df["cp"] * df["dl"]
    trv = tr.values
    keys = {"ao": ["ap", "op", "nm"], "a": ["ap"]}
    res = pd.DataFrame(index=df.index, dtype=float)
    tot = {n_: df[trv].groupby(c + ["mo"])[["dl", "cp"]].sum() for n_, c in keys.items()}
    for n_, c in keys.items():
        t = tot[n_].reset_index()
        whole = t.groupby(c)[["dl", "cp"]].sum()
        j = df[c].merge(whole, left_on=c, right_index=True, how="left")[["dl", "cp"]].fillna(0.0)
        mo_tab = t.set_index(c + ["mo"])[["dl", "cp"]]
        own = df[c + ["mo"]].merge(mo_tab, left_on=c + ["mo"], right_index=True, how="left")[["dl", "cp"]].fillna(0.0)
        sub = np.where(trv[:, None], own.values, 0.0)  # remove the row's own month for training rows
        res[n_ + "_dl"] = j["dl"].values - sub[:, 0]
        res[n_ + "_cp"] = j["cp"].values - sub[:, 1]
    apr = res["a_cp"] / res["a_dl"].where(res["a_dl"] > 0)
    rate = (res["ao_cp"] + k * apr.fillna(0)) / (res["ao_dl"] + k)
    return pd.DataFrame({"x_cprate_delayed": rate.values, "x_cprate_delayed_n": res["ao_dl"].values})


def _eurocontrol_daily(P):
    """EUROCONTROL daily airport series joined on (airport, UTC date of takeoff)."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opdi", "eurocontrol")
    parts = {}
    for name in ["atfm_slot_adherence", "all_pre_departure_delays", "atc_pre_departure_delays"]:
        parts[name] = pd.concat([pd.read_csv(os.path.join(base, f"{name}_{y}.csv")) for y in (2025, 2026)])
    s = parts["atfm_slot_adherence"]
    f = pd.DataFrame({"APT_ICAO": s.APT_ICAO, "date": pd.to_datetime(s.FLT_DATE),
                      "x_dly_traffic": s.FLT_DEP_1, "x_dly_reg_share": s.FLT_DEP_REG_1 / s.FLT_DEP_1,
                      "x_dly_late_share": s.FLT_DEP_OUT_LATE_1 / s.FLT_DEP_1})
    a = parts["all_pre_departure_delays"]
    a = pd.DataFrame({"APT_ICAO": a.APT_ICAO, "date": pd.to_datetime(a.FLT_DATE), "x_dly_pre_per_flt": a.DLY_ALL_PRE_2 / a.FLT_DEP_IFR_2})
    c = parts["atc_pre_departure_delays"]
    c = pd.DataFrame({"APT_ICAO": c.APT_ICAO, "date": pd.to_datetime(c.FLT_DATE), "x_dly_atc_per_flt": c.DLY_ATC_PRE_3 / c.FLT_DEP_3})
    f = f.merge(a, on=["APT_ICAO", "date"], how="left").merge(c, on=["APT_ICAO", "date"], how="left")
    f = f.replace([np.inf, -np.inf], np.nan).drop_duplicates(["APT_ICAO", "date"])
    d = P["d"]
    key = pd.DataFrame({"APT_ICAO": d["ADEP_mvt"].values, "date": d["MVT_TIME_UTC_mvt"].dt.normalize().values})
    return key.merge(f, on=["APT_ICAO", "date"], how="left").drop(columns=["APT_ICAO", "date"])


def submit(a):
    P = prepare_all(a.data)
    y, month, d, n = P["y"], P["month"], P["d"], len(P["y"])
    tr, rk = P["ok"], P["is_rank"]
    base = {k: np.full(n, np.nan) for k in ["pred", "p", "r"]}
    cache = os.path.join(CACHE, "stack_submit_base.npz" if a.seeds == 1 else f"stack_submit_base_s{a.seeds}.npz")
    if os.path.exists(cache) and not a.refit:
        z = np.load(cache); base = {k: z[k] for k in ["pred", "p", "r"]}
        print("loaded cached base predictions")
    else:
        blocks = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
        jobs = [(tr & ~month.isin(b), tr & month.isin(b), f"block {b}") for b in blocks] + [(tr, rk, "ranking")]
        for trm, apm, name in jobs:
            t0 = time.time()
            h, p, r = base_run(P, trm, apm, seeds=a.seeds)
            base["pred"][apm.values], base["p"][apm.values], base["r"][apm.values] = h, p, r
            print(f"base {name}: {int(apm.sum()):,} rows in {time.time()-t0:.0f}s", flush=True)
            np.savez(cache, **base)
    Xtr_g, Xrk_g = with_cells(P, tr, rk)
    Xg = pd.concat([Xtr_g, Xrk_g]).reindex(range(n))
    for c in T.CATS:
        Xg[c] = Xg[c].astype(str)
    groups = [_past_future_means(P), _delayed_copy_rate(P, tr, rk), _eurocontrol_daily(P)]
    Ftr = corrector_frame(P, Xg, base, tr.values)
    Frk = corrector_frame(P, Xg, base, rk.values)
    for g in groups:
        Ftr = pd.concat([Ftr.reset_index(drop=True), g[tr.values].reset_index(drop=True)], axis=1)
        Frk = pd.concat([Frk.reset_index(drop=True), g[rk.values].reset_index(drop=True)], axis=1)
    for c in T.CATS:
        cats = sorted(set(Ftr[c]) | set(Frk[c]))
        Ftr[c] = pd.Categorical(Ftr[c], categories=cats)
        Frk[c] = pd.Categorical(Frk[c], categories=cats)
    yt, pt = y[tr].values, base["pred"][tr.values]
    fit = (yt >= 0) & (yt <= 10800) & (pt <= 7200)
    res = np.clip(yt - pt, -3600, 3600)
    Af, rf = Ftr[fit], res[fit]
    P_ = dict(T.PARAMS, num_leaves=63, min_data_in_leaf=200, learning_rate=0.05, lambda_l2=10)
    g = lgb.train({**P_, "objective": "regression"}, lgb.Dataset(Af, rf), a.rounds)
    c_g = g.predict(Frk)
    c_pa = c_g.copy()
    apx, apb = Af["AIRPORT"].astype(str).values, Frk["AIRPORT"].astype(str).values
    for a_ in np.unique(apb):
        sel_t = apx == a_
        if sel_t.sum() < 500:
            continue
        ma = lgb.train({**dict(P_, num_leaves=31, min_data_in_leaf=100), "objective": "regression"},
                       lgb.Dataset(Af[sel_t], rf[sel_t]), a.rounds)
        c_pa[apb == a_] = ma.predict(Frk[apb == a_])
    from catboost import CatBoostRegressor, Pool
    cats = [c for c in T.CATS if c in Af]
    f_ = lambda X: X.assign(**{c: X[c].astype(str) for c in cats})
    cb = CatBoostRegressor(iterations=600, depth=8, learning_rate=0.1, thread_count=8, verbose=0, random_seed=0, one_hot_max_size=2)
    cb.fit(Pool(f_(Af), rf, cat_features=cats))
    c_cb = cb.predict(Pool(f_(Frk), cat_features=cats))
    corr = (c_g + c_pa + c_cb) / 3
    pv = base["pred"][rk.values]
    new = pv.copy()
    app = pv <= 7200
    new[app] = pv[app] + corr[app]
    drk = d[rk.values]
    final = np.clip(T.lobt_clip(new, drk), 30, None)
    sub = pd.read_parquet(os.path.join(a.data, "submitting.parquet"))
    pm = pd.Series(final, index=drk["MVT_ID_mvt"].values)
    sub["TAXITIME_SEC_mvt"] = sub["MVT_ID_mvt"].map(pm).astype(float)
    assert sub["TAXITIME_SEC_mvt"].notna().all(), "unmatched ids"
    sub.to_parquet(a.out, index=False)
    print("wrote", a.out, len(sub), "rows; corrector changed", int(app.sum()), "rows, mean abs change %.1f s" % np.abs(corr[app]).mean())


def rm(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def validate(a):
    P = prepare(a.data)
    y, month, ok, d = P["y"], P["month"], P["ok"], P["d"]
    n = len(y)
    va = month.isin([1, 7]) & ok
    tr = ~month.isin([1, 7]) & ok
    base = {k: np.full(n, np.nan) for k in ["pred", "p", "r"]}
    cache = os.path.join(CACHE, "stack_val_base.npz")
    if os.path.exists(cache) and not a.refit:
        z = np.load(cache)
        base = {k: z[k] for k in ["pred", "p", "r"]}
        print("loaded cached out-of-block base predictions")
    else:
        jobs = [(tr & ~month.isin(b), tr & month.isin(b), f"block {b}") for b in BLOCKS] + [(tr, va, "holdout Jan+Jul")]
        for trm, apm, name in jobs:
            t0 = time.time()
            h, p, r = base_run(P, trm, apm)
            base["pred"][apm.values], base["p"][apm.values], base["r"][apm.values] = h, p, r
            print(f"base {name}: {int(apm.sum()):,} rows predicted in {time.time()-t0:.0f}s, RMSE {rm(y[apm], h):.1f}", flush=True)
        np.savez(cache, **base)
    # corrector input features: cells out-of-month for training rows, training-only for held-out rows
    Xtr_g, Xva_g = with_cells(P, tr, va)
    Xg = pd.concat([Xtr_g, Xva_g]).reindex(range(n))
    for c in T.CATS:
        Xg[c] = Xg[c].astype(str)
    Ftr = corrector_frame(P, Xg, base, tr.values)
    Fva = corrector_frame(P, Xg, base, va.values)
    for c in T.CATS:
        cats = sorted(set(Ftr[c]) | set(Fva[c]))
        Ftr[c] = pd.Categorical(Ftr[c], categories=cats)
        Fva[c] = pd.Categorical(Fva[c], categories=cats)
    yt, pt = y[tr].values, base["pred"][tr.values]
    # train on normal flights whose base prediction is not a tail bet; target = clipped residual
    fit = (yt >= 0) & (yt <= 10800) & (pt <= 7200)
    res = np.clip(yt - pt, -3600, 3600)
    pv = base["pred"][va.values]
    yv = y[va].values
    apply = pv <= 7200
    dv = d[va.values]
    mo = month[va].values
    norm = (yv >= 0) & (yv <= 10800)

    def report(tag, p_):
        e = p_ - yv
        print(f"{tag:34s} all {rm(p_, yv):.1f} | Jan {np.sqrt((e[mo==1]**2).mean()):.1f} Jul {np.sqrt((e[mo==7]**2).mean()):.1f} | normal {rm(p_[norm], yv[norm]):.1f}", flush=True)

    report("base (out-of-block run)", T.lobt_clip(pv, dv))
    groups = {}
    if a.variants:
        t0 = time.time()
        E1 = _past_future_means(P); E2 = _delayed_copy_rate(P, tr, va); E3 = _eurocontrol_daily(P)
        print(f"extra features built in {time.time()-t0:.0f}s")
        groups = {"neighbours (future, relative)": E1, "delayed-flight copy rate": E2, "EUROCONTROL daily series": E3}
    P_ = dict(T.PARAMS, num_leaves=63, min_data_in_leaf=200, learning_rate=0.05, lambda_l2=10)

    def run(names, tag):
        A, B = Ftr, Fva
        for nme in names:
            g = groups[nme]
            A = pd.concat([A.reset_index(drop=True), g[tr.values].reset_index(drop=True)], axis=1)
            B = pd.concat([B.reset_index(drop=True), g[va.values].reset_index(drop=True)], axis=1)
        m = lgb.train({**P_, "objective": "regression"}, lgb.Dataset(A[fit], res[fit]), a.rounds)
        corr = m.predict(B)
        new = pv.copy()
        new[apply] = pv[apply] + corr[apply]
        report(tag, T.lobt_clip(new, dv))
        return m

    m = run([], "base + corrector")
    for nme in groups:
        run([nme], "  + " + nme)
    if groups:
        run(list(groups), "  + all three")
    imp = pd.Series(m.feature_importance("gain"), index=m.feature_name()).sort_values(ascending=False)
    print("top corrector features by gain:", imp.head(6).round(0).to_dict())
    if a.ensemble:
        A, B = Ftr, Fva
        for nme in groups:
            A = pd.concat([A.reset_index(drop=True), groups[nme][tr.values].reset_index(drop=True)], axis=1)
            B = pd.concat([B.reset_index(drop=True), groups[nme][va.values].reset_index(drop=True)], axis=1)
        Af, rf = A[fit], res[fit]
        g = lgb.train({**P_, "objective": "regression"}, lgb.Dataset(Af, rf), a.rounds)
        c_g = g.predict(B)
        c_pa = c_g.copy()
        apx, apb = Af["AIRPORT"].astype(str).values, B["AIRPORT"].astype(str).values
        for a_ in np.unique(apb):
            sel_t = apx == a_
            if sel_t.sum() < 500:
                continue
            Pa = dict(P_, num_leaves=31, min_data_in_leaf=100)
            ma = lgb.train({**Pa, "objective": "regression"}, lgb.Dataset(Af[sel_t], rf[sel_t]), a.rounds)
            c_pa[apb == a_] = ma.predict(B[apb == a_])
        from catboost import CatBoostRegressor, Pool
        cats = [c for c in T.CATS if c in Af]
        f_ = lambda X: X.assign(**{c: X[c].astype(str) for c in cats})
        cb = CatBoostRegressor(iterations=600, depth=8, learning_rate=0.1, thread_count=8, verbose=0, random_seed=0, one_hot_max_size=2)
        t0 = time.time()
        cb.fit(Pool(f_(Af), rf, cat_features=cats))
        c_cb = cb.predict(Pool(f_(B), cat_features=cats))
        print(f"catboost corrector trained in {time.time()-t0:.0f}s")
        def apply_(c):
            n_ = pv.copy(); n_[apply] = pv[apply] + c[apply]; return T.lobt_clip(n_, dv)
        report("ensemble: global LightGBM", apply_(c_g))
        report("ensemble: per-airport LightGBM", apply_(c_pa))
        report("ensemble: CatBoost", apply_(c_cb))
        report("ensemble: average of the three", apply_((c_g + c_pa + c_cb) / 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "submit"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", type=int, default=1, help="LightGBM members in the base model (submit)")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--variants", action="store_true", help="also test the extra feature groups")
    ap.add_argument("--ensemble", action="store_true", help="also test global + per-airport + CatBoost correctors")
    ap.add_argument("--out", default="quirky-honey_v6.parquet")
    ap.add_argument("--refit", action="store_true", help="recompute the out-of-block base predictions")
    a = ap.parse_args()
    validate(a) if a.cmd == "validate" else submit(a)
