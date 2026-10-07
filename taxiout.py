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


EPOCH = pd.Timestamp("1970-01-01")


def secs(s):
    """Seconds since epoch, independent of the datetime unit (pandas 3 stores microseconds)."""
    return (s - EPOCH).dt.total_seconds()


def add_congestion(df):
    """Traffic counts around each movement at its airport. Uses only times that are
    present in ranking data (MVT_TIME for all rows, BLOCK_TIME only for ARR)."""
    df = df.copy()
    df["AIRPORT"] = np.where(df["PHASE_mvt"] == "DEP", df["ADEP_mvt"], df["ADES_mvt"])
    df["_t"] = secs(df["MVT_TIME_UTC_mvt"])
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


def add_queue(d):
    """Surface-queue features at the estimated push-back time.

    Block times of other departures are blanked in the ranking set, so push-back is proxied by
    AOBT_3 (NM actual off-block; falls back to takeoff - 15 min). Arrivals keep their real
    in-block time. For each departure i with push time p_i and takeoff m_i:
      q_dep_surf   departures already pushed (p_j <= p_i) but not yet airborne (m_j > p_i)
      q_rwy_surf   the same, on i's runway
      q_arr_surf   arrivals landed and not yet in-block at p_i
      q_rwy_ahead  takeoffs on i's runway between p_i and m_i (what i waited behind)
      q_rwy_prev / q_rwy_next  gap to previous / next takeoff on the runway
    """
    m = secs(d["MVT_TIME_UTC_mvt"]).values
    a = secs(d["AOBT_3_flt"]).values if "AOBT_3_flt" in d else np.full(len(d), np.nan)
    have = np.isfinite(a)
    p = np.where(have, a, m - 900.0)
    p = np.minimum(p, m)  # never after takeoff
    b = secs(d["BLOCK_TIME_UTC_mvt"]).values
    arr = (d["PHASE_mvt"] == "ARR").values
    dep = ~arr
    ap = d["AIRPORT"].values
    rw = d["RUNWAY_mvt"].astype(str).values
    out = {k: np.full(len(d), np.nan) for k in
           ["q_dep_surf", "q_rwy_surf", "q_arr_surf", "q_rwy_ahead", "q_rwy_prev", "q_rwy_next",
            "q_stand_since_arr", "q_nbr_taxi_20m", "q_nbr_taxi_60m", "q_rwy_nbr_taxi_30m"]}
    stand = d["STAND_mvt"].astype(str).values
    v_proxy = np.clip(m - p, 0, 7200.0)  # taxi proxy of each departure: takeoff - (estimated) push
    out["q_push_proxied"] = (~have).astype(float)
    for code in np.unique(ap):
        ia = np.where(ap == code)[0]
        idep = ia[dep[ia]]
        iarr = ia[arr[ia]]
        P, M = np.sort(p[idep]), np.sort(m[idep])
        x = p[idep]
        out["q_dep_surf"][idep] = np.searchsorted(P, x, "right") - np.searchsorted(M, x, "right") - (x < m[idep])
        if len(iarr):
            land = np.sort(m[iarr])
            inb = np.sort(np.where(np.isfinite(b[iarr]), b[iarr], m[iarr] + 300.0))
            out["q_arr_surf"][idep] = np.searchsorted(land, x, "right") - np.searchsorted(inb, x, "right")
        # recent taxi level at the airport: mean proxy of departures that took off just before i
        Mi = np.argsort(m[idep]); Ms = m[idep][Mi]; C = np.concatenate([[0.0], np.cumsum(v_proxy[idep][Mi])])
        for w, name in [(1200, "q_nbr_taxi_20m"), (3600, "q_nbr_taxi_60m")]:
            lo = np.searchsorted(Ms, m[idep] - w, "left"); hi = np.searchsorted(Ms, m[idep], "left")
            cnt = hi - lo
            out[name][idep] = np.where(cnt > 0, (C[hi] - C[lo]) / np.maximum(cnt, 1), np.nan)
        # turnaround at the gate: time from the last arrival in-block at this stand to push-back
        ok_a = iarr[np.isfinite(b[iarr])] if len(iarr) else iarr
        if len(ok_a):
            sa = pd.DataFrame({"st": stand[ok_a], "b": b[ok_a]}).sort_values("b")
            grp = {k: g["b"].values for k, g in sa.groupby("st")}
            dd = pd.DataFrame({"i": idep, "st": stand[idep], "x": p[idep]})
            for k, g in dd.groupby("st"):
                arrb = grp.get(k)
                if arrb is None:
                    continue
                pos = np.searchsorted(arrb, g["x"].values, "right") - 1
                out["q_stand_since_arr"][g["i"].values] = np.where(pos >= 0, g["x"].values - arrb[np.maximum(pos, 0)], np.nan)
        for r in np.unique(rw[idep]):
            ir = idep[rw[idep] == r]
            Pr, Mr = np.sort(p[ir]), np.sort(m[ir])
            xr = p[ir]
            out["q_rwy_surf"][ir] = np.searchsorted(Pr, xr, "right") - np.searchsorted(Mr, xr, "right") - (xr < m[ir])
            lo = np.searchsorted(Mr, xr, "right")
            hi = np.searchsorted(Mr, m[ir], "left")
            out["q_rwy_ahead"][ir] = np.maximum(hi - lo, 0)
            k = np.searchsorted(Mr, m[ir], "left")
            prev = np.where(k > 0, Mr[np.maximum(k - 1, 0)], np.nan)
            nxt_i = np.searchsorted(Mr, m[ir], "right")
            nxt = np.where(nxt_i < len(Mr), Mr[np.minimum(nxt_i, len(Mr) - 1)], np.nan)
            Ri = np.argsort(m[ir]); Rs = m[ir][Ri]; Cr = np.concatenate([[0.0], np.cumsum(v_proxy[ir][Ri])])
            lo = np.searchsorted(Rs, m[ir] - 1800, "left"); hi = np.searchsorted(Rs, m[ir], "left")
            out["q_rwy_nbr_taxi_30m"][ir] = np.where(hi > lo, (Cr[hi] - Cr[lo]) / np.maximum(hi - lo, 1), np.nan)
            out["q_rwy_prev"][ir] = m[ir] - prev
            out["q_rwy_next"][ir] = nxt - m[ir]
    return pd.DataFrame(out, index=d.index)


def features(df):
    d = add_congestion(df)
    Q = add_queue(d)
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
    # schedule-copy signals: does SCHED line up with the other planned/actual stamps, and is it a round time?
    S = d["SCHED_TIME_UTC_mvt"]
    for c, nm_ in [("AOBT_3_flt", "aobt"), ("EOBT_1_flt", "eobt"), ("LOBT_flt", "lobt"), ("IOBT_flt", "iobt")]:
        if c in d:
            X["dsched_" + nm_] = (S - d[c]).dt.total_seconds()
    if "AOBT_3_flt" in d and "EOBT_1_flt" in d:
        X["daobt_eobt"] = (d["AOBT_3_flt"] - d["EOBT_1_flt"]).dt.total_seconds()
    X["sched_min5"] = S.dt.minute % 5
    X["sched_sec"] = S.dt.second
    X["mvt_sec"] = t.dt.second
    if "ARVT_3_flt" in d:
        X["flight_min"] = (d["ARVT_3_flt"] - d["AOBT_3_flt"]).dt.total_seconds() / 60
    X = pd.concat([X, Q], axis=1)
    for c in CATS:
        X[c] = X[c].astype("string").fillna("NA").astype("category")
    return X


def align_cats(*frames):
    for c in CATS:
        cats = pd.api.types.union_categoricals([f[c].astype(str).astype("category") for f in frames]).categories
        for f in frames:
            f[c] = pd.Categorical(f[c].astype(str), categories=cats)


PARAMS = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5, cat_smooth=20, cat_l2=10,
              max_cat_to_onehot=8, verbose=-1, num_threads=8)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def member_params(i):
    """Diverse LightGBM members: different seeds, tree sizes and column sampling."""
    leaves = [127, 255, 63, 191][i % 4]
    ff = [0.8, 0.7, 0.9, 0.75][i % 4]
    return dict(PARAMS, num_leaves=leaves, feature_fraction=ff, seed=i, bagging_seed=i,
                feature_fraction_seed=i)


def catboost_reg(Xtr, ytr, Xte, iters=600):
    from catboost import CatBoostRegressor, Pool
    cats = [c for c in CATS if c in Xtr]
    f = lambda X: X.assign(**{c: X[c].astype(str) for c in cats})
    m = CatBoostRegressor(iterations=iters, depth=8, learning_rate=0.1, loss_function="RMSE",
                          thread_count=8, verbose=0, random_seed=0, one_hot_max_size=2)
    m.fit(Pool(f(Xtr), ytr, cat_features=cats))
    return m.predict(Pool(f(Xte), cat_features=cats))


def two_stage(Xtr, ytr, eq_tr, Xte, sched_gap_te, rounds_reg=700, rounds_clf=300, seeds=1,
              use_cat=False, members=None):
    """Mixture of two regimes (see README).

    eq = |BLOCK - SCHED| <= 60 s, i.e. the off-block stamp is a copy of the schedule. For those
    flights the taxi time equals MVT - SCHED, which is known at prediction time.
      stage 1: classifier p = P(eq), averaged over `seeds` LightGBM members
      stage 2: regressor on the normal flights only (target clipped): the mean of the LightGBM
               members and, with use_cat, a CatBoost member (equal weight to the LightGBM mean)
      prediction = p * (MVT - SCHED) + (1 - p) * regressor    (expectation, never argmax)
    If `members` is a dict it receives the individual regressor/classifier predictions.
    """
    align_cats(Xtr, Xte)
    n = ~eq_tr.values
    yn = ytr[n].clip(30, 7200)
    ps, rs = [], []
    for i in range(seeds):
        P = member_params(i)
        clf = lgb.train({**P, "objective": "binary"}, lgb.Dataset(Xtr, eq_tr.astype(int)), rounds_clf)
        ps.append(clf.predict(Xte))
        reg = lgb.train({**P, "objective": "regression"}, lgb.Dataset(Xtr[n], yn), rounds_reg)
        rs.append(reg.predict(Xte))
        print(f"  member {i} done", flush=True)
    p = np.mean(ps, axis=0)
    r_lgb = np.mean(rs, axis=0)
    r_cat = catboost_reg(Xtr[n], yn, Xte) if use_cat else None
    if members is not None:
        members.update(ps=ps, rs=rs, r_cat=r_cat)
    r = r_lgb if r_cat is None else 0.5 * r_lgb + 0.5 * r_cat
    return p * sched_gap_te + (1 - p) * r, p, r


CELLS = {"apst_rwy": ["ap", "stand", "rwy"], "apst": ["ap", "stand"], "aprwy": ["ap", "rwy"]}


def cell_stats(K_fit, y_fit, K_apply):
    """Typical taxi time per (airport, stand, runway) cell and coarser cells: median, P10, P90, mean, count.
    Fitted on clean taxi times only (not schedule copies), clipped to [30, 7200]."""
    out = pd.DataFrame(index=K_apply.index)
    fit = K_fit.assign(y=np.asarray(y_fit, dtype=float))
    for name, cols in CELLS.items():
        g = fit.groupby(cols)["y"].agg(med="median", p10=lambda v: v.quantile(.1),
                                       p90=lambda v: v.quantile(.9), mean="mean", n="size")
        g.columns = [f"{name}_{c}" for c in g.columns]
        out = out.join(K_apply[cols].join(g, on=cols).drop(columns=cols))
    return out


def cell_oof(K, y, usable, fold):
    """Out-of-fold cell stats for training rows (fold = month), so a row never sees its own label."""
    parts = []
    for f in sorted(set(fold)):
        te = (fold == f)
        fit = ~te & usable
        parts.append(cell_stats(K[fit], y[fit], K[te]))
    return pd.concat(parts).loc[K.index]


def copy_rates(K_fit, eq_fit, K_apply):
    """Share of departures whose off-block stamp is a schedule copy, per stand / operator / runway."""
    out = pd.DataFrame(index=K_apply.index)
    fit = K_fit.assign(cp=np.asarray(eq_fit, dtype=float))
    for name, cols in {"cp_apst": ["ap", "stand"], "cp_apop": ["ap", "op"], "cp_aprwy": ["ap", "rwy"]}.items():
        g = fit.groupby(cols)["cp"].agg(rate="mean", n="size")
        g.columns = [f"{name}_{c}" for c in g.columns]
        out = out.join(K_apply[cols].join(g, on=cols).drop(columns=cols))
    return out


def copy_oof(K, eq, fold):
    parts = []
    for f in sorted(set(fold)):
        te = fold == f
        parts.append(copy_rates(K[~te], eq[~te], K[te]))
    return pd.concat(parts).loc[K.index]


def nm_stage(d_tr, y_tr, cp_tr, d_te):
    """Dedicated mixture for departures with no NM flight record (~1% of rows, but they hold
    ~90% of the multi-hour outliers, almost all at LIRF). Small model on a few columns only.
    d_* have columns ap, gap (MVT - SCHED, s), hour, dow, month."""
    cols = ["ap", "gap", "hour", "dow", "month"]
    aps = sorted(set(d_tr["ap"]) | set(d_te["ap"]))
    f = lambda d: d[cols].assign(ap=pd.Categorical(d["ap"], categories=aps))
    P = dict(PARAMS, num_leaves=15, min_data_in_leaf=20, lambda_l2=10, learning_rate=0.05)
    clf = lgb.train({**P, "objective": "binary"}, lgb.Dataset(f(d_tr), cp_tr.astype(int)), 200)
    p = clf.predict(f(d_te))
    n = ~cp_tr.values
    reg = lgb.train({**P, "objective": "regression"},
                    lgb.Dataset(f(d_tr)[n], y_tr[n].clip(30, 7200)), 200)
    return p * d_te["gap"].values + (1 - p) * reg.predict(f(d_te))


def nm_frame(d):
    t = d["MVT_TIME_UTC_mvt"]
    ap = d["ADEP_mvt"]
    return pd.DataFrame({"ap": ap.values, "gap": (t - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds().values,
                         "hour": (t.dt.hour + t.dt.minute / 60).values, "dow": t.dt.dayofweek.values,
                         "month": t.dt.month.values})


def single(Xtr, ytr, Xte, rounds=700):
    m = lgb.train({**PARAMS, "objective": "regression"}, lgb.Dataset(Xtr, ytr), rounds)
    return m.predict(Xte), m


def dep_frame(df, Xall):
    mask = (df["PHASE_mvt"] == "DEP").values
    return mask, Xall[mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--no-submit", action="store_true")
    ap.add_argument("--seeds", type=int, default=3, help="LightGBM members")
    ap.add_argument("--cat", action="store_true", help="add a CatBoost regressor member")
    ap.add_argument("--dump", help="write validation predictions to this parquet")
    ap.add_argument("--out", default="submission.parquet")
    a = ap.parse_args()

    train = load(os.path.join(a.data, "training_2025-*.parquet"))
    Xall = features(train)  # congestion needs ALL movements (arrivals too)
    mask, X = dep_frame(train, Xall)
    d = train[mask]
    y = d["TAXITIME_SEC_mvt"].astype(float).reset_index(drop=True)
    X = X.reset_index(drop=True)
    gap = (d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds().reset_index(drop=True)
    eq = ((d["BLOCK_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).abs().dt.total_seconds() <= 60).reset_index(drop=True)
    month = d["MVT_TIME_UTC_mvt"].dt.month.reset_index(drop=True)
    ok = y.notna() & gap.notna()  # every departure with a label counts, outliers included

    # Validation mimics the ranking setup: hold out Jan and Jul, train on the rest.
    # No early stopping on the held-out months, so the score is not tuned on them.
    va = month.isin([1, 7]) & ok
    tr = ~month.isin([1, 7]) & ok
    print(f"train {int(tr.sum()):,}  valid {int(va.sum()):,}  copy rate {eq[tr].mean():.3%}")
    print(f"mean-prediction RMSE {rmse(y[va], y[tr].mean()):.1f}s")

    K = pd.DataFrame({"ap": d["ADEP_mvt"].values, "stand": d["STAND_mvt"].astype(str).values,
                      "rwy": d["RUNWAY_mvt"].astype(str).values,
                      "op": d["AIRCRAFT_OPERATOR_flt"].astype(str).values})
    yc = y.clip(30, 7200)
    usable = ok & ~eq
    Ftr = cell_oof(K[tr], yc[tr].values, usable[tr].values, month[tr].values)
    Fva = cell_stats(K[tr][usable[tr].values], yc[tr][usable[tr]].values, K[va])
    Ctr = copy_oof(K[tr], eq[tr].values, month[tr].values)
    Cva = copy_rates(K[tr], eq[tr].values, K[va])
    Xbase = X
    X = pd.concat([Xbase, pd.concat([Ftr, Fva]).reindex(Xbase.index),
                   pd.concat([Ctr, Cva]).reindex(Xbase.index)], axis=1)
    print(f"cell features: {Ftr.shape[1]} columns; stand/runway cell hit rate on valid "
          f"{Fva['apst_rwy_n'].notna().mean():.1%}")

    ps, _ = single(X[tr].copy(), y[tr], X[va].copy())
    print(f"single regressor, all rows        RMSE {rmse(y[va], ps):.1f}s")
    mem = {}
    mix, p, r = two_stage(X[tr].copy(), y[tr].reset_index(drop=True), eq[tr].reset_index(drop=True),
                          X[va].copy(), gap[va].values, seeds=a.seeds, use_cat=a.cat, members=mem)
    g_va = gap[va].values
    for i, ri in enumerate(mem["rs"]):
        print(f"  member {i} alone (plain mixture)        RMSE {rmse(y[va], mem['ps'][i] * g_va + (1 - mem['ps'][i]) * ri):.1f}s")
    if mem["r_cat"] is not None:
        print(f"  CatBoost regressor alone                 RMSE {rmse(y[va], p * g_va + (1 - p) * mem['r_cat']):.1f}s")
    print(f"two-stage mixture                 RMSE {rmse(y[va], mix):.1f}s")
    nm = d["FLIGHT_ID_mvt"].isna().reset_index(drop=True)
    nmf = nm_frame(d.reset_index(drop=True))
    tn, vn = tr & nm, va & nm
    mix2 = mix.copy()
    sel = nm[va].values
    mix2[sel] = nm_stage(nmf[tn].reset_index(drop=True), y[tn].reset_index(drop=True),
                         eq[tn].reset_index(drop=True), nmf[vn].reset_index(drop=True))
    hyb = mix.copy()
    gv = gap[va].values
    use = sel & (gv > 3600)
    hyb[use] = mix2[use]
    print(f"  hybrid (NM stage only where MVT-SCHED > 1h, {int(use.sum())} rows) RMSE {rmse(y[va], hyb):.1f}s")
    mix_all_nm = mix2
    print(f"  + NM stage on all NM-missing rows ({int(tn.sum()):,} train rows, {int(vn.sum()):,} valid rows) RMSE {rmse(y[va], mix2):.1f}s")
    mix = hyb
    if a.dump:
        vv = d.reset_index(drop=True)[va.values]
        pd.DataFrame({"mvt_id": vv["MVT_ID_mvt"].values, "ap": vv["ADEP_mvt"].values, "flight": vv["FLIGHT_mvt"].values,
                      "y": y[va].values, "pred": mix, "p_copy": p, "reg": r, "gap": gap[va].values,
                      "is_copy": eq[va].values, "nm": nm[va].values, "stand": vv["STAND_mvt"].values,
                      "month": month[va].values}).to_parquet(a.dump)
    print(f"FINAL (mixture + hybrid NM stage)     RMSE {rmse(y[va], mix):.1f}s")
    yv = y[va].values
    big = yv > 10800
    print(f"  RMSE on y<=3h {rmse(yv[~big], mix[~big]):.1f}s  ({(~big).sum():,} rows),"
          f" y>3h {rmse(yv[big], mix[big]):.1f}s ({big.sum():,} rows); y>3h share of SSE "
          f"{((yv[big]-mix[big])**2).sum()/((yv-mix)**2).sum():.1%}")
    from sklearn.metrics import roc_auc_score
    print(f"  copy classifier AUC {roc_auc_score(eq[va], p):.4f}")

    if a.no_submit:
        return
    rank = load(os.path.join(a.data, "ranking.parquet"))
    Xr = features(rank)
    rmask, Xr = dep_frame(rank, Xr)
    rd = rank[rmask]
    rgap = (rd["MVT_TIME_UTC_mvt"] - rd["SCHED_TIME_UTC_mvt"]).dt.total_seconds().fillna(y.median()).values
    full = ok
    Kr = pd.DataFrame({"ap": rd["ADEP_mvt"].values, "stand": rd["STAND_mvt"].astype(str).values,
                       "rwy": rd["RUNWAY_mvt"].astype(str).values})
    Fall = cell_oof(K[full], yc[full].values, usable[full].values, month[full].values)
    Xfull = pd.concat([Xbase[full], Fall.reindex(Xbase[full].index)], axis=1)
    Kr["op"] = rd["AIRCRAFT_OPERATOR_flt"].astype(str).values
    Call = copy_oof(K[full], eq[full].values, month[full].values)
    Xfull = pd.concat([Xfull, Call.reindex(Xfull.index)], axis=1)
    Fr = pd.concat([cell_stats(K[usable], yc[usable].values, Kr), copy_rates(K[full], eq[full].values, Kr)], axis=1)
    Xr = pd.concat([Xr.reset_index(drop=True), Fr.reset_index(drop=True)], axis=1)
    pr, _, _ = two_stage(Xfull.copy(), y[full].reset_index(drop=True), eq[full].reset_index(drop=True),
                         Xr.reset_index(drop=True), rgap, seeds=a.seeds, use_cat=a.cat)
    rnm = rd["FLIGHT_ID_mvt"].isna().values
    if rnm.any():
        fn = ok & nm
        nmp = nm_stage(nmf[fn].reset_index(drop=True), y[fn].reset_index(drop=True),
                       eq[fn].reset_index(drop=True), nm_frame(rd[rnm]).reset_index(drop=True))
        use = rgap[rnm] > 3600
        pr[np.where(rnm)[0][use]] = nmp[use]
    sub = pd.read_parquet(os.path.join(a.data, "submitting.parquet"))
    pm = pd.Series(np.clip(pr, 30, None), index=rd["MVT_ID_mvt"].values)
    sub["TAXITIME_SEC_mvt"] = sub["MVT_ID_mvt"].map(pm).fillna(y.median()).astype(float)
    sub.to_parquet(a.out, index=False)
    print("wrote", a.out, len(sub), "rows; matched:", int(sub["MVT_ID_mvt"].isin(pm.index).sum()))


if __name__ == "__main__":
    main()
