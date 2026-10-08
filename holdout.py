"""Out-of-sample v11-equivalent predictions for the Jan+Jul 2025 holdout (the input of the ADS-B stage in adsb_stage.py).

Train on the other ten months: out-of-block base predictions (the cache written by `stack.py validate`), the three-model corrector with
all input groups, and the LIRF no-NM specialist predicted leave-one-month-out. Writes one row per Jan+Jul 2025 departure:
MVT_ID_mvt, y, pred (final v11-equivalent prediction), base, p_copy, spec (row is in the specialist scope), month.

  python stack.py validate --data DIR            # once: writes stack_val_base.npz into $PRC_CACHE (the base out-of-block run)
  python holdout.py --data DIR --out holdout.parquet
"""
import argparse
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

import specialist as SP
import stack as S
import taxiout as T


def main(a):
    P = S.prepare(a.data)
    y, month, ok, d = P["y"], P["month"], P["ok"], P["d"]
    n = len(y)
    va, tr = month.isin([1, 7]) & ok, ~month.isin([1, 7]) & ok
    z = np.load(os.path.join(S.CACHE, "stack_val_base.npz"))
    base = {k: z[k] for k in ["pred", "p", "r"]}
    Xtr_g, Xva_g = S.with_cells(P, tr, va)
    Xg = pd.concat([Xtr_g, Xva_g]).reindex(range(n))
    for c in T.CATS:
        Xg[c] = Xg[c].astype(str)
    Ftr, Fva = S.corrector_frame(P, Xg, base, tr.values), S.corrector_frame(P, Xg, base, va.values)
    for g in [S._past_future_means(P), S._delayed_copy_rate(P, tr, va), S._eurocontrol_daily(P), S._neighbour_delay(P)]:
        Ftr = pd.concat([Ftr.reset_index(drop=True), g[tr.values].reset_index(drop=True)], axis=1)
        Fva = pd.concat([Fva.reset_index(drop=True), g[va.values].reset_index(drop=True)], axis=1)
    for c in T.CATS:
        cats = sorted(set(Ftr[c]) | set(Fva[c]))
        Ftr[c], Fva[c] = pd.Categorical(Ftr[c], categories=cats), pd.Categorical(Fva[c], categories=cats)
    yt, pt = y[tr].values, base["pred"][tr.values]
    fit = (yt >= 0) & (yt <= 10800) & (pt <= 7200)
    res = np.clip(yt - pt, -3600, 3600)
    Af, rf = Ftr[fit], res[fit]
    P_ = dict(T.PARAMS, num_leaves=63, min_data_in_leaf=200, learning_rate=0.05, lambda_l2=10)
    c_g = lgb.train({**P_, "objective": "regression"}, lgb.Dataset(Af, rf), 400).predict(Fva)
    c_pa = c_g.copy()
    apx, apb = Af["AIRPORT"].astype(str).values, Fva["AIRPORT"].astype(str).values
    for a_ in np.unique(apb):
        s = apx == a_
        if s.sum() < 500:
            continue
        m = lgb.train({**dict(P_, num_leaves=31, min_data_in_leaf=100), "objective": "regression"}, lgb.Dataset(Af[s], rf[s]), 400)
        c_pa[apb == a_] = m.predict(Fva[apb == a_])
    cats = [c for c in T.CATS if c in Af]
    f_ = lambda X: X.assign(**{c: X[c].astype(str) for c in cats})
    cb = CatBoostRegressor(iterations=600, depth=8, learning_rate=0.1, thread_count=8, verbose=0, random_seed=0, one_hot_max_size=2)
    cb.fit(Pool(f_(Af), rf, cat_features=cats))
    c_cb = cb.predict(Pool(f_(Fva), cat_features=cats))
    pv = base["pred"][va.values]
    app = pv <= 7200
    pred = pv.copy()
    pred[app] = pv[app] + ((c_g + c_pa + c_cb) / 3)[app]
    pred = np.clip(T.lobt_clip(pred, d[va.values]), 30, None)
    # the specialist replaces the LIRF no-NM rows, predicted out of month
    idx, sp = SP.lomo_predictions(P, seeds=a.seeds)
    vidx = np.where(va.values)[0]
    pos = {j: k for k, j in enumerate(vidx)}
    spec = np.zeros(len(vidx), bool)
    for j, v in zip(idx, sp):
        if j in pos:
            pred[pos[j]] = v
            spec[pos[j]] = True
    dv = d[va.values].reset_index(drop=True)
    out = pd.DataFrame({"MVT_ID_mvt": dv["MVT_ID_mvt"].values, "y": y[va].values, "pred": pred, "base": pv,
                        "p_copy": base["p"][va.values], "spec": spec, "month": month[va].values})
    out.to_parquet(a.out)
    rm = lambda m: np.sqrt(np.mean((out.pred[m] - out.y[m]) ** 2))
    print("holdout: all %.1f | Jan %.1f | Jul %.1f | specialist rows %d | wrote %s" % (
        rm(np.ones(len(out), bool)), rm(out.month == 1), rm(out.month == 7), spec.sum(), a.out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="holdout.parquet")
    ap.add_argument("--seeds", type=int, default=3, help="CatBoost seeds of the specialist")
    main(ap.parse_args())
