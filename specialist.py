"""Specialist for LIRF departures without a Network Manager flight record (the "no-NM" scope).

These ~1,500 rows a year hold most of the multi-hour and one-day-shift outliers, which dominate squared error.
A global model cannot represent them, so a small CatBoost model is fitted on that scope alone. Its target is the
residual over max(0, MVT - SCHED), which is what the stamp equals when the off-block time copies the schedule.

Idea (a separate specialist for unmatched LIRF rows on a MVT-SCHED baseline) credited to Phoenix-Ops-LTD/prc2026-taxiout
(GPLv3, team zestful-fountain); implemented here independently. Hyperparameters are not tuned on our data.

  python specialist.py validate --data DIR            leave-one-month-out over all 12 months of 2025
  python specialist.py apply --data DIR --base quirky-honey_v6.parquet --out quirky-honey_v8.parquet
"""
import argparse
import os

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import stack as S
import taxiout as T


# In 2025 training, a no-NM LIRF label is either ~ the schedule gap or sits in the one-day-shift cluster (max 88,392 s),
# so a prediction above max(gap, DAY_SHIFT_MAX) is never supported by the data.
DAY_SHIFT_MAX = 88500.0


def scope_mask(P, scope="lirf"):
    """lirf: no-NM LIRF rows; other: no-NM rows at every other airport; all: both."""
    nm, lirf = P["nm"].values, (P["K"]["ap"] == "LIRF").values
    return {"lirf": nm & lirf, "other": nm & ~lirf, "all": nm}[scope]


def frame(P, idx):
    X = P["X"].iloc[idx].copy().reset_index(drop=True)
    cats = [c for c in T.CATS if c in X]
    for c in cats:
        X[c] = X[c].astype(str)
    X["gap"] = P["gap"].values[idx]
    return X, cats


def fit_predict(Xtr, ytr, gtr, Xte, gte, cats, seeds):
    """Mean of `seeds` CatBoost fits on y - max(0, gap); returns max(0, gap) + residual, floored at 30 s."""
    out = np.zeros(len(Xte))
    for s in range(seeds):
        m = CatBoostRegressor(depth=5, iterations=1000, learning_rate=0.04, l2_leaf_reg=10, random_seed=s,
                              verbose=0, thread_count=4, cat_features=cats)
        m.fit(Xtr, ytr - gtr)
        out += m.predict(Xte) / seeds
    return np.clip(out + gte, 30, np.maximum(gte, DAY_SHIFT_MAX))


def lomo_predictions(P, seeds=3, scope="lirf"):
    """Out-of-month specialist predictions for every labelled row of the scope: (row index, prediction)."""
    y, month = P["y"].values, P["month"].values
    idx = np.where(scope_mask(P, scope) & P["ok"].values & (y >= 0))[0]
    X, cats = frame(P, idx)
    ys, ms, gs = y[idx], month[idx], np.maximum(P["gap"].values[idx], 0)
    pr = np.full(len(idx), np.nan)
    for mth in range(1, 13):
        te, tr = np.where(ms == mth)[0], np.where(ms != mth)[0]
        pr[te] = fit_predict(X.iloc[tr], ys[tr], gs[tr], X.iloc[te], gs[te], cats, seeds)
    return idx, pr


def validate(a):
    P = S.prepare(a.data)
    y, month = P["y"].values, P["month"].values
    sc = scope_mask(P, a.scope) & P["ok"].values & (y >= 0)
    idx = np.where(sc)[0]
    X, cats = frame(P, idx)
    ys, ms, gs = y[idx], month[idx], np.maximum(P["gap"].values[idx], 0)
    z = np.load(os.path.join(S.CACHE, "stack_val_base.npz"))
    base = T.lobt_clip(z["pred"][idx], P["d"].iloc[idx])
    pr = np.full(len(idx), np.nan)
    for mth in range(1, 13):
        te = np.where(ms == mth)[0]
        tr = np.where(ms != mth)[0]
        pr[te] = fit_predict(X.iloc[tr], ys[tr], gs[tr], X.iloc[te], gs[te], cats, a.seeds)
    r = lambda p, m: float(np.sqrt(np.mean((p[m] - ys[m]) ** 2)))
    print(f"month    n   base  special.   (RMSE on the no-NM {a.scope} scope, leave-one-month-out)")
    for mth in range(1, 13):
        m = ms == mth
        print(f"{mth:5d} {m.sum():4d} {r(base, m):6.0f} {r(pr, m):8.0f}")
    jj = np.isin(ms, [1, 7])
    print(f"all12 {len(ms):4d} {r(base, np.ones(len(ms), bool)):6.0f} {r(pr, np.ones(len(ms), bool)):8.0f}")
    print(f"Jan+Jul {jj.sum():3d} {r(base, jj):6.0f} {r(pr, jj):8.0f}")
    # effect on the whole Jan+Jul holdout: replace the scope rows in the cached base predictions
    va = (P["month"].isin([1, 7]) & P["ok"]).values
    full = T.lobt_clip(z["pred"][va], P["d"][va])
    yv = y[va]
    pos = {j: k for k, j in enumerate(np.where(va)[0])}
    new = full.copy()
    for k, j in enumerate(idx):
        if j in pos:
            new[pos[j]] = pr[k]
    f = lambda p: float(np.sqrt(np.mean((p - yv) ** 2)))
    mo = month[va]
    g = lambda p, m_: float(np.sqrt(np.mean((p[mo == m_] - yv[mo == m_]) ** 2)))
    print(f"whole holdout (base only): {f(full):.1f} (Jan {g(full, 1):.1f}, Jul {g(full, 7):.1f}) -> {f(new):.1f} (Jan {g(new, 1):.1f}, Jul {g(new, 7):.1f})")


def apply(a):
    P = S.prepare_all(a.data)
    y, d = P["y"].values, P["d"]
    sc = scope_mask(P, a.scope)
    tr = np.where(sc & P["ok"].values & (y >= 0))[0]
    rk = np.where(sc & P["is_rank"].values)[0]
    X, cats = frame(P, np.r_[tr, rk])
    gap = np.maximum(P["gap"].values, 0)
    p = fit_predict(X.iloc[:len(tr)], y[tr], gap[tr], X.iloc[len(tr):], gap[rk], cats, a.seeds)
    sub = pd.read_parquet(a.base)
    ids = d["MVT_ID_mvt"].values[rk]
    pm = pd.Series(p, index=ids)
    hit = sub["MVT_ID_mvt"].isin(pm.index).values
    old = sub.loc[hit, "TAXITIME_SEC_mvt"].values.copy()
    sub.loc[hit, "TAXITIME_SEC_mvt"] = sub.loc[hit, "MVT_ID_mvt"].map(pm).astype(float).values
    assert hit.sum() == len(rk), (hit.sum(), len(rk))
    sub.to_parquet(a.out, index=False)
    print(f"trained on {len(tr)} rows, replaced {int(hit.sum())} ranking rows; mean {old.mean():.0f} -> {sub.loc[hit, 'TAXITIME_SEC_mvt'].mean():.0f} s,"
          f" max {old.max():.0f} -> {sub.loc[hit, 'TAXITIME_SEC_mvt'].max():.0f} s; wrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "apply"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--scope", choices=["lirf", "other", "all"], default="lirf")
    ap.add_argument("--base")
    ap.add_argument("--out")
    a = ap.parse_args()
    {"validate": validate, "apply": apply}[a.cmd](a)
