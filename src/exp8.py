"""실험 B: CatBoost 튜닝 (exp2 와 같은 분할 seed 100+r).
1단계: 후보 5개를 rep 0 만으로 빠르게 평가 (기준: exp6 CatBoost rep0, 그리고 v5 혼합 .55 slow + .30 cb + .15 lr)
2단계: python exp8.py confirm c3 c5  → 지정 후보를 rep 1, 2 까지 돌려 3회 평균.
OOF: oof8_<name>_r<rep>.npy"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

from exp2 import CAT, evaluate
from exp3 import combos_ext
from exp2 import combos
from train_lgbm import fe

warnings.filterwarnings("ignore")
CFG = {
    "c1": dict(depth=4, learning_rate=0.05, l2_leaf_reg=5, iterations=1200, te12=False),
    "c2": dict(depth=8, learning_rate=0.04, l2_leaf_reg=10, iterations=800, te12=False),
    "c3": dict(depth=6, learning_rate=0.04, l2_leaf_reg=10, iterations=1000, te12=True),
    "c4": dict(depth=5, learning_rate=0.03, l2_leaf_reg=3, iterations=1500, te12=False),
    "c5": dict(depth=6, learning_rate=0.04, l2_leaf_reg=5, iterations=1000, te12=True),
}


def cb_frame(df, te12):
    X = fe(df)
    C = combos_ext(df) if te12 else combos(df)
    X = pd.concat([X, C], axis=1)
    cats = CAT + list(C.columns)
    for c in cats:
        X[c] = X[c].astype("Int64").astype(str) if c in CAT else X[c]
        X[c] = X[c].astype(object).where(X[c].notna(), "NA").astype(str)
    return X, cats


def make_cb(cfg, seed=0):
    p = {k: v for k, v in cfg.items() if k != "te12"}
    return CatBoostClassifier(rsm=0.5, random_seed=seed, thread_count=4, verbose=0, allow_writing_files=False, **p)


def run(name, reps, y, frames):
    cfg = CFG[name]
    X, cats = frames[cfg["te12"]]
    out = []
    for r in reps:
        t0, oof = time.time(), np.zeros(len(y))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(X, y):
            oof[va] = make_cb(cfg, r).fit(X.iloc[tr], y[tr], cat_features=cats).predict_proba(X.iloc[va])[:, 1]
        np.save(f"oof8_{name}_r{r}.npy", oof)
        sl, lr = np.load(f"oof5_slow_r{r}.npy"), np.load(f"oof2_M4_C0.02_r{r}.npy")
        a, bl = evaluate(y, oof)[2], evaluate(y, 0.55 * sl + 0.30 * oof + 0.15 * lr)[2]
        out.append((a, bl))
        print(f"{name} rep{r}: cb={a:.4f} blend={bl:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/valid.csv")], ignore_index=True)
    y = df["vdd"].values
    frames = {False: cb_frame(df, False), True: cb_frame(df, True)}
    for r in (0,):
        o6 = np.load(f"oof6_cb_r{r}.npy")
        sl, lr = np.load(f"oof5_slow_r{r}.npy"), np.load(f"oof2_M4_C0.02_r{r}.npy")
        print(f"baseline exp6 rep{r}: cb={evaluate(y, o6)[2]:.4f} blend={evaluate(y, .55*sl+.30*o6+.15*lr)[2]:.4f}")
    if sys.argv[1:2] == ["confirm"]:
        for name in sys.argv[2:]:
            res = run(name, (1, 2), y, frames)
            print(f"{name} reps1-2 done: {res}", flush=True)
    else:
        for name in sys.argv[1:] or list(CFG):
            run(name, (0,), y, frames)
