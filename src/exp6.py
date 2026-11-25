"""CatBoost 반복 CV (5-fold x 3, 같은 분할). 범주형 10개 + 조합 6개를 CatBoost 자체 인코딩(ordered target stats)에 맡김.
비교: 트리(te6 빠른 설정) 0.6857, 느린 설정 0.6875, 트리+로지스틱 0.6881(느린 설정).
OOF: oof6_cb_r<rep>.npy"""
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

from exp2 import CAT, combos, evaluate
from train_lgbm import fe

warnings.filterwarnings("ignore")


def cb_frame(df):
    X = fe(df)
    C = combos(df)
    X = pd.concat([X, C], axis=1)
    cats = CAT + list(C.columns)
    for c in cats:                                     # pandas 3.0: Int64→str 변환 후에도 NA 가 남음 → "NA" 문자열로
        X[c] = X[c].astype("Int64").astype(str) if c in CAT else X[c]
        X[c] = X[c].astype(object).where(X[c].notna(), "NA").astype(str)
    return X, cats


def make_cb(seed=0):
    return CatBoostClassifier(iterations=1000, learning_rate=0.04, depth=6, l2_leaf_reg=5, rsm=0.5,
                              random_seed=seed, thread_count=4, verbose=0, allow_writing_files=False)


if __name__ == "__main__":
    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/valid.csv")], ignore_index=True)
    y = df["vdd"].values
    X, cats = cb_frame(df)
    res, rb, rb3 = [], [], []
    for r in (0, 1, 2):
        t0, oof = time.time(), np.zeros(len(y))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(X, y):
            oof[va] = make_cb(r).fit(X.iloc[tr], y[tr], cat_features=cats).predict_proba(X.iloc[va])[:, 1]
        np.save(f"oof6_cb_r{r}.npy", oof)
        lr, lg = np.load(f"oof2_M4_C0.02_r{r}.npy"), np.load(f"oof5_slow_r{r}.npy")
        res.append(evaluate(y, oof))
        rb.append(evaluate(y, 0.75 * oof + 0.25 * lr))
        rb3.append(evaluate(y, 0.4 * oof + 0.4 * lg + 0.2 * lr))
        print(f"rep {r}: cb={res[-1][2]:.4f} corr(cb,lgb)={np.corrcoef(oof, lg)[0, 1]:.3f} ({time.time() - t0:.0f}s)",
              flush=True)
    for name, rr in [("CatBoost alone", res), ("CB.75+LR.25", rb), ("CB.4+LGB.4+LR.2", rb3)]:
        m, s = np.mean(rr, 0), np.std(rr, 0)
        print(f"{name:16s} AUC={m[0]:.4f} F1={m[1]:.4f} SCORE={m[2]:.4f}±{s[2]:.4f}")
