"""중첩 검증: 각 fold 의 학습 데이터만으로 null importance 피처 선택(thr=0) → 검증 fold 예측.
선택 과정의 정보 누수 없이 피처 선택 효과를 측정 (exp4 의 낙관 편향 확인용)."""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

from exp2 import CAT, combos, evaluate, lgb_fast, run_fold
from exp3 import FAST_BEST
from train_lgbm import fe

warnings.filterwarnings("ignore")
N_NULL = 10


def select(X, C, y, seed):
    Xt = X.copy()
    for c in CAT:
        Xt[c] = Xt[c].astype("category")
    te = TargetEncoder(target_type="binary", random_state=seed).fit_transform(C, y)
    Xt = pd.concat([Xt, pd.DataFrame(te, index=Xt.index, columns=C.columns)], axis=1)
    g = lambda yy, s: pd.Series(lgb_fast(s, **FAST_BEST).fit(Xt, yy).booster_.feature_importance("gain"),
                                index=Xt.columns)
    real = g(y, seed)
    rng = np.random.default_rng(seed)
    null = pd.concat([g(rng.permutation(y), 1000 + i) for i in range(N_NULL)], axis=1)
    score = np.log((real + 1) / (null.quantile(0.75, axis=1) + 1))[X.columns]
    return [c for c in X.columns if score[c] > 0 or c in CAT]


if __name__ == "__main__":
    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/valid.csv")], ignore_index=True)
    y, X, C = df["vdd"].values, fe(df), combos(df)
    res = []
    for r in (0, 1, 2):
        t0, oof, nsel = time.time(), np.zeros(len(y)), []
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(X, y):
            keep = select(X.iloc[tr], C.iloc[tr], y[tr], r)
            nsel.append(len(keep))
            oof[va] = run_fold("M2", X[keep], C, y, tr, va, r, FAST_BEST)
        np.save(f"oof4_nested_r{r}.npy", oof)
        res.append(evaluate(y, oof))
        print(f"rep {r}: SCORE={res[-1][2]:.4f} vars/fold={nsel} ({time.time() - t0:.0f}s)", flush=True)
    m, s = np.mean(res, 0), np.std(res, 0)
    print(f"NESTED: AUC={m[0]:.4f} F1={m[1]:.4f} SCORE={m[2]:.4f}±{s[2]:.4f}  (baseline te6 0.6857±0.0007)")
