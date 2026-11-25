"""실험 C: XGBoost (반복 CV 5-fold x 3, exp2 와 같은 분할). 입력은 M2 와 동일 (fe + 범주형 + 타깃 인코딩 6개).
v5 구성(slow .55 + cb .30 + lr .15)에 XGBoost 를 섞었을 때 점수를 격자로 비교. OOF: oof9_xgb_r<rep>.npy"""
import itertools
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

from exp2 import CAT, combos, evaluate
from train_lgbm import fe

warnings.filterwarnings("ignore")
PARAMS = dict(n_estimators=600, learning_rate=0.02, max_depth=5, min_child_weight=20, subsample=0.7,
              colsample_bytree=0.4, reg_lambda=5.0, tree_method="hist", enable_categorical=True,
              max_cat_to_onehot=1, n_jobs=4, eval_metric="auc")


def fold(X, C, y, tr, va, seed):
    Xtr, Xva = X.iloc[tr].copy(), X.iloc[va].copy()
    for c in CAT:
        Xtr[c], Xva[c] = Xtr[c].astype("Int64"), Xva[c].astype("Int64")      # XGBoost: 범주 값은 정수여야 함
        cats = sorted(int(v) for v in Xtr[c].dropna().unique())
        Xtr[c] = pd.Categorical(Xtr[c], categories=cats)
        Xva[c] = pd.Categorical(Xva[c], categories=cats)
    te = TargetEncoder(target_type="binary", random_state=seed)
    a, b = te.fit_transform(C.iloc[tr], y[tr]), te.transform(C.iloc[va])
    Xtr = pd.concat([Xtr, pd.DataFrame(a, index=Xtr.index, columns=C.columns)], axis=1)
    Xva = pd.concat([Xva, pd.DataFrame(b, index=Xva.index, columns=C.columns)], axis=1)
    return xgb.XGBClassifier(random_state=seed, **PARAMS).fit(Xtr, y[tr]).predict_proba(Xva)[:, 1]


if __name__ == "__main__":
    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/valid.csv")], ignore_index=True)
    y, X, C = df["vdd"].values, fe(df), combos(df)
    O = {k: [] for k in ("xgb", "slow", "cb", "lr")}
    for r in (0, 1, 2):
        t0, oof = time.time(), np.zeros(len(y))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(X, y):
            oof[va] = fold(X, C, y, tr, va, r)
        np.save(f"oof9_xgb_r{r}.npy", oof)
        O["xgb"].append(oof)
        O["slow"].append(np.load(f"oof5_slow_r{r}.npy"))
        O["cb"].append(np.load(f"oof6_cb_r{r}.npy"))
        O["lr"].append(np.load(f"oof2_M4_C0.02_r{r}.npy"))
        print(f"rep{r}: xgb={evaluate(y, oof)[2]:.4f} corr(xgb,lgb)={np.corrcoef(oof, O['slow'][-1])[0, 1]:.3f} "
              f"corr(xgb,cb)={np.corrcoef(oof, O['cb'][-1])[0, 1]:.3f} ({time.time() - t0:.0f}s)", flush=True)

    m = np.mean([evaluate(y, o) for o in O["xgb"]], 0)
    print(f"XGB alone: AUC={m[0]:.4f} F1={m[1]:.4f} SCORE={m[2]:.4f}")
    rows = []
    for wx, wc, wl in itertools.product([0, .1, .2, .3, .4], [0, .1, .2, .3], [.1, .15, .2]):
        ws = 1 - wx - wc - wl
        if ws < 0:
            continue
        sc = [evaluate(y, ws * O["slow"][r] + wc * O["cb"][r] + wl * O["lr"][r] + wx * O["xgb"][r])[2]
              for r in range(3)]
        rows.append((np.mean(sc), np.std(sc), ws, wc, wl, wx, sc))
    rows.sort(key=lambda r: -r[0])
    v5 = [r for r in rows if r[5] == 0 and abs(r[2] - .55) < 1e-9 and abs(r[3] - .3) < 1e-9 and abs(r[4] - .15) < 1e-9]
    if v5:
        print(f"v5 blend: SCORE={v5[0][0]:.4f} reps={[round(s, 4) for s in v5[0][6]]}")
    for s, sd, ws, wc, wl, wx, sc in rows[:8]:
        print(f"lgb={ws:.2f} cb={wc:.2f} lr={wl:.2f} xgb={wx:.2f} SCORE={s:.4f}±{sd:.4f} reps={[round(v, 4) for v in sc]}")
