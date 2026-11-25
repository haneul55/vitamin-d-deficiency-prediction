"""2차 개선 실험: 반복 CV(5-fold x 3) 로 M0/M1/M2 비교.
M0 = 현재 LightGBM 피처, M1 = + 범주형 지정, M2 = M1 + 타깃 인코딩(조합 변수).
결과 OOF 는 oof2_<name>_r<rep>.npy 로 저장."""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

from train_lgbm import fe

warnings.filterwarnings("ignore")

CAT = ["region", "occp", "ec_occp", "marri_2", "house", "live_t", "cfam", "genertn", "lk_lb_it", "n_diet_why"]


def combos(R):
    """타깃 인코딩할 조합 변수 (문자열 키)."""
    age5 = (R["age"] // 5).astype("Int64").astype(str)
    s = R["sex"].astype("Int64").astype(str)
    return pd.DataFrame({
        "te_region_town": R["region"].astype("Int64").astype(str) + "_" + R["town_t"].astype("Int64").astype(str),
        "te_occp_sex": R["occp"].astype("Int64").astype(str) + "_" + s,
        "te_sex_age5": s + "_" + age5,
        "te_sex_month": s + "_" + R["dr_month"].astype("Int64").astype(str),
        "te_region_occp": R["region"].astype("Int64").astype(str) + "_" + R["occp"].astype("Int64").astype(str),
        "te_ecoccp_sex": R["ec_occp"].astype("Int64").astype(str) + "_" + s,
    }, index=R.index)


def as_cat(X):
    X = X.copy()
    for c in CAT:
        X[c] = X[c].astype("category")
    return X


def lgb_fast(seed=0, **kw):
    p = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.4, reg_lambda=2.0, verbose=-1, random_state=seed, n_jobs=4,
             cat_smooth=20, min_data_per_group=50)
    p.update(kw)
    return lgb.LGBMClassifier(**p)


def run_fold(name, X, C, y, tr, va, seed, params):
    Xtr, Xva = X.iloc[tr], X.iloc[va]
    if name in ("M1", "M2"):
        Xtr, Xva = as_cat(Xtr), as_cat(Xva)
        # 학습 fold 에 없는 범주를 맞추기 위해 카테고리 목록 통일
        for c in CAT:
            cats = Xtr[c].cat.categories
            Xva[c] = pd.Categorical(Xva[c].astype(float), categories=cats)
    if name == "M2":
        te = TargetEncoder(target_type="binary", random_state=seed)
        a = te.fit_transform(C.iloc[tr], y[tr])          # 학습 fold 안에서 cross-fitting (누수 방지)
        b = te.transform(C.iloc[va])
        Xtr = pd.concat([Xtr, pd.DataFrame(a, index=Xtr.index, columns=C.columns)], axis=1)
        Xva = pd.concat([Xva, pd.DataFrame(b, index=Xva.index, columns=C.columns)], axis=1)
    m = lgb_fast(seed, **params).fit(Xtr, y[tr])
    return m.predict_proba(Xva)[:, 1]


def evaluate(y, oof):
    ts = np.arange(0.5, 0.75, 0.005)
    f1 = max(f1_score(y, (oof >= t).astype(int), average="macro") for t in ts)
    auc = roc_auc_score(y, oof)
    return auc, f1, (auc + f1) / 2


def repeated_cv(name, X, C, y, params=None, reps=(0, 1, 2), save=True):
    params = params or {}
    res = []
    for r in reps:
        oof = np.zeros(len(y))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(X, y):
            oof[va] = run_fold(name, X, C, y, tr, va, r, params)
        if save:
            np.save(f"oof2_{name}_r{r}.npy", oof)
        res.append(evaluate(y, oof))
    res = np.array(res)
    return res.mean(0), res.std(0)


if __name__ == "__main__":
    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/valid.csv")], ignore_index=True)
    y = df["vdd"].values
    X = fe(df)
    C = combos(df)
    np.save("oof2_y.npy", y)
    for name in sys.argv[1:] or ["M0", "M1", "M2"]:
        t0 = time.time()
        (auc, f1, sc), (sa, sf, ss) = repeated_cv(name, X, C, y)
        print(f"{name}: AUC={auc:.4f}±{sa:.4f}  F1={f1:.4f}±{sf:.4f}  SCORE={sc:.4f}±{ss:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
