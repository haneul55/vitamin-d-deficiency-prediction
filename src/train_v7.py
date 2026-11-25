"""v7 / v8 학습 (train+valid 전체). 반복 CV 와 똑같은 설정으로 모델을 학습하고, 후보별 묶음을 저장.
  v7 (G_z+): 트리 .55 = mean(lgb_z, lgb_deep, lgb_et, lgb_lr01) | CatBoost .30 = mean(cb_z, cb) | 로지스틱 .15 = lr_z
             반복 CV 0.6898 (v5 0.6887 대비 +0.0011, 3회 모두 상승, AUC +0.0010)
  v8 (G_all): 트리 = mean(slow, lgb_z, lgb_deep, lgb_et, lgb_lr01, lgb_nomonth) | CB = mean(cb, cb_z, cb_slow) | LR = mean(lr, lr_z)
             반복 CV 0.6897 (+0.0009) — 가중치 선택이 전혀 없는 가장 보수적인 후보
슬롯 안 평균·.55/.30/.15 가중치는 v5 그대로(스태킹으로 확인한 값) → 혼합 가중치 선택 편향 거의 없음.
임계값 t 는 각 후보의 OOF(3회 평균 macro-F1)로 결정."""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import TargetEncoder

from exp2 import CAT, as_cat, combos, lgb_fast
from exp7 import LABS, bsig, build, find_blocks, lab_stats
from exp11 import SLOW
from exp_lr import make_lr
from train_lgbm import fe

ROOT = Path(__file__).resolve().parent
LGB = {"lgb_z": (SLOW, "Z", None),
       "lgb_deep": (dict(SLOW, num_leaves=63, min_child_samples=100, colsample_bytree=0.3), "base", None),
       "lgb_et": (dict(SLOW, extra_trees=True, n_estimators=800), "base", None),
       "lgb_lr01": (dict(SLOW, learning_rate=0.01, n_estimators=1000), "base", None),
       "lgb_nomonth": (SLOW, "base", ["te_sex_month"])}
CB = {"cb_z": (dict(iterations=1000, learning_rate=0.04, depth=6, l2_leaf_reg=5), "Z"),
      "cb_slow": (dict(iterations=2000, learning_rate=0.02, depth=6, l2_leaf_reg=5), "base")}
CANDS = {
    "v7": {"t": (.55, ["lgb_z", "lgb_deep", "lgb_et", "lgb_lr01"]), "c": (.30, ["cb_z", "cb"]), "l": (.15, ["lr_z"])},
    "v8": {"t": (.55, ["slow", "lgb_z", "lgb_deep", "lgb_et", "lgb_lr01", "lgb_nomonth"]),
           "c": (.30, ["cb", "cb_z", "cb_slow"]), "l": (.15, ["lr", "lr_z"])},
}
OOF = {"slow": "oof5_slow", "cb": "oof6_cb", "lr": "oof2_M4_C0.02"}


def cb_cast(X, C):
    X = pd.concat([X, C], axis=1)
    for c in CAT + list(C.columns):
        X[c] = X[c].astype("Int64").astype(str) if c in CAT else X[c]
        X[c] = X[c].astype(object).where(X[c].notna(), "NA").astype(str)
    return X, CAT + list(C.columns)


def best_t(groups):
    y = np.load(ROOT / "oof2_y.npy")
    ts = np.arange(0.55, 0.70, 0.0025)
    f = np.zeros(len(ts))
    for r in range(3):
        p = sum(w * np.mean([np.load(ROOT / f"{OOF.get(m, 'oof11_' + m)}_r{r}.npy") for m in ms], 0)
                for w, ms in groups.values())
        f += [f1_score(y, (p >= t).astype(int), average="macro") for t in ts]
    return float(ts[f.argmax()]), float(f.max() / 3)


if __name__ == "__main__":
    t0 = time.time()
    df = pd.concat([pd.read_csv(ROOT / "data/train.csv"), pd.read_csv(ROOT / "data/valid.csv")], ignore_index=True)
    y = df["vdd"].values
    v5 = joblib.load(ROOT / "submit_v5/model.joblib")
    blocks = find_blocks(df)
    g = bsig(df, blocks)
    mu, sd, gm, gs = lab_stats(df, g)
    z = {"blocks": blocks, "groups": [int(i) for i in mu.index], "mu": mu[LABS].values, "sd": sd[LABS].values,
         "gm": gm[LABS].values, "gs": gs[LABS].values}
    F = {"base": (fe(df), combos(df)), "Z": build(df, "Z", blocks)}
    cats = {c: list(as_cat(F["base"][0])[c].cat.categories) for c in CAT}
    M = {}

    for name, (params, feat, drop) in LGB.items():
        X, C = F[feat]
        C = C.drop(columns=drop) if drop else C
        Xt = as_cat(X)
        models, te0 = [], None
        for s in range(3):
            te = TargetEncoder(target_type="binary", random_state=s)
            a = te.fit_transform(C, y)
            te0 = te0 or te
            Xs = pd.concat([Xt, pd.DataFrame(a, index=Xt.index, columns=C.columns)], axis=1)
            models.append(lgb_fast(s, **params).fit(Xs, y))
        M[name] = {"kind": "lgb", "feat": feat, "te": te0, "te_cols": list(C.columns), "cols": list(Xs.columns),
                   "models": models}
        print(f"{name} done ({time.time() - t0:.0f}s)", flush=True)

    M["slow"] = {"kind": "lgb", "feat": "base", "te": v5["te"], "te_cols": v5["te_cols"], "cols": v5["tree_cols"],
                 "models": v5["trees"]}
    M["cb"] = {"kind": "cb", "feat": "base", "cols": v5["cb_cols"], "models": v5["cbs"]}
    M["lr"] = {"kind": "lr", "feat": "base", "cols": v5["lr_cols"], "models": [v5["lr"]]}

    X, C = F["Z"]
    Xz = X.loc[:, X.notna().any()]
    M["lr_z"] = {"kind": "lr", "feat": "Z", "cols": list(Xz.columns), "models": [make_lr(list(Xz.columns), 0.02).fit(Xz, y)]}
    print(f"lr_z done ({time.time() - t0:.0f}s)", flush=True)

    from catboost import CatBoostClassifier
    for name, (cfg, feat) in CB.items():
        Xc, cc = cb_cast(*F[feat])
        models = []
        for s in range(3):
            models.append(CatBoostClassifier(rsm=0.5, random_seed=s, thread_count=4, verbose=0,
                                             allow_writing_files=False, **cfg).fit(Xc, y, cat_features=cc))
            print(f"{name} seed {s} done ({time.time() - t0:.0f}s)", flush=True)
        M[name] = {"kind": "cb", "feat": feat, "cols": list(Xc.columns), "models": models}

    joblib.dump({"M": M, "z": z, "cats": cats}, ROOT / "models_all_v7v8.joblib", compress=3)
    for cand, groups in CANDS.items():
        t, f = best_t(groups)
        need = {m for _, ms in groups.values() for m in ms}
        out = ROOT / f"submit_{cand}"
        out.mkdir(exist_ok=True)
        joblib.dump({"M": {k: M[k] for k in need}, "groups": groups, "z": z, "cats": cats, "t": t},
                    out / "model.joblib", compress=3)
        print(f"{cand}: t={t:.4f} (OOF macro-F1 {f:.4f}) saved {out}", flush=True)
