"""피처 엔지니어링 실험: LightGBM 5-fold CV (빠름) 로 피처 세트 비교."""
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/valid.csv")], ignore_index=True)
y = df["vdd"].values
RAW = df.drop(columns=["id", "vdd"])


def fe(R, sets):
    X = R.drop(columns=["he_dprg", "he_mpls"]).replace(-2, np.nan)
    new = {}
    if "miss" in sets:          # 결측 패턴 → 조사 기수(연도) 대리 변수
        new["n_nan"] = R.isna().sum(1)
        new["n_neg1"] = (R == -1).sum(1)
        new["n_neg2"] = (R == -2).sum(1)
        groups = {"bm": [c for c in R if c.startswith("bm1_")], "lk": ["lk_edu", "lk_lb_it", "lk_lb_ef"],
                  "hba1c": ["he_hba1c"], "seat": ["sc_seatblt"], "htg": ["he_htg"], "ucrea": ["he_ucrea"],
                  "ba": ["ba1_1", "ba1_2", "ba1_3"], "n": ["n_ca", "n_en"], "or": ["or1", "or1_2"],
                  "sm": ["sm_presnt"], "ndiet": ["n_diet_why"], "mens": ["he_mens"], "prg": ["he_prg"]}
        sig = np.zeros(len(R), dtype=np.int64)
        for i, (k, cs) in enumerate(groups.items()):
            cs = [c for c in cs if c in R]
            f = R[cs].isna().all(1).astype(int)
            new[f"miss_{k}"] = f
            sig += f.values << i
        new["miss_sig"] = sig
    if "ratio" in sets:         # 임상/영양 비율
        new["bun_crea"] = R["he_bun"] / R["he_crea"]
        new["ast_alt"] = R["he_ast"] / R["he_alt"]
        new["tg_hdl"] = R["he_tg"] / R["he_hdl_st2"]
        en = R["n_en"].where(R["n_en"] > 0)
        for c in ["n_prot", "n_fat", "n_cho", "n_ca", "n_phos", "n_na", "n_k", "n_fe", "n_carot", "n_retin",
                  "n_b1", "n_b2", "n_niac", "n_vitc"]:
            new[f"{c}_per_en"] = R[c] / en
    if "sun" in sets:           # 햇빛 노출 대리 변수 조합
        new["outdoor_job"] = (R["occp"] == 4).astype(int)
        new["rural_x_job"] = R["town_t"] * 10 + R["occp"].fillna(0)
        new["sex_age"] = R["sex"] * 100 + R["age"]
        new["sex_x_month"] = R["sex"] * 10 + R["dr_month"].fillna(-1)
    return pd.concat([X, pd.DataFrame(new, index=R.index)], axis=1)


def params(**kw):
    p = dict(n_estimators=800, learning_rate=0.02, num_leaves=31, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.4, reg_lambda=2.0, verbose=-1, random_state=0, n_jobs=4)
    p.update(kw)
    return p


skf = list(StratifiedKFold(5, shuffle=True, random_state=42).split(RAW, y))


def cv(X, p):
    oof = np.zeros(len(y))
    for tr, va in skf:
        m = lgb.LGBMClassifier(**p).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    ts = np.arange(0.4, 0.8, 0.005)
    f1s = [f1_score(y, (oof >= t).astype(int), average="macro") for t in ts]
    auc = roc_auc_score(y, oof)
    return auc, max(f1s), ts[int(np.argmax(f1s))], oof


if __name__ == "__main__":
    for sets in [(), ("miss",), ("ratio",), ("sun",), ("miss", "ratio", "sun")]:
        X = fe(RAW, sets)
        auc, f1, t, _ = cv(X, params())
        print(f"{'+'.join(sets) or 'base':20s} AUC={auc:.4f} F1={f1:.4f} t={t:.3f} SCORE={(auc+f1)/2:.4f}", flush=True)
