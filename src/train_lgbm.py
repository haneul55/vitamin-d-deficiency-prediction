"""LightGBM 최종 모델: 결측패턴 + 비율 피처, 3-seed 평균.
5-fold OOF 로 임계값 t 를 찾고, 전체 데이터로 3개 seed 학습 → submit_lgbm/model.joblib.
fe() 는 submit_lgbm/inference.py 와 반드시 동일하게 유지할 것."""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submit_lgbm"
OUT.mkdir(exist_ok=True)
SEEDS = [0, 1, 2]


def fe(R):
    R = R.drop(columns=[c for c in ["id", "vdd"] if c in R.columns])
    X = R.drop(columns=["he_dprg", "he_mpls"]).replace(-2, np.nan)
    new = {"n_nan": R.isna().sum(1), "n_neg1": (R == -1).sum(1), "n_neg2": (R == -2).sum(1)}
    groups = {"bm": [c for c in R if c.startswith("bm1_")], "lk": ["lk_edu", "lk_lb_it", "lk_lb_ef"],
              "hba1c": ["he_hba1c"], "seat": ["sc_seatblt"], "htg": ["he_htg"], "ucrea": ["he_ucrea"],
              "ba": ["ba1_1", "ba1_2", "ba1_3"], "n": ["n_ca", "n_en"], "or": ["or1", "or1_2"],
              "sm": ["sm_presnt"], "ndiet": ["n_diet_why"], "mens": ["he_mens"], "prg": ["he_prg"]}
    sig = np.zeros(len(R), dtype=np.int64)
    for i, cs in enumerate(groups.values()):
        f = R[cs].isna().all(1).astype(int)
        new[f"miss_{i}"] = f
        sig += f.values << i
    new["miss_sig"] = sig
    new["bun_crea"] = R["he_bun"] / R["he_crea"]
    new["ast_alt"] = R["he_ast"] / R["he_alt"]
    new["tg_hdl"] = R["he_tg"] / R["he_hdl_st2"]
    en = R["n_en"].where(R["n_en"] > 0)
    for c in ["n_prot", "n_fat", "n_cho", "n_ca", "n_phos", "n_na", "n_k", "n_fe", "n_carot", "n_retin",
              "n_b1", "n_b2", "n_niac", "n_vitc"]:
        new[f"{c}_per_en"] = R[c] / en
    X = pd.concat([X, pd.DataFrame(new, index=R.index)], axis=1)
    return X.replace([np.inf, -np.inf], np.nan)


def make(seed):
    return lgb.LGBMClassifier(n_estimators=800, learning_rate=0.02, num_leaves=31, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.4, reg_lambda=2.0,
                              verbose=-1, random_state=seed, n_jobs=4)


def calibrate(p, t):
    return np.where(p < t, 0.5 * p / t, 0.5 + 0.5 * (p - t) / (1 - t))


if __name__ == "__main__":
    df = pd.concat([pd.read_csv(ROOT / "data/train.csv"), pd.read_csv(ROOT / "data/valid.csv")], ignore_index=True)
    y = df["vdd"].values
    X = fe(df)
    oof = np.zeros(len(y))
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
        oof[va] = make(0).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
        print("fold", k, flush=True)
    ts = np.arange(0.4, 0.8, 0.005)
    t = float(ts[np.argmax([f1_score(y, (oof >= s).astype(int), average="macro") for s in ts])])
    pc = calibrate(oof, t)
    auc, f1 = roc_auc_score(y, pc), f1_score(y, (pc >= 0.5).astype(int), average="macro")
    print(f"OOF AUC={auc:.4f} F1@0.5(calibrated)={f1:.4f} t={t:.3f} SCORE={(auc + f1) / 2:.4f}")
    models = [make(s).fit(X, y) for s in SEEDS]
    joblib.dump({"models": models, "cols": list(X.columns), "t": t}, OUT / "model.joblib", compress=3)
    print("saved", OUT / "model.joblib")
