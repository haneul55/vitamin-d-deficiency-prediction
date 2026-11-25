# ===== 고정 (수정 금지) : 데이터 읽기 =====
import sys
from pathlib import Path

import pandas as pd

path = sys.argv[1]                                   # 사용법: python inference.py 데이터.csv  (.tsv 도 가능)
df = pd.read_csv(path, sep="\t" if path.lower().endswith(".tsv") else ",")   # df: id + 설명변수 (정답 vdd 없음)

# ===== 아래부터 커스터마이징 하세요 =====
import joblib
import numpy as np


# (1) 전처리 — train_lgbm.fe(), exp2.combos() 와 동일
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


def combos(R):
    s = R["sex"].astype("Int64").astype(str)
    age5 = (R["age"] // 5).astype("Int64").astype(str)
    g = lambda c: R[c].astype("Int64").astype(str)
    return pd.DataFrame({
        "te_region_town": g("region") + "_" + g("town_t"),
        "te_occp_sex": g("occp") + "_" + s,
        "te_sex_age5": s + "_" + age5,
        "te_sex_month": s + "_" + g("dr_month"),
        "te_region_occp": g("region") + "_" + g("occp"),
        "te_ecoccp_sex": g("ec_occp") + "_" + s,
    }, index=R.index)


CAT = ["region", "occp", "ec_occp", "marri_2", "house", "live_t", "cfam", "genertn", "lk_lb_it", "n_diet_why"]
LABS = ["he_bun", "he_crea", "he_ucrea", "he_ast", "he_alt", "he_bplt", "he_hb", "he_glu", "he_chol", "he_wbc"]


def zfeat(R, X, z):
    """exp7.build(arm="Z") 와 동일: 결측 묶음 패턴 번호 bsig + bsig 그룹 안에서 표준화한 log 검사 수치.
    묶음 목록·그룹별 평균/표준편차는 학습 데이터에서 구해 model.joblib 에 저장한 값 사용."""
    X = X.copy()
    g = sum((R[cs].isna().mean(1) > 0.5).astype(int) * (2 ** i) for i, cs in enumerate(z["blocks"])).astype(int)
    X["bsig"] = g
    L = np.log1p(R[LABS].clip(lower=0))
    mu = pd.DataFrame(z["mu"], index=z["groups"], columns=LABS)
    sd = pd.DataFrame(z["sd"], index=z["groups"], columns=LABS)
    gm, gs = pd.Series(z["gm"], index=LABS), pd.Series(z["gs"], index=LABS)
    M = mu.reindex(g.values).set_index(R.index).fillna(gm)
    S = sd.reindex(g.values).set_index(R.index).fillna(gs).replace(0, np.nan)
    return pd.concat([X, ((L - M) / S).add_suffix("_gz")], axis=1)


def cb_cast(X, C):
    """exp6.cb_frame() 과 동일 (범주형 10개 + 조합 변수를 문자열로)."""
    X = pd.concat([X, C], axis=1)
    for c in CAT + list(C.columns):
        X[c] = X[c].astype("Int64").astype(str) if c in CAT else X[c]
        X[c] = X[c].astype(object).where(X[c].notna(), "NA").astype(str)
    return X


# (2) 모델 로딩 — model.joblib = {M: 모델 종류별 {kind, feat, cols, models, (te)}, groups: 슬롯별 (가중치, 모델 이름들), z, cats, t}
b = joblib.load(Path(__file__).resolve().parent / "model.joblib")
X0, C0 = fe(df), combos(df)
F = {"base": X0, "Z": zfeat(df, X0, b["z"])}


def predict(m):
    X = F[m["feat"]]
    if m["kind"] == "lgb":                          # 범주형(학습 때 범주 목록 그대로) + 타깃 인코딩
        Xt = X.copy()
        for c, cats in b["cats"].items():
            Xt[c] = pd.Categorical(Xt[c], categories=cats)
        te = pd.DataFrame(m["te"].transform(C0[m["te_cols"]]), index=Xt.index, columns=m["te_cols"])
        Xt = pd.concat([Xt, te], axis=1)[m["cols"]]
    elif m["kind"] == "cb":
        Xt = cb_cast(X, C0)[m["cols"]]
    else:
        Xt = X[m["cols"]]
    return np.mean([mm.predict_proba(Xt)[:, 1] for mm in m["models"]], axis=0)


# (3) 예측 — 슬롯(LightGBM .55 / CatBoost .30 / 로지스틱 .15) 마다 모델 평균 → 가중합 → 임계값 t 가 0.5 가 되도록 단조 변환
P = {k: predict(m) for k, m in b["M"].items()}
p = sum(w * np.mean([P[k] for k in ks], axis=0) for w, ks in b["groups"].values())
t = b["t"]
pred = np.where(p < t, 0.5 * p / t, 0.5 + 0.5 * (p - t) / (1 - t))
# ===== 여기까지 =====

# ===== 고정 (수정 금지) : 제출 형식으로 저장 =====
pd.DataFrame({"id": df["id"], "vdd": pred}).to_csv("submission.csv", index=False)
