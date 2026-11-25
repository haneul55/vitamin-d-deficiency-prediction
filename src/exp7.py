"""실험 A: 조사 연도(기수) 추정 변수 (반복 CV 5-fold x 3, exp2 와 같은 분할, FAST_BEST LightGBM).
비교 기준: te6 (같은 설정, 새 변수 없음) 0.6856.
  A1 : 결측 묶음 자동 탐지 → 묶음 패턴 번호 bsig + bsig 조합 타깃 인코딩 (성별/나이5/지역)
  A3 : A1 + bsig 그룹 안에서 표준화한 검사 수치 (검사기기 차이 제거)
  A2 : A3 + 검사 수치 분포로 GMM 군집 (bsig 안의 연도 차이 추정용)
y 를 쓰지 않는 변수라 전체(train+valid) X 로 한 번 계산. OOF: oof7_<arm>_r<rep>.npy"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

from exp2 import combos, evaluate, run_fold
from exp3 import FAST_BEST
from train_lgbm import fe

warnings.filterwarnings("ignore")
LABS = ["he_bun", "he_crea", "he_ucrea", "he_ast", "he_alt", "he_bplt", "he_hb", "he_glu", "he_chol", "he_wbc"]


def find_blocks(R):
    """함께 비는 열 묶음 찾기 (y 미사용). 반환: 묶음별 열 목록."""
    X = R.drop(columns=[c for c in ["id", "vdd"] if c in R.columns])
    mr = X.isna().mean()
    cand = list(mr[(mr > 0.05) & (mr < 0.95)].index)
    pat = X[cand].isna().astype(float)
    d = 1 - pat.corr().abs().fillna(0).values
    np.fill_diagonal(d, 0)
    cl = pd.Series(fcluster(linkage(squareform(d, checks=False), "average"), 0.1, "distance"), index=cand)
    vc = cl.value_counts()
    return [list(cl[cl == b].index) for b in sorted(vc[vc >= 2].index)]


def bsig(R, blocks):
    return sum((R[cs].isna().mean(1) > 0.5).astype(int) * (2 ** i) for i, cs in enumerate(blocks)).astype(int)


def lab_stats(R, g):
    """bsig 그룹별 log 검사 수치 평균/표준편차 (inference 때 그대로 재사용)."""
    L = np.log1p(R[LABS].clip(lower=0))
    return L.groupby(g.values).mean(), L.groupby(g.values).std(), L.mean(), L.std()


def lab_z(R, g, st):
    mu, sd, gm, gs = st
    L = np.log1p(R[LABS].clip(lower=0))
    M = mu.reindex(g.values).set_index(R.index).fillna(gm)
    S = sd.reindex(g.values).set_index(R.index).fillna(gs).replace(0, np.nan)
    return ((L - M) / S).add_suffix("_gz")


def fit_gmm(R, k=6, seed=0):
    L = np.log1p(R[LABS].clip(lower=0))
    med = L.median()
    return GaussianMixture(k, covariance_type="full", random_state=seed).fit(L.fillna(med).values), med


def gmm_feat(R, gm):
    g, med = gm
    L = np.log1p(R[LABS].clip(lower=0)).fillna(med).values
    return pd.Series(g.predict(L), index=R.index)


def build(df, arm, blocks, st=None, gm=None):
    X, C = fe(df), combos(df)
    g = bsig(df, blocks)
    X["bsig"] = g
    s = df["sex"].astype("Int64").astype(str)
    age5 = (df["age"] // 5).astype("Int64").astype(str)
    gs = g.astype(str)
    C["te_bsig"] = gs
    C["te_bsig_sex"] = gs + "_" + s
    C["te_bsig_age5"] = gs + "_" + age5
    C["te_bsig_region"] = gs + "_" + df["region"].astype("Int64").astype(str)
    if arm == "Z":                       # 타깃 인코딩 조합 없이 그룹 표준화 검사 수치만
        C = combos(df)
        X = pd.concat([X, lab_z(df, g, st or lab_stats(df, g))], axis=1)
    if arm in ("A3", "A2"):
        X = pd.concat([X, lab_z(df, g, st or lab_stats(df, g))], axis=1)
    if arm == "A2":
        k = gmm_feat(df, gm or fit_gmm(df))
        X["gmm"] = k
        C["te_bsig_gmm"] = gs + "_" + k.astype(str)
    return X, C


if __name__ == "__main__":
    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/valid.csv")], ignore_index=True)
    y = df["vdd"].values
    blocks = find_blocks(df)
    print("blocks:", [len(b) for b in blocks], flush=True)
    for arm in sys.argv[1:] or ["A1", "A3", "A2"]:
        X, C = build(df, arm, blocks)
        t0, res = time.time(), []
        for r in (0, 1, 2):
            oof = np.zeros(len(y))
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(X, y):
                oof[va] = run_fold("M2", X, C, y, tr, va, r, FAST_BEST)
            np.save(f"oof7_{arm}_r{r}.npy", oof)
            res.append(evaluate(y, oof))
        m, s = np.mean(res, 0), np.std(res, 0)
        per = " ".join(f"{x[2]:.4f}" for x in res)
        print(f"{arm}: AUC={m[0]:.4f} F1={m[1]:.4f} SCORE={m[2]:.4f}±{s[2]:.4f} reps=[{per}] "
              f"(base te6 0.6856)  ({time.time() - t0:.0f}s)", flush=True)
