"""마지막 라운드 (반복 CV 5-fold x 3, 같은 분할). 기준: te6 트리 단독 0.6857, 트리.75+로지스틱.25 0.6868.
  slow : 학습률 0.02 / 트리 500 / fold 마다 seed 3개 평균
  knn  : 비슷한 사람 50명의 결핍률 피처 추가 (학습 fold 는 내부 5-fold 로 OOF 계산 → 누수 방지)
각 결과를 로지스틱 OOF 와 0.75:0.25 로 섞은 점수도 같이 출력."""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from exp2 import combos, evaluate, run_fold
from exp3 import FAST_BEST
from train_lgbm import fe

warnings.filterwarnings("ignore")
KNN_COLS = ["age", "sex", "region", "town_t", "occp", "he_bmi", "he_wc", "dr_month", "ho_incm5", "edu"]
SLOW = dict(FAST_BEST, learning_rate=0.02, n_estimators=500)


def make_knn():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         KNeighborsClassifier(n_neighbors=50, weights="uniform", n_jobs=4))


def knn_feature(K, y, tr, va, seed):
    """train 행: 내부 OOF, val 행: 학습 fold 전체로 적합한 kNN 예측."""
    f = np.full(len(y), np.nan)
    for itr, iva in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y[tr]):
        a, b = tr[itr], tr[iva]
        f[b] = make_knn().fit(K.iloc[a], y[a]).predict_proba(K.iloc[b])[:, 1]
    f[va] = make_knn().fit(K.iloc[tr], y[tr]).predict_proba(K.iloc[va])[:, 1]
    return f


if __name__ == "__main__":
    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/valid.csv")], ignore_index=True)
    y, X, C = df["vdd"].values, fe(df), combos(df)
    K = X[KNN_COLS]
    for name in sys.argv[1:] or ["slow", "knn"]:
        t0, res, resb = time.time(), [], []
        for r in (0, 1, 2):
            oof = np.zeros(len(y))
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(X, y):
                if name == "slow":
                    oof[va] = np.mean([run_fold("M2", X, C, y, tr, va, 10 * r + s, SLOW) for s in range(3)], axis=0)
                else:
                    Xk = X.copy()
                    Xk["knn50"] = knn_feature(K, y, tr, va, r)
                    oof[va] = run_fold("M2", Xk, C, y, tr, va, r, FAST_BEST)
            np.save(f"oof5_{name}_r{r}.npy", oof)
            res.append(evaluate(y, oof))
            resb.append(evaluate(y, 0.75 * oof + 0.25 * np.load(f"oof2_M4_C0.02_r{r}.npy")))
        m, s = np.mean(res, 0), np.std(res, 0)
        mb, sb = np.mean(resb, 0), np.std(resb, 0)
        print(f"{name:5s} tree: SCORE={m[2]:.4f}±{s[2]:.4f} (AUC {m[0]:.4f})  |  +lr blend: SCORE={mb[2]:.4f}±{sb[2]:.4f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
