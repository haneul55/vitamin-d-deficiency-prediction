"""M4: 트리와 성격이 다른 선형 모델 (앙상블 다양성용).
- 범주형: 원-핫, 나이/BMI 등: 스플라인, 나머지 수치: 결측표시 + 중앙값 대체 + 표준화
- 같은 반복 CV(5-fold x 3, seed 100+r) 로 OOF 저장: oof2_M4_r<rep>.npy"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

from exp2 import CAT, evaluate
from train_lgbm import fe

warnings.filterwarnings("ignore")
SPL = ["age", "he_bmi", "he_bun", "he_crea", "he_wc", "he_hb", "he_ast"]


def make_lr(cols, C=0.05):
    num = [c for c in cols if c not in CAT + SPL]
    pre = ColumnTransformer([
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value=-9),
                              OneHotEncoder(handle_unknown="ignore", min_frequency=30)), CAT),
        # SplineTransformer 는 scipy BSpline 객체를 저장해 채점 서버(다른 scipy 버전)에서 로딩 실패 → 구간 원-핫으로 대체
        ("spl", make_pipeline(SimpleImputer(strategy="median"),
                              KBinsDiscretizer(n_bins=8, encode="onehot", strategy="quantile",
                                               quantile_method="averaged_inverted_cdf")), SPL),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
    ])
    return make_pipeline(pre, LogisticRegression(C=C, max_iter=3000))


if __name__ == "__main__":
    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/valid.csv")], ignore_index=True)
    y = df["vdd"].values
    X = fe(df)
    X = X.loc[:, X.notna().any()]
    for C in [float(c) for c in (sys.argv[1:] or ["0.05"])]:
        t0, res = time.time(), []
        for r in (0, 1, 2):
            oof = np.zeros(len(y))
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(X, y):
                oof[va] = make_lr(list(X.columns), C).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
            np.save(f"oof2_M4_C{C}_r{r}.npy", oof)
            res.append(evaluate(y, oof))
        m, s = np.mean(res, 0), np.std(res, 0)
        print(f"M4 C={C}: AUC={m[0]:.4f}±{s[0]:.4f} F1={m[1]:.4f}±{s[1]:.4f} SCORE={m[2]:.4f}±{s[2]:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
