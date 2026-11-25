"""최종 모델 학습: train+valid 전체로 HistGradientBoosting 학습 → submit/model.joblib 저장.
1) 5-fold OOF 예측으로 macro-F1 최적 임계값 t 를 찾고
2) 전체 데이터로 다시 학습한 뒤 {model, cols, t} 를 저장.
inference.py 는 확률 p 를 t→0.5 로 옮기는 단조 변환을 적용합니다 (AUROC 불변, F1@0.5 향상)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
D = ROOT / "data"
OUT = ROOT / "submit"
OUT.mkdir(exist_ok=True)

DROP = ["id", "vdd", "he_dprg", "he_mpls"]


def preprocess(df):
    """inference.py 의 전처리와 반드시 동일하게 유지할 것."""
    X = df.drop(columns=[c for c in DROP if c in df.columns])
    X = X.replace(-2, np.nan).copy()                 # 모름·무응답 → 결측, -1(해당 없음)은 값으로 유지
    X["month_sin"] = np.sin(2 * np.pi * X["dr_month"] / 12)   # 비타민D 계절성
    X["month_cos"] = np.cos(2 * np.pi * X["dr_month"] / 12)
    return X


def make_model(seed=0):
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          early_stopping=False, random_state=seed)


def calibrate(p, t):
    p = np.asarray(p, dtype=float)
    return np.where(p < t, 0.5 * p / t, 0.5 + 0.5 * (p - t) / (1 - t))


def score(y, p):
    return (roc_auc_score(y, p) + f1_score(y, (p >= 0.5).astype(int), average="macro")) / 2


if __name__ == "__main__":
    df = pd.concat([pd.read_csv(D / "train.csv"), pd.read_csv(D / "valid.csv")], ignore_index=True)
    y = df["vdd"].values
    X = preprocess(df)
    cols = list(X.columns)

    if "--fast" in sys.argv:      # CV 생략: 이전 CV(neg2_nan)에서 찾은 임계값 0.610~0.625 의 중간값 사용
        t = 0.62
    else:
      oof = np.zeros(len(y))
      for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
        m = make_model().fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        print(f"fold {k} done", flush=True)
      ts = np.arange(0.30, 0.80, 0.005)
      t = float(ts[np.argmax([f1_score(y, (oof >= s).astype(int), average="macro") for s in ts])])
      print(f"OOF  raw score={score(y, oof):.4f}  |  t={t:.3f}  calibrated score={score(y, calibrate(oof, t)):.4f}")

    model = make_model().fit(X, y)
    joblib.dump({"model": model, "cols": cols, "t": t}, OUT / "model.joblib", compress=3)
    print("saved", OUT / "model.joblib")
