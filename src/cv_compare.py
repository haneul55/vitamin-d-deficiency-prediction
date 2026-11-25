"""train+valid 합쳐 5-fold CV 로 전처리/모델 후보 비교.
점수 = (AUROC + macro-F1@0.5) / 2, 그리고 OOF 에서 최적 임계값으로 보정했을 때 점수도 같이 출력."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
D = Path(__file__).resolve().parent / "data"
df = pd.concat([pd.read_csv(D / "train.csv"), pd.read_csv(D / "valid.csv")], ignore_index=True)
y = df["vdd"].values
X0 = df.drop(columns=["id", "vdd"])


def score(y, p, t=0.5):
    return (roc_auc_score(y, p) + f1_score(y, (p >= t).astype(int), average="macro")) / 2


def best_t(y, p):
    ts = np.arange(0.30, 0.80, 0.005)
    s = [f1_score(y, (p >= t).astype(int), average="macro") for t in ts]
    return ts[int(np.argmax(s))]


def prep(X, mode):
    X = X.drop(columns=["he_dprg", "he_mpls"])
    if mode == "nan_all":       # -1, -2 모두 결측
        X = X.replace([-1, -2], np.nan)
    elif mode == "neg2_nan":    # -2(모름)만 결측, -1(해당없음)은 값으로 유지
        X = X.replace(-2, np.nan)
    # "raw": 그대로
    # 계절성: 조사 월을 원형 인코딩
    m = X["dr_month"]
    X["month_sin"] = np.sin(2 * np.pi * m / 12)
    X["month_cos"] = np.cos(2 * np.pi * m / 12)
    return X.loc[:, X.notna().any()]


skf = StratifiedKFold(5, shuffle=True, random_state=42)


def run(name, make_model, mode):
    X = prep(X0, mode)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = make_model()
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    t = best_t(y, oof)
    print(f"{name:32s} AUC={roc_auc_score(y, oof):.4f}  score@0.5={score(y, oof):.4f}  "
          f"best_t={t:.3f} score@t={score(y, oof, t):.4f}", flush=True)
    return oof


lr = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
hgb = lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
                                             min_samples_leaf=40, l2_regularization=1.0,
                                             early_stopping=False, random_state=0)

for mode in ["neg2_nan", "nan_all"]:
    run(f"HGB {mode}", hgb, mode)

try:
    import lightgbm as lgb
    lgbm = lambda: lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                                      verbose=-1, random_state=0)
    run("LGBM neg2_nan", lgbm, "neg2_nan")
except ImportError:
    pass
