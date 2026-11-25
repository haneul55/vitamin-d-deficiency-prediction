"""같은 fold 로 HGB / LightGBM OOF 예측을 만들어 oof_*.npy 로 저장 (앙상블 가중치·임계값 탐색용).
두 모델 모두 train_lgbm.fe() 피처를 사용."""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from train_lgbm import fe, make

warnings.filterwarnings("ignore")


def make_hgb(seed=0):
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          early_stopping=False, random_state=seed)


if __name__ == "__main__":
    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/valid.csv")], ignore_index=True)
    y = df["vdd"].values
    X = fe(df)
    np.save("oof_y.npy", y)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    for name, mk in [("lgb", make), ("hgb", make_hgb)]:
        oof = np.zeros(len(y))
        for k, (tr, va) in enumerate(folds):
            oof[va] = mk(0).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
            print(name, "fold", k, flush=True)
        np.save(f"oof_{name}.npy", oof)
