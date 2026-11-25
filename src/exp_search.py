"""M3: M2 피처에서 LightGBM 하이퍼파라미터 랜덤 탐색.
1차: 후보 14개를 5-fold x 1회로 빠르게 평가 → 상위 2개를 5-fold x 3회로 재확인."""
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

from exp2 import combos, repeated_cv
from train_lgbm import fe

warnings.filterwarnings("ignore")
SPACE = {
    "num_leaves": [15, 23, 31, 47, 63],
    "min_child_samples": [20, 40, 80, 150, 250],
    "colsample_bytree": [0.2, 0.3, 0.4, 0.6],
    "subsample": [0.7, 0.8, 1.0],
    "reg_lambda": [0.0, 2.0, 5.0, 10.0, 20.0],
    "n_estimators": [200, 300, 450, 600],
    "cat_smooth": [10, 20, 50],
    "class_weight": [None, "balanced"],
    "min_split_gain": [0.0, 0.01],
}

if __name__ == "__main__":
    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/valid.csv")], ignore_index=True)
    y, X, C = df["vdd"].values, fe(df), combos(df)
    results = []
    for i, p in enumerate(ParameterSampler(SPACE, n_iter=14, random_state=7)):
        t0 = time.time()
        (auc, f1, sc), _ = repeated_cv("M2", X, C, y, params=p, reps=(0,), save=False)
        results.append((sc, p))
        print(f"[{i:2d}] SCORE={sc:.4f} AUC={auc:.4f} F1={f1:.4f} ({time.time() - t0:.0f}s) {p}", flush=True)
    results.sort(key=lambda r: -r[0])
    best = []
    for sc1, p in results[:2]:
        (auc, f1, sc), (sa, sf, ss) = repeated_cv("M2", X, C, y, params=p, reps=(0, 1, 2), save=False)
        best.append({"score": sc, "std": ss, "auc": auc, "f1": f1, "params": p})
        print(f"CONFIRM SCORE={sc:.4f}±{ss:.4f} AUC={auc:.4f} F1={f1:.4f} {p}", flush=True)
    json.dump(best, open("search_best.json", "w"), indent=1)
