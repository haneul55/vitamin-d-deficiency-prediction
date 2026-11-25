"""피처 선택 (null importance).
1) 실제 정답으로 1회, 정답을 섞어서 15회 LightGBM 학습 → 변수별 gain 비교
2) score = log((실제 gain + 1) / (가짜 gain 75퍼센타일 + 1)), score <= 기준 인 변수 제거
3) 선택된 변수로 반복 CV (exp2.run_fold, 같은 분할) → te6 기준선(0.6857)과 비교
주의: 선택을 전체 데이터에서 하므로 CV 가 약간 낙관적일 수 있음 → 개선 폭이 표준편차보다 확실히 커야 채택."""
import json
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

from exp2 import CAT, combos, evaluate, lgb_fast, run_fold
from exp3 import FAST_BEST
from train_lgbm import fe

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/valid.csv")], ignore_index=True)
    y, X, C = df["vdd"].values, fe(df), combos(df)

    Xt = X.copy()
    for c in CAT:
        Xt[c] = Xt[c].astype("category")
    te = TargetEncoder(target_type="binary", random_state=0).fit_transform(C, y)
    Xt = pd.concat([Xt, pd.DataFrame(te, index=Xt.index, columns=C.columns)], axis=1)

    def gains(yy, seed):
        m = lgb_fast(seed, **FAST_BEST).fit(Xt, yy)
        return pd.Series(m.booster_.feature_importance("gain"), index=Xt.columns)

    t0 = time.time()
    real = gains(y, 0)
    rng = np.random.default_rng(0)
    null = pd.concat([gains(rng.permutation(y), i) for i in range(15)], axis=1)
    score = np.log((real + 1) / (null.quantile(0.75, axis=1) + 1))
    score = score[X.columns]                          # 선택 대상은 fe() 변수만 (타깃 인코딩 6개는 유지)
    print(f"null importance done ({time.time() - t0:.0f}s)")
    print("lowest 15:", score.sort_values().head(15).round(2).to_dict())

    results = {}
    for thr in [0.0, 0.5, 1.0]:
        keep = [c for c in X.columns if score[c] > thr or c in CAT]
        t0, res = time.time(), []
        for r in (0, 1, 2):
            oof = np.zeros(len(y))
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(X, y):
                oof[va] = run_fold("M2", X[keep], C, y, tr, va, r, FAST_BEST)
            np.save(f"oof4_sel{thr}_r{r}.npy", oof)
            res.append(evaluate(y, oof))
        m, s = np.mean(res, 0), np.std(res, 0)
        results[thr] = keep
        print(f"thr={thr}: {len(keep)}/{X.shape[1]} vars  AUC={m[0]:.4f}  F1={m[1]:.4f}  "
              f"SCORE={m[2]:.4f}±{s[2]:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump({str(k): v for k, v in results.items()}, open("selected_features.json", "w"))
