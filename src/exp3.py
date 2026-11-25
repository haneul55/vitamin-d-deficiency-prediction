"""4차 개선 실험 (반복 CV 5-fold x 3, seed 100+r — exp2 와 같은 분할).
  te6  : 기존 타깃 인코딩 6개 조합 + 탐색된 LightGBM 파라미터
  te12 : + 조합 6개 추가
  mlp  : 로지스틱과 같은 전처리 + MLP 신경망
  et   : ExtraTrees (중앙값 대체 + 결측 표시)
OOF 는 oof3_<name>_r<rep>.npy 로 저장."""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

from exp2 import combos, evaluate, run_fold
from exp_lr import make_lr
from train_lgbm import fe

warnings.filterwarnings("ignore")
FAST_BEST = dict(subsample=0.7, reg_lambda=2.0, num_leaves=23, min_split_gain=0.01, min_child_samples=250,
                 colsample_bytree=0.4, cat_smooth=50, n_estimators=200)   # lr 0.05 (lgb_fast 기본)


def combos_ext(R):
    C = combos(R)
    g = lambda c: R[c].astype("Int64").astype(str)
    age5 = (R["age"] // 5).astype("Int64").astype(str)
    age10 = (R["age"] // 10).astype("Int64").astype(str)
    C["te_region_age5"] = g("region") + "_" + age5
    C["te_occp_age5"] = g("occp") + "_" + age5
    C["te_region_sex"] = g("region") + "_" + g("sex")
    C["te_incm_sex"] = g("ho_incm5") + "_" + g("sex")
    C["te_edu_sex"] = g("edu") + "_" + g("sex")
    C["te_town_sex_age10"] = g("town_t") + "_" + g("sex") + "_" + age10
    return C


def make_mlp(cols):
    lr = make_lr(cols)                      # 같은 전처리(원-핫 + 구간 + 표준화) 재사용
    pre = lr.steps[0][1]
    return make_pipeline(pre, MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2, learning_rate_init=1e-3,
                                            batch_size=256, max_iter=200, early_stopping=True,
                                            validation_fraction=0.1, n_iter_no_change=10, random_state=0))


def make_et():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         ExtraTreesClassifier(n_estimators=300, min_samples_leaf=20, max_features=0.3,
                                              n_jobs=4, random_state=0))


def run(name, X, y, C=None):
    t0, res = time.time(), []
    for r in (0, 1, 2):
        oof = np.zeros(len(y))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(X, y):
            if name.startswith("te"):
                oof[va] = run_fold("M2", X, C, y, tr, va, r, FAST_BEST)
            else:
                Xl = X.loc[:, X.notna().any()]
                m = make_mlp(list(Xl.columns)) if name == "mlp" else make_et()
                oof[va] = m.fit(Xl.iloc[tr], y[tr]).predict_proba(Xl.iloc[va])[:, 1]
        np.save(f"oof3_{name}_r{r}.npy", oof)
        res.append(evaluate(y, oof))
    m, s = np.mean(res, 0), np.std(res, 0)
    print(f"{name:5s} AUC={m[0]:.4f}±{s[0]:.4f} F1={m[1]:.4f}±{s[1]:.4f} SCORE={m[2]:.4f}±{s[2]:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/valid.csv")], ignore_index=True)
    y, X = df["vdd"].values, fe(df)
    for name in sys.argv[1:] or ["te6", "te12", "mlp", "et"]:
        run(name, X, y, combos(df) if name == "te6" else combos_ext(df) if name == "te12" else None)
