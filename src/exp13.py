"""밤샘 2차 실험 (exp11 이 끝난 뒤 실행). exp11 과 같은 러너·분할·로그(overnight_log.txt)를 사용.
  lgb_inter : 사람이 만든 상호작용 변수 (나이x성별, BUN/나이, BMI x 성별, 결측 묶음 x 나이 등)
  lgb_goss  : GOSS 샘플링 (다른 방식의 부스팅 → 다양성)
  lgb_linear: 리프마다 선형식 (linear_tree)
  nys_lr    : RBF 커널 근사(Nystroem) + 로지스틱 (트리와 전혀 다른 모델)
  cb_inter  : CatBoost + 상호작용 변수
끝나면 전체 OOF 로 혼합 탐색을 다시 함."""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

import exp11
from exp11 import SLOW, blend_search, cb_arm, lgb_arm, log, run_arms

warnings.filterwarnings("ignore")


def inter_feats(df, X):
    X = X.copy()
    a, s = df["age"], df["sex"]
    X["age_x_sex"] = a * (s == 2)
    X["bun_per_age"] = df["he_bun"] / a
    X["crea_x_sex"] = df["he_crea"] * (s == 2)
    X["bmi_x_sex"] = df["he_bmi"] * (s == 2)
    X["hb_x_sex"] = df["he_hb"] * (s == 2)
    X["miss_sig_x_age10"] = X["miss_sig"] * 10 + (a // 10)
    X["wt_ht"] = df["he_wt"] / df["he_ht"]
    return X.replace([np.inf, -np.inf], np.nan)


def nys_arm():
    def run(df, y, r, tr, va, cache):
        X = cache["base"][0]
        X = X.loc[:, X.notna().any()]
        m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                          QuantileTransformer(output_distribution="normal", random_state=r),
                          Nystroem(gamma=0.002, n_components=1500, random_state=r),
                          LogisticRegression(C=0.05, max_iter=3000))
        return m.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    return run


exp11.ARMS.update({
    "lgb_inter":  lgb_arm(SLOW, feat="I"),
    "lgb_goss":   lgb_arm(dict(SLOW, data_sample_strategy="goss", subsample=1.0, subsample_freq=0)),
    "lgb_linear": lgb_arm(dict(SLOW, linear_tree=True, linear_lambda=1.0), seeds=1),
    "nys_lr":     nys_arm(),
    "cb_inter":   cb_arm(dict(iterations=1000, learning_rate=0.04, depth=6, l2_leaf_reg=5), feat="I"),
})

def _patched_run_arms(names):
    """exp11.run_arms 의 cache 에 "I" 피처 세트를 추가해서 실행."""
    from exp2 import combos
    from train_lgbm import fe

    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/valid.csv")], ignore_index=True)
    XI = inter_feats(df, fe(df))
    exp11._EXTRA_CACHE = {"I": (XI, combos(df))}
    run_arms(names)


if __name__ == "__main__":
    names = sys.argv[1:] or ["lgb_inter", "lgb_goss", "nys_lr", "cb_inter", "lgb_linear"]
    log(f"=== 2차 실험 시작: {names}")
    _patched_run_arms(names)
    blend_search()
    log("=== 2차 끝")
