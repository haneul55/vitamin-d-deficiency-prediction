"""최종 모델 (3차 개선): LightGBM(범주형 + 타깃 인코딩 + 탐색된 파라미터, 3 seed) 0.75 + 로지스틱 0.25.
반복 CV 근거: M2 0.6845 → 탐색 파라미터 0.6857, 로지스틱 혼합 시 +0.0012. 임계값 t=0.625.
submit_v4/model.joblib 저장 (로지스틱: 스플라인 → 구간 원-핫). 전처리 함수는 submit_v3/inference.py 와 동일하게 유지."""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

from exp2 import CAT, combos
from exp_lr import make_lr
from train_lgbm import fe

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submit_v4"
OUT.mkdir(exist_ok=True)
BEST = dict(subsample=0.7, subsample_freq=1, reg_lambda=2.0, num_leaves=23, min_split_gain=0.01,
            min_child_samples=250, colsample_bytree=0.4, cat_smooth=50, min_data_per_group=50,
            learning_rate=0.025, n_estimators=400, verbose=-1, n_jobs=4)   # 탐색값(lr .05, 200그루)을 lr 절반·트리 2배로
W_TREE, T = 0.75, 0.625

if __name__ == "__main__":
    df = pd.concat([pd.read_csv(ROOT / "data/train.csv"), pd.read_csv(ROOT / "data/valid.csv")], ignore_index=True)
    y = df["vdd"].values
    X = fe(df)

    # 로지스틱 (전부 결측인 열 제외)
    lr_cols = list(X.columns[X.notna().any()])
    lr_model = make_lr(lr_cols, C=0.02).fit(X[lr_cols], y)

    # 트리: 범주형 + 타깃 인코딩
    cats = {}
    Xt = X.copy()
    for c in CAT:
        Xt[c] = Xt[c].astype("category")
        cats[c] = list(Xt[c].cat.categories)
    Cb = combos(df)
    te = TargetEncoder(target_type="binary", random_state=0)
    te_tr = te.fit_transform(Cb, y)                       # cross-fitting 으로 학습용 값 생성
    Xt = pd.concat([Xt, pd.DataFrame(te_tr, index=Xt.index, columns=Cb.columns)], axis=1)
    trees = [lgb.LGBMClassifier(random_state=s, **BEST).fit(Xt, y) for s in (0, 1, 2)]

    joblib.dump({"trees": trees, "tree_cols": list(Xt.columns), "cats": cats, "te": te, "te_cols": list(Cb.columns),
                 "lr": lr_model, "lr_cols": lr_cols, "w_tree": W_TREE, "t": T}, OUT / "model.joblib", compress=3)
    print("saved", OUT / "model.joblib")
