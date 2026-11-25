"""v5: v4 (LightGBM 3 seed + 로지스틱) 에 CatBoost(3 seed) 추가.
반복 CV(OOF) 근거: LGBM .55 + CatBoost .30 + 로지스틱 .15 → 0.6887 (v4 와 비슷한 구성 0.6878). 임계값 t=0.6175.
v4 model.joblib 의 트리·로지스틱은 그대로 쓰고, CatBoost 만 train+valid 전체로 새로 학습해 묶어서 저장."""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from exp6 import cb_frame, make_cb

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submit_v5"
OUT.mkdir(exist_ok=True)
W_TREE, W_CB, W_LR, T = 0.55, 0.30, 0.15, 0.6175

if __name__ == "__main__":
    df = pd.concat([pd.read_csv(ROOT / "data/train.csv"), pd.read_csv(ROOT / "data/valid.csv")], ignore_index=True)
    y = df["vdd"].values
    X, cats = cb_frame(df)
    cbs = []
    for s in (0, 1, 2):
        t0 = time.time()
        cbs.append(make_cb(s).fit(X, y, cat_features=cats))
        print(f"catboost seed {s} done ({time.time() - t0:.0f}s)", flush=True)

    b = joblib.load(ROOT / "submit_v4/model.joblib")
    b.update({"cbs": cbs, "cb_cols": list(X.columns), "cb_cats": cats,
              "w_tree": W_TREE, "w_cb": W_CB, "w_lr": W_LR, "t": T})
    joblib.dump(b, OUT / "model.joblib", compress=3)
    print("saved", OUT / "model.joblib")
