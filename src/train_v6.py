"""v6: v5 와 같은 구성·가중치·임계값에 seed 만 늘림 (LightGBM 3 → 10, CatBoost 3 → 6).
리더보드를 보지 않고 예측 분산만 줄이는 변경. 근거: fold 당 seed 3개 평균(slow) 0.6872 > seed 1개(te6) 0.6856.
v5 model.joblib 의 트리·CatBoost(seed 0~2)·로지스틱·타깃 인코더를 그대로 두고 seed 를 추가로 학습해 덧붙임."""
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

from exp2 import CAT, combos
from exp6 import cb_frame, make_cb
from train_final import BEST
from train_lgbm import fe

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submit_v6"
OUT.mkdir(exist_ok=True)
LGB_SEEDS, CB_SEEDS = range(3, 10), range(3, 6)

if __name__ == "__main__":
    df = pd.concat([pd.read_csv(ROOT / "data/train.csv"), pd.read_csv(ROOT / "data/valid.csv")], ignore_index=True)
    y = df["vdd"].values
    b = joblib.load(ROOT / "submit_v5/model.joblib")

    # 트리 입력을 train_final.py 와 똑같이 재구성 (타깃 인코딩 cross-fitting 값 포함)
    X = fe(df)
    Xt = X.copy()
    for c in CAT:
        Xt[c] = Xt[c].astype("category")
        assert list(Xt[c].cat.categories) == list(b["cats"][c])
    Cb = combos(df)
    te = TargetEncoder(target_type="binary", random_state=0)
    te_tr = te.fit_transform(Cb, y)
    for e1, e2 in zip(te.encodings_, b["te"].encodings_):          # v4 와 같은 인코더인지 확인
        assert np.allclose(e1, e2)
    Xt = pd.concat([Xt, pd.DataFrame(te_tr, index=Xt.index, columns=Cb.columns)], axis=1)[b["tree_cols"]]

    # 재현 확인: seed 0 을 다시 학습하면 v4 의 seed 0 트리와 같은 예측이어야 함
    chk = lgb.LGBMClassifier(random_state=0, **BEST).fit(Xt, y).predict_proba(Xt.iloc[:500])[:, 1]
    assert np.allclose(chk, b["trees"][0].predict_proba(Xt.iloc[:500])[:, 1]), "LightGBM 재현 실패"
    print("reproduce check ok", flush=True)

    for s in LGB_SEEDS:
        b["trees"].append(lgb.LGBMClassifier(random_state=s, **BEST).fit(Xt, y))
    print(f"lightgbm: {len(b['trees'])} models", flush=True)

    Xc, cats = cb_frame(df)
    assert list(Xc.columns) == b["cb_cols"]
    for s in CB_SEEDS:
        t0 = time.time()
        b["cbs"].append(make_cb(s).fit(Xc, y, cat_features=cats))
        print(f"catboost seed {s} done ({time.time() - t0:.0f}s)", flush=True)

    joblib.dump(b, OUT / "model.joblib", compress=3)
    print("saved", OUT / "model.joblib", "trees", len(b["trees"]), "cbs", len(b["cbs"]))
