"""실험 D: 스태킹 (가중치 격자 대신 로지스틱 메타모델).
각 반복(rep)의 OOF 예측을 logit 으로 바꿔 입력으로 쓰고, 메타모델 성능은 다시 5-fold (seed 200+r) 로 측정 → 메타 학습 누수 없음.
비교 기준: v5 고정 가중치 (slow .55 + cb .30 + lr .15) 0.6887. 채택 기준 +0.0010 이상, 3회 모두 상승."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from exp2 import evaluate

SRC = {"slow": "oof5_slow", "cb": "oof6_cb", "lr": "oof2_M4_C0.02", "xgb": "oof9_xgb", "et": "oof3_et",
       "mlp": "oof3_mlp", "knn": "oof5_knn", "te6": "oof3_te6"}
SETS = {"v5 3개": ["slow", "cb", "lr"], "+xgb": ["slow", "cb", "lr", "xgb"],
        "+et,mlp": ["slow", "cb", "lr", "et", "mlp"], "전부 8개": list(SRC)}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


if __name__ == "__main__":
    y = np.load("oof2_y.npy")
    base = []
    for r in range(3):
        o = {k: np.load(f"{v}_r{r}.npy") for k, v in SRC.items()}
        base.append(evaluate(y, .55 * o["slow"] + .30 * o["cb"] + .15 * o["lr"])[2])
    print(f"v5 고정 가중치: {np.mean(base):.4f} reps={[round(b, 4) for b in base]}")
    for name, cols in SETS.items():
        for C in (0.01, 1.0):
            sc, coefs = [], []
            for r in range(3):
                Z = np.column_stack([logit(np.load(f"{SRC[k]}_r{r}.npy")) for k in cols])
                meta = np.zeros(len(y))
                for tr, va in StratifiedKFold(5, shuffle=True, random_state=200 + r).split(Z, y):
                    m = LogisticRegression(C=C, max_iter=2000).fit(Z[tr], y[tr])
                    meta[va] = m.predict_proba(Z[va])[:, 1]
                    coefs.append(m.coef_[0])
                sc.append(evaluate(y, meta)[2])
            d = np.array(sc) - np.array(base)
            w = np.mean(coefs, 0)
            print(f"{name:8s} C={C:<5} SCORE={np.mean(sc):.4f} 차이={d.mean():+.4f} reps차이={[round(x, 4) for x in d]} "
                  f"계수={dict(zip(cols, np.round(w, 2)))}")
