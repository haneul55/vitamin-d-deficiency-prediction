"""OOF 로 앙상블 가중치 w (lgb 비중) 와 안정적인 임계값 t 를 찾는다.
임계값: 부트스트랩 200회에서 각 t 의 macro-F1 평균이 최대인 값 (한 번의 OOF 에 과적합 방지)."""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import f1_score, roc_auc_score

y = np.load("oof_y.npy")
L, H = np.load("oof_lgb.npy"), np.load("oof_hgb.npy")
ts = np.arange(0.50, 0.75, 0.0025)
rng = np.random.default_rng(0)
boots = [rng.integers(0, len(y), len(y)) for _ in range(200)]


def f1_at(p, idx, t):
    return f1_score(y[idx], (p[idx] >= t).astype(int), average="macro")


print(f"corr(lgb,hgb)={np.corrcoef(L, H)[0, 1]:.3f}")
for w in [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0]:
    p = w * L + (1 - w) * H
    auc = roc_auc_score(y, p)
    full = np.array([f1_at(p, slice(None), t) for t in ts])
    print(f"w_lgb={w:.1f} AUC={auc:.4f} bestF1={full.max():.4f} t*={ts[full.argmax()]:.4f} "
          f"SCORE={(auc + full.max()) / 2:.4f}", flush=True)

# 최종 후보 가중치에서 부트스트랩 안정 임계값
for w in [0.5, 0.7]:
    p = w * L + (1 - w) * H
    m = np.array([[f1_at(p, b, t) for t in ts[::2]] for b in boots[:60]]).mean(0)
    print(f"w={w} bootstrap-stable t={ts[::2][m.argmax()]:.4f} meanF1={m.max():.4f}", flush=True)
