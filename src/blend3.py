"""트리 / 로지스틱 / 3번째 모델(mlp 또는 et) 가중치를 격자 탐색 (반복 CV 3회 평균).
사용: python blend3.py <tree_name> <third_name>   예) python blend3.py te12 mlp"""
import itertools
import sys

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

y = np.load("oof2_y.npy")
ts = np.arange(0.55, 0.70, 0.0025)
tree, third = sys.argv[1], sys.argv[2]
T = [np.load(f"oof3_{tree}_r{r}.npy") for r in range(3)]
L = [np.load(f"oof2_M4_C0.02_r{r}.npy") for r in range(3)]
M = [np.load(f"oof3_{third}_r{r}.npy") for r in range(3)]
print(f"corr tree-lr={np.corrcoef(T[0], L[0])[0, 1]:.3f} tree-{third}={np.corrcoef(T[0], M[0])[0, 1]:.3f} "
      f"lr-{third}={np.corrcoef(L[0], M[0])[0, 1]:.3f}")

rows = []
for wl, wm in itertools.product([0, 0.1, 0.2, 0.25, 0.3, 0.4], [0, 0.1, 0.2, 0.3]):
    wt = 1 - wl - wm
    if wt < 0.3:
        continue
    sc, f1s = [], []
    for r in range(3):
        p = wt * T[r] + wl * L[r] + wm * M[r]
        f = np.array([f1_score(y, (p >= t).astype(int), average="macro") for t in ts])
        f1s.append(f)
        sc.append((roc_auc_score(y, p) + f.max()) / 2)
    tstar = ts[np.mean(f1s, 0).argmax()]               # 3회 평균 F1 이 최대인 공통 임계값
    rows.append((np.mean(sc), np.std(sc), wt, wl, wm, tstar))
rows.sort(key=lambda r: -r[0])
for s, sd, wt, wl, wm, t in rows[:8]:
    print(f"tree={wt:.2f} lr={wl:.2f} {third}={wm:.2f}  SCORE={s:.4f}±{sd:.4f}  t={t:.4f}")
