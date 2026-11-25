"""선택 편향 검증: 행을 반으로 나눠 한쪽(A)에서 Caruana 혼합을 고르고, 다른 쪽(B)에서 v5 와 비교 (그리고 반대로)."""
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from exp2 import evaluate
from exp11 import EXIST
y = np.load("oof2_y.npy")
pool = dict(EXIST)
for f in sorted(Path(".").glob("oof11_*_r0.npy")):
    pool[f.name[6:-7]] = f.name[:-7]
O = {k: [np.load(f"{v}_r{r}.npy") for r in range(3)] for k, v in pool.items() if all(Path(f"{v}_r{r}.npy").exists() for r in range(3))}
V5 = {"slow": .55, "cb": .30, "lr": .15}
FIX = {"7model": {"cb_z": 1, "lgb_deep": 2, "lgb_et": 1, "lgb_lr01": 1, "lgb_z": 1, "lr_z": 1},
       "simple_z": {"lgb_z": .30, "lgb_deep": .25, "cb_z": .30, "lr_z": .15}}
def sc(w, idx):
    t = sum(w.values())
    return [evaluate(y[idx], sum(wk / t * O[k][r][idx] for k, wk in w.items()))[2] for r in range(3)]
def caruana(idx):
    bag, best = [], -1
    for _ in range(20):
        c = max(((np.mean(sc({kk: (bag + [k]).count(kk) for kk in set(bag + [k])}, idx)), k) for k in O))
        if c[0] <= best + 1e-5: break
        best = c[0]; bag.append(c[1])
    return {kk: bag.count(kk) for kk in set(bag)}
for seed in (0, 1):
    for a, b in StratifiedKFold(2, shuffle=True, random_state=seed).split(y, y):
        w = caruana(a)
        d = np.array(sc(w, b)) - np.array(sc(V5, b))
        dfix = {n: (np.array(sc(f, b)) - np.array(sc(V5, b))).mean() for n, f in FIX.items()}
        print(f"seed{seed}: A에서 고른 {w} → B에서 v5 대비 {d.mean():+.4f} {np.round(d,4)} | 고정후보 B {dict((k, round(v,4)) for k,v in dfix.items())}", flush=True)
all_idx = np.arange(len(y))
for n, f in FIX.items():
    d = np.array(sc(f, all_idx)) - np.array(sc(V5, all_idx))
    print(n, "전체", np.round(np.mean(sc(f, all_idx)), 4), np.round(d, 4))
