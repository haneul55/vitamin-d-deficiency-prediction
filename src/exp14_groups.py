"""선택 없는 혼합: 종류별 평균 (LGBM 변형 평균 / CatBoost 평균 / 로지스틱 평균) 에 v5 가중치 .55/.30/.15."""
import numpy as np
from exp2 import evaluate
y = np.load("oof2_y.npy")
L = lambda n: [np.load(f"{n}_r{r}.npy") for r in range(3)]
src = {"slow": "oof5_slow", "cb": "oof6_cb", "lr": "oof2_M4_C0.02"}
for k in ["lgb_z", "lgb_deep", "lgb_et", "lgb_lr01", "lgb_nomonth", "cb_z", "cb_slow", "lr_z"]:
    src[k] = f"oof11_{k}"
O = {k: L(v) for k, v in src.items()}
def blend(g):  # g: {group: (weight, [members])}
    return [sum(w * np.mean([O[m][r] for m in ms], 0) for w, ms in g.values()) for r in range(3)]
C = {
 "v5":        {"t": (.55, ["slow"]), "c": (.30, ["cb"]), "l": (.15, ["lr"])},
 "G_all":     {"t": (.55, ["slow", "lgb_z", "lgb_deep", "lgb_et", "lgb_lr01", "lgb_nomonth"]), "c": (.30, ["cb", "cb_z", "cb_slow"]), "l": (.15, ["lr", "lr_z"])},
 "G_z":       {"t": (.55, ["lgb_z"]), "c": (.30, ["cb_z"]), "l": (.15, ["lr_z"])},
 "G_z+":      {"t": (.55, ["lgb_z", "lgb_deep", "lgb_et", "lgb_lr01"]), "c": (.30, ["cb_z", "cb"]), "l": (.15, ["lr_z"])},
 "7model":    {"a": (1/7, ["cb_z"]), "b": (2/7, ["lgb_deep"]), "c": (1/7, ["lgb_et"]), "d": (1/7, ["lgb_lr01"]), "e": (1/7, ["lgb_z"]), "f": (1/7, ["lr_z"])},
}
base = [evaluate(y, p) for p in blend(C["v5"])]
for n, g in C.items():
    res = [evaluate(y, p) for p in blend(g)]
    d = np.array([x[2] for x in res]) - np.array([x[2] for x in base])
    da = np.array([x[0] for x in res]) - np.array([x[0] for x in base])
    print(f"{n:8s} SCORE={np.mean([x[2] for x in res]):.4f} dSCORE={d.mean():+.4f} {np.round(d,4)}  AUC={np.mean([x[0] for x in res]):.4f} dAUC={da.mean():+.4f} {np.round(da,4)}  F1={np.mean([x[1] for x in res]):.4f}")
