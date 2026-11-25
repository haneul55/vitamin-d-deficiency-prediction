"""밤샘 실험 러너 (반복 CV 5-fold x 3, exp2 와 같은 분할 seed 100+r). 리더보드는 전혀 보지 않음.
각 실험의 OOF 를 oof11_<arm>_r<rep>.npy 로 저장하고, 이미 있으면 건너뜀 (중단돼도 이어서 실행 가능).
마지막에 기존 OOF 전부 + 새 OOF 로 혼합 탐색 → 결과를 overnight_log.txt 에 기록.
채택 기준: v5 혼합(0.6887)보다 평균 +0.0010 이상, 3회 모두 상승."""
import itertools
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from exp2 import combos, evaluate, run_fold
from exp3 import FAST_BEST
from exp7 import build, find_blocks
from exp_lr import make_lr
from train_lgbm import fe

warnings.filterwarnings("ignore")
LOG = Path("overnight_log.txt")
SLOW = dict(FAST_BEST, learning_rate=0.02, n_estimators=500)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def lgb_arm(params, feat="base", drop_te=None, seeds=3):
    def run(df, y, r, tr, va, cache):
        X, C = cache[feat]
        if drop_te:
            C = C.drop(columns=drop_te)
        return np.mean([run_fold("M2", X, C, y, tr, va, 10 * r + s, params) for s in range(seeds)], axis=0)
    return run


def cb_arm(cfg, feat="base"):
    def run(df, y, r, tr, va, cache):
        from catboost import CatBoostClassifier
        from exp2 import CAT
        X, C = cache[feat]
        X = pd.concat([X, C], axis=1)
        cats = CAT + list(C.columns)
        for c in cats:
            X[c] = X[c].astype("Int64").astype(str) if c in CAT else X[c]
            X[c] = X[c].astype(object).where(X[c].notna(), "NA").astype(str)
        m = CatBoostClassifier(rsm=0.5, random_seed=r, thread_count=4, verbose=0, allow_writing_files=False, **cfg)
        return m.fit(X.iloc[tr], y[tr], cat_features=cats).predict_proba(X.iloc[va])[:, 1]
    return run


def lr_arm(C, feat="base"):
    def run(df, y, r, tr, va, cache):
        X = cache[feat][0]
        X = X.loc[:, X.notna().any()]
        return make_lr(list(X.columns), C).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    return run


ARMS = {
    # LightGBM 변형 (fold 마다 seed 3개 평균 = oof5_slow 와 같은 방식)
    "lgb_z":       lgb_arm(SLOW, feat="Z"),                                   # 그룹 표준화 검사 수치 추가
    "lgb_nomonth": lgb_arm(SLOW, drop_te=["te_sex_month"]),                   # dr_month 는 월이 아님 → 조합 제거
    "lgb_et":      lgb_arm(dict(SLOW, extra_trees=True, n_estimators=800)),   # 무작위 분할 (다양성)
    "lgb_lr01":    lgb_arm(dict(SLOW, learning_rate=0.01, n_estimators=1000)),
    "lgb_deep":    lgb_arm(dict(SLOW, num_leaves=63, min_child_samples=100, colsample_bytree=0.3)),
    "lgb_dart":    lgb_arm(dict(FAST_BEST, boosting_type="dart", learning_rate=0.05, n_estimators=400,
                                drop_rate=0.1, skip_drop=0.5), seeds=2),
    # 로지스틱 변형
    "lr_z":        lr_arm(0.02, feat="Z"),
    "lr_c005":     lr_arm(0.005),
    # CatBoost 변형
    "cb_z":        cb_arm(dict(iterations=1000, learning_rate=0.04, depth=6, l2_leaf_reg=5), feat="Z"),
    "cb_slow":     cb_arm(dict(iterations=2000, learning_rate=0.02, depth=6, l2_leaf_reg=5)),
}

EXIST = {"slow": "oof5_slow", "cb": "oof6_cb", "lr": "oof2_M4_C0.02", "xgb": "oof9_xgb", "et": "oof3_et",
         "knn": "oof5_knn", "te6": "oof3_te6", "te12": "oof3_te12", "cbc3": "oof8_c3"}


def run_arms(names):
    df = pd.concat([pd.read_csv("data/train.csv"), pd.read_csv("data/valid.csv")], ignore_index=True)
    y = df["vdd"].values
    blocks = find_blocks(df)
    cache = {"base": (fe(df), combos(df)), "Z": build(df, "Z", blocks)}
    cache.update(globals().get("_EXTRA_CACHE", {}))          # exp13 에서 추가 피처 세트를 넣을 때 사용
    lr5 = [np.load(f"oof2_M4_C0.02_r{r}.npy") for r in range(3)]
    sl = [np.load(f"oof5_slow_r{r}.npy") for r in range(3)]
    cb = [np.load(f"oof6_cb_r{r}.npy") for r in range(3)]
    for name in names:
        if all(Path(f"oof11_{name}_r{r}.npy").exists() for r in range(3)):
            log(f"{name}: 이미 있음, 건너뜀")
            continue
        try:
            t0, res, rb = time.time(), [], []
            for r in range(3):
                oof = np.zeros(len(y))
                for tr, va in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(df, y):
                    oof[va] = ARMS[name](df, y, r, tr, va, cache)
                np.save(f"oof11_{name}_r{r}.npy", oof)
                res.append(evaluate(y, oof)[2])
                # v5 혼합에서 같은 종류 모델 자리를 바꿔 끼운 점수
                if name.startswith("lgb"):
                    rb.append(evaluate(y, .55 * oof + .30 * cb[r] + .15 * lr5[r])[2])
                elif name.startswith("cb"):
                    rb.append(evaluate(y, .55 * sl[r] + .30 * oof + .15 * lr5[r])[2])
                else:
                    rb.append(evaluate(y, .55 * sl[r] + .30 * cb[r] + .15 * oof)[2])
            log(f"{name}: 단독 {np.mean(res):.4f} {[round(v, 4) for v in res]} | v5 자리 교체 {np.mean(rb):.4f} "
                f"{[round(v, 4) for v in rb]} ({time.time() - t0:.0f}s)")
        except Exception:
            log(f"{name}: 실패\n{traceback.format_exc()}")


def blend_search():
    y = np.load("oof2_y.npy")
    pool = dict(EXIST)
    for f in sorted(Path(".").glob("oof11_*_r0.npy")):
        pool[f.name[6:-7]] = f.name[:-7]
    O = {}
    for k, v in pool.items():
        if all(Path(f"{v}_r{r}.npy").exists() for r in range(3)):
            O[k] = [np.load(f"{v}_r{r}.npy") for r in range(3)]
    base = [evaluate(y, .55 * O["slow"][r] + .30 * O["cb"][r] + .15 * O["lr"][r])[2] for r in range(3)]
    log(f"혼합 탐색 시작: 후보 {len(O)}개 {list(O)} | v5 기준 {np.mean(base):.4f} {[round(b, 4) for b in base]}")

    def sc(w):
        return [evaluate(y, sum(wk * O[k][r] for k, wk in w.items()))[2] for r in range(3)]

    # 앙상블 선택 (Caruana, 0.05 단위, 같은 모델 반복 선택 허용, 최대 20단계)
    bag, best_hist = [], []
    for step in range(20):
        cand = []
        for k in O:
            trial = bag + [k]
            w = {kk: trial.count(kk) / len(trial) for kk in set(trial)}
            s = sc(w)
            cand.append((np.mean(s), k, s, w))
        cand.sort(key=lambda c: -c[0])
        m, k, s, w = cand[0]
        if best_hist and m <= best_hist[-1][0] + 1e-5:
            break
        bag.append(k)
        best_hist.append((m, s, w))
        log(f"  선택 {step + 1}: +{k} → {m:.4f} {[round(v, 4) for v in s]} 가중치={{{', '.join(f'{a}:{b:.2f}' for a, b in sorted(w.items()))}}}")
    m, s, w = best_hist[-1]
    d = np.array(s) - np.array(base)
    ok = d.mean() >= 0.0010 and (d > 0).all()
    log(f"최종 혼합 {m:.4f} (v5 대비 {d.mean():+.4f}, 회차별 {[round(x, 4) for x in d]}) → {'채택 기준 통과' if ok else '채택 기준 미달'}")
    return ok, w


if __name__ == "__main__":
    names = sys.argv[1:] or list(ARMS)
    log(f"=== 밤샘 실험 시작: {names}")
    run_arms(names)
    blend_search()
    log("=== 끝")
