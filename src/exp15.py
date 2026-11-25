"""밤샘 3차: Z 피처(bsig + 그룹 표준화 검사 수치)가 가장 효과적이었으므로 나머지 LightGBM 변형도 Z 피처로 학습."""
import exp11
from exp11 import SLOW, lgb_arm, log, run_arms
exp11.ARMS.update({
    "lgb_deep_z": lgb_arm(dict(SLOW, num_leaves=63, min_child_samples=100, colsample_bytree=0.3), feat="Z"),
    "lgb_et_z": lgb_arm(dict(SLOW, extra_trees=True, n_estimators=800), feat="Z"),
    "lgb_lr01_z": lgb_arm(dict(SLOW, learning_rate=0.01, n_estimators=1000), feat="Z"),
})
if __name__ == "__main__":
    names = ["lgb_deep_z", "lgb_et_z", "lgb_lr01_z"]
    log(f"=== 3차 실험 시작: {names}")
    run_arms(names)
    log("=== 3차 끝")
