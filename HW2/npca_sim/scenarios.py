"""實驗場景定義：主通道 OBSS 負載 x 副通道 OBSS 負載 x 目標 BSS 負載。"""

import json
import os

TARGET_MBPS = {"light": 5.0, "heavy": 30.0}          # 目標 BSS 上行總提供負載 (5 台 STA 平分)
P_BUSY_TARGETS = {"low": 0.13, "mid": 0.31, "high": 0.44}   # Legacy 試跑：目標輕載/重載兩種情況下 OBSS_P 忙碌比例的平均 (重載時飽和於約 0.45)
S_BUSY_TARGETS = {"light": 0.15, "heavy": 0.60}      # 副通道 OBSS 期望忙碌比例

P_LEVELS = ["low", "mid", "high"]
S_LEVELS = ["light", "heavy"]
T_LEVELS = ["light", "heavy"]

CHOSEN_LOADS_PATH = os.path.join("results", "pilot", "chosen_loads.json")


def scenario_id(p, s, t):
    return f"P{p}_S{s}_T{t}"


def load_chosen():
    with open(CHOSEN_LOADS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def all_scenarios():
    """主圖使用的固定排列順序：目標負載 -> 副通道負載 -> 主通道負載。"""
    return [(p, s, t) for t in T_LEVELS for s in S_LEVELS for p in P_LEVELS]
