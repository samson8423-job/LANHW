"""試跑：(1) 以 Legacy 找出主/副通道 OBSS 的低中高負載設定；(2) 確認暖機時間、量測時間與種子數。

執行：
  .\.venv\Scripts\python.exe pilot.py load
  .\.venv\Scripts\python.exe pilot.py timing
試跑種子 (1001 起) 與正式種子 (1..30) 分開，避免用同一批資料選參數又拿來報結果。
"""

import csv
import json
import os
import sys

import numpy as np

from npca_sim.experiment import run_many
from npca_sim.scenarios import (
    CHOSEN_LOADS_PATH, P_BUSY_TARGETS, S_BUSY_TARGETS, TARGET_MBPS, P_LEVELS,
)
from npca_sim.stats import mean_ci

PILOT_DIR = os.path.join("results", "pilot")
PILOT_SEEDS = list(range(1001, 1006))


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def interp_rate(rates, ratios, target):
    """在 (速率, 平均忙碌比例) 曲線上線性內插，找出達到 target 的速率。"""
    for i in range(len(rates) - 1):
        if ratios[i] <= target <= ratios[i + 1]:
            f = (target - ratios[i]) / (ratios[i + 1] - ratios[i])
            return rates[i] + f * (rates[i + 1] - rates[i])
    raise ValueError(f"目標 {target} 不在已掃描範圍 {ratios[0]:.3f}..{ratios[-1]:.3f}")


def cmd_load():
    t_warm, t_meas = 1.0, 4.0
    # --- 主通道：Legacy，副通道輕載，目標輕/重載 ---
    grid_p = list(range(50, 801, 50))
    jobs = [
        dict(stage="P_grid", method="legacy", seed=s, rate_p_pps=r, rate_s_pps=150.0,
             target_mbps=TARGET_MBPS[t], target_level=t, t_warm_s=t_warm, t_meas_s=t_meas)
        for r in grid_p for t in ("light", "heavy") for s in PILOT_SEEDS
    ]
    # --- 副通道：Legacy，主通道無 OBSS (只看 OBSS_S 自己的忙碌比例) ---
    grid_s = list(range(50, 801, 50))
    jobs += [
        dict(stage="S_grid", method="legacy", seed=s, rate_p_pps=0.0, rate_s_pps=r,
             target_mbps=TARGET_MBPS["light"], target_level="light", t_warm_s=t_warm, t_meas_s=t_meas)
        for r in grid_s for s in PILOT_SEEDS
    ]
    rows = run_many(jobs)
    write_csv(os.path.join(PILOT_DIR, "pilot_load_grid_raw.csv"), rows)

    def curve(stage, key, level=None, rate_key="rate_p_pps"):
        xs, ys = [], []
        for r in sorted({x[rate_key] for x in rows if x["stage"] == stage}):
            v = [x[key] for x in rows if x["stage"] == stage and x[rate_key] == r and (level is None or x["target_level"] == level)]
            xs.append(r)
            ys.append(float(np.mean(v)))
        return xs, ys

    px, py_light = curve("P_grid", "obss_p_busy_ratio", "light")
    _, py_heavy = curve("P_grid", "obss_p_busy_ratio", "heavy")
    py = [(a + b) / 2 for a, b in zip(py_light, py_heavy)]  # 兩種目標負載下的平均曲線，重載時忙碡比例飽和在約 0.45
    sx, sy = curve("S_grid", "obss_s_busy_ratio", rate_key="rate_s_pps")
    print("主通道 OBSS 速率 -> 平均忙碡比例 (Legacy, 目標輕載 | 目標重載 | 兩者平均)")
    for r, a, b, c in zip(px, py_light, py_heavy, py):
        print(f"  {r:6.0f} pps  {a:.3f} | {b:.3f} | {c:.3f}")
    print("副通道 OBSS 速率 -> 平均忙碌比例")
    for r, a in zip(sx, sy):
        print(f"  {r:6.0f} pps  {a:.3f}")

    chosen = {"rate_p_pps": {}, "rate_s_pps": {}, "target_mbps": TARGET_MBPS,
              "pilot_seeds": PILOT_SEEDS, "pilot_t_warm_s": t_warm, "pilot_t_meas_s": t_meas}
    for lvl in P_LEVELS:
        chosen["rate_p_pps"][lvl] = round(interp_rate(px, py, P_BUSY_TARGETS[lvl]), 1)
    for lvl, tgt in S_BUSY_TARGETS.items():
        chosen["rate_s_pps"][lvl] = round(interp_rate(sx, sy, tgt), 1)

    # --- 以選定速率再驗證一次 (獨立的驗證種子) ---
    ver = []
    for lvl, r in chosen["rate_p_pps"].items():
        for t in ("light", "heavy"):
            for s in range(2001, 2006):
                ver.append(dict(stage="verify_P", level=lvl, method="legacy", seed=s, rate_p_pps=r,
                                rate_s_pps=chosen["rate_s_pps"]["light"], target_mbps=TARGET_MBPS[t],
                                target_level=t, t_warm_s=t_warm, t_meas_s=t_meas))
    for lvl, r in chosen["rate_s_pps"].items():
        for s in range(2001, 2006):
            ver.append(dict(stage="verify_S", level=lvl, method="legacy", seed=s, rate_p_pps=0.0, rate_s_pps=r,
                            target_mbps=TARGET_MBPS["light"], target_level="light", t_warm_s=t_warm, t_meas_s=t_meas))
    vrows = run_many(ver)
    write_csv(os.path.join(PILOT_DIR, "pilot_load_verify_raw.csv"), vrows)
    chosen["verified"] = {}
    print("\n選定負載與 Legacy 驗證 (獨立種子 2001-2005)：")
    for lvl in P_LEVELS:
        for t in ("light", "heavy"):
            v = [x["obss_p_busy_ratio"] for x in vrows if x["stage"] == "verify_P" and x["level"] == lvl and x["target_level"] == t]
            chosen["verified"][f"P_{lvl}_T{t}"] = round(float(np.mean(v)), 4)
            print(f"  主通道 {lvl:4s} rate={chosen['rate_p_pps'][lvl]:6.1f} pps, 目標{t}: 測得忙碌比例 {np.mean(v):.3f}")
    for lvl in S_BUSY_TARGETS:
        v = [x["obss_s_busy_ratio"] for x in vrows if x["stage"] == "verify_S" and x["level"] == lvl]
        chosen["verified"][f"S_{lvl}"] = round(float(np.mean(v)), 4)
        print(f"  副通道 {lvl:5s} rate={chosen['rate_s_pps'][lvl]:6.1f} pps: 測得忙碌比例 {np.mean(v):.3f}")
    os.makedirs(PILOT_DIR, exist_ok=True)
    with open(CHOSEN_LOADS_PATH, "w", encoding="utf-8") as f:
        json.dump(chosen, f, ensure_ascii=False, indent=2)
    print("已寫入", CHOSEN_LOADS_PATH)


def cmd_timing():
    from npca_sim.scenarios import load_chosen
    ch = load_chosen()
    # 代表性極端場景：主通道中/高負載、目標輕/重載、副通道輕/重載；Legacy 與固定門檻 NPCA
    combos = [("mid", "light", "light"), ("high", "heavy", "heavy"), ("mid", "heavy", "light"), ("high", "light", "heavy")]
    seeds = PILOT_SEEDS
    block_s = 0.5
    jobs = []
    for p, s, t in combos:
        for m in ("legacy", "fixed"):
            for sd in seeds:
                jobs.append(dict(stage="blocks", scen=f"P{p}_S{s}_T{t}", method=m, seed=sd, rate_p_pps=ch["rate_p_pps"][p],
                                 rate_s_pps=ch["rate_s_pps"][s], target_mbps=TARGET_MBPS[t], t_warm_s=0.0, t_meas_s=20.0,
                                 block_s=block_s, theta0_us=500))
    rows = run_many(jobs)
    # 每 0.5 s 區塊：跨種子平均吞吐量與平均延遲
    out = []
    print("每 0.5 s 區塊的跨種子平均吞吐量 (Mbps) / 平均延遲 (ms)，由 t=0 (空佇列) 開始：")
    for p, s, t in combos:
        for m in ("legacy", "fixed"):
            sel = [r for r in rows if r["scen"] == f"P{p}_S{s}_T{t}" and r["method"] == m]
            n_blk = int(20.0 / block_s)
            tp, dl = [], []
            for k in range(n_blk):
                bits = [r["blocks"].get(k, [0, 0, 0])[0] for r in sel]
                cnt = [r["blocks"].get(k, [0, 0, 0])[1] for r in sel]
                dsum = [r["blocks"].get(k, [0, 0, 0])[2] for r in sel]
                tp.append(np.mean(bits) / block_s / 1e6)
                dl.append(np.sum(dsum) / max(1, np.sum(cnt)) / 1e6)
                out.append(dict(scen=f"P{p}_S{s}_T{t}", method=m, block_start_s=k * block_s,
                                throughput_mbps=tp[-1], mean_delay_ms=dl[-1]))
            print(f"  {p}/{s}/{t} {m:6s} tput:", " ".join(f"{x:5.1f}" for x in tp[:12]), "...", f"| 後半平均 {np.mean(tp[20:]):.2f}")
            print(f"  {'':16s}   delay:", " ".join(f"{x:5.0f}" for x in dl[:12]), "...", f"| 後半平均 {np.mean(dl[20:]):.1f}")
            for name, xs in (("throughput", tp), ("delay", dl)):
                plateau = np.mean(xs[20:])
                k = next(k for k in range(len(xs)) if all(abs(x - plateau) <= 0.10 * plateau for x in xs[k:]))
                print(f"  {'':16s}   {name}: 自 t={k * block_s:.1f} s 起各 0.5 s 區塊均落在後半平均 ±10% 內")
    write_csv(os.path.join(PILOT_DIR, "pilot_timing_blocks.csv"), out)

    # 量測長度 vs 信賴區間半寬：暖機固定 2 s，20 個試跑種子
    print("\n量測長度對信賴區間的影響 (暖機 2 s；20 個試跑種子；Legacy)：")
    seeds20 = list(range(3001, 3021))
    jobs, meta = [], []
    for p, s, t in combos:
        for tm in (2.0, 5.0, 10.0, 20.0):
            for sd in seeds20:
                jobs.append(dict(stage="len", scen=f"P{p}_S{s}_T{t}", method="legacy", seed=sd, rate_p_pps=ch["rate_p_pps"][p],
                                 rate_s_pps=ch["rate_s_pps"][s], target_mbps=TARGET_MBPS[t], t_warm_s=2.0, t_meas_s=tm))
    rows = run_many(jobs)
    write_csv(os.path.join(PILOT_DIR, "pilot_length_raw.csv"), rows)
    for p, s, t in combos:
        for tm in (2.0, 5.0, 10.0, 20.0):
            sel = [r for r in rows if r["scen"] == f"P{p}_S{s}_T{t}" and r["t_meas_s"] == tm]
            m1, _, h1, _ = mean_ci([r["throughput_mbps"] for r in sel])
            m2, _, h2, _ = mean_ci([r["mean_delay_ms"] for r in sel])
            print(f"  {p}/{s}/{t} T_meas={tm:4.0f}s  tput {m1:6.2f} ±{h1:5.2f} ({100 * h1 / m1:4.1f}%)   delay {m2:7.1f} ±{h2:6.1f} ({100 * h2 / m2:4.1f}%)")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("load", "timing"):
        print(__doc__)
        sys.exit(1)
    os.makedirs(PILOT_DIR, exist_ok=True)
    with open(os.path.join(PILOT_DIR, f"pilot_{sys.argv[1]}_log.txt"), "w", encoding="utf-8") as logf:
        sys.stdout = Tee(sys.__stdout__, logf)
        {"load": cmd_load, "timing": cmd_timing}[sys.argv[1]]()
        sys.stdout = sys.__stdout__
