"""正式實驗與固定門檻敏感度實驗。

  .\.venv\Scripts\python.exe run_experiment.py main   # 12 情境 x 3 方法 x 30 種子 -> results/raw_per_seed.csv
  .\.venv\Scripts\python.exe run_experiment.py sens   # 基礎門檻 / 係數敏感度 -> results/raw_sensitivity.csv
  .\.venv\Scripts\python.exe run_experiment.py events # 事件紀錄小片段 -> results/event_log_sample.csv
  .\.venv\Scripts\python.exe run_experiment.py improve # 改善變體 (佇列感知 / 視窗長度 / oracle 診斷) -> results/raw_improve.csv
  .\.venv\Scripts\python.exe run_experiment.py rate     # 目標負載延伸 10/15/20/25 Mbps -> results/raw_rate.csv

暖機/量測長度與種子數依 pilot.py timing 的結果決定 (見 README)。
"""

import csv
import json
import os
import platform
import sys
import time

import numpy as np

from npca_sim.config import US
from npca_sim.experiment import N_STA, run_many
from npca_sim.scenarios import P_LEVELS, S_LEVELS, TARGET_MBPS, all_scenarios, load_chosen, scenario_id

T_WARM_S = 3.0
T_MEAS_S = 10.0
SEEDS = list(range(1, 31))           # 30 個獨立種子 (試跑用 1001 起，互不重疊)
BASE_THETA_US = 500                  # 固定/自適應共用的基礎門檻
K_US = 2000                          # 自適應係數：門檻 = 基礎門檻 + K * 副通道近期忙碌比例
SENS_THETAS_US = [0, 200, 500, 1000, 2000]
SENS_KS_US = [1000, 4000]

OUT = "results"

META_COLS = ["scenario", "p_level", "s_level", "t_level", "variant", "method", "theta0_us", "k_us", "seed",
             "rate_p_pps", "rate_s_pps", "target_mbps", "t_warm_s", "t_meas_s"]


def write_csv(path, rows, first_cols=()):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = [c for c in first_cols if c in rows[0]]
    for k in rows[0]:
        if k not in keys:
            keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.9g}" if isinstance(v, float) else v) for k, v in r.items()})


def variant_name(method, theta0_us, k_us):
    if method == "legacy":
        return "legacy"
    if method == "fixed":
        return f"fixed_th{theta0_us}"
    return f"adaptive_th{theta0_us}_k{k_us}"


def make_jobs(variants):
    """variants: [(method, theta0_us, k_us)]。同一 (情境, 種子) 下所有變體共用相同的到達序列。"""
    ch = load_chosen()
    jobs = []
    for p, s, t in all_scenarios():
        for method, th, k in variants:
            for seed in SEEDS:
                jobs.append(dict(
                    scenario=scenario_id(p, s, t), p_level=p, s_level=s, t_level=t,
                    variant=variant_name(method, th, k), method=method, theta0_us=th, k_us=k, seed=seed,
                    rate_p_pps=ch["rate_p_pps"][p], rate_s_pps=ch["rate_s_pps"][s], target_mbps=TARGET_MBPS[t],
                    t_warm_s=T_WARM_S, t_meas_s=T_MEAS_S,
                ))
    return jobs


def save_run_config(extra=None):
    cfg = {
        "t_warm_s": T_WARM_S, "t_meas_s": T_MEAS_S, "seeds": SEEDS, "n_sta": N_STA,
        "base_theta_us": BASE_THETA_US, "k_us": K_US, "sens_thetas_us": SENS_THETAS_US, "sens_ks_us": SENS_KS_US,
        "chosen_loads": load_chosen(),
        "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
    }
    try:
        import matplotlib
        cfg["matplotlib"] = matplotlib.__version__
    except ImportError:
        pass
    if extra:
        cfg.update(extra)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def cmd_main():
    variants = [("legacy", BASE_THETA_US, K_US), ("fixed", BASE_THETA_US, K_US), ("adaptive", BASE_THETA_US, K_US)]
    jobs = make_jobs(variants)
    print(f"正式實驗：{len(jobs)} 次模擬 (12 情境 x 3 方法 x {len(SEEDS)} 種子)")
    t0 = time.time()
    rows = run_many(jobs)
    order = {v: i for i, v in enumerate(["legacy", f"fixed_th{BASE_THETA_US}", f"adaptive_th{BASE_THETA_US}_k{K_US}"])}
    rows.sort(key=lambda r: (r["scenario"], order[r["variant"]], r["seed"]))
    write_csv(os.path.join(OUT, "raw_per_seed.csv"), rows, META_COLS)
    save_run_config({"main_elapsed_s": round(time.time() - t0, 1)})
    print(f"完成，耗時 {time.time() - t0:.0f} s -> results/raw_per_seed.csv")


def cmd_sens():
    variants = []
    for th in SENS_THETAS_US:
        variants.append(("fixed", th, K_US))
        variants.append(("adaptive", th, K_US))
    for k in SENS_KS_US:
        variants.append(("adaptive", BASE_THETA_US, k))
    # 與主實驗重複的變體 (th=500,k=2000) 不重跑，分析時由 raw_per_seed.csv 併入
    variants = [v for v in variants if not (v[1] == BASE_THETA_US and v[2] == K_US)]
    jobs = make_jobs(variants)
    print(f"敏感度實驗：{len(jobs)} 次模擬")
    t0 = time.time()
    rows = run_many(jobs)
    rows.sort(key=lambda r: (r["scenario"], r["variant"], r["seed"]))
    write_csv(os.path.join(OUT, "raw_sensitivity.csv"), rows, META_COLS)
    print(f"完成，耗時 {time.time() - t0:.0f} s -> results/raw_sensitivity.csv")


IMPROVE_DEFAULTS = dict(theta0_us=500, k_us=2000, theta_hi_us=500, qa_min_backlog=20, busy_window_ms=20)
IMPROVE_SPECS = [
    dict(variant="legacy", method="legacy"),
    dict(variant="fixed_th0", method="fixed", theta0_us=0),
    dict(variant="fixed_th500", method="fixed"),
    dict(variant="adaptive_th500_k2000_w20", method="adaptive"),
    dict(variant="adaptive_th500_k2000_w5", method="adaptive", busy_window_ms=5),
    dict(variant="adaptive_th500_k2000_w2", method="adaptive", busy_window_ms=2),
    dict(variant="queue_lo0_hi500_B20", method="queue", theta0_us=0),
    dict(variant="oracle_th0", method="oracle", theta0_us=0),
]
RATE_MBPS = [10, 15, 20, 25]
RATE_SPECS = [
    dict(variant="legacy", method="legacy"),
    dict(variant="fixed_th500", method="fixed"),
    dict(variant="adaptive_th500_k2000", method="adaptive"),
]


def make_jobs_spec(specs, scen):
    """specs: 變體參數 dict 列表；scen: [(p, s, t_level, target_mbps)]。未指定的參數用 IMPROVE_DEFAULTS。"""
    ch = load_chosen()
    jobs = []
    for p, s, tl, mbps in scen:
        for sp in specs:
            cfg = {**IMPROVE_DEFAULTS, **sp}
            for seed in SEEDS:
                jobs.append(dict(
                    scenario=scenario_id(p, s, tl), p_level=p, s_level=s, t_level=tl, seed=seed,
                    rate_p_pps=ch["rate_p_pps"][p], rate_s_pps=ch["rate_s_pps"][s], target_mbps=mbps,
                    t_warm_s=T_WARM_S, t_meas_s=T_MEAS_S, **cfg,
                ))
    return jobs


def cmd_improve():
    jobs = make_jobs_spec(IMPROVE_SPECS, [(p, s, t, TARGET_MBPS[t]) for p, s, t in all_scenarios()])
    print(f"改善實驗：{len(jobs)} 次模擬 ({len(IMPROVE_SPECS)} 變體 x 12 情境 x {len(SEEDS)} 種子)")
    t0 = time.time()
    rows = run_many(jobs)
    rows.sort(key=lambda r: (r["scenario"], r["variant"], r["seed"]))
    write_csv(os.path.join(OUT, "raw_improve.csv"), rows, META_COLS)
    print(f"完成，耗時 {time.time() - t0:.0f} s -> results/raw_improve.csv")


def cmd_rate():
    scen = [(p, s, str(m), float(m)) for p in P_LEVELS for s in S_LEVELS for m in RATE_MBPS]
    jobs = make_jobs_spec(RATE_SPECS, scen)
    print(f"負載延伸實驗：{len(jobs)} 次模擬 ({len(RATE_MBPS)} 個負載 x 6 情境 x 3 方法 x {len(SEEDS)} 種子)")
    t0 = time.time()
    rows = run_many(jobs)
    rows.sort(key=lambda r: (r["scenario"], r["variant"], r["seed"]))
    write_csv(os.path.join(OUT, "raw_rate.csv"), rows, META_COLS)
    print(f"完成，耗時 {time.time() - t0:.0f} s -> results/raw_rate.csv")


def cmd_events():
    """輸出一小段事件紀錄 (自適應門檻、P_mid/S_heavy/T_heavy、seed 1)，涵蓋切換、退避凍結/續倒數、碰撞與返回。"""
    from npca_sim.sim import SimConfig, Simulator
    from npca_sim.traffic import make_arrivals, make_backoff_rngs, sta_rate_pps
    ch = load_chosen()
    seed = 1
    cfg = SimConfig(method="adaptive", theta0_ns=BASE_THETA_US * US, k_ns=float(K_US * US),
                    t_warm_ns=int(T_WARM_S * 1e9), t_meas_ns=int(T_MEAS_S * 1e9), n_sta=N_STA,
                    trace=True, trace_limit=400_000)
    arr = make_arrivals(seed, N_STA, sta_rate_pps(TARGET_MBPS["heavy"], N_STA), ch["rate_p_pps"]["mid"],
                        ch["rate_s_pps"]["heavy"], int((T_WARM_S + T_MEAS_S) * 1e9) + 10_000_000)
    cfg.t_meas_ns = int(0.2e9)
    cfg.t_warm_ns = int(T_WARM_S * 1e9)
    sim = Simulator(cfg, arr, make_backoff_rngs(seed, N_STA))
    sim.run()
    log = [e for e in sim.log if e[0] >= cfg.t_warm_ns and e[1] not in ("ARRIVAL", "ARRIVAL_OVERFLOW")]
    # 窗口：第一個「有成功送達」的切換 (含其前一個 OBSS_P 傳輸；若 8 ms 內有被拒絕的判斷則一併收入)
    i_back = next(i for i, e in enumerate(log) if e[1] == "NPCA_SWITCH_BACK" and not e[4].startswith("deliveries=0"))
    i_out = max(i for i in range(i_back) if log[i][1] == "NPCA_SWITCH_OUT")
    i_start = max(i for i in range(i_out) if log[i][1] == "TX_START" and log[i][2] == "OBSS_P")
    prev = [i for i in range(i_start) if log[i][1] in ("NPCA_REJECT", "NPCA_SKIP")]
    if prev and log[i_start][0] - log[prev[-1]][0] < 8 * 1_000_000:
        i_start = max(i for i in range(prev[-1]) if log[i][1] == "TX_START" and log[i][2] == "OBSS_P")
    seg = log[i_start:i_back + 16]
    rows = [dict(time_us=f"{e[0] / US:.1f}", event=e[1], node=e[2], channel=("P" if e[3] == 0 else "S"), detail=e[4]) for e in seg]
    write_csv(os.path.join(OUT, "event_log_sample.csv"), rows)
    print(f"事件紀錄片段 {len(rows)} 列，t = {seg[0][0] / US:.1f} .. {seg[-1][0] / US:.1f} us -> results/event_log_sample.csv")


if __name__ == "__main__":
    cmds = {"main": cmd_main, "sens": cmd_sens, "events": cmd_events, "improve": cmd_improve, "rate": cmd_rate}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        sys.exit(1)
    cmds[sys.argv[1]]()
