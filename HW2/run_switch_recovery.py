"""切回主通道後的恢復時間敏感度實驗 (補充實驗；恢復時間為假設性參數)。

恢復時間不是 IEEE 802.11bn 的規定，也不是實測硬體數值。
目標 BSS 切回主通道後，在 recovery 時間內不開始倒數 (退避計數保留)；鄰網照常競爭。
情境：主通道 OBSS 高負載 (P-high)、副通道 OBSS 輕/重載 (S-light/S-heavy)、目標 BSS 30 Mbps。

  .\\.venv\\Scripts\\python.exe run_switch_recovery.py check   # 種子 1 短測：0 µs 對齊舊結果 + 事件紀錄檢查
  .\\.venv\\Scripts\\python.exe run_switch_recovery.py run     # 種子 1-30 正式實驗 -> results/switch_recovery/
"""

import bisect
import csv
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from npca_sim.config import DIFS, US
from npca_sim.experiment import N_STA, run_many
from npca_sim.scenarios import TARGET_MBPS, load_chosen, scenario_id
from npca_sim.sim import SimConfig, Simulator
from npca_sim.stats import mean_ci
from npca_sim.traffic import make_arrivals, make_backoff_rngs, sta_rate_pps

OUT = os.path.join("results", "switch_recovery")
OLD = os.path.join("results")
P_LEVEL = "high"
S_LEVELS = ["light", "heavy"]
T_LEVEL = "heavy"
SEEDS = list(range(1, 31))
T_WARM_S = 3.0
T_MEAS_S = 10.0
RECOVERY_US = [0, 100, 300]          # 假設性敏感度值，非標準規定
CHECK_SEED = 1
CHECK_MEAS_S = 5.0

ADAPT = "adaptive_th500_k2000_w20"
REF0 = "fixed_th0"
REF500 = "fixed_th500"
VARIANTS = [
    dict(variant=REF0, method="fixed", theta0_us=0, k_us=2000, busy_window_ms=20),
    dict(variant=REF500, method="fixed", theta0_us=500, k_us=2000, busy_window_ms=20),
    dict(variant=ADAPT, method="adaptive", theta0_us=500, k_us=2000, busy_window_ms=20),
]
# 0 µs 應與既有結果逐項相同：(檔名, 舊 variant 名稱)
OLD_MAP = {
    REF0: [("raw_improve.csv", "fixed_th0")],
    REF500: [("raw_per_seed.csv", "fixed_th500"), ("raw_improve.csv", "fixed_th500")],
    ADAPT: [("raw_per_seed.csv", "adaptive_th500_k2000"), ("raw_improve.csv", "adaptive_th500_k2000_w20")],
}
CHECK_METRICS = ["throughput_mbps", "mean_delay_ms", "drop_rate", "delivered_win", "tx_attempts", "collisions",
                 "n_switch", "n_switch_no_delivery", "n_switch_no_tx", "npca_delivered",
                 "obss_p_busy_ratio", "obss_s_busy_ratio"]
METRICS = ["throughput_mbps", "mean_delay_ms", "drop_rate", "n_switch", "n_switch_no_delivery"]
PAIRS = [("adaptive - fixed_th0", ADAPT, REF0), ("adaptive - fixed_th500", ADAPT, REF500)]


def make_jobs(seeds, recoveries, t_meas_s=T_MEAS_S, t_warm_s=T_WARM_S):
    ch = load_chosen()
    jobs = []
    for s in S_LEVELS:
        for r in recoveries:
            for v in VARIANTS:
                for seed in seeds:
                    jobs.append(dict(
                        scenario=scenario_id(P_LEVEL, s, T_LEVEL), p_level=P_LEVEL, s_level=s, t_level=T_LEVEL,
                        variant=v["variant"], method=v["method"], theta0_us=v["theta0_us"], k_us=v["k_us"],
                        busy_window_ms=v["busy_window_ms"], recovery_us=r, seed=seed,
                        rate_p_pps=ch["rate_p_pps"][P_LEVEL], rate_s_pps=ch["rate_s_pps"][s],
                        target_mbps=TARGET_MBPS[T_LEVEL], t_warm_s=t_warm_s, t_meas_s=t_meas_s,
                    ))
    return jobs


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


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]


def _num(v):
    try:
        return float(v)
    except ValueError:
        return v


def close(a, b):
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)


# ------------------------------------------------------------------ check
def check_zero_vs_old(say):
    """0 µs 的新結果是否與既有結果逐項相同 (種子 1)。"""
    rows = run_many(make_jobs([CHECK_SEED], [0]), progress=False)
    old_cache = {}
    ok_all = True
    n_cmp = 0
    for r in rows:
        for fname, oldv in OLD_MAP[r["variant"]]:
            key = (fname, oldv, r["s_level"])
            if key not in old_cache:
                old_cache[key] = [o for o in read_csv(os.path.join(OLD, fname))
                                  if o["variant"] == oldv and o["p_level"] == P_LEVEL
                                  and o["s_level"] == r["s_level"] and o["t_level"] == T_LEVEL]
            match = [o for o in old_cache[key] if o["seed"] == r["seed"]]
            if len(match) != 1:
                ok_all = False
                say(f"    [FAIL] 找不到舊結果：{fname} {oldv} S-{r['s_level']} seed {r['seed']}")
                continue
            o = match[0]
            bad = [m for m in CHECK_METRICS if not close(float(r[m]), float(o[m]))]
            n_cmp += 1
            if bad:
                ok_all = False
                say(f"    [FAIL] {r['variant']} S-{r['s_level']} vs {fname}:{oldv}：不一致欄位 {bad}")
    say(f"    [{'PASS' if ok_all else 'FAIL'}] recovery=0 的 {len(rows)} 筆結果，與既有結果 {n_cmp} 組比對，"
        f"{len(CHECK_METRICS)} 個欄位相對誤差 <= 1e-6")
    return ok_all


def check_events(say):
    """非零恢復期：恢復期內目標 BSS 不在主通道起送、鄰網仍傳輸，且恢復期後才起送。"""
    ch = load_chosen()
    ok_all = True
    for s in S_LEVELS:
        for v in VARIANTS:
            for r in (100, 300):
                cfg = SimConfig(method=v["method"], theta0_ns=v["theta0_us"] * US, k_ns=float(v["k_us"] * US),
                                busy_window_ns=v["busy_window_ms"] * 1_000_000,
                                t_warm_ns=0, t_meas_ns=int(CHECK_MEAS_S * 1e9), n_sta=N_STA,
                                trace=True, trace_limit=3_000_000, recovery_ns=r * US)
                arr = make_arrivals(CHECK_SEED, N_STA, sta_rate_pps(TARGET_MBPS[T_LEVEL], N_STA),
                                    ch["rate_p_pps"][P_LEVEL], ch["rate_s_pps"][s],
                                    int(CHECK_MEAS_S * 1e9) + 10_000_000)
                sim = Simulator(cfg, arr, make_backoff_rngs(CHECK_SEED, N_STA))
                sim.run()
                log = sim.log
                backs = sorted(e[0] for e in log if e[1] == "NPCA_SWITCH_BACK")
                tgt_p_tx = sorted(e[0] for e in log if e[1] == "TX_START" and e[2].startswith("STA") and e[3] == 0)
                op_tx = sorted(e[0] for e in log if e[1] == "TX_START" and e[2] == "OBSS_P")
                rec = r * US
                # 1) 切回後的每次主通道起送 (目標 BSS)：距最近一次切回至少 R + DIFS
                min_gap = None
                n_tgt_p = 0
                viol = 0
                for t in tgt_p_tx:
                    i = bisect.bisect_right(backs, t)
                    if i == 0:
                        continue
                    gap = t - backs[i - 1]
                    n_tgt_p += 1
                    min_gap = gap if min_gap is None else min(min_gap, gap)
                    if gap < rec + DIFS:
                        viol += 1
                # 2) 恢復期內鄰網仍起送
                op_in_win = sum(bisect.bisect_left(op_tx, b + rec) - bisect.bisect_right(op_tx, b) for b in backs)
                ok = viol == 0 and (r == 0 or op_in_win > 0) and n_tgt_p > 0
                ok_all &= ok
                say(f"    [{'PASS' if ok else 'FAIL'}] S-{s} {v['variant']} R={r} us：切回 {len(backs)} 次，"
                    f"切回後目標 BSS 主通道起送 {n_tgt_p} 次，最小間隔 {(min_gap or 0) / US:.1f} us "
                    f"(需 >= {(rec + DIFS) / US:.0f} us)，違規 {viol} 次；恢復期內 OBSS_P 起送 {op_in_win} 次")
    return ok_all


def cmd_check():
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def say(m=""):
        print(m)
        lines.append(m)

    say(f"切回恢復時間短測 (種子 {CHECK_SEED}，P-high，T-heavy，S-light/heavy)")
    say("[A] recovery = 0 µs 與既有結果逐項比對")
    a_ok = check_zero_vs_old(say)
    say(f"[B] recovery > 0 的事件紀錄檢查 (量測 {CHECK_MEAS_S:g} s)")
    b_ok = check_events(say)
    verdict = "全部通過" if (a_ok and b_ok) else "有失敗項目"
    say(f"結論：{verdict}")
    with open(os.path.join(OUT, "check_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if not (a_ok and b_ok):
        sys.exit(1)


# ------------------------------------------------------------------ run
def cmd_run():
    os.makedirs(OUT, exist_ok=True)
    jobs = make_jobs(SEEDS, RECOVERY_US)
    print(f"切回恢復時間實驗：{len(jobs)} 次模擬 ({len(S_LEVELS)} 副通道負載 x {len(RECOVERY_US)} 恢復時間 x "
          f"{len(VARIANTS)} 方法 x {len(SEEDS)} 種子)")
    t0 = time.time()
    rows = run_many(jobs)
    rows.sort(key=lambda r: (r["s_level"], r["recovery_us"], r["variant"], r["seed"]))
    write_csv(os.path.join(OUT, "raw_recovery_per_seed.csv"), rows,
              ["scenario", "p_level", "s_level", "t_level", "variant", "method", "theta0_us", "k_us", "recovery_us", "seed"])
    print(f"模擬完成，耗時 {time.time() - t0:.0f} s")
    summarize(rows)
    cfg = {
        "hypothetical_parameter": "recovery_us 為假設性敏感度參數，非 IEEE 802.11bn 規定或實測硬體數值",
        "p_level": P_LEVEL, "s_levels": S_LEVELS, "t_level": T_LEVEL, "target_mbps": TARGET_MBPS[T_LEVEL],
        "recovery_us": RECOVERY_US, "seeds": SEEDS, "t_warm_s": T_WARM_S, "t_meas_s": T_MEAS_S,
        "variants": VARIANTS, "python": platform.python_version(), "numpy": np.__version__,
        "matplotlib": matplotlib.__version__, "platform": platform.platform(),
    }
    with open(os.path.join(OUT, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def summarize(rows):
    idx = defaultdict(dict)
    for r in rows:
        idx[(r["s_level"], int(r["recovery_us"]), r["variant"])][r["seed"]] = r

    summ = []
    for s in S_LEVELS:
        for rec in RECOVERY_US:
            for v in VARIANTS:
                d = idx[(s, rec, v["variant"])]
                seeds = sorted(d)
                row = {"s_level": s, "recovery_us": rec, "variant": v["variant"], "n_seeds": len(seeds)}
                for m in METRICS:
                    mean, _, half, _ = mean_ci([d[x][m] for x in seeds])
                    row[f"{m}_mean"] = mean
                    row[f"{m}_ci95"] = half
                row["wasted_switch_frac_of_means"] = row["n_switch_no_delivery_mean"] / row["n_switch_mean"]
                summ.append(row)
    write_csv(os.path.join(OUT, "summary_recovery.csv"), summ, ["s_level", "recovery_us", "variant", "n_seeds"])

    pair_rows = []
    for s in S_LEVELS:
        for rec in RECOVERY_US:
            for label, a, b in PAIRS:
                da, db = idx[(s, rec, a)], idx[(s, rec, b)]
                seeds = sorted(set(da) & set(db))
                for m in METRICS:
                    diffs = np.array([da[x][m] - db[x][m] for x in seeds], dtype=float)
                    mean, sd, half, n = mean_ci(diffs)
                    pair_rows.append({
                        "s_level": s, "recovery_us": rec, "comparison": label, "metric": m,
                        "mean_diff": mean, "sd_diff": sd, "ci95_halfwidth": half,
                        "ci_low": mean - half, "ci_high": mean + half,
                        "ci_excludes_zero": int(mean - half > 0 or mean + half < 0), "n_pairs": n,
                    })
    # 相對於 recovery = 0 的差距變化
    base = {(r["s_level"], r["comparison"], r["metric"]): r["mean_diff"] for r in pair_rows if r["recovery_us"] == 0}
    for r in pair_rows:
        r["mean_diff_change_vs_0us"] = r["mean_diff"] - base[(r["s_level"], r["comparison"], r["metric"])]
    write_csv(os.path.join(OUT, "paired_recovery.csv"), pair_rows,
              ["s_level", "recovery_us", "comparison", "metric"])

    # 判定：吞吐量 (+ 為改善)、延遲 (- 為改善)、無效切換 (- 為改善)
    good_sign = {"throughput_mbps": 1, "mean_delay_ms": -1, "n_switch_no_delivery": -1}
    pr = {(r["s_level"], r["recovery_us"], r["comparison"], r["metric"]): r for r in pair_rows}
    verdict_rows = []
    for s in S_LEVELS:
        for rec in RECOVERY_US:
            for m, sgn in good_sign.items():
                st = {}
                for label, _, _ in PAIRS:
                    r = pr[(s, rec, label, m)]
                    if not r["ci_excludes_zero"]:
                        st[label] = "無顯著差異"
                    elif r["mean_diff"] * sgn > 0:
                        st[label] = "改善"
                    else:
                        st[label] = "變差"
                good0 = st[PAIRS[0][0]] == "改善"
                good500 = st[PAIRS[1][0]] == "改善"
                worse = "變差" in st.values()
                if worse:
                    v = "至少一個固定門檻上變差"
                elif good0 and good500:
                    v = "相對兩個固定門檻皆改善"
                elif good500:
                    v = "只相對 500 µs 固定門檻改善"
                elif good0:
                    v = "只相對 0 µs 固定門檻改善"
                else:
                    v = "無顯著差異"
                verdict_rows.append({"s_level": s, "recovery_us": rec, "metric": m,
                                     "vs_fixed_th0": st[PAIRS[0][0]], "vs_fixed_th500": st[PAIRS[1][0]],
                                     "verdict": v})
    write_csv(os.path.join(OUT, "verdict_recovery.csv"), verdict_rows, ["s_level", "recovery_us", "metric"])

    print("\n[彙總] 平均 ± 95% CI")
    for r in summ:
        print(f"S-{r['s_level']:<5} R={r['recovery_us']:>3} {r['variant']:<26} "
              f"tput {r['throughput_mbps_mean']:.3f}±{r['throughput_mbps_ci95']:.3f}  "
              f"delay {r['mean_delay_ms_mean']:.2f}±{r['mean_delay_ms_ci95']:.2f}  "
              f"drop {r['drop_rate_mean']:.4f}  sw {r['n_switch_mean']:.1f}  "
              f"sw0 {r['n_switch_no_delivery_mean']:.1f}")
    print("\n[成對差值] adaptive 減固定門檻 (同種子)")
    for r in pair_rows:
        star = "*" if r["ci_excludes_zero"] else " "
        print(f"S-{r['s_level']:<5} R={r['recovery_us']:>3} {r['comparison']:<24} {r['metric']:<20} "
              f"{r['mean_diff']:+10.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]{star}")
    print("\n[判定]")
    for r in verdict_rows:
        print(f"S-{r['s_level']:<5} R={r['recovery_us']:>3} {r['metric']:<20} {r['verdict']}")
    fig_recovery(pr)
    print(f"\n已寫入 {OUT}/ (raw_recovery_per_seed.csv, summary_recovery.csv, paired_recovery.csv, "
          f"verdict_recovery.csv, fig_recovery_paired.png)")


def fig_recovery(pr):
    panels = [("throughput_mbps", "Throughput diff (Mbps)"), ("mean_delay_ms", "Mean delay diff (ms)"),
              ("n_switch_no_delivery", "No-delivery switches diff (count)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {"light": "#2ca02c", "heavy": "#d62728"}
    xs = RECOVERY_US
    for ax, (m, title) in zip(axes, panels):
        for s in S_LEVELS:
            for label, ls in ((PAIRS[0][0], "-"), (PAIRS[1][0], "--")):
                ys = [pr[(s, rec, label, m)]["mean_diff"] for rec in xs]
                es = [pr[(s, rec, label, m)]["ci95_halfwidth"] for rec in xs]
                ax.errorbar(xs, ys, yerr=es, color=colors[s], ls=ls, marker="o", ms=4, capsize=3,
                            label=f"S-{s}, {label}")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Hypothetical recovery time after switch-back (us)")
        ax.set_xticks(xs)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("P-high, T-heavy (30 Mbps): adaptive minus fixed, paired by seed (seeds 1-30, 95% CI). "
                 "Recovery time is a hypothetical sensitivity parameter.", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_recovery_paired.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("check", "run"):
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "check":
        cmd_check()
    else:
        cmd_run()
