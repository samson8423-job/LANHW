"""改善實驗與速率延伸的彙總 (沿用 analyze.py 的讀取與成對統計)。

  輸入：results/raw_improve.csv                                 -> improve_summary.csv, improve_paired.csv
        results/raw_rate.csv + results/raw_per_seed.csv (5/30 Mbps) -> rate_summary.csv, rate_paired.csv,
                                                                      rate_changepoints.csv, fig7_rate_sweep.png

  .\.venv\Scripts\python.exe analyze_extra.py
"""

import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import LEGACY, FIXED, ADAPT, OUT, col, index, load, pair_table, paired, write_csv
from npca_sim.scenarios import P_LEVELS, S_LEVELS, all_scenarios, scenario_id
from npca_sim.stats import mean_ci

# ------------------------------------------------------------------ 改善實驗
IMP_REF = "fixed_th0"
IMP_VARIANTS = [
    "legacy", "fixed_th0", "fixed_th500",
    "adaptive_th500_k2000_w20", "adaptive_th500_k2000_w5", "adaptive_th500_k2000_w2",
    "queue_lo0_hi500_B20", "oracle_th0",
]
IMP_METRICS = [
    "throughput_mbps", "mean_delay_ms", "drop_rate", "n_switch", "n_switch_no_tx", "n_switch_no_delivery",
    "wasted_switch_frac", "npca_delivered", "n_rej_theta", "n_rej_theta_s_idle", "n_rej_oracle", "n_sw_s_busy0",
    "obss_s_tput_mbps", "obss_p_tput_mbps",
]


def print_grid(idx, sc_list, metric, title):
    print(f"\n{title}：平均值；括號為相對 {IMP_REF} 的同種子差值 (* = 95% CI 不含 0)")
    print("情境 | " + " | ".join(IMP_VARIANTS))
    for p, s, t, sc in sc_list:
        cells = []
        for v in IMP_VARIANTS:
            mean = col(idx, sc, v, metric).mean()
            if v == IMP_REF:
                cells.append(f"{mean:.3f}")
                continue
            m, _, half, _ = paired(idx, sc, v, IMP_REF, metric)
            star = "*" if (m - half > 0 or m + half < 0) else ""
            cells.append(f"{mean:.3f} ({m:+.3f}{star})")
        print(f"P-{p} S-{s} T-{t} | " + " | ".join(cells))


def improve_section():
    rows = load(os.path.join(OUT, "raw_improve.csv"))
    idx = index(rows)
    sc_list = [(p, s, t, scenario_id(p, s, t)) for p, s, t in all_scenarios()]

    summ = []
    for p, s, t, sc in sc_list:
        for v in IMP_VARIANTS:
            row = {"scenario": sc, "p_level": p, "s_level": s, "t_level": t, "variant": v, "n_seeds": len(idx[(sc, v)])}
            for m in IMP_METRICS:
                mean, _, half, _ = mean_ci(col(idx, sc, v, m))
                row[f"{m}_mean"] = mean
                row[f"{m}_ci95"] = half
            summ.append(row)
    write_csv(os.path.join(OUT, "improve_summary.csv"), summ)

    pairs = [(f"{v} - {IMP_REF}", v, IMP_REF) for v in IMP_VARIANTS if v != IMP_REF]
    pairs.append((f"{IMP_REF} - {LEGACY}", IMP_REF, LEGACY))
    write_csv(os.path.join(OUT, "improve_paired.csv"), pair_table(idx, pairs))

    print_grid(idx, sc_list, "throughput_mbps", "吞吐量 (Mbps)")
    print_grid(idx, sc_list, "mean_delay_ms", "平均延遲 (ms)")


# ------------------------------------------------------------------ 速率延伸與變化點
RATE_VARIANTS = [LEGACY, FIXED, ADAPT]
RATE_PAIRS = [("fixed - legacy", FIXED, LEGACY), ("adaptive - fixed", ADAPT, FIXED), ("adaptive - legacy", ADAPT, LEGACY)]
RATE_METRICS = ["throughput_mbps", "mean_delay_ms", "drop_rate", "obss_p_busy_ratio", "obss_s_busy_ratio", "wasted_switch_frac", "n_switch"]
RATE_PAIR_METRICS = ["throughput_mbps", "mean_delay_ms", "drop_rate"]
SAT_RATIO = 0.95


def paired2(idx, ka, kb, metric):
    seeds = sorted(set(idx[ka]) & set(idx[kb]))
    a = np.array([idx[ka][x][metric] for x in seeds], dtype=float)
    b = np.array([idx[kb][x][metric] for x in seeds], dtype=float)
    return mean_ci(a - b)


def rate_section():
    rate = load(os.path.join(OUT, "raw_rate.csv"))
    main = [r for r in load(os.path.join(OUT, "raw_per_seed.csv"))
            if r["variant"] in RATE_VARIANTS and r["target_mbps"] in (5.0, 30.0)]
    idx = defaultdict(dict)
    for r in rate + main:
        idx[(r["p_level"], r["s_level"], r["target_mbps"], r["variant"])][r["seed"]] = r
    mbs = sorted({k[2] for k in idx})

    sum_rows, pair_rows = [], []
    for p in P_LEVELS:
        for s in S_LEVELS:
            for mb in mbs:
                for v in RATE_VARIANTS:
                    d = idx[(p, s, mb, v)]
                    seeds = sorted(d)
                    row = {"p_level": p, "s_level": s, "target_mbps": mb, "variant": v, "n_seeds": len(seeds)}
                    for m in RATE_METRICS:
                        mean, _, half, _ = mean_ci(np.array([d[x][m] for x in seeds], dtype=float))
                        row[f"{m}_mean"] = mean
                        row[f"{m}_ci95"] = half
                    row["tput_over_offered"] = row["throughput_mbps_mean"] / mb
                    sum_rows.append(row)
                for label, a, b in RATE_PAIRS:
                    for m in RATE_PAIR_METRICS:
                        mean, sd, half, n = paired2(idx, (p, s, mb, a), (p, s, mb, b), m)
                        pair_rows.append({"p_level": p, "s_level": s, "target_mbps": mb, "comparison": label, "metric": m,
                                          "mean_diff": mean, "sd_diff": sd, "ci95_halfwidth": half,
                                          "ci_low": mean - half, "ci_high": mean + half,
                                          "ci_excludes_zero": int(mean - half > 0 or mean + half < 0), "n_pairs": n})
    write_csv(os.path.join(OUT, "rate_summary.csv"), sum_rows)
    write_csv(os.path.join(OUT, "rate_paired.csv"), pair_rows)

    sum_map = {(r["p_level"], r["s_level"], r["target_mbps"], r["variant"]): r for r in sum_rows}
    pr_map = {(r["p_level"], r["s_level"], r["target_mbps"], r["comparison"], r["metric"]): r for r in pair_rows}

    def first(p, s, pred):
        for mb in mbs:
            if pred(mb):
                return mb
        return float("nan")

    cp_rows = []
    for p in P_LEVELS:
        for s in S_LEVELS:
            sat = first(p, s, lambda mb: sum_map[(p, s, mb, LEGACY)]["tput_over_offered"] < SAT_RATIO)
            gain = first(p, s, lambda mb: pr_map[(p, s, mb, "fixed - legacy", "throughput_mbps")]["mean_diff"] > 0
                         and pr_map[(p, s, mb, "fixed - legacy", "throughput_mbps")]["ci_excludes_zero"])
            dgain = first(p, s, lambda mb: pr_map[(p, s, mb, "fixed - legacy", "mean_delay_ms")]["mean_diff"] < 0
                          and pr_map[(p, s, mb, "fixed - legacy", "mean_delay_ms")]["ci_excludes_zero"])
            pen = first(p, s, lambda mb: pr_map[(p, s, mb, "adaptive - fixed", "throughput_mbps")]["mean_diff"] < 0
                        and pr_map[(p, s, mb, "adaptive - fixed", "throughput_mbps")]["ci_excludes_zero"])
            cp_rows.append({"p_level": p, "s_level": s, "legacy_saturation_mbps": sat,
                            "first_sig_tput_gain_mbps": gain, "first_sig_delay_gain_mbps": dgain,
                            "first_sig_adaptive_penalty_mbps": pen})
    write_csv(os.path.join(OUT, "rate_changepoints.csv"), cp_rows)

    print("\n[速率延伸] 每組 (P,S) 在各目標負載的吞吐量與變化點")
    print("P S | " + " | ".join(f"{mb:g}Mbps leg/fix/adp" for mb in mbs))
    for p in P_LEVELS:
        for s in S_LEVELS:
            cells = []
            for mb in mbs:
                cells.append("/".join(f"{sum_map[(p, s, mb, v)]['throughput_mbps_mean']:.2f}" for v in RATE_VARIANTS))
            print(f"{p} {s} | " + " | ".join(cells))
    print("\n變化點 (Mbps；nan 表示範圍內未出現)")
    for r in cp_rows:
        print(r)

    fig_rate(sum_map, pr_map, mbs)


def fig_rate(sum_map, pr_map, mbs):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    cols = {"low": "#2ca02c", "mid": "#ff7f0e", "high": "#9467bd"}
    for p in P_LEVELS:
        for j, s in enumerate(S_LEVELS):
            ls = "-" if j == 0 else "--"
            lab = f"P-{p}, S-{s}"
            c = cols.get(p, None)
            ys = [sum_map[(p, s, mb, LEGACY)]["throughput_mbps_mean"] for mb in mbs]
            es = [sum_map[(p, s, mb, LEGACY)]["throughput_mbps_ci95"] for mb in mbs]
            axes[0].errorbar(mbs, ys, yerr=es, color=c, ls=ls, marker="o", ms=4, capsize=2, label=lab)
            for ax, comp, title in ((axes[1], "fixed - legacy", "Fixed - Legacy throughput diff (Mbps)"),
                                    (axes[2], "adaptive - fixed", "Adaptive - Fixed throughput diff (Mbps)")):
                ys = [pr_map[(p, s, mb, comp, "throughput_mbps")]["mean_diff"] for mb in mbs]
                es = [pr_map[(p, s, mb, comp, "throughput_mbps")]["ci95_halfwidth"] for mb in mbs]
                ax.errorbar(mbs, ys, yerr=es, color=c, ls=ls, marker="o", ms=4, capsize=2, label=lab)
                ax.set_title(title)
    axes[0].plot(mbs, mbs, "k:", lw=1, label="offered = target")
    axes[0].set_title("Legacy throughput vs target load (Mbps)")
    for ax in axes:
        ax.set_xlabel("Target BSS aggregate load (Mbps)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[2].axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("Mbps")
    fig.suptitle("Rate sweep: 10/15/20/25 Mbps added to main 5/30 Mbps; error bars = 95% CI", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_rate_sweep.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    improve_section()
    rate_section()
    print("\n已寫入 results/improve_*.csv, rate_*.csv, fig7_rate_sweep.png")


if __name__ == "__main__":
    main()
