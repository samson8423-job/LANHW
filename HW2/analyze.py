"""彙總與繪圖：讀取 results/raw_per_seed.csv (與 raw_sensitivity.csv)，輸出彙總 CSV、成對差值 CSV、PNG 圖與 tables.md。

  .\.venv\Scripts\python.exe analyze.py
"""

import csv
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from npca_sim.scenarios import P_LEVELS, S_LEVELS, T_LEVELS, all_scenarios, scenario_id
from npca_sim.stats import mean_ci

OUT = "results"
BASE_TH, BASE_K = 500, 2000
LEGACY = "legacy"
FIXED = f"fixed_th{BASE_TH}"
ADAPT = f"adaptive_th{BASE_TH}_k{BASE_K}"
METHOD_LABEL = {LEGACY: "Legacy", FIXED: f"Fixed NPCA (th={BASE_TH}us)", ADAPT: f"Adaptive NPCA (th0={BASE_TH}us, k={BASE_K}us)"}
COLORS = {LEGACY: "#7f7f7f", FIXED: "#1f77b4", ADAPT: "#d62728"}

METRICS = [
    "throughput_mbps", "mean_delay_ms", "drop_rate", "drop_retry_win", "drop_overflow_win", "arrivals_win",
    "delivered_win", "backlog_end", "backlog_age_mean_ms", "arrivals_total", "delivered_total", "drop_retry_total", "drop_overflow_total",
    "tx_attempts", "collisions", "n_eligible", "n_rej_theta", "n_rej_nobacklog", "n_switch", "n_switch_no_delivery",
    "n_switch_no_tx", "n_switch_useful", "wasted_switch_frac", "npca_delivered", "npca_tx_attempts", "npca_collisions", "npca_window_ms",
    "mean_theta_us", "obss_p_busy_ratio", "obss_s_busy_ratio", "obss_p_tput_mbps", "obss_s_tput_mbps",
    "obss_p_delay_ms", "obss_s_delay_ms", "obss_s_drop_overflow", "obss_s_drop_retry", "obss_s_arrivals",
]
PAIR_METRICS = [
    "throughput_mbps", "mean_delay_ms", "drop_rate", "backlog_end", "n_switch", "n_switch_no_delivery", "n_switch_useful",
    "wasted_switch_frac", "npca_delivered", "npca_collisions", "obss_s_tput_mbps", "obss_p_tput_mbps",
    "obss_s_delay_ms", "obss_p_delay_ms", "obss_p_busy_ratio", "obss_s_busy_ratio",
]
STR_COLS = {"scenario", "p_level", "s_level", "t_level", "variant", "method"}


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                if k not in STR_COLS:
                    r[k] = float(v) if v not in ("", "nan") else float("nan")
            r["seed"] = int(r["seed"])
            r["wasted_switch_frac"] = (r["n_switch_no_delivery"] / r["n_switch"]) if r["n_switch"] > 0 else float("nan")
            r["n_switch_useful"] = r["n_switch"] - r["n_switch_no_delivery"]
            rows.append(r)
    return rows


def index(rows):
    idx = defaultdict(dict)
    for r in rows:
        idx[(r["scenario"], r["variant"])][r["seed"]] = r
    return idx


def col(idx, scen, var, metric, seeds=None):
    d = idx[(scen, var)]
    seeds = sorted(d) if seeds is None else seeds
    return np.array([d[s][metric] for s in seeds], dtype=float)


def paired(idx, scen, var_a, var_b, metric):
    """同一種子下 a - b；任一邊為 NaN 的種子略過。回傳 (mean, sd, half, n)。"""
    seeds = sorted(set(idx[(scen, var_a)]) & set(idx[(scen, var_b)]))
    a = col(idx, scen, var_a, metric, seeds)
    b = col(idx, scen, var_b, metric, seeds)
    return mean_ci(a - b)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def scen_label(p, s, t):
    return f"P-{p}\nS-{s}\nT-{t}"


# ------------------------------------------------------------------ 彙總
def summarize(idx, variants):
    out = []
    for p, s, t in all_scenarios():
        sc = scenario_id(p, s, t)
        for v in variants:
            if (sc, v) not in idx:
                continue
            row = {"scenario": sc, "p_level": p, "s_level": s, "t_level": t, "variant": v, "n_seeds": len(idx[(sc, v)])}
            for m in METRICS:
                mean, sd, half, n = mean_ci(col(idx, sc, v, m))
                row[f"{m}_mean"] = mean
                row[f"{m}_ci95"] = half
            out.append(row)
    return out


def pair_table(idx, pairs, scenarios=None):
    out = []
    for p, s, t in (scenarios or all_scenarios()):
        sc = scenario_id(p, s, t)
        for label, a, b in pairs:
            if (sc, a) not in idx or (sc, b) not in idx:
                continue
            for m in PAIR_METRICS:
                mean, sd, half, n = paired(idx, sc, a, b, m)
                out.append({"scenario": sc, "p_level": p, "s_level": s, "t_level": t, "comparison": label, "metric": m,
                            "mean_diff": mean, "sd_diff": sd, "ci95_halfwidth": half,
                            "ci_low": mean - half, "ci_high": mean + half,
                            "ci_excludes_zero": int(not math.isnan(half) and (mean - half > 0 or mean + half < 0)), "n_pairs": n})
    return out


# ------------------------------------------------------------------ 繪圖
def panel_grid(title_metric, ylabel, fname, summ, metric, ylog=False, sharey_row=True):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharey="row" if sharey_row else False)
    S = {(r["scenario"], r["variant"]): r for r in summ}
    w = 0.26
    for i, t in enumerate(T_LEVELS):
        for j, s in enumerate(S_LEVELS):
            ax = axes[i][j]
            for k, v in enumerate([LEGACY, FIXED, ADAPT]):
                ys = [S[(scenario_id(p, s, t), v)][f"{metric}_mean"] for p in P_LEVELS]
                es = [S[(scenario_id(p, s, t), v)][f"{metric}_ci95"] for p in P_LEVELS]
                ax.bar(np.arange(3) + (k - 1) * w, ys, w, yerr=es, capsize=3, color=COLORS[v], label=METHOD_LABEL[v], error_kw={"lw": 1})
            ticks = []
            for p in P_LEVELS:
                b = [S[(scenario_id(p, s, t), v)]["obss_p_busy_ratio_mean"] for v in (LEGACY, FIXED, ADAPT)]
                ticks.append(f"P-{p}\nmeasured OBSS_P busy:\nL {b[0]:.2f} / F {b[1]:.2f} / A {b[2]:.2f}")
            ax.set_xticks(range(3))
            ax.set_xticklabels(ticks, fontsize=8)
            ax.set_title(f"Target load: {t}   |   Secondary OBSS: {s}", fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            if ylog:
                ax.set_yscale("log")
            if j == 0:
                ax.set_ylabel(ylabel)
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle(title_metric + "  (mean over 30 seeds, error bars = 95% CI)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


def fig_paired(idx, fname):
    scs = all_scenarios()
    x = np.arange(len(scs))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, m, yl in ((axes[0], "throughput_mbps", "Adaptive - Fixed throughput (Mbps)"), (axes[1], "mean_delay_ms", "Adaptive - Fixed mean delay (ms)")):
        ms, hs = [], []
        for p, s, t in scs:
            mean, sd, half, n = paired(idx, scenario_id(p, s, t), ADAPT, FIXED, m)
            ms.append(mean)
            hs.append(half)
        ax.errorbar(x, ms, yerr=hs, fmt="o", capsize=4, color="#d62728")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([scen_label(*sc) for sc in scs], fontsize=7)
    fig.suptitle(f"Paired difference (same seed): Adaptive(th0={BASE_TH}us, k={BASE_K}us) - Fixed(th={BASE_TH}us), 95% CI, n=30", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


def fig_switches(summ, idx, fname):
    scs = all_scenarios()
    S = {(r["scenario"], r["variant"]): r for r in summ}
    x = np.arange(len(scs))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    w = 0.2
    for v, off, c in ((FIXED, -1.5, "#1f77b4"), (ADAPT, 0.5, "#d62728")):
        axes[0].bar(x + off * w, [S[(scenario_id(*sc), v)]["n_switch_mean"] for sc in scs], w, yerr=[S[(scenario_id(*sc), v)]["n_switch_ci95"] for sc in scs],
                    capsize=2, color=c, alpha=0.45, label=f"{METHOD_LABEL[v]}: all switches")
        axes[0].bar(x + (off + 1) * w, [S[(scenario_id(*sc), v)]["n_switch_no_delivery_mean"] for sc in scs], w,
                    yerr=[S[(scenario_id(*sc), v)]["n_switch_no_delivery_ci95"] for sc in scs], capsize=2, color=c, label=f"{METHOD_LABEL[v]}: switches w/o any delivery")
    axes[0].set_ylabel("Switches per run (10 s measurement window)")
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="y", alpha=0.3)
    # 無效切換比例：跨種子平均 (種子內比值，忽略無切換的種子)
    for v, c, off in ((FIXED, "#1f77b4", -0.1), (ADAPT, "#d62728", 0.1)):
        ms = [S[(scenario_id(*sc), v)]["wasted_switch_frac_mean"] for sc in scs]
        hs = [S[(scenario_id(*sc), v)]["wasted_switch_frac_ci95"] for sc in scs]
        axes[1].errorbar(x + off, ms, yerr=hs, fmt="o", capsize=3, color=c, label=METHOD_LABEL[v])
    axes[1].set_ylabel("Fraction of switches with zero delivery")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([scen_label(*sc) for sc in scs], fontsize=7)
    fig.suptitle("NPCA switching activity (mean, 95% CI over 30 seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


def fig_obss(summ, fname):
    scs = all_scenarios()
    S = {(r["scenario"], r["variant"]): r for r in summ}
    x = np.arange(len(scs))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    w = 0.26
    for ax, m, yl in ((axes[0], "obss_s_tput_mbps", "Secondary-channel OBSS throughput (Mbps)"), (axes[1], "obss_p_tput_mbps", "Primary-channel OBSS throughput (Mbps)")):
        for k, v in enumerate([LEGACY, FIXED, ADAPT]):
            ax.bar(x + (k - 1) * w, [S[(scenario_id(*sc), v)][f"{m}_mean"] for sc in scs], w, yerr=[S[(scenario_id(*sc), v)][f"{m}_ci95"] for sc in scs],
                   capsize=2, color=COLORS[v], label=METHOD_LABEL[v])
        ax.set_ylabel(yl)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([scen_label(*sc) for sc in scs], fontsize=7)
    fig.suptitle("Neighbouring-BSS impact (mean, 95% CI over 30 seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


def heatmap(ax, mat, rows, cols, title, fmt, sig=None, cmap="RdBu_r"):
    mat = np.array(mat, dtype=float)
    lim = np.nanmax(np.abs(mat)) or 1.0
    im = ax.imshow(mat, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                star = "*" if sig is not None and sig[i][j] else ""
                ax.text(j, i, fmt.format(mat[i, j]) + star, ha="center", va="center", fontsize=7)
    ax.set_title(title, fontsize=9)
    return im


def fig_sensitivity(idx, thetas, fname):
    scs = all_scenarios()
    rows = [f"P-{p} S-{s} T-{t}" for p, s, t in scs]
    cols = [f"th0={th}" for th in thetas]
    mats = {}
    for key, a_fn, b_fn, m in (
        ("A_tp", lambda th: f"adaptive_th{th}_k{BASE_K}", lambda th: f"fixed_th{th}", "throughput_mbps"),
        ("A_dl", lambda th: f"adaptive_th{th}_k{BASE_K}", lambda th: f"fixed_th{th}", "mean_delay_ms"),
        ("F_tp", lambda th: f"fixed_th{th}", lambda th: LEGACY, "throughput_mbps"),
        ("F_dl", lambda th: f"fixed_th{th}", lambda th: LEGACY, "mean_delay_ms"),
    ):
        mat, sig = [], []
        for p, s, t in scs:
            sc = scenario_id(p, s, t)
            mr, sr = [], []
            for th in thetas:
                mean, sd, half, n = paired(idx, sc, a_fn(th), b_fn(th), m)
                mr.append(mean)
                sr.append(not math.isnan(half) and (mean - half > 0 or mean + half < 0))
            mat.append(mr)
            sig.append(sr)
        mats[key] = (mat, sig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    heatmap(axes[0][0], *[mats["A_tp"][0]], rows, cols, "Adaptive - Fixed (same th0): throughput, Mbps  (* = 95% CI excludes 0)", "{:+.2f}", mats["A_tp"][1])
    heatmap(axes[0][1], mats["A_dl"][0], rows, cols, "Adaptive - Fixed (same th0): mean delay, ms  (blue = adaptive lower)", "{:+.1f}", mats["A_dl"][1])
    heatmap(axes[1][0], mats["F_tp"][0], rows, cols, "Fixed(th0) - Legacy: throughput, Mbps", "{:+.2f}", mats["F_tp"][1])
    heatmap(axes[1][1], mats["F_dl"][0], rows, cols, "Fixed(th0) - Legacy: mean delay, ms", "{:+.1f}", mats["F_dl"][1])
    fig.suptitle(f"Fixed-threshold sensitivity (adaptive k={BASE_K}us, 30 paired seeds per cell)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 主程式
def fmt_pm(m, h, d=2):
    return f"{m:.{d}f} ± {h:.{d}f}" if not math.isnan(h) else f"{m:.{d}f}"


def main():
    main_rows = load(os.path.join(OUT, "raw_per_seed.csv"))
    sens_path = os.path.join(OUT, "raw_sensitivity.csv")
    sens_rows = load(sens_path) if os.path.exists(sens_path) else []
    idx = index(main_rows + sens_rows)
    scs = all_scenarios()

    summ = summarize(index(main_rows), [LEGACY, FIXED, ADAPT])
    write_csv(os.path.join(OUT, "summary_by_scenario.csv"), summ)
    pairs = [("adaptive_minus_fixed", ADAPT, FIXED), ("fixed_minus_legacy", FIXED, LEGACY), ("adaptive_minus_legacy", ADAPT, LEGACY)]
    pt = pair_table(index(main_rows), pairs)
    write_csv(os.path.join(OUT, "paired_differences.csv"), pt)

    panel_grid("Target BSS throughput", "Throughput (Mbps)", "fig1_throughput.png", summ, "throughput_mbps")
    panel_grid("Target BSS average delay (successfully delivered packets)", "Average delay (ms)", "fig2_delay.png", summ, "mean_delay_ms")
    fig_paired(index(main_rows), "fig3_paired_adaptive_minus_fixed.png")
    fig_switches(summ, index(main_rows), "fig4_switching.png")
    fig_obss(summ, "fig5_neighbour_obss.png")

    if sens_rows:
        thetas = sorted({int(r["theta0_us"]) for r in sens_rows} | {BASE_TH})
        sens_pairs = []
        for th in thetas:
            sens_pairs.append((f"adaptive_th{th}_minus_fixed_th{th}", f"adaptive_th{th}_k{BASE_K}", f"fixed_th{th}"))
            sens_pairs.append((f"fixed_th{th}_minus_legacy", f"fixed_th{th}", LEGACY))
        ks = sorted({int(r["k_us"]) for r in sens_rows if r["method"] == "adaptive" and int(r["theta0_us"]) == BASE_TH} | {BASE_K})
        for k in ks:
            sens_pairs.append((f"adaptive_th{BASE_TH}_k{k}_minus_fixed_th{BASE_TH}", f"adaptive_th{BASE_TH}_k{k}", f"fixed_th{BASE_TH}"))
        spt = pair_table(idx, sens_pairs)
        write_csv(os.path.join(OUT, "sensitivity_paired.csv"), spt)
        all_vars = sorted({r["variant"] for r in main_rows + sens_rows})
        write_csv(os.path.join(OUT, "sensitivity_summary.csv"), summarize(idx, all_vars))
        fig_sensitivity(idx, thetas, "fig6_threshold_sensitivity.png")

    # ---- tables.md ----
    L = []
    S = {(r["scenario"], r["variant"]): r for r in summ}
    L.append("### 各情境平均值 ± 95% CI (30 種子)\n")
    L.append("| 情境 | 方法 | 實測 OBSS_P 忙碌比 | 吞吐量 Mbps | 平均延遲 ms | 丟棄率 | 期末未送達 | 期末積壓平均年齡 ms | 切換次數 | 無送達切換 | OBSS_S Mbps | OBSS_P Mbps |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p, s, t in scs:
        sc = scenario_id(p, s, t)
        for v, nm in ((LEGACY, "Legacy"), (FIXED, "固定"), (ADAPT, "自適應")):
            r = S[(sc, v)]
            L.append(f"| {sc} | {nm} | {r['obss_p_busy_ratio_mean']:.3f} | {fmt_pm(r['throughput_mbps_mean'], r['throughput_mbps_ci95'])} | "
                     f"{fmt_pm(r['mean_delay_ms_mean'], r['mean_delay_ms_ci95'], 1)} | {r['drop_rate_mean']:.3f} | {r['backlog_end_mean']:.0f} | {r['backlog_age_mean_ms_mean']:.0f} | "
                     f"{r['n_switch_mean']:.0f} | {r['n_switch_no_delivery_mean']:.1f} | {r['obss_s_tput_mbps_mean']:.2f} | {r['obss_p_tput_mbps_mean']:.2f} |")
    L.append("\n### 成對差值：自適應 − 固定 (同種子，平均 [95% CI]，* 表示 CI 不含 0)\n")
    L.append("| 情境 | Δ吞吐量 Mbps | Δ平均延遲 ms | Δ切換次數 | Δ有效切換次數 | Δ無效切換次數 | Δ無效切換比例 | ΔOBSS_S Mbps | ΔOBSS_S 延遲 ms | ΔOBSS_P Mbps |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    PT = {(r["scenario"], r["comparison"], r["metric"]): r for r in pt}
    for p, s, t in scs:
        sc = scenario_id(p, s, t)
        cells = []
        for m, d in (("throughput_mbps", 2), ("mean_delay_ms", 1), ("n_switch", 1), ("n_switch_useful", 1), ("n_switch_no_delivery", 1), ("wasted_switch_frac", 3), ("obss_s_tput_mbps", 3), ("obss_s_delay_ms", 3), ("obss_p_tput_mbps", 3)):
            r = PT[(sc, "adaptive_minus_fixed", m)]
            cells.append(f"{r['mean_diff']:+.{d}f} [{r['ci_low']:+.{d}f}, {r['ci_high']:+.{d}f}]{'*' if r['ci_excludes_zero'] else ''}")
        L.append(f"| {sc} | " + " | ".join(cells) + " |")
    L.append("\n### 成對差值：固定 − Legacy 與 自適應 − Legacy (吞吐量 Mbps / 延遲 ms)\n")
    L.append("| 情境 | 固定−Legacy 吞吐量 | 固定−Legacy 延遲 | 自適應−Legacy 吞吐量 | 自適應−Legacy 延遲 |")
    L.append("|---|---|---|---|---|")
    for p, s, t in scs:
        sc = scenario_id(p, s, t)
        cells = []
        for comp in ("fixed_minus_legacy", "adaptive_minus_legacy"):
            for m, d in (("throughput_mbps", 2), ("mean_delay_ms", 1)):
                r = PT[(sc, comp, m)]
                cells.append(f"{r['mean_diff']:+.{d}f} [{r['ci_low']:+.{d}f}, {r['ci_high']:+.{d}f}]{'*' if r['ci_excludes_zero'] else ''}")
        L.append(f"| {sc} | " + " | ".join(cells) + " |")
    if sens_rows:
        SP = {(r["scenario"], r["comparison"], r["metric"]): r for r in spt}
        L.append("\n### 敏感度：自適應(th0, k=2000) − 固定(th0)，吞吐量 Mbps (* = CI 不含 0)\n")
        L.append("| 情境 | " + " | ".join(f"th0={th}" for th in thetas) + " |")
        L.append("|---|" + "---|" * len(thetas))
        for p, s, t in scs:
            sc = scenario_id(p, s, t)
            cells = []
            for th in thetas:
                r = SP[(sc, f"adaptive_th{th}_minus_fixed_th{th}", "throughput_mbps")]
                cells.append(f"{r['mean_diff']:+.2f}{'*' if r['ci_excludes_zero'] else ''}")
            L.append(f"| {sc} | " + " | ".join(cells) + " |")
        L.append("\n### 敏感度：自適應(th0, k=2000) − 固定(th0)，平均延遲 ms (* = CI 不含 0)\n")
        L.append("| 情境 | " + " | ".join(f"th0={th}" for th in thetas) + " |")
        L.append("|---|" + "---|" * len(thetas))
        for p, s, t in scs:
            sc = scenario_id(p, s, t)
            cells = []
            for th in thetas:
                r = SP[(sc, f"adaptive_th{th}_minus_fixed_th{th}", "mean_delay_ms")]
                cells.append(f"{r['mean_diff']:+.1f}{'*' if r['ci_excludes_zero'] else ''}")
            L.append(f"| {sc} | " + " | ".join(cells) + " |")
        L.append("\n### 敏感度：固定(th0) − Legacy，吞吐量 Mbps (* = CI 不含 0)\n")
        L.append("| 情境 | " + " | ".join(f"th0={th}" for th in thetas) + " |")
        L.append("|---|" + "---|" * len(thetas))
        for p, s, t in scs:
            sc = scenario_id(p, s, t)
            cells = []
            for th in thetas:
                r = SP[(sc, f"fixed_th{th}_minus_legacy", "throughput_mbps")]
                cells.append(f"{r['mean_diff']:+.2f}{'*' if r['ci_excludes_zero'] else ''}")
            L.append(f"| {sc} | " + " | ".join(cells) + " |")
        L.append("\n### 敏感度：自適應係數 k (th0=500)，相對固定(th0=500) 的吞吐量差 Mbps / 延遲差 ms\n")
        L.append("| 情境 | " + " | ".join(f"k={k}" for k in ks) + " |")
        L.append("|---|" + "---|" * len(ks))
        for p, s, t in scs:
            sc = scenario_id(p, s, t)
            cells = []
            for k in ks:
                a = SP[(sc, f"adaptive_th{BASE_TH}_k{k}_minus_fixed_th{BASE_TH}", "throughput_mbps")]
                b = SP[(sc, f"adaptive_th{BASE_TH}_k{k}_minus_fixed_th{BASE_TH}", "mean_delay_ms")]
                cells.append(f"{a['mean_diff']:+.2f}{'*' if a['ci_excludes_zero'] else ''} / {b['mean_diff']:+.1f}{'*' if b['ci_excludes_zero'] else ''}")
            L.append(f"| {sc} | " + " | ".join(cells) + " |")
    with open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
