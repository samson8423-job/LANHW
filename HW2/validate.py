"""驗證腳本：先用短而可追蹤的案例確認模型行為，再跑正式實驗。

執行：.\.venv\Scripts\python.exe validate.py
結果同時寫入 results/validation_report.txt；任一項失敗則以非 0 結束。
"""

import os
import random
import re
import sys

from npca_sim.config import DIFS, EXCH, T_BACK, T_OUT, US, exchange_ns
from npca_sim.sim import SimConfig, Simulator
from npca_sim.traffic import make_arrivals, make_backoff_rngs, sta_rate_pps

LINES = []
FAILS = []


def say(msg=""):
    print(msg)
    LINES.append(msg)


def check(cond, desc):
    say(f"    [{'PASS' if cond else 'FAIL'}] {desc}")
    if not cond:
        FAILS.append(desc)


class ScriptRng:
    """依序回傳預先指定的退避值；用完後改用固定種子亂數。"""

    def __init__(self, seq, fallback_seed=0):
        self.seq = list(seq)
        self.fb = random.Random(fallback_seed)
        self.drawn = []

    def randint(self, a, b):
        v = self.seq.pop(0) if self.seq else self.fb.randint(a, b)
        assert a <= v <= b, f"scripted backoff {v} 超出 [{a},{b}]"
        self.drawn.append(v)
        return v


def build(seed, method, p_pps, s_pps, target_mbps, t_meas_s=1.0, t_warm_s=0.0, n_sta=5, **kw):
    horizon = int((t_warm_s + t_meas_s) * 1e9) + 10_000_000
    cfg = SimConfig(method=method, t_warm_ns=int(t_warm_s * 1e9), t_meas_ns=int(t_meas_s * 1e9), n_sta=n_sta, **kw)
    arr = make_arrivals(seed, n_sta, sta_rate_pps(target_mbps, n_sta), p_pps, s_pps, horizon)
    return Simulator(cfg, arr, make_backoff_rngs(seed, n_sta))


def sig(d):
    return repr(sorted(d.items()))


def trace_sig(sim):
    return [(t, ev, nd, ch) for (t, ev, nd, ch, _info) in sim.log]


# ---------------------------------------------------------------------------
def test1():
    say("[1] 主通道沒有 OBSS 時，三組主通道存取邏輯一致且不觸發 NPCA")
    ok_all = True
    n_checked = 0
    for seed in (1, 2, 3, 4, 5):
        for target in (6.0, 30.0):
            outs = {}
            for m in ("legacy", "fixed", "adaptive"):
                sim = build(seed, m, 0.0, 600.0, target, t_meas_s=1.0, trace=True)
                outs[m] = (sim.run(), trace_sig(sim), sim)
            same = all(sig(outs[m][0]) == sig(outs["legacy"][0]) and outs[m][1] == outs["legacy"][1] for m in outs)
            no_sw = all(outs[m][0]["n_switch"] == 0 and outs[m][0]["n_eligible"] == 0 for m in outs)
            ok_all &= same and no_sw
            n_checked += 1
    check(ok_all, f"{n_checked} 組 (seed x 目標負載)：Legacy/固定/自適應 的結果欄位與完整事件序列逐項相同，n_eligible = n_switch = 0")
    sim = build(1, "legacy", 0.0, 600.0, 30.0, t_meas_s=1.0)
    r = sim.run()
    check(r["throughput_mbps"] > 0 and r["obss_s_tput_mbps"] > 0, f"非空測試：目標 {r['throughput_mbps']:.2f} Mbps、副通道 OBSS {r['obss_s_tput_mbps']:.2f} Mbps")


def test2():
    say("[2] 自適應係數 = 0 時，同一種子下與固定門檻 NPCA 結果一致")
    ok = True
    n_sw_total = 0
    for seed in (1, 2, 3, 4, 5):
        for s_pps in (150.0, 700.0):
            sf = build(seed, "fixed", 500.0, s_pps, 20.0, t_meas_s=1.0, trace=True, theta0_ns=500 * US)
            sa = build(seed, "adaptive", 500.0, s_pps, 20.0, t_meas_s=1.0, trace=True, theta0_ns=500 * US, k_ns=0.0)
            rf, ra = sf.run(), sa.run()
            ok &= sig(rf) == sig(ra) and trace_sig(sf) == trace_sig(sa)
            n_sw_total += rf["n_switch"]
    check(ok, "10 組 (seed x 副通道負載)：結果欄位與完整事件序列逐項相同")
    check(n_sw_total > 50, f"非空測試：這些案例共發生 {n_sw_total} 次切換")
    sf = build(1, "fixed", 500.0, 700.0, 20.0, t_meas_s=1.0, theta0_ns=500 * US).run()
    sa = build(1, "adaptive", 500.0, 700.0, 20.0, t_meas_s=1.0, theta0_ns=500 * US, k_ns=2000.0 * US).run()
    check(sig(sf) != sig(sa), f"對照：k = 2000 us 時結果確實不同 (切換 {sf['n_switch']} vs {sa['n_switch']})")


def crafted(n_sta, sta_arr, obss_p_arr, obss_s_arr, rng_seqs, method="fixed", **kw):
    """sta_arr: [(t_ns, n_mpdu)...] 或每 STA 各自的 list；其餘同。"""
    arrivals = []
    for lst in sta_arr:
        arrivals.append(([t for t, _ in lst], [n for _, n in lst]))
    arrivals.append(([t for t, _ in obss_p_arr], [n for _, n in obss_p_arr]))
    arrivals.append(([t for t, _ in obss_s_arr], [n for _, n in obss_s_arr]))
    rngs = [ScriptRng(s, fallback_seed=100 + i) for i, s in enumerate(rng_seqs)]
    cfg = SimConfig(method=method, t_meas_ns=20_000_000, n_sta=n_sta, trace=True, **kw)
    return Simulator(cfg, arrivals, rngs)


def ev(sim, name=None, node=None, ch=None):
    return [e for e in sim.log if (name is None or e[1] == name) and (node is None or e[2] == node) and (ch is None or e[3] == ch)]


def test3():
    say("[3] 副通道競爭失敗，或窗口只剩切回時間時，NPCA 不計入未完成傳輸、不越過切回期限")
    # --- 3a：副通道被 OBSS_S 持續占用，目標 STA 競爭不到 ---
    obss_s = [(i * 1000, 24) for i in range(6)]
    sim = crafted(1, [[(20 * US, 1)]], [(0, 24)], obss_s,
                  [[15], [0] * 3, [0] * 20], theta0_ns=500 * US)
    r = sim.run()
    sw = ev(sim, "NPCA_SWITCH_OUT")
    back = ev(sim, "NPCA_SWITCH_BACK")
    t_end_obss = DIFS + EXCH[24]
    ack_s = ev(sim, "ACK_OK", "STA0", 1)
    ack_p = ev(sim, "ACK_OK", "STA0", 0)
    say("  3a 副通道被 OBSS_S 持續占用 (OBSS_S 退避恆為 0)")
    check(len(sw) == 1 and sw[0][0] == DIFS + T_OUT, f"在 OBSS_P 傳輸開始後 T_OUT={T_OUT // US} us 切換 (t={sw[0][0] / US:.1f} us)")
    check(r["n_switch"] == 1 and r["n_switch_no_tx"] == 1 and r["npca_tx_attempts"] == 0, "切換 1 次，但副通道沒有任何起送 (n_switch_no_tx = 1)")
    check(len(ack_s) == 0 and r["npca_delivered"] == 0, "副通道上沒有 ACK_OK，npca_delivered = 0")
    check(len(back) == 1 and back[0][0] == t_end_obss, f"在 OBSS_P 傳輸結束時刻返回主通道 (t={back[0][0] / US:.1f} us = {t_end_obss / US:.1f} us)")
    check(len(ack_p) == 1 and ack_p[0][0] > t_end_obss and r["delivered_total"] == 1, f"封包在返回主通道後才於主通道送達 (t={ack_p[0][0] / US:.1f} us)")
    # --- 3b：窗口只剩切回時間 ---
    sim = crafted(1, [[(20 * US, 1)]], [(0, 1)], [], [[15], [0], []], theta0_ns=0)
    r = sim.run()
    sw, back = ev(sim, "NPCA_SWITCH_OUT"), ev(sim, "NPCA_SWITCH_BACK")
    t_end_obss = DIFS + EXCH[1]
    win = t_end_obss - T_BACK - sw[0][0] if sw else None
    say("  3b 窗口只剩切回時間 (門檻設 0，OBSS_P 只傳 1 個 MPDU)")
    check(len(sw) == 1, f"門檻 0 時仍觸發切換；切換後到「必須開始切回」只剩 {win / US:.1f} us，小於一次完整交換 {EXCH[1] / US:.1f} us")
    check(r["n_switch_no_tx"] == 1 and len(ev(sim, "TX_START", "STA0", 1)) == 0, "副通道沒有任何起送")
    check(r["npca_delivered"] == 0 and len(ev(sim, "ACK_OK", "STA0", 1)) == 0, "沒有把未完成傳輸計入 NPCA 送達 / 吞吐量")
    check(back[0][0] == t_end_obss, f"返回主通道時刻 = OBSS_P 傳輸結束時刻 ({back[0][0] / US:.1f} us)")
    # --- 3c：副通道與 OBSS_S 同槽碰撞 ---
    sim = crafted(1, [[(20 * US, 1)]], [(0, 24)], [(150 * US, 1)],
                  [[10, 3], [0], [12, 20]], theta0_ns=500 * US)
    r = sim.run()
    txs = ev(sim, "TX_START", None, 1)
    coll = ev(sim, "COLLISION", None, 1)
    ok_pair = len(txs) >= 2 and txs[0][0] == txs[1][0]
    say("  3c 副通道與 OBSS_S 同槽起送 (人工指定退避值使其同槽)")
    check(ok_pair and len(coll) == 1, f"副通道發生碰撞 (t={coll[0][0] / US:.1f} us, 參與者 {coll[0][2]})" if coll else "副通道發生碰撞")
    check(r["npca_collisions"] == 1 and r["npca_tx_attempts"] == 2 and r["npca_delivered"] == 1,
          "碰撞那次不計送達；之後重傳成功 1 次 (tx=2, coll=1, delivered=1)")
    check(r["delivered_total"] == 1 and abs(r["throughput_mbps"] - 12000 / 0.02 / 1e6) < 1e-9, "吞吐量只含成功那 1 個封包 (12000 bit)")
    to = ev(sim, "ACK_TIMEOUT", "STA0", 1)
    check(len(to) == 1 and "cw=31" in to[0][4], "碰撞後 CW 由 15 加倍為 31 (沿用同一組重傳規則)")
    # --- 3d：隨機重載下檢查所有副通道起送都在切回期限前結束 ---
    n_tx = 0
    min_slack = None
    ok = True
    for seed in (1, 2, 3):
        for m in ("fixed", "adaptive"):
            sim = build(seed, m, 700.0, 500.0, 30.0, t_meas_s=1.0, theta0_ns=0)  # 門檻 0：每次 OBSS 都切，壓力測試
            sim.run()
            for (st, en, win_end, t_end, succ) in sim.s_tx_log:
                n_tx += 1
                slack = win_end - en
                min_slack = slack if min_slack is None else min(min_slack, slack)
                ok &= en <= win_end <= t_end - T_BACK
    check(ok and n_tx > 200, f"3d 隨機壓力測試 (門檻 0)：{n_tx} 次副通道起送皆在 (OBSS 結束 - T_BACK) 之前完成，最小餘裕 {min_slack / US:.1f} us")


def test4():
    say("[4] 同槽起送會碰撞；封包數對帳 (到達 = 送達 + 丟棄 + 期末未完成，每個封包只計一次)")
    sim = crafted(2, [[(100 * US, 1)], [(100 * US, 1)]], [], [], [[3, 1], [3, 5], [], []], method="legacy")
    r = sim.run()
    col = ev(sim, "COLLISION")
    say("  4a 兩台 STA 人工指定相同退避值 (同槽起送)")
    check(len(col) == 1 and col[0][2] == "STA0+STA1", f"同槽起送 -> COLLISION ({col[0][2]} @ {col[0][0] / US:.1f} us)")
    check(r["collisions"] == 2 and r["delivered_total"] == 2 and r["arrivals_total"] == 2, "雙方各失敗 1 次後都成功送達 (collisions=2, delivered=2)")
    say("  4b 隨機重載對帳 (由空佇列開跑；retry_limit=2、佇列上限 8，強迫出現兩種丟棄)")
    ok = True
    tot = dict(arr=0, dlv=0, dr=0, do=0, bl=0)
    for seed in (1, 2, 3):
        for m in ("legacy", "fixed", "adaptive"):
            sim = build(seed, m, 500.0, 300.0, 40.0, t_meas_s=1.0, retry_limit=2, qlimit_target=8, qlimit_obss=8, track_ids=True)
            sim.run()
            for nd in sim.nodes:
                n_arr = nd.arrivals
                ids = nd.delivered_ids + nd.dropped_ids + nd.overflow_ids + [p[2] for p in nd.queue]
                ok &= (n_arr == nd.delivered + nd.drop_retry + nd.drop_overflow + len(nd.queue))
                ok &= len(ids) == len(set(ids)) == n_arr and set(ids) == set(range(n_arr))
                if nd.is_target:
                    tot["arr"] += n_arr
                    tot["dlv"] += nd.delivered
                    tot["dr"] += nd.drop_retry
                    tot["do"] += nd.drop_overflow
                    tot["bl"] += len(nd.queue)
    check(ok, "9 次模擬 x 每節點：到達數 = 送達 + 重試丟棄 + 佇列溢位丟棄 + 期末佇列內；ID 集合互斥且恰好涵蓋所有到達封包")
    check(tot["dr"] > 0 and tot["do"] > 0, f"非空測試：目標 BSS 合計到達 {tot['arr']}、送達 {tot['dlv']}、重試丟棄 {tot['dr']}、溢位丟棄 {tot['do']}、期末未完成 {tot['bl']}")


def test5():
    say("[5] 自適應門檻的副通道忙碌比例：只用判斷時刻以前的樣本，且排除目標 BSS 自己的傳輸")
    t_cut = 600 * 1_000_000  # 0.6 s
    sim_a = build(7, "adaptive", 400.0, 600.0, 30.0, t_meas_s=1.0, trace=True)
    arr_b = [list(x) for x in make_arrivals(7, 5, sta_rate_pps(30.0, 5), 400.0, 600.0, 1_010_000_000)]
    k = sum(1 for t in arr_b[6][0] if t <= t_cut)
    arr_b[6] = (arr_b[6][0][:k], arr_b[6][1][:k])  # 刪除 t_cut 之後的副通道 OBSS 到達
    cfg = SimConfig(method="adaptive", t_meas_ns=1_000_000_000, n_sta=5, trace=True)
    sim_b = Simulator(cfg, arr_b, make_backoff_rngs(7, 5))
    sim_a.run()
    sim_b.run()
    ea = [e for e in sim_a.log if e[0] <= t_cut]
    eb = [e for e in sim_b.log if e[0] <= t_cut]
    check(ea == eb and len(ea) > 1000, f"改變 t=0.6 s 之後的副通道 OBSS 到達，t <= 0.6 s 的 {len(ea)} 個事件 (含各次門檻/忙碌比例) 完全不變 -> 無偷看未來")
    # 以 trace 獨立重算：只用 OBSS_S 的 TX_START，視窗 [t - 20 ms, t]
    eps = [(e[0], float(re.search(r"end_us=([\d.]+)", e[4]).group(1)) * US) for e in sim_a.log if e[1] == "TX_START" and e[2] == "OBSS_S"]
    worst = 0.0
    n_cmp = 0
    own_vs_all_gap = 0.0
    s_all = [(e[0], float(re.search(r"end_us=([\d.]+)", e[4]).group(1)) * US) for e in sim_a.log if e[1] == "TX_START" and e[3] == 1]
    for e in sim_a.log:
        if e[1] in ("NPCA_SWITCH_OUT", "NPCA_REJECT") and e[0] >= 20_000_000:
            t = e[0]
            a = t - 20_000_000
            busy = sum(max(0.0, min(en, t) - max(st, a)) for st, en in eps if st < t)
            busy_all = sum(max(0.0, min(en, t) - max(st, a)) for st, en in s_all if st < t)
            ratio = float(re.search(r"ratio=([\d.]+)", e[4]).group(1))
            worst = max(worst, abs(ratio - busy / 20_000_000))
            own_vs_all_gap = max(own_vs_all_gap, abs(busy_all - busy) / 20_000_000)
            n_cmp += 1
    check(n_cmp > 100 and worst < 0.0011, f"{n_cmp} 次判斷：模擬器記錄的忙碌比例與由 OBSS_S 傳輸紀錄獨立重算的值最大差 {worst:.4f}")
    check(own_vs_all_gap > 0.02, f"對照：若把目標 BSS 在副通道的傳輸也算進去，忙碌比例最多會差 {own_vs_all_gap:.3f}，故確實已排除")


def test6():
    say("[6] 改善變體：佇列門檻的 θ_hi = θ_0 時與固定門檻一致、無未來資訊；oracle 可行性判斷正確")
    ok = True
    n_sw = 0
    for seed in (1, 2, 3):
        for target in (6.0, 30.0):
            sf = build(seed, "fixed", 400.0, 500.0, target, t_meas_s=1.0, trace=True, theta0_ns=500 * US)
            sq = build(seed, "queue", 400.0, 500.0, target, t_meas_s=1.0, trace=True, theta0_ns=500 * US, theta_hi_ns=500 * US)
            rf, rq = sf.run(), sq.run()
            ok &= sig(rf) == sig(rq) and trace_sig(sf) == trace_sig(sq)
            n_sw += rf["n_switch"]
    check(ok and n_sw > 50, f"6a 6 組 (seed x 目標負載)：queue 的 θ_hi = θ_0 時與固定門檻結果與事件序列逐項相同 (共 {n_sw} 次切換)")
    t_cut = 600 * 1_000_000
    cfg_kw = dict(method="queue", theta0_ns=0, theta_hi_ns=500 * US, t_meas_ns=1_000_000_000, n_sta=5, trace=True)

    def arrivals():
        return [list(x) for x in make_arrivals(7, 5, sta_rate_pps(30.0, 5), 400.0, 600.0, 1_010_000_000)]
    arr_b = arrivals()
    k = sum(1 for t in arr_b[6][0] if t <= t_cut)
    arr_b[6] = (arr_b[6][0][:k], arr_b[6][1][:k])  # 刪除 t_cut 之後的副通道 OBSS 到達
    sa = Simulator(SimConfig(**cfg_kw), arrivals(), make_backoff_rngs(7, 5))
    sb = Simulator(SimConfig(**cfg_kw), arr_b, make_backoff_rngs(7, 5))
    sa.run()
    sb.run()
    ea = [e for e in sa.log if e[0] <= t_cut]
    eb = [e for e in sb.log if e[0] <= t_cut]
    check(ea == eb and len(ea) > 1000, f"6b queue：改變 t = 0.6 s 之後的副通道到達，t <= 0.6 s 的 {len(ea)} 個事件完全不變")
    # oracle 可行性：副通道 OBSS 傳輸 (100,200) 與 (300,400) us，判斷 [t+DIFS, win_end] 能否容納 need
    o = build(1, "oracle", 0.0, 0.0, 30.0, t_meas_s=1.0, oracle_s_busy=((100 * US, 200 * US), (300 * US, 400 * US)))
    cases = [(0, 1000, 50, True), (0, 1000, 80, True), (0, 600, 500, False), (0, 120, 50, True), (0, 80, 50, False)]
    got = [o._oracle_feasible(t * US, w * US, n * US) for t, w, n, _ in cases]
    check(got == [c[3] for c in cases], "6c oracle 可行性判斷在 5 個人工案例上正確 (含窗口尾端不足、間隙容納、跨越忙碌區間)")


def main():
    say("NPCA 模擬器驗證報告")
    say("=" * 60)
    for t in (test1, test2, test3, test4, test5, test6):
        t()
        say()
    say("=" * 60)
    say("全部通過" if not FAILS else f"失敗 {len(FAILS)} 項：" + "; ".join(FAILS))
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "validation_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
