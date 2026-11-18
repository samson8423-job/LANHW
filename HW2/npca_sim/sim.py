"""封包層級、事件驅動的 NPCA 簡化 MAC 模擬器 (Legacy / 固定門檻 NPCA / 自適應門檻 NPCA)。

通道 0 = 主通道 P，通道 1 = 副通道 S，皆為單一 20 MHz。
節點：target STA x n_sta (上行，往 AP)，OBSS_P (主通道鄰近 BSS 傳送端)，OBSS_S (副通道鄰近 BSS 傳送端)。
規則細節見 README「模型假設」。
"""

import bisect
import heapq
from collections import deque
from dataclasses import dataclass

from .config import (
    ACK_TIME, CW_MAX, CW_MIN, DIFS, EXCH, PAYLOAD_BITS, RETRY_LIMIT, SIFS, SLOT,
    T_BACK, T_OUT, US,
)

INF = 1 << 62

# 事件優先序 (同一時刻)：先結算各節點的傳輸結果，再結束通道忙碌，最後做 NPCA 判斷
P_DONE, P_BEND, P_DEC = 0, 1, 2


@dataclass
class SimConfig:
    method: str = "legacy"            # legacy | fixed | adaptive | queue | oracle (診斷用)
    theta0_ns: int = 500 * US         # 固定門檻 / 自適應基礎門檻 / queue 在積壓足夠時的門檻
    k_ns: float = 2000.0 * US         # 自適應係數：門檻 = theta0 + k * 副通道近期忙碌比例
    busy_window_ns: int = 20_000 * US  # 近期忙碌比例的回看視窗
    theta_hi_ns: int = 500 * US       # queue：積壓不足時改用的門檻
    qa_min_backlog: int = 20          # queue：目標 BSS 待傳封包總數 >= 此值時使用 theta0
    oracle_s_busy: tuple = ()         # oracle 專用：副通道 OBSS 傳輸區間 (start, end)，取自前一趟模擬 (含未來資訊)
    t_warm_ns: int = 0
    t_meas_ns: int = 1_000_000_000
    n_sta: int = 5
    qlimit_target: int = 100
    qlimit_obss: int = 50
    retry_limit: int = RETRY_LIMIT
    trace: bool = False
    trace_limit: int = 100_000
    track_ids: bool = False
    block_ns: int = 0                 # >0 時另外輸出每個時間區塊的目標 BSS 送達統計 (試跑用)


class Node:
    __slots__ = (
        "idx", "name", "group", "is_target", "ch", "arr_t", "arr_n", "ai", "next_arr",
        "queue", "qlimit", "rng", "cw", "retry", "b", "active", "count_start", "in_tx",
        "arrivals", "delivered", "drop_retry", "drop_overflow", "tx_attempts", "collisions",
        "w_arrivals", "w_delivered", "w_bits", "w_delay_sum", "w_drop_retry", "w_drop_overflow",
        "w_tx_attempts", "w_collisions",
        "ep_starts", "ep_ends", "ep_cum", "delivered_ids", "dropped_ids", "overflow_ids",
    )

    def __init__(self, idx, name, group, home_ch, arr_t, arr_n, rng, qlimit):
        self.idx = idx
        self.name = name
        self.group = group
        self.is_target = group == "T"
        self.ch = home_ch
        self.arr_t = arr_t
        self.arr_n = arr_n
        self.ai = 0
        self.next_arr = arr_t[0] if len(arr_t) else INF
        self.queue = deque()
        self.qlimit = qlimit
        self.rng = rng
        self.cw = CW_MIN
        self.retry = 0
        self.b = None
        self.active = False
        self.count_start = 0
        self.in_tx = False
        self.arrivals = self.delivered = self.drop_retry = self.drop_overflow = 0
        self.tx_attempts = self.collisions = 0
        self.w_arrivals = self.w_delivered = self.w_bits = self.w_delay_sum = 0
        self.w_drop_retry = self.w_drop_overflow = 0
        self.w_tx_attempts = self.w_collisions = 0
        self.ep_starts, self.ep_ends, self.ep_cum = [], [], []
        self.delivered_ids, self.dropped_ids, self.overflow_ids = [], [], []

    def busy_in(self, a, b):
        """本節點自己的傳輸 (PPDU+SIFS+ACK) 在 [a, b] 內的總長度。只用已開始的紀錄，並以 b 截斷。"""
        if b <= a or not self.ep_starts:
            return 0
        i = bisect.bisect_right(self.ep_ends, a)
        j = bisect.bisect_left(self.ep_starts, b)
        if i >= j:
            return 0
        tot = self.ep_cum[j - 1] - (self.ep_cum[i - 1] if i > 0 else 0)
        tot -= max(0, a - self.ep_starts[i])
        tot -= max(0, self.ep_ends[j - 1] - b)
        return tot


class Simulator:
    def __init__(self, cfg: SimConfig, arrivals, rngs):
        """arrivals: 每個節點一組 (到達時間 ns 串列, 每封包 MPDU 數串列)。
        節點順序：target STA 0..n_sta-1、OBSS_P、OBSS_S。rngs: 每節點一個具 randint(a, b) 的退避亂數源。"""
        self.cfg = cfg
        n = cfg.n_sta
        self.nodes = []
        for i in range(n):
            self.nodes.append(Node(i, f"STA{i}", "T", 0, arrivals[i][0], arrivals[i][1], rngs[i], cfg.qlimit_target))
        self.nodes.append(Node(n, "OBSS_P", "OP", 0, arrivals[n][0], arrivals[n][1], rngs[n], cfg.qlimit_obss))
        self.nodes.append(Node(n + 1, "OBSS_S", "OS", 1, arrivals[n + 1][0], arrivals[n + 1][1], rngs[n + 1], cfg.qlimit_obss))
        self.targets = self.nodes[:n]
        self.obss_p = self.nodes[n]
        self.obss_s = self.nodes[n + 1]
        self.t_end = cfg.t_warm_ns + cfg.t_meas_ns
        self.heap = []
        self.seq = 0
        self.busy_until = [0, 0]
        self.last_end = [0, 0]
        self.session = None
        self.session_log = []
        self.s_tx_log = []  # 目標 BSS 在副通道的每次起送 (供不變量檢查)
        self.n_eligible = self.n_rej_theta = self.n_rej_nobacklog = 0
        self.theta_sum = 0.0
        self.now = 0
        self.log = [] if cfg.trace else None
        self.tracing = cfg.trace
        self.blk = {}
        self.npca_k_ns = cfg.k_ns
        self.oracle_iv = cfg.oracle_s_busy
        self.oracle_en = [b for _, b in cfg.oracle_s_busy]
        self.n_rej_oracle = self.n_sw_s_busy0 = self.n_rej_theta_s_idle = 0

    # ---------------------------------------------------------------- 工具
    def _trace(self, t, ev, node, ch, info=""):
        if self.log is not None and len(self.log) < self.cfg.trace_limit:
            self.log.append((t, ev, node, ch, info))

    def _push(self, t, prio, kind, payload):
        self.seq += 1
        heapq.heappush(self.heap, (t, prio, self.seq, kind, payload))

    def _align(self, c, ready):
        """把開始倒數時間對齊到通道 c 的 slot 格線 (以最近一次忙碌結束 + DIFS 為原點)。"""
        anchor = self.last_end[c] + DIFS
        if ready <= anchor:
            return anchor
        return anchor + ((ready - anchor + SLOT - 1) // SLOT) * SLOT

    def _tx_time(self, nd):
        if not nd.active:
            return INF
        t = nd.count_start + nd.b * SLOT
        if nd.is_target and nd.ch == 1:
            s = self.session
            if s is None or t + EXCH[nd.queue[0][1]] > s["win_end"]:
                return INF  # 窗口內放不下一次完整交換 -> 本次切換不再起送
        return t

    def _elapsed(self, nd, t):
        """nd 自 count_start 起到時刻 t 已倒數的 slot 數。NPCA 窗口放不下完整交換後即停止倒數。"""
        if nd.is_target and nd.ch == 1:
            t = min(t, self.session["win_end"] - EXCH[nd.queue[0][1]])
        return (t - nd.count_start) // SLOT if t > nd.count_start else 0

    def _in_win(self, t):
        return self.cfg.t_warm_ns <= t <= self.t_end

    def _s_busy_ratio(self, t):
        """判斷時刻 t 以前、副通道 OBSS 自己的傳輸占比 (不含目標 BSS)。理想歷史觀測假設。"""
        a = max(0, t - self.cfg.busy_window_ns)
        if t <= a:
            return 0.0
        return self.obss_s.busy_in(a, t) / (t - a)

    def _oracle_feasible(self, t, win_end, need):
        """診斷用：依前一趟的副通道 OBSS 傳輸紀錄，判斷 [t+DIFS, win_end] 內是否有閒置段可容納 need (含未來資訊，非可實作方法)。"""
        a = t + DIFS
        for st, en in self.oracle_iv[bisect.bisect_right(self.oracle_en, t):]:
            if a + need <= min(st, win_end):
                return True
            if st >= win_end:
                return False
            a = max(a, en + DIFS)
        return a + need <= win_end

    # ---------------------------------------------------------------- 事件處理
    def _arrival(self, nd, t):
        idx = nd.ai
        n_mpdu = nd.arr_n[idx]
        nd.ai += 1
        nd.next_arr = nd.arr_t[nd.ai] if nd.ai < len(nd.arr_t) else INF
        nd.arrivals += 1
        win = self._in_win(t)
        if win:
            nd.w_arrivals += 1
        if len(nd.queue) >= nd.qlimit:
            nd.drop_overflow += 1
            if win:
                nd.w_drop_overflow += 1
            if self.cfg.track_ids:
                nd.overflow_ids.append(idx)
            self.tracing and self._trace(t, "ARRIVAL_OVERFLOW", nd.name, nd.ch, f"pkt={idx}")
            return
        nd.queue.append((t, n_mpdu, idx))
        if len(nd.queue) == 1:
            nd.b = nd.rng.randint(0, nd.cw)
            c = nd.ch
            self.tracing and self._trace(t, "ARRIVAL_HOL", nd.name, c, f"pkt={idx} draw_b={nd.b} cw={nd.cw}")
            if self.busy_until[c] > t:
                nd.active = False
            else:
                ready = t
                if nd.is_target and c == 1:
                    ready = max(t, self.session["sense_ready"])
                nd.count_start = self._align(c, ready)
                nd.active = True
        else:
            self.tracing and self._trace(t, "ARRIVAL", nd.name, nd.ch, f"pkt={idx} qlen={len(nd.queue)}")

    def _start_tx(self, t):
        by_ch = {}
        for nd in self.nodes:
            if self._tx_time(nd) == t:
                by_ch.setdefault(nd.ch, []).append(nd)
        for c, starters in by_ch.items():
            for nd in self.nodes:
                if nd.ch == c and nd.active and nd not in starters:
                    nd.b = max(0, nd.b - self._elapsed(nd, t))
                    nd.active = False
                    self.tracing and self._trace(t, "FREEZE", nd.name, c, f"b_left={nd.b}")
            success = len(starters) == 1
            end_busy = 0
            for nd in starters:
                nd.active = False
                nd.in_tx = True
                nd.tx_attempts += 1
                if self._in_win(t):
                    nd.w_tx_attempts += 1
                n_mpdu = nd.queue[0][1]
                end = t + EXCH[n_mpdu]
                end_busy = max(end_busy, end)
                self._push(end, P_DONE, "DONE", (nd, success))
                if not nd.is_target:
                    nd.ep_starts.append(t)
                    nd.ep_ends.append(end)
                    nd.ep_cum.append((nd.ep_cum[-1] if nd.ep_cum else 0) + (end - t))
                self.tracing and self._trace(t, "TX_START", nd.name, c,
                            f"n_mpdu={n_mpdu} end_us={end / US:.1f} retry={nd.retry} cw={nd.cw} {'ALONE' if success else 'COLLIDE'}")
                if nd.is_target and c == 1:
                    s = self.session
                    assert t + EXCH[n_mpdu] <= s["win_end"], "NPCA 傳輸越過切回期限"
                    s["tx"] += 1
                    if not success:
                        s["coll"] += 1
                    self.s_tx_log.append((t, end, s["win_end"], s["t_end"], success))
            if not success:
                self.tracing and self._trace(t, "COLLISION", "+".join(n.name for n in starters), c, "同槽起送")
            self.busy_until[c] = end_busy
            self._push(end_busy, P_BEND, "BEND", c)
            if c == 0 and success and starters[0].group == "OP" and self.cfg.method != "legacy":
                self._push(t + T_OUT, P_DEC, "DEC", t + EXCH[starters[0].queue[0][1]])

    def _done(self, nd, success, t):
        nd.in_tx = False
        pkt = nd.queue[0]
        win = self._in_win(t)
        if success:
            nd.queue.popleft()
            nd.delivered += 1
            if win:
                nd.w_delivered += 1
                nd.w_bits += pkt[1] * PAYLOAD_BITS
                nd.w_delay_sum += t - pkt[0]
            if self.cfg.track_ids:
                nd.delivered_ids.append(pkt[2])
            if self.cfg.block_ns and nd.is_target:
                b = self.blk.setdefault(t // self.cfg.block_ns, [0, 0, 0])
                b[0] += pkt[1] * PAYLOAD_BITS
                b[1] += 1
                b[2] += t - pkt[0]
            nd.cw = CW_MIN
            nd.retry = 0
            if nd.is_target and nd.ch == 1:
                self.session["deliv"] += 1
            self.tracing and self._trace(t, "ACK_OK", nd.name, nd.ch, f"pkt={pkt[2]} delay_us={(t - pkt[0]) / US:.1f}")
        else:
            nd.collisions += 1
            if win:
                nd.w_collisions += 1
            nd.retry += 1
            if nd.retry > self.cfg.retry_limit:
                nd.queue.popleft()
                nd.drop_retry += 1
                if win:
                    nd.w_drop_retry += 1
                if self.cfg.track_ids:
                    nd.dropped_ids.append(pkt[2])
                nd.cw = CW_MIN
                nd.retry = 0
                self.tracing and self._trace(t, "DROP_RETRY", nd.name, nd.ch, f"pkt={pkt[2]}")
            else:
                nd.cw = min(2 * nd.cw + 1, CW_MAX)
                self.tracing and self._trace(t, "ACK_TIMEOUT", nd.name, nd.ch, f"retry={nd.retry} cw={nd.cw}")
        if nd.queue:
            nd.b = nd.rng.randint(0, nd.cw)
            self.tracing and self._trace(t, "DRAW_BACKOFF", nd.name, nd.ch, f"b={nd.b} cw={nd.cw}")
        else:
            nd.b = None

    def _bend(self, c, t):
        self.last_end[c] = t
        s = self.session
        if c == 0 and s is not None and s["t_end"] == t:
            self._end_session(t)
        for nd in self.nodes:
            if nd.ch == c and not nd.in_tx and nd.queue:
                nd.active = True
                nd.count_start = t + DIFS

    def _decide(self, t, t_end_est):
        win = self._in_win(t)
        if win:
            self.n_eligible += 1
        remaining = t_end_est - t
        backlog = any(nd.queue for nd in self.targets)
        ratio = None
        m = self.cfg.method
        if m == "adaptive":
            ratio = self._s_busy_ratio(t)
            theta = self.cfg.theta0_ns + self.npca_k_ns * ratio
        elif m == "queue":
            n_pkts = sum(len(nd.queue) for nd in self.targets)
            theta = self.cfg.theta0_ns if n_pkts >= self.cfg.qa_min_backlog else self.cfg.theta_hi_ns
        else:
            theta = self.cfg.theta0_ns
        if not backlog:
            if win:
                self.n_rej_nobacklog += 1
            self.tracing and self._trace(t, "NPCA_SKIP", "TARGET", 0, "目標 BSS 無待傳封包")
            return
        if remaining <= theta:
            if win:
                self.n_rej_theta += 1
                self.n_rej_theta_s_idle += self.busy_until[1] <= t
            self.tracing and self._trace(t, "NPCA_REJECT", "TARGET", 0,
                        f"remaining_us={remaining / US:.1f} theta_us={theta / US:.1f} ratio={'-' if ratio is None else format(ratio, '.3f')}")
            return
        if m == "oracle":
            need = min(EXCH[nd.queue[0][1]] for nd in self.targets if nd.queue)
            if not self._oracle_feasible(t, t_end_est - T_BACK, need):
                if win:
                    self.n_rej_oracle += 1
                self.tracing and self._trace(t, "NPCA_REJECT", "TARGET", 0, "oracle: 副通道無可容納交換的閒置窗口")
                return
        if win:
            self.theta_sum += theta
            self.n_sw_s_busy0 += self.busy_until[1] > t
        self.session = {
            "t_start": t, "t_end": t_end_est, "win_end": t_end_est - T_BACK,
            "sense_ready": t + DIFS, "deliv": 0, "tx": 0, "coll": 0,
            "theta": theta, "ratio": ratio, "remaining": remaining,
        }
        self.tracing and self._trace(t, "NPCA_SWITCH_OUT", "TARGET", 1,
                    f"remaining_us={remaining / US:.1f} theta_us={theta / US:.1f} ratio={'-' if ratio is None else format(ratio, '.3f')} win_end_us={(t_end_est - T_BACK) / US:.1f}")
        for nd in self.targets:
            nd.ch = 1
            nd.active = False
            if nd.queue and not nd.in_tx and self.busy_until[1] <= t:
                nd.count_start = self._align(1, t + DIFS)
                nd.active = True
            self.tracing and self._trace(t, "RESUME_BACKOFF_ON_S" if nd.queue else "IDLE_ON_S", nd.name, 1,
                        f"b={nd.b}" if nd.queue else "")

    def _end_session(self, t):
        s = self.session
        t_cut = t - T_BACK
        for nd in self.targets:
            assert not nd.in_tx, "切回時仍有傳輸進行"
            if nd.active:
                nd.b = max(0, nd.b - self._elapsed(nd, t_cut))
                nd.active = False
            nd.ch = 0
            if nd.queue:
                self.tracing and self._trace(t, "RETURN_TO_P", nd.name, 0, f"b_left={nd.b}")
        self.tracing and self._trace(t, "NPCA_SWITCH_BACK", "TARGET", 0, f"deliveries={s['deliv']} tx={s['tx']} coll={s['coll']}")
        self.session_log.append(s)
        self.session = None

    # ---------------------------------------------------------------- 主迴圈
    def run(self):
        nodes = self.nodes
        heap = self.heap
        t_end = self.t_end
        tx_time = self._tx_time
        while True:
            t_ev = heap[0][0] if heap else INF
            t_ar = INF
            t_tx = INF
            for nd in nodes:
                if nd.next_arr < t_ar:
                    t_ar = nd.next_arr
                x = tx_time(nd)
                if x < t_tx:
                    t_tx = x
            t_next = t_ev if t_ev < t_ar else t_ar
            if t_tx < t_next:
                if t_tx > t_end:
                    break
                self.now = t_tx
                self._start_tx(t_tx)
                continue
            if t_next > t_end:
                break
            self.now = t_next
            if t_ev <= t_ar:
                _, _, _, kind, payload = heapq.heappop(heap)
                if kind == "DONE":
                    self._done(payload[0], payload[1], t_next)
                elif kind == "BEND":
                    self._bend(payload, t_next)
                else:
                    self._decide(t_next, payload)
            else:
                for nd in nodes:
                    if nd.next_arr == t_ar:
                        self._arrival(nd, t_ar)
        return self.results()

    # ---------------------------------------------------------------- 統計
    def results(self):
        cfg = self.cfg
        T = cfg.t_meas_ns
        tg = self.targets
        w_del = sum(n.w_delivered for n in tg)
        w_bits = sum(n.w_bits for n in tg)
        w_dsum = sum(n.w_delay_sum for n in tg)
        w_arr = sum(n.w_arrivals for n in tg)
        w_dr = sum(n.w_drop_retry for n in tg)
        w_do = sum(n.w_drop_overflow for n in tg)
        t0, t1 = cfg.t_warm_ns, self.t_end
        sess = [s for s in self.session_log + ([self.session] if self.session else []) if t0 <= s["t_start"] <= t1]
        n_sw = len(sess)
        sw_nodeliv = sum(1 for s in sess if s["deliv"] == 0)
        sw_notx = sum(1 for s in sess if s["tx"] == 0)
        win_total = sum(s["win_end"] - (s["t_start"] + DIFS) for s in sess if s["win_end"] > s["t_start"] + DIFS)
        npca_deliv = sum(s["deliv"] for s in sess)
        op, os_ = self.obss_p, self.obss_s
        ages = [self.t_end - p[0] for n in tg for p in n.queue]
        res = {
            "throughput_mbps": w_bits / (T / 1e9) / 1e6,
            "mean_delay_ms": (w_dsum / w_del / 1e6) if w_del else float("nan"),
            "delivered_win": w_del,
            "arrivals_win": w_arr,
            "drop_retry_win": w_dr,
            "drop_overflow_win": w_do,
            "drop_rate": ((w_dr + w_do) / w_arr) if w_arr else float("nan"),
            "arrivals_total": sum(n.arrivals for n in tg),
            "delivered_total": sum(n.delivered for n in tg),
            "drop_retry_total": sum(n.drop_retry for n in tg),
            "drop_overflow_total": sum(n.drop_overflow for n in tg),
            "backlog_end": sum(len(n.queue) for n in tg),
            "backlog_age_mean_ms": (sum(ages) / len(ages) / 1e6) if ages else float("nan"),
            "tx_attempts": sum(n.w_tx_attempts for n in tg),
            "collisions": sum(n.w_collisions for n in tg),
            "n_eligible": self.n_eligible,
            "n_rej_theta": self.n_rej_theta,
            "n_rej_nobacklog": self.n_rej_nobacklog,
            "n_rej_oracle": self.n_rej_oracle,
            "n_rej_theta_s_idle": self.n_rej_theta_s_idle,
            "n_sw_s_busy0": self.n_sw_s_busy0,
            "n_switch": n_sw,
            "n_switch_no_delivery": sw_nodeliv,
            "n_switch_no_tx": sw_notx,
            "npca_delivered": npca_deliv,
            "npca_tx_attempts": sum(s["tx"] for s in sess),
            "npca_collisions": sum(s["coll"] for s in sess),
            "npca_window_ms": win_total / 1e6,
            "mean_theta_us": (self.theta_sum / n_sw / US) if n_sw else float("nan"),
            "obss_p_busy_ratio": op.busy_in(t0, t1) / T,
            "obss_s_busy_ratio": os_.busy_in(t0, t1) / T,
            "obss_p_tput_mbps": op.w_bits / (T / 1e9) / 1e6,
            "obss_s_tput_mbps": os_.w_bits / (T / 1e9) / 1e6,
            "obss_p_delay_ms": (op.w_delay_sum / op.w_delivered / 1e6) if op.w_delivered else float("nan"),
            "obss_s_delay_ms": (os_.w_delay_sum / os_.w_delivered / 1e6) if os_.w_delivered else float("nan"),
            "obss_p_delivered": op.w_delivered,
            "obss_s_delivered": os_.w_delivered,
            "obss_s_drop_overflow": os_.w_drop_overflow,
            "obss_s_drop_retry": os_.w_drop_retry,
            "obss_s_arrivals": os_.w_arrivals,
        }
        if cfg.block_ns:
            res["blocks"] = dict(self.blk)
        return res
