"""封包到達序列與退避亂數源。到達與退避使用互相獨立的 SeedSequence 子流。

同一 seed 下：
- 各節點的到達序列只由 (seed, 節點, 速率) 決定，與 method / 門檻無關 -> 三種方法共用到達序列。
- 單位指數間隔序列與封包大小序列各用獨立子流；改變負載只會縮放間隔，不改動底層亂數。
- 每個節點各有一條獨立退避亂數流 (Python random.Random)，三種方法各自消耗，不會影響到達序列。
"""

import random

import numpy as np

from .config import OBSS_MPDU_CHOICES, OBSS_MPDU_PROBS, PAYLOAD_BITS

N_STREAMS_PER_NODE = 2  # 0: 單位指數間隔, 1: 封包大小


def node_names(n_sta):
    return [f"STA{i}" for i in range(n_sta)] + ["OBSS_P", "OBSS_S"]


def make_arrivals(seed, n_sta, rate_sta_pps, rate_p_pps, rate_s_pps, horizon_ns):
    """回傳每節點 (到達時間 ns 串列, 每封包 MPDU 數串列)。"""
    n_nodes = n_sta + 2
    ss = np.random.SeedSequence([int(seed), 1]).spawn(n_nodes * N_STREAMS_PER_NODE)
    rates = [rate_sta_pps] * n_sta + [rate_p_pps, rate_s_pps]
    out = []
    horizon_s = horizon_ns / 1e9
    for i, rate in enumerate(rates):
        if rate <= 0:
            out.append(([], []))
            continue
        g_iat = np.random.Generator(np.random.PCG64(ss[2 * i]))
        g_sz = np.random.Generator(np.random.PCG64(ss[2 * i + 1]))
        # 以固定區塊大小抽取單位指數間隔，不同速率共用同一底層序列 (只做縮放)
        chunks, total = [], 0.0
        while total < horizon_s * rate:
            c = g_iat.standard_exponential(4096)
            chunks.append(c)
            total += float(c.sum())
        unit = np.cumsum(np.concatenate(chunks))
        times = (unit[unit < horizon_s * rate] / rate * 1e9).astype(np.int64)
        n = len(times)
        if i < n_sta:
            sizes = [1] * n
        else:
            sizes = g_sz.choice(OBSS_MPDU_CHOICES, size=n, p=OBSS_MPDU_PROBS).tolist()
        out.append((times.tolist(), sizes))
    return out


def make_backoff_rngs(seed, n_sta):
    ss = np.random.SeedSequence([int(seed), 2]).spawn(n_sta + 2)
    return [random.Random(int(s.generate_state(1, dtype=np.uint64)[0])) for s in ss]


def sta_rate_pps(aggregate_mbps, n_sta):
    return aggregate_mbps * 1e6 / PAYLOAD_BITS / n_sta
