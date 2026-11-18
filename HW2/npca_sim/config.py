"""PHY/MAC 常數。時間單位一律為整數奈秒 (ns)，避免浮點誤差影響「同槽起送」判斷。"""

US = 1_000
MS = 1_000_000

# --- 802.11 時序 (取自 802.11 5 GHz OFDM 慣用值，與 Bellalta 等人 Table I 一致) ---
SLOT = 9 * US
SIFS = 16 * US
DIFS = 34 * US

# --- 簡化 PHY：單一 20 MHz、單一空間串流、HE MCS7 (64-QAM 5/6)，NDBPS=1170, 符元 13.6 us ---
PHY_PREAMBLE = 40 * US
OFDM_SYM = 13_600
NDBPS = 1170
ACK_TIME = 28 * US  # ACK 與 Block ACK 皆簡化為同一長度 (24 Mbps 基本速率)

PAYLOAD_BYTES = 1500
MAC_OVERHEAD_BYTES = 34  # MAC header + FCS
MPDU_DELIM_BYTES = 4
PAYLOAD_BITS = PAYLOAD_BYTES * 8

# --- 存取參數 ---
CW_MIN = 15
CW_MAX = 1023
RETRY_LIMIT = 7  # 重傳次數上限：第 8 次失敗後丟棄

# --- NPCA 時間參數 (本作業假設) ---
T_OUT = 100 * US  # 從 OBSS 傳輸開始到完成偵測並切到副通道 (對應論文 T_NPCA)
T_BACK = 50 * US  # 從副通道切回主通道所需時間 (對應論文 T_switch)

# --- OBSS PPDU 長度分佈：聚合 n 個 MPDU ---
OBSS_MPDU_CHOICES = (1, 3, 6, 12, 24)
OBSS_MPDU_PROBS = (0.30, 0.25, 0.20, 0.15, 0.10)


def ppdu_ns(n_mpdu: int) -> int:
    """n 個 1500 B MPDU 聚合成一個 PPDU 的傳輸時間。"""
    if n_mpdu == 1:
        bits = (PAYLOAD_BYTES + MAC_OVERHEAD_BYTES) * 8 + 22
    else:
        bits = n_mpdu * (PAYLOAD_BYTES + MAC_OVERHEAD_BYTES + MPDU_DELIM_BYTES) * 8 + 22
    syms = -(-bits // NDBPS)
    return PHY_PREAMBLE + syms * OFDM_SYM


def exchange_ns(n_mpdu: int) -> int:
    """PPDU + SIFS + ACK。ACK 逾時也設為同一長度 (ACK timeout = SIFS + ACK 時間)。"""
    return ppdu_ns(n_mpdu) + SIFS + ACK_TIME


class _ExchCache(dict):
    def __missing__(self, n):
        v = self[n] = exchange_ns(n)
        return v


EXCH = _ExchCache()
