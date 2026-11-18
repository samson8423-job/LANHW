"""統計工具：平均、Student t 95% 信賴區間 (僅用標準函式庫與 NumPy)。"""

import math

import numpy as np


def _betacf(a, b, x):
    # Numerical Recipes 連分式，用於正規化不完全 Beta 函數
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    x = df / (df + t * t)
    p = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - p if t > 0 else p


def t_crit(df, conf=0.95):
    """雙尾 Student t 臨界值 (例如 df=19 -> 2.093)。"""
    target = 1.0 - (1.0 - conf) / 2.0
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def mean_ci(x, conf=0.95):
    """回傳 (平均, 標準差, 95% CI 半寬, n)；忽略 NaN。以各次獨立模擬為樣本。"""
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = float(a.mean())
    if n < 2:
        return m, float("nan"), float("nan"), n
    s = float(a.std(ddof=1))
    return m, s, t_crit(n - 1, conf) * s / math.sqrt(n), n
