"""單次模擬執行與平行批次執行 (僅用標準函式庫 multiprocessing)。"""

import multiprocessing as mp
from dataclasses import replace

from .config import US
from .sim import SimConfig, Simulator
from .traffic import make_arrivals, make_backoff_rngs, sta_rate_pps

N_STA = 5


def run_one(job):
    """job: dict，欄位見下。回傳 meta + 結果欄位的 dict。"""
    t_warm = int(job["t_warm_s"] * 1e9)
    t_meas = int(job["t_meas_s"] * 1e9)
    cfg = SimConfig(
        method=job["method"],
        theta0_ns=int(job.get("theta0_us", 500) * US),
        k_ns=float(job.get("k_us", 2000) * US),
        theta_hi_ns=int(job.get("theta_hi_us", 500) * US),
        qa_min_backlog=int(job.get("qa_min_backlog", 20)),
        busy_window_ns=int(job.get("busy_window_ms", 20) * 1000 * US),
        t_warm_ns=t_warm,
        t_meas_ns=t_meas,
        n_sta=N_STA,
        block_ns=int(job.get("block_s", 0) * 1e9),
    )
    arr = make_arrivals(
        job["seed"], N_STA, sta_rate_pps(job["target_mbps"], N_STA),
        job["rate_p_pps"], job["rate_s_pps"], t_warm + t_meas + 10_000_000,
    )
    if cfg.method == "oracle":
        # 第一趥：同一種子的固定門檻 0 模擬，取得副通道 OBSS 傳輸區間 (近似的未來資訊)
        pre = Simulator(replace(cfg, method="fixed", theta0_ns=0), arr, make_backoff_rngs(job["seed"], N_STA))
        pre.run()
        cfg = replace(cfg, theta0_ns=0, oracle_s_busy=tuple(zip(pre.obss_s.ep_starts, pre.obss_s.ep_ends)))
    rngs = make_backoff_rngs(job["seed"], N_STA)
    res = Simulator(cfg, arr, rngs).run()
    row = {k: v for k, v in job.items()}
    row.update(res)
    return row


def run_many(jobs, processes=None, chunksize=4, progress=True):
    jobs = list(jobs)
    out = []
    if processes == 1:
        for i, j in enumerate(jobs):
            out.append(run_one(j))
            if progress and (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
        return out
    with mp.Pool(processes) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, jobs, chunksize=chunksize)):
            out.append(r)
            if progress and (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    return out
