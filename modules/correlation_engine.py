# modules/correlation_engine.py
# ============================================================
# 相關性引擎 - Pearson / Rolling / Volatility Correlation
# ============================================================

import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st
from config.settings import CORR_THRESHOLDS


# ── 1. Pearson 相關矩陣 ────────────────────────────────────
def pearson_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """全樣本 Pearson 相關矩陣"""
    if returns.empty or len(returns) < 5:
        return pd.DataFrame()
    return returns.corr(method="pearson")


# ── 2. Rolling 相關矩陣（最新一個窗口）────────────────────
def rolling_correlation(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """滾動窗口相關矩陣（取最後 window 個觀測）"""
    if returns.empty or len(returns) < window:
        return pearson_correlation(returns)
    tail = returns.tail(window)
    return tail.corr(method="pearson")


# ── 3. Volatility-Weighted 相關 ────────────────────────────
def volatility_correlation(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    波動率加權相關：先用 vol 標準化收益，再計算相關
    反映高波動時期的真實共動關係
    """
    if returns.empty or len(returns) < window + 2:
        return pearson_correlation(returns)
    vol = returns.rolling(window).std()
    normalized = (returns / vol).replace([np.inf, -np.inf], np.nan).dropna()
    return normalized.corr(method="pearson")


# ── 4. 相關矩陣時間序列（用於滾動圖）──────────────────────
def rolling_corr_series(
    returns: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window: int = 20,
) -> pd.Series:
    """返回兩個 ticker 之間的滾動相關序列"""
    if ticker_a not in returns.columns or ticker_b not in returns.columns:
        return pd.Series(dtype=float)
    return returns[ticker_a].rolling(window).corr(returns[ticker_b]).dropna()


# ── 5. Lead-Lag 檢測 ───────────────────────────────────────
def lead_lag_analysis(returns: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """
    計算所有對的 cross-correlation，找 lag=-max_lag..+max_lag
    返回 DataFrame: [leader, follower, lag, corr]
    """
    tickers = returns.columns.tolist()
    records = []

    for i, t1 in enumerate(tickers):
        for t2 in tickers[i+1:]:
            best_lag  = 0
            best_corr = 0.0
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    c = returns[t1].corr(returns[t2])
                elif lag > 0:
                    c = returns[t1].corr(returns[t2].shift(lag))
                else:
                    c = returns[t1].shift(-lag).corr(returns[t2])

                if abs(c) > abs(best_corr):
                    best_corr = c
                    best_lag  = lag

            if best_lag < 0:
                # t2 leads t1
                records.append({"leader": t2, "follower": t1,
                                 "lag": abs(best_lag), "corr": best_corr})
            elif best_lag > 0:
                # t1 leads t2
                records.append({"leader": t1, "follower": t2,
                                 "lag": best_lag, "corr": best_corr})
            else:
                records.append({"leader": t1, "follower": t2,
                                 "lag": 0, "corr": best_corr})

    if not records:
        return pd.DataFrame(columns=["leader", "follower", "lag", "corr"])

    df = pd.DataFrame(records)
    return df.sort_values("lag").reset_index(drop=True)


def leader_score(lead_lag_df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    計算每個 ticker 的 leader score（領先次數 × 平均相關強度）
    """
    if lead_lag_df.empty:
        return pd.DataFrame()

    scores = []
    for t in tickers:
        led   = lead_lag_df[lead_lag_df["leader"] == t]
        score = len(led) * (led["corr"].abs().mean() if not led.empty else 0)
        scores.append({"ticker": t, "lead_count": len(led), "leader_score": round(score, 3)})

    return pd.DataFrame(scores).sort_values("leader_score", ascending=False).reset_index(drop=True)


# ── 6. 相關性快照工具 ─────────────────────────────────────
def get_corr_matrix(
    returns: pd.DataFrame,
    method: str = "pearson",
    window: int = 60,
) -> pd.DataFrame:
    """統一入口：選擇相關計算方法"""
    if method == "rolling":
        return rolling_correlation(returns, window)
    elif method == "volatility":
        return volatility_correlation(returns, window)
    else:
        return pearson_correlation(returns)


def edge_list_from_corr(corr_matrix: pd.DataFrame, threshold: float = 0.0) -> list:
    """
    將相關矩陣轉換為邊列表（用於 Force Graph）
    只保留 |corr| >= threshold 的邊
    返回: [(ticker_a, ticker_b, corr_value), ...]
    """
    edges = []
    tickers = corr_matrix.columns.tolist()
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            c = corr_matrix.iloc[i, j]
            if not np.isnan(c) and abs(c) >= threshold:
                edges.append((tickers[i], tickers[j], float(c)))
    return edges


def detect_corr_spikes(
    returns: pd.DataFrame,
    window_short: int = 10,
    window_long: int = 60,
) -> dict:
    """
    偵測相關性爆升：短期相關 vs 長期相關
    返回 {(t1, t2): spike_intensity}
    """
    spikes = {}
    tickers = returns.columns.tolist()
    threshold = CORR_THRESHOLDS["strong_pos"]

    for i, t1 in enumerate(tickers):
        for t2 in tickers[i+1:]:
            short_c = returns[t1].tail(window_short).corr(returns[t2].tail(window_short))
            long_c  = returns[t1].tail(window_long).corr(returns[t2].tail(window_long))
            if pd.isna(short_c) or pd.isna(long_c):
                continue
            spike = short_c - long_c
            if short_c > threshold and spike > 0.2:
                spikes[(t1, t2)] = round(spike, 3)

    return spikes
