# modules/data_engine.py
# ============================================================
# 數據引擎 - yfinance + curl_cffi 防封鎖
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# curl_cffi 為可選依賴（加速 yfinance），沒有也可正常運行
try:
    import curl_cffi  # noqa
except ImportError:
    pass

from config.settings import TIMEFRAMES, MARKET_CAP_B, DEFAULT_MARKET_CAP_B


# ── 四層數據回退 ──────────────────────────────────────────
def _fetch_yfinance(tickers: list, period: str, interval: str) -> pd.DataFrame:
    """主層：yfinance 批量下載"""
    try:
        raw = yf.download(
            tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw.empty:
            return pd.DataFrame()

        # 展平 MultiIndex
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw.xs("Close", axis=1, level=0) if "Close" in raw.columns.get_level_values(0) else pd.DataFrame()
        else:
            close = raw[["Close"]] if "Close" in raw.columns else pd.DataFrame()

        return close.dropna(how="all")
    except Exception as e:
        return pd.DataFrame()


def _fetch_yfinance_single(tickers: list, period: str, interval: str) -> pd.DataFrame:
    """第二層：逐個下載合并"""
    frames = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            df = tk.history(period=period, interval=interval, auto_adjust=True)
            if not df.empty and "Close" in df.columns:
                frames[t] = df["Close"]
        except Exception:
            pass
    if frames:
        return pd.DataFrame(frames)
    return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_price_data(tickers: list, timeframe: str = "1d") -> pd.DataFrame:
    """
    主入口：獲取收盤價矩陣
    返回: DataFrame [datetime index, ticker columns]
    """
    if not tickers:
        return pd.DataFrame()

    cfg = TIMEFRAMES.get(timeframe, TIMEFRAMES["1d"])
    period   = cfg["period"]
    interval = cfg["interval"]

    # 嘗試批量
    df = _fetch_yfinance(tickers, period, interval)

    # 如果批量失敗，逐個
    if df.empty or len(df.columns) < len(tickers) * 0.5:
        df2 = _fetch_yfinance_single(tickers, period, interval)
        if not df2.empty:
            df = df2

    if df.empty:
        return pd.DataFrame()

    # 只保留請求的 ticker
    available = [t for t in tickers if t in df.columns]
    df = df[available].copy()

    # 移除全空列
    df = df.dropna(axis=1, how="all")

    # 向前填充（最多 3 格）
    df = df.ffill(limit=3)

    return df


@st.cache_data(ttl=60, show_spinner=False)
def fetch_quote_data(tickers: list) -> pd.DataFrame:
    """
    獲取即時報價：price, change%, volume, market_cap
    """
    records = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            info = tk.fast_info
            hist = tk.history(period="2d", interval="1d", auto_adjust=True)

            price = float(getattr(info, "last_price", 0) or 0)
            prev  = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price
            chg_pct = ((price - prev) / prev * 100) if prev else 0.0
            vol   = float(getattr(info, "three_month_average_volume", 0) or 0)
            mcap  = MARKET_CAP_B.get(t, DEFAULT_MARKET_CAP_B)

            records.append({
                "ticker":    t,
                "price":     price,
                "chg_pct":   chg_pct,
                "volume":    vol,
                "mcap_b":    mcap,
            })
        except Exception:
            records.append({
                "ticker":  t,
                "price":   0.0,
                "chg_pct": 0.0,
                "volume":  0.0,
                "mcap_b":  MARKET_CAP_B.get(t, DEFAULT_MARKET_CAP_B),
            })

    return pd.DataFrame(records).set_index("ticker")


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """計算對數收益率"""
    if prices.empty:
        return pd.DataFrame()
    return np.log(prices / prices.shift(1)).dropna()


def compute_volatility(returns: pd.DataFrame, window: int = 20) -> pd.Series:
    """年化波動率"""
    if returns.empty:
        return pd.Series(dtype=float)
    return returns.rolling(window).std().iloc[-1] * np.sqrt(252)
