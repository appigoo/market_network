# modules/risk_engine.py
# ============================================================
# 風險引擎 - 系統性風險偵測 + Smart Signals
# ============================================================

import pandas as pd
import numpy as np
from config.settings import RISK_THRESHOLDS


# ── 1. Volatility Explosion 偵測 ──────────────────────────
def detect_vol_explosion(returns: pd.DataFrame, window: int = 5) -> dict:
    """
    偵測近期波動率爆炸（短期 vol >> 長期 vol）
    返回: {ticker: vol_ratio}
    """
    if returns.empty or len(returns) < 20:
        return {}

    threshold = RISK_THRESHOLDS["vol_explosion"]
    alerts = {}

    for t in returns.columns:
        s = returns[t].dropna()
        if len(s) < 20:
            continue
        short_vol = s.tail(window).std() * np.sqrt(252)
        long_vol  = s.tail(60).std() * np.sqrt(252)
        if long_vol > 0:
            ratio = short_vol / long_vol
            if short_vol > threshold and ratio > 1.5:
                alerts[t] = round(ratio, 2)

    return dict(sorted(alerts.items(), key=lambda x: -x[1]))


# ── 2. Correlation Spike 偵測 ─────────────────────────────
def detect_correlation_spike(returns: pd.DataFrame) -> dict:
    """
    整體市場相關性上升（systemic risk signal）
    返回: {metric: value}
    """
    if returns.empty or len(returns) < 20:
        return {}

    short_corr = returns.tail(10).corr()
    long_corr  = returns.tail(60).corr()

    # 平均相關（排除對角線）
    def avg_corr(m):
        vals = m.values.copy()
        np.fill_diagonal(vals, np.nan)
        return float(np.nanmean(vals))

    sc = avg_corr(short_corr)
    lc = avg_corr(long_corr)
    spike = sc - lc

    return {
        "short_avg_corr": round(sc, 3),
        "long_avg_corr":  round(lc, 3),
        "corr_spike":     round(spike, 3),
        "is_spiking":     spike > 0.15 and sc > RISK_THRESHOLDS["corr_spike"] * 0.7,
    }


# ── 3. Market Panic 偵測（VXX-based）────────────────────
def detect_market_panic(quotes_df: pd.DataFrame) -> dict:
    """
    基於 VXX 和整體市場跌幅偵測恐慌
    """
    if quotes_df.empty:
        return {"is_panic": False}

    result = {"is_panic": False, "vxx_chg": 0.0, "spy_chg": 0.0, "panic_score": 0}

    if "VXX" in quotes_df.index:
        vxx_chg = quotes_df.loc["VXX", "chg_pct"] / 100
        result["vxx_chg"] = round(vxx_chg * 100, 2)
        if vxx_chg > RISK_THRESHOLDS["panic_vxx_pct"]:
            result["panic_score"] += 3

    if "SPY" in quotes_df.index:
        spy_chg = quotes_df.loc["SPY", "chg_pct"] / 100
        result["spy_chg"] = round(spy_chg * 100, 2)
        if spy_chg < -0.02:
            result["panic_score"] += 2

    if "TLT" in quotes_df.index:
        tlt_chg = quotes_df.loc["TLT", "chg_pct"] / 100
        # TLT 上漲 = 資金逃往避險
        if tlt_chg > 0.01:
            result["panic_score"] += 1

    result["is_panic"] = result["panic_score"] >= 4
    return result


# ── 4. Risk-On / Risk-Off Regime ────────────────────────
def detect_market_regime(quotes_df: pd.DataFrame, returns: pd.DataFrame) -> str:
    """
    判斷當前市場 Regime
    返回: "RISK-ON" / "RISK-OFF" / "NEUTRAL" / "VOLATILE"
    """
    if quotes_df.empty:
        return "UNKNOWN"

    score = 0  # 正 = risk-on，負 = risk-off

    # SPY/QQQ 漲 → risk-on
    for etf in ["SPY", "QQQ"]:
        if etf in quotes_df.index:
            chg = quotes_df.loc[etf, "chg_pct"]
            score += 1 if chg > 0.5 else (-1 if chg < -0.5 else 0)

    # IWM 漲 → risk-on（小盤更敏感）
    if "IWM" in quotes_df.index:
        chg = quotes_df.loc["IWM", "chg_pct"]
        score += 2 if chg > 0.5 else (-2 if chg < -0.5 else 0)

    # VXX 跌 → risk-on
    if "VXX" in quotes_df.index:
        chg = quotes_df.loc["VXX", "chg_pct"]
        score += 2 if chg < -2 else (-2 if chg > 3 else 0)

    # TLT 漲 = 逃往避險 → risk-off
    if "TLT" in quotes_df.index:
        chg = quotes_df.loc["TLT", "chg_pct"]
        score -= 1 if chg > 0.5 else (-1 if chg < -0.5 else 0)

    # GLD 漲 = 避險 → risk-off
    if "GLD" in quotes_df.index:
        chg = quotes_df.loc["GLD", "chg_pct"]
        score -= 1 if chg > 0.5 else 0

    # NVDA/AI 漲 → risk-on
    if "NVDA" in quotes_df.index:
        chg = quotes_df.loc["NVDA", "chg_pct"]
        score += 1 if chg > 1 else (-1 if chg < -1.5 else 0)

    if score >= 3:
        return "RISK-ON 🟢"
    elif score <= -3:
        return "RISK-OFF 🔴"
    elif abs(score) <= 1:
        if not returns.empty:
            recent_vol = returns.tail(5).std().mean() * np.sqrt(252)
            if recent_vol > 0.35:
                return "VOLATILE ⚡"
    return "NEUTRAL ⚪"


# ── 5. Capital Flow 識別 ─────────────────────────────────
def detect_capital_flow(quotes_df: pd.DataFrame) -> dict:
    """
    識別資金流向：哪個板塊在吸引資金
    """
    if quotes_df.empty:
        return {}

    sector_map = {
        "Technology":  ["NVDA", "MSFT", "AVGO", "SMH", "QQQ"],
        "EV/AI":       ["TSLA"],
        "Financials":  ["XLF"],
        "Energy":      ["XLE"],
        "Healthcare":  ["XLV"],
        "ConsDisc":    ["XLY"],
        "Bonds":       ["TLT", "HYG"],
        "Gold":        ["GLD"],
        "Crypto":      ["IBIT"],
        "China":       ["FXI"],
    }

    flows = {}
    for sector, tickers in sector_map.items():
        available = [t for t in tickers if t in quotes_df.index]
        if not available:
            continue
        avg_chg = quotes_df.loc[available, "chg_pct"].mean()
        flows[sector] = round(avg_chg, 3)

    return dict(sorted(flows.items(), key=lambda x: -x[1]))


# ── 6. Smart Signals ─────────────────────────────────────
def generate_smart_signals(
    returns: pd.DataFrame,
    quotes_df: pd.DataFrame,
    corr_matrix: pd.DataFrame,
) -> list:
    """
    生成智能信號列表
    返回: [{"type": str, "message": str, "severity": "high"/"med"/"low"}]
    """
    signals = []

    # 波動率爆炸
    vol_alerts = detect_vol_explosion(returns)
    for t, ratio in list(vol_alerts.items())[:3]:
        signals.append({
            "type":     "VOL_EXPLOSION",
            "ticker":   t,
            "message":  f"{t} 波動率爆升 {ratio:.1f}x 正常水平",
            "severity": "high" if ratio > 2.5 else "med",
        })

    # 相關性爆升
    spike_info = detect_correlation_spike(returns)
    if spike_info.get("is_spiking"):
        signals.append({
            "type":     "CORR_SPIKE",
            "ticker":   "MARKET",
            "message":  f"市場整體相關性急升 +{spike_info['corr_spike']:.2f}，系統性風險上升",
            "severity": "high",
        })

    # 恐慌信號
    panic = detect_market_panic(quotes_df)
    if panic.get("is_panic"):
        signals.append({
            "type":     "MARKET_PANIC",
            "ticker":   "VXX",
            "message":  f"市場恐慌偵測！VXX +{panic['vxx_chg']:.1f}% / SPY {panic['spy_chg']:.1f}%",
            "severity": "high",
        })

    # 資金流向
    flows = detect_capital_flow(quotes_df)
    if flows:
        top_in  = list(flows.items())[0]
        top_out = list(flows.items())[-1]
        if top_in[1] > 0.5:
            signals.append({
                "type":     "CAPITAL_FLOW_IN",
                "ticker":   top_in[0],
                "message":  f"資金流入 {top_in[0]} ({top_in[1]:+.2f}%)",
                "severity": "low",
            })
        if top_out[1] < -0.5:
            signals.append({
                "type":     "CAPITAL_FLOW_OUT",
                "ticker":   top_out[0],
                "message":  f"資金流出 {top_out[0]} ({top_out[1]:+.2f}%)",
                "severity": "med",
            })

    # 動量擴張
    if not quotes_df.empty:
        strong_movers = quotes_df[quotes_df["chg_pct"].abs() > 2]
        if not strong_movers.empty:
            for t, row in strong_movers.iterrows():
                signals.append({
                    "type":     "MOMENTUM",
                    "ticker":   t,
                    "message":  f"{t} 強勢動量 {row['chg_pct']:+.2f}%",
                    "severity": "low" if abs(row["chg_pct"]) < 4 else "med",
                })

    return signals[:12]  # 最多返回 12 個信號
