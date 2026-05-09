# modules/ai_engine.py
# ============================================================
# AI 摘要引擎 - Groq LLM 市場分析
# ============================================================

import os
import streamlit as st
from groq import Groq
from config.settings import GROQ_MODEL, GROQ_MAX_TOKENS


def _get_groq_client() -> Groq | None:
    """獲取 Groq 客戶端"""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        api_key = st.session_state.get("groq_api_key", "")
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None


def generate_market_summary(
    regime: str,
    flows: dict,
    signals: list,
    leaders: list,
    risk_info: dict,
    cluster_info: str,
    timeframe: str = "1d",
) -> str:
    """
    調用 Groq 生成中文市場結構分析
    """
    client = _get_groq_client()
    if not client:
        return _fallback_summary(regime, flows, signals, leaders, risk_info)

    # 格式化輸入
    flow_str = " | ".join([f"{k}: {v:+.2f}%" for k, v in list(flows.items())[:6]])
    signal_str = "\n".join([f"- [{s['severity'].upper()}] {s['message']}" for s in signals[:6]])
    leader_str = ", ".join([str(l) for l in leaders[:5]])

    prompt = f"""你是一位頂級對沖基金量化分析師。根據以下即時市場數據，用繁體中文生成一份簡潔有力的市場結構分析報告（300字內）。

**時間週期**: {timeframe}
**市場 Regime**: {regime}
**資金流向**: {flow_str}
**領漲股票**: {leader_str}
**市場聚類**: {cluster_info}
**風險信號**:
{signal_str}
**恐慌偵測**: {risk_info}

請按以下格式輸出：
1. 市場結構概覽（1-2句）
2. 資金流向分析（板塊強弱）
3. 領漲核心（誰在帶動）
4. 風險提示（如有）
5. 交易建議方向（謹慎措辭）

語言：繁體中文，專業但直接，用數字說話。"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=GROQ_MAX_TOKENS,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return _fallback_summary(regime, flows, signals, leaders, risk_info)


def _fallback_summary(regime, flows, signals, leaders, risk_info) -> str:
    """無 Groq API 時的本地摘要"""
    top_in  = list(flows.items())[0]  if flows else ("N/A", 0)
    top_out = list(flows.items())[-1] if flows else ("N/A", 0)
    high_signals = [s for s in signals if s["severity"] == "high"]

    lines = [
        f"**市場 Regime**: {regime}",
        "",
        f"**資金流向**: 資金流入 {top_in[0]}（{top_in[1]:+.2f}%），流出 {top_out[0]}（{top_out[1]:+.2f}%）",
        "",
        f"**領漲股票**: {', '.join([str(l) for l in leaders[:5]])}",
        "",
    ]

    if high_signals:
        lines.append("**⚠️ 高級風險提示**:")
        for s in high_signals:
            lines.append(f"- {s['message']}")
    else:
        lines.append("**風險狀態**: 暫無高級風險信號，市場結構相對穩定")

    lines += [
        "",
        "*（連接 Groq API 以獲取完整 AI 分析）*",
    ]
    return "\n".join(lines)


def generate_telegram_alert(signals: list, regime: str) -> str:
    """生成 Telegram 通知文本"""
    high = [s for s in signals if s["severity"] == "high"]
    if not high:
        return ""

    lines = [
        "🚨 *市場網絡警報*",
        f"📊 Regime: `{regime}`",
        "",
    ]
    for s in high[:5]:
        icon = "🔴" if "PANIC" in s["type"] or "VOL" in s["type"] else "⚠️"
        lines.append(f"{icon} {s['message']}")

    return "\n".join(lines)
