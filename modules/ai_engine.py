# modules/ai_engine.py
# ============================================================
# AI 摘要引擎 - Groq LLM 市場分析（含各 Tab 一鍵解讀）
# ============================================================

import os
import streamlit as st
from groq import Groq
from config.settings import GROQ_MODEL, GROQ_MAX_TOKENS

SYSTEM_PROMPT = """你是一位管理10億美元的對沖基金首席量化分析師。
規則：
1. 用繁體中文回答
2. 直接給出明確方向，不說「可能」「也許」「需要注意」這種廢話
3. 每個判斷都要有數字支撐
4. 格式用 emoji 開頭讓每行清晰易讀
5. 字數控制在400字內，精煉有力
6. 最後一定要有「✅ 今日操作方向」，給出具體做多/做空/觀望的股票名稱"""


def _get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        api_key = st.session_state.get("groq_api_key", "")
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None


def _call_groq(prompt: str) -> str:
    client = _get_groq_client()
    if not client:
        return "⚠️ 請在左側欄輸入 Groq API Key（免費申請：console.groq.com）"
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=GROQ_MAX_TOKENS,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI 分析失敗：{str(e)}"


def analyze_force_graph(regime, quotes_df, signals, flows, node_colors, edge_count, risk_nodes):
    if not quotes_df.empty:
        sorted_q = quotes_df["chg_pct"].sort_values(ascending=False)
        top3 = [(t, v) for t, v in sorted_q.head(3).items()]
        bot3 = [(t, v) for t, v in sorted_q.tail(3).items()]
        top_str = " | ".join([f"{t} {v:+.2f}%" for t, v in top3])
        bot_str = " | ".join([f"{t} {v:+.2f}%" for t, v in bot3])
    else:
        top_str = bot_str = "N/A"

    flow_str = " | ".join([f"{k} {v:+.2f}%" for k, v in list(flows.items())[:6]])
    sig_str  = "\n".join([f"- [{s['severity'].upper()}] {s['message']}" for s in signals[:8]])
    risk_str = "、".join(risk_nodes) if risk_nodes else "無"

    prompt = f"""根據以下 Force Graph 市場網絡數據，給出完整解讀：

市場 Regime：{regime}
網絡連接邊數：{edge_count}（邊越多=市場越同步）
系統風險節點（高中心性）：{risk_str}
今日漲幅前三：{top_str}
今日跌幅前三：{bot_str}
板塊資金流向：{flow_str}
Smart Signals：
{sig_str}

請解讀：
1. 📊 市場網絡結構（今天是分散還是同步行情？）
2. 💰 資金在哪裡（哪個板塊最強，哪個被拋棄）
3. ⚠️ 風險節點分析（{risk_str} 為何被高亮，代表什麼）
4. 🎯 Force Graph 最重要的交易信號是什麼
5. ✅ 今日操作方向（明確說做多哪隻/做空哪隻/觀望）"""
    return _call_groq(prompt)


def analyze_heatmap(corr_matrix, spike_info, spikes, pair_a, pair_b, current_corr):
    import numpy as np
    if corr_matrix.empty:
        return "⚠️ 暫無相關性數據"

    vals = []
    tickers = corr_matrix.columns.tolist()
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            c = corr_matrix.iloc[i, j]
            if not np.isnan(c):
                vals.append((tickers[i], tickers[j], float(c)))
    vals.sort(key=lambda x: -abs(x[2]))

    avg_corr = float(sum(v[2] for v in vals) / len(vals)) if vals else 0
    top_str  = "\n".join([f"  {a}/{b}: {c:.3f}" for a, b, c in vals[:5]])
    pos_str  = "\n".join([f"  {a}/{b}: {c:.3f}" for a, b, c in vals if c > 0.7][:3]) or "  無"
    neg_str  = "\n".join([f"  {a}/{b}: {c:.3f}" for a, b, c in vals if c < -0.3][:3]) or "  無"
    spike_str = "\n".join([f"  {a}/{b}: 爆升 +{v:.3f}" for (a,b),v in list(spikes.items())[:5]]) or "  無"

    prompt = f"""根據以下相關性熱力圖數據，給出完整解讀：

市場平均相關係數：{avg_corr:.3f}（>0.5=高度同步，<0.3=分化行情）
相關性爆升狀態：{'是！短期相關急升 +' + str(spike_info.get('corr_spike', 0)) if spike_info.get('is_spiking') else '否，正常水平'}
最強相關配對：
{top_str}
強正相關（>0.7）：
{pos_str}
負相關（<-0.3）：
{neg_str}
相關性爆升配對：
{spike_str}
當前追蹤配對 {pair_a}/{pair_b}：滾動相關 = {current_corr:.3f}

請解讀：
1. 📊 今天市場是分化還是抱團？
2. 🔗 最重要的正相關意味著什麼（哪些股票可以互相替代做多）
3. ↕️ 負相關如何利用（對沖還是輪動）
4. 🚨 相關性爆升代表什麼風險
5. ✅ 今日操作方向（根據相關性結構，做多哪些/避開哪些）"""
    return _call_groq(prompt)


def analyze_cluster(communities, cluster_stats, centrality, regime, flows):
    comm_str = "\n".join([
        f"  Cluster {cid}（{len(members)}隻）: {', '.join(members)}"
        for cid, members in sorted(communities.items())
    ])

    if not cluster_stats.empty:
        stats_str = "\n".join([
            f"  Cluster {int(row['cluster_id'])}: 內部相關={row['intra_corr']:.3f}, 波動率={row['avg_vol']:.1f}%"
            for _, row in cluster_stats.iterrows()
        ])
    else:
        stats_str = "  暫無統計"

    if centrality:
        top_central = sorted(centrality.items(), key=lambda x: -x[1].get("composite", 0))[:5]
        central_str = " | ".join([f"{t}({v['composite']:.2f})" for t, v in top_central])
    else:
        central_str = "N/A"

    flow_str = " | ".join([f"{k} {v:+.2f}%" for k, v in list(flows.items())[:6]])

    prompt = f"""根據以下市場聚類分析數據，給出完整解讀：

市場 Regime：{regime}
板塊資金流向：{flow_str}
Louvain 自動聚類結果：
{comm_str}
各聚類統計：
{stats_str}
系統核心節點（中心性最高）：{central_str}

請解讀：
1. 🧩 今天市場分成幾個陣營？每個陣營的特徵是什麼
2. 🏆 哪個 Cluster 是今天的主力（內部相關高+波動大=抱團拉升）
3. 🌐 中心性最高的股票為何重要（牽一髮動全身）
4. 🔄 哪些股票聚類結果出乎意料
5. ✅ 今日操作方向（做最強 Cluster 的龍頭，說出具體股票名）"""
    return _call_groq(prompt)


def analyze_lead_lag(ll_scores, ll_df, flows, regime, timeframe):
    if ll_scores.empty:
        return "⚠️ 數據不足，無法計算 Lead-Lag（建議切換到日線或1小時週期）"

    scores_str = "\n".join([
        f"  {row['ticker']}: Leader Score={row['leader_score']:.2f}, 領先次數={row['lead_count']}"
        for _, row in ll_scores.head(8).iterrows()
    ])

    if not ll_df.empty:
        top_pairs = ll_df[ll_df["lag"] > 0].head(8)
        pairs_str = "\n".join([
            f"  {row['leader']} 領先 {row['follower']} {row['lag']} 期（相關={abs(row['corr']):.3f}）"
            for _, row in top_pairs.iterrows()
        ])
    else:
        pairs_str = "  暫無配對數據"

    flow_str = " | ".join([f"{k} {v:+.2f}%" for k, v in list(flows.items())[:5]])

    prompt = f"""根據以下 Lead-Lag 領漲分析數據，給出完整解讀：

時間週期：{timeframe}
市場 Regime：{regime}
板塊資金流向：{flow_str}
Leader Score 排名：
{scores_str}
具體領漲配對：
{pairs_str}

請解讀：
1. 👑 今天誰是真正的市場領袖（為何排名第一）
2. ⏱ 最實用的跟隨交易機會（A漲了，幾期後買B）
3. 🔄 資金傳導鏈（從哪隻股票擴散到整個市場）
4. ⚠️ 哪些股票只是跟隨者（不要把跟隨者當領漲追）
5. ✅ 今日操作方向（具體 Leader→Follower 策略，看什麼信號入場）"""
    return _call_groq(prompt)


def analyze_risk(signals, regime, risk_score, panic_info, vol_alerts, spike_info, flows, risk_nodes):
    sig_str = "\n".join([
        f"  [{s['severity'].upper()}] {s['message']}" for s in signals
    ]) or "  無信號"

    vol_str = "\n".join([
        f"  {t}: 波動率是正常的 {v:.1f}倍" for t, v in list(vol_alerts.items())[:5]
    ]) or "  無異常"

    flow_str = " | ".join([f"{k} {v:+.2f}%" for k, v in list(flows.items())[:6]])

    prompt = f"""根據以下市場風險數據，給出完整風險評估：

市場 Regime：{regime}
綜合風險分數：{risk_score}/100（0-30正常，30-60警戒，60+危險）
恐慌狀態：{'🔴 是！VXX +' + str(panic_info.get('vxx_chg',0)) + '%' if panic_info.get('is_panic') else '🟢 否'}
VXX 今日：{panic_info.get('vxx_chg', 0):+.1f}%
SPY 今日：{panic_info.get('spy_chg', 0):+.1f}%
系統風險節點：{'、'.join(risk_nodes) if risk_nodes else '無'}
相關性爆升：{'是，+' + str(spike_info.get('corr_spike',0)) if spike_info.get('is_spiking') else '否'}
波動率異常：
{vol_str}
所有風險信號：
{sig_str}
板塊資金流向：{flow_str}

請解讀：
1. 🚦 現在市場風險等級（用紅/黃/綠燈，一句話說清楚）
2. 🔴 最危險的信號是哪個（為什麼優先處理）
3. 💉 系統性風險有多高（會不會一跌全跌）
4. 🛡 如何對沖今天的風險（具體工具：VXX/TLT/GLD 及比例）
5. ✅ 今日操作方向（根據風險水平，說明最大可接受倉位比例和止損位）"""
    return _call_groq(prompt)


def generate_market_summary(regime, flows, signals, leaders, risk_info, cluster_info, timeframe="1d"):
    flow_str   = " | ".join([f"{k}: {v:+.2f}%" for k, v in list(flows.items())[:6]])
    signal_str = "\n".join([f"- [{s['severity'].upper()}] {s['message']}" for s in signals[:6]])
    leader_str = ", ".join([str(l) for l in leaders[:5]])

    prompt = f"""根據以下即時市場數據，生成完整市場結構總覽：

時間週期：{timeframe}
市場 Regime：{regime}
資金流向：{flow_str}
領漲股票：{leader_str}
市場聚類：{cluster_info}
風險信號：
{signal_str}
恐慌偵測：{risk_info}

請輸出：
1. 📊 市場結構總覽
2. 💰 資金流向分析
3. 👑 領漲核心
4. ⚠️ 風險提示
5. ✅ 今日操作方向（明確股票名稱和方向）"""
    return _call_groq(prompt)


def generate_telegram_alert(signals: list, regime: str) -> str:
    high = [s for s in signals if s["severity"] == "high"]
    if not high:
        return ""
    lines = ["🚨 *市場網絡警報*", f"📊 Regime: `{regime}`", ""]
    for s in high[:5]:
        icon = "🔴" if "PANIC" in s["type"] or "VOL" in s["type"] else "⚠️"
        lines.append(f"{icon} {s['message']}")
    return "\n".join(lines)
