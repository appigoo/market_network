# modules/ai_engine.py
# ============================================================
# AI 摘要引擎 - Groq LLM 市場分析（各 Tab 詳細一鍵解讀）
# ============================================================

import os
import streamlit as st
from groq import Groq
from config.settings import GROQ_MODEL, GROQ_MAX_TOKENS

SYSTEM_PROMPT = """你是一位管理10億美元的對沖基金首席量化分析師，同時是一位優秀的市場教師。

你的解讀風格必須嚴格遵守以下規則：

【格式規則】
1. 每個分析點用數字編號（1. 2. 3.）
2. 每個點都要有小標題（粗體格式：**標題**）
3. 標題下面逐行列出具體股票和數字，每行用「→」說明含義
4. 重要結論用「→」引導，說清楚「是什麼」「意味著什麼」「該怎麼做」
5. 最後必須有「✅ 今日交易結論」，用表格或列點格式，說清楚做多哪隻/做空哪隻/不碰哪隻
6. 最後一行必須是「**一句話總結：**」，一句話概括今天最重要的操作方向

【內容規則】
1. 直接說結論，禁止說「可能」「也許」「建議關注」「需要注意」
2. 每個判斷必須有數字支撐（漲跌%、相關係數、倍數等）
3. 解釋圖表的每個視覺元素代表什麼（顏色、位置、線條、圓圈）
4. 把專業術語翻譯成交易員能立刻用的語言
5. 字數400-600字，詳細但不囉嗦

【語言】繁體中文"""


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
            max_tokens=1800,
            temperature=0.15,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI 分析失敗：{str(e)}"


# ══════════════════════════════════════════════════════════
# TAB 1: Force Graph 解讀
# ══════════════════════════════════════════════════════════
def analyze_force_graph(regime, quotes_df, signals, flows, node_colors, edge_count, risk_nodes):
    if not quotes_df.empty:
        sorted_q  = quotes_df["chg_pct"].sort_values(ascending=False)
        top5      = [(t, v) for t, v in sorted_q.head(5).items()]
        bot5      = [(t, v) for t, v in sorted_q.tail(5).items()]
        top_str   = "\n".join([f"  {t}: {v:+.2f}%" for t, v in top5])
        bot_str   = "\n".join([f"  {t}: {v:+.2f}%" for t, v in bot5])
        total     = len(quotes_df)
        green_ct  = int((quotes_df["chg_pct"] > 0).sum())
        red_ct    = int((quotes_df["chg_pct"] < 0).sum())
    else:
        top_str = bot_str = "N/A"
        total = green_ct = red_ct = 0

    flow_str  = "\n".join([f"  {k}: {v:+.2f}%" for k, v in list(flows.items())])
    sig_str   = "\n".join([f"  [{s['severity'].upper()}] {s['message']}" for s in signals])
    risk_str  = "、".join(risk_nodes) if risk_nodes else "無"

    # 計算正負邊比例（edge_count 是總邊數）
    prompt = f"""根據以下 Force Graph 市場網絡的完整數據，像分析師解讀真實圖表一樣，逐點詳細解讀。

=== 市場數據 ===
市場 Regime：{regime}
網絡邊數：{edge_count}條（越多代表市場越同步）
上漲股票數：{green_ct}/{total}隻
下跌股票數：{red_ct}/{total}隻
系統風險節點（紅圈高亮）：{risk_str}

今日漲幅排名（節點顏色越綠=漲幅越大）：
{top_str}

今日跌幅排名（節點顏色越紅=跌幅越大）：
{bot_str}

板塊資金流向：
{flow_str}

Smart Signals 面板：
{sig_str}

=== 你的解讀任務 ===
請完整解讀以下6個方面，每個方面都要說清楚「視覺上看到什麼」→「意味著什麼」→「交易含義是什麼」：

1. **整體市場結構**
解讀：網絡邊數{edge_count}條、{green_ct}隻上漲/{red_ct}隻下跌，說明今天是什麼類型的行情（同步/分化/恐慌/Risk-On/Risk-Off）

2. **節點顏色 = 漲跌分析**
逐一解讀漲幅前三和跌幅前三的股票，每隻都要說明：它今天漲/跌意味著哪類資金在行動

3. **節點位置 = 相關性距離**
解讀：哪些股票聚在中心（高度同步）、哪些在邊緣孤立（獨立行情），各自的交易含義

4. **邊線顏色分析**
藍線（正相關）密集還是稀疏？紅線（負相關）連接哪些股票？說明資金流動方向

5. **紅圈高亮節點（{risk_str}）**
這些節點為何被系統標記為風險節點？今天需要怎麼對待它們

6. **Smart Signals 解讀**
逐條解讀每個信號的含義和對應操作

✅ 今日交易結論（必須有明確的做多/做空/不碰列表，每項附理由和數字）

**一句話總結：**（概括今天最重要的一個操作方向）"""

    return _call_groq(prompt)


# ══════════════════════════════════════════════════════════
# TAB 2: Heatmap 解讀
# ══════════════════════════════════════════════════════════
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

    avg_corr  = float(sum(v[2] for v in vals) / len(vals)) if vals else 0
    top5_str  = "\n".join([f"  {a}/{b}: {c:+.3f}" for a, b, c in vals[:5]])
    pos3_str  = "\n".join([f"  {a}/{b}: {c:.3f}" for a, b, c in vals if c > 0.7][:5]) or "  無強正相關對"
    neg3_str  = "\n".join([f"  {a}/{b}: {c:.3f}" for a, b, c in vals if c < -0.3][:5]) or "  無負相關對"
    spike_str = "\n".join([f"  {a}/{b}: 短期比長期高 +{v:.3f}" for (a,b),v in list(spikes.items())[:5]]) or "  無"

    prompt = f"""根據以下相關性熱力圖數據，像解讀真實熱力圖一樣，逐點詳細解讀。

=== 相關性數據 ===
市場平均相關係數：{avg_corr:.3f}
（解讀標準：>0.6=高度抱團，0.3-0.6=正常，<0.3=高度分化）

相關性爆升狀態：{'⚠️ 是！短期相關比長期高 +' + str(round(spike_info.get('corr_spike',0),3)) + '，系統性風險上升' if spike_info.get('is_spiking') else '✅ 正常，無爆升'}

相關係數最強的5對（熱力圖最深色格子）：
{top5_str}

強正相關對（>0.7，熱力圖深綠/深藍格子）：
{pos3_str}

負相關對（<-0.3，熱力圖深紅格子）：
{neg3_str}

相關性爆升配對（短期異常同步）：
{spike_str}

當前追蹤配對 {pair_a} / {pair_b}：
  滾動相關係數 = {current_corr:.3f}
  （>0.7=高度同步，0~0.3=輕度，<0=反向）

=== 你的解讀任務 ===
請完整解讀以下5個方面：

1. **熱力圖整體顏色解讀**
平均相關{avg_corr:.3f}說明今天市場是抱團還是分化？對比正常水平意味著什麼？

2. **最強正相關對的交易含義**
這些高度同步的股票對，在交易上可以怎麼利用（做多其中一隻時，另一隻可以作為確認信號）

3. **負相關對的交易機會**
紅色格子代表的反向關係，如何用來做板塊輪動或對沖

4. **相關性爆升的風險含義**
{('⚠️ 發現爆升：' + str(len(spikes)) + '對異常，說明什麼風險') if spikes else '✅ 無爆升，市場結構正常'}

5. **{pair_a}/{pair_b} 配對解讀**
當前相關{current_corr:.3f}，這個數字說明什麼？對交易這兩隻股票有什麼具體含義？

✅ 今日交易結論（基於相關性結構，明確說做多哪組/做空哪組/用哪對做對沖）

**一句話總結：**"""

    return _call_groq(prompt)


# ══════════════════════════════════════════════════════════
# TAB 3: Cluster 解讀
# ══════════════════════════════════════════════════════════
def analyze_cluster(communities, cluster_stats, centrality, regime, flows):
    comm_lines = []
    for cid, members in sorted(communities.items()):
        comm_lines.append(f"  Cluster {cid}（{len(members)}隻）: {', '.join(members)}")
    comm_str = "\n".join(comm_lines)

    if not cluster_stats.empty:
        stats_lines = []
        for _, row in cluster_stats.iterrows():
            stats_lines.append(
                f"  Cluster {int(row['cluster_id'])}: 內部相關={row['intra_corr']:.3f}, "
                f"平均波動率={row['avg_vol']:.1f}%, 成員數={int(row['size'])}"
            )
        stats_str = "\n".join(stats_lines)
    else:
        stats_str = "  暫無統計"

    if centrality:
        top_central = sorted(centrality.items(), key=lambda x: -x[1].get("composite", 0))[:8]
        central_str = "\n".join([
            f"  {t}: 中心性={v['composite']:.3f} (中介={v['betweenness']:.3f}, 度={v['degree']:.3f})"
            for t, v in top_central
        ])
    else:
        central_str = "  N/A"

    flow_str = "\n".join([f"  {k}: {v:+.2f}%" for k, v in list(flows.items())])

    prompt = f"""根據以下市場聚類分析數據，像解讀真實聚類圖一樣，逐點詳細解讀。

=== 聚類數據 ===
市場 Regime：{regime}

Louvain 算法自動聚類結果（算法根據相關性自動把相似股票分組）：
{comm_str}

各聚類詳細統計：
{stats_str}

（解讀標準：內部相關>0.7=高度抱團，波動率高=這個cluster今天很活躍）

系統核心節點（中心性排名，數值越高=對整個市場影響越大）：
{central_str}

板塊資金流向：
{flow_str}

=== 你的解讀任務 ===
請完整解讀以下5個方面：

1. **市場今天分成幾個陣營**
解讀每個 Cluster 的組成，說明算法為什麼把這些股票分在一起（它們今天有什麼共同點）

2. **最強 Cluster 是哪個**
根據內部相關+波動率，找出今天最活躍的陣營，說明它代表什麼主題（AI/防守/流動性等）

3. **意外的聚類結果**
有沒有本來不應該在同一組的股票被分在一起？這說明今天有什麼特殊的資金行為？

4. **中心性最高的股票**
這些股票為什麼中心性高？今天如果它們突然反轉，會拖累哪些其他股票？

5. **Cluster 間的資金輪動**
對照資金流向數據，資金正在從哪個 Cluster 流向哪個 Cluster？

✅ 今日交易結論（做最強 Cluster 的龍頭股，具體說出股票名稱和理由）

**一句話總結：**"""

    return _call_groq(prompt)


# ══════════════════════════════════════════════════════════
# TAB 4: Lead-Lag 解讀
# ══════════════════════════════════════════════════════════
def analyze_lead_lag(ll_scores, ll_df, flows, regime, timeframe):
    if ll_scores.empty:
        return "⚠️ 數據不足，無法計算 Lead-Lag（建議切換到日線或1小時週期後重新分析）"

    scores_str = "\n".join([
        f"  {row['ticker']}: Leader Score={row['leader_score']:.2f}, 領先次數={row['lead_count']}次"
        for _, row in ll_scores.head(10).iterrows()
    ])

    if not ll_df.empty:
        top_pairs = ll_df[ll_df["lag"] > 0].head(10)
        pairs_str = "\n".join([
            f"  {row['leader']} 領先 {row['follower']} {row['lag']} 期，相關強度={abs(row['corr']):.3f}"
            for _, row in top_pairs.iterrows()
        ])
    else:
        pairs_str = "  暫無配對數據"

    flow_str = "\n".join([f"  {k}: {v:+.2f}%" for k, v in list(flows.items())])

    prompt = f"""根據以下 Lead-Lag 領漲分析數據，像分析師解讀真實領漲關係一樣，逐點詳細解讀。

=== Lead-Lag 數據 ===
時間週期：{timeframe}（每「1期」= 1個{timeframe}）
市場 Regime：{regime}

Leader Score 完整排名（分數=領先次數×平均相關強度，分數越高=領漲能力越強）：
{scores_str}

具體領漲配對（A領先B N期=A動了N個{timeframe}後B才跟）：
{pairs_str}

板塊資金流向：
{flow_str}

=== 你的解讀任務 ===
請完整解讀以下5個方面：

1. **今天的市場領袖是誰**
解讀排名第一的股票：它的 Leader Score 說明什麼？為什麼是它在帶動市場而不是其他股票？

2. **最值得交易的 Leader→Follower 配對**
從配對列表中找出最實用的跟隨策略：當A股發出信號後，幾個{timeframe}內可以買B股，預期的跟隨相關性是多少

3. **資金傳導鏈**
今天資金是從哪隻股票開始，逐步擴散到整個市場的？畫出傳導路徑（如：NVDA→SMH→QQQ→IWM）

4. **假領漲陷阱**
哪些股票的 Leader Score 低但漲幅大？這說明它們只是跟隨者，不能用它們的動作來判斷市場方向

5. **Lead-Lag 實戰操作方法**
在{timeframe}時間框架下，具體說明：看到什麼信號入場、等多少期確認、止損放在哪裡

✅ 今日交易結論（具體的 Leader→Follower 策略，說清楚A漲了多少%後買B，目標和止損）

**一句話總結：**"""

    return _call_groq(prompt)


# ══════════════════════════════════════════════════════════
# TAB 5: Risk 解讀
# ══════════════════════════════════════════════════════════
def analyze_risk(signals, regime, risk_score, panic_info, vol_alerts, spike_info, flows, risk_nodes):
    sig_high = [s for s in signals if s["severity"] == "high"]
    sig_med  = [s for s in signals if s["severity"] == "med"]
    sig_low  = [s for s in signals if s["severity"] == "low"]

    def fmt_sigs(lst):
        return "\n".join([f"  • {s['message']}" for s in lst]) or "  無"

    vol_str = "\n".join([
        f"  {t}: 波動率是正常的{v:.1f}倍（正常=1x，>2x=危險）"
        for t, v in list(vol_alerts.items())
    ]) or "  無異常"

    flow_str = "\n".join([f"  {k}: {v:+.2f}%" for k, v in list(flows.items())])

    # 風險燈號
    if risk_score >= 60:
        traffic = "🔴 紅燈 — 高風險，大幅減倉或觀望"
    elif risk_score >= 30:
        traffic = "🟡 黃燈 — 警戒，縮小倉位，收緊止損"
    else:
        traffic = "🟢 綠燈 — 正常，按計劃操作"

    prompt = f"""根據以下市場風險數據，像風控主管解讀真實風險儀表板一樣，逐點詳細解讀。

=== 風險數據 ===
市場 Regime：{regime}
綜合風險分數：{risk_score}/100 → {traffic}
恐慌狀態：{'🔴 觸發！' if panic_info.get('is_panic') else '🟢 未觸發'}
VXX 今日：{panic_info.get('vxx_chg', 0):+.2f}%（>+10%=市場恐慌）
SPY 今日：{panic_info.get('spy_chg', 0):+.2f}%
恐慌評分：{panic_info.get('panic_score', 0)}/6分（≥4分觸發恐慌警報）

系統風險節點（高中心性，一旦崩潰會拖累全市場）：
  {', '.join(risk_nodes) if risk_nodes else '無'}

相關性爆升：{'⚠️ 是！短期相關比長期高 +' + str(round(spike_info.get('corr_spike',0),3)) if spike_info.get('is_spiking') else '✅ 否'}

波動率異常（>1.5x即預警）：
{vol_str}

🔴 高級警報（{len(sig_high)}個）：
{fmt_sigs(sig_high)}

🟡 中級警報（{len(sig_med)}個）：
{fmt_sigs(sig_med)}

🔵 低級信號（{len(sig_low)}個）：
{fmt_sigs(sig_low)}

板塊資金流向：
{flow_str}

=== 你的解讀任務 ===
請完整解讀以下5個方面：

1. **風險燈號判斷**
風險分數{risk_score}/100，是{traffic}。逐一解釋這個分數是怎麼來的（哪些信號造成了高/低分）

2. **最危險的信號是哪個**
從所有警報中找出今天最需要優先處理的一個，說明為什麼它最危險，以及不處理的後果

3. **系統性風險評估**
今天如果{', '.join(risk_nodes[:2]) if risk_nodes else '主要股票'}突然大跌，會拖累哪些股票？系統性崩潰的概率有多高？

4. **具體對沖方案**
根據當前風險水平，給出具體的對沖工具和比例：
- VXX：買多少%倉位
- TLT：買多少%倉位  
- GLD：買多少%倉位
- 或者：直接減倉到什麼比例

5. **各警報的應對方法**
逐條高/中級警報說明具體應對：看到這個信號，立刻做什麼動作

✅ 今日交易結論（根據風險水平：最大持倉比例是多少/止損位置/對沖工具）

**一句話總結：**"""

    return _call_groq(prompt)


# ══════════════════════════════════════════════════════════
# TAB 6: 總覽 AI 分析
# ══════════════════════════════════════════════════════════
def generate_market_summary(regime, flows, signals, leaders, risk_info, cluster_info, timeframe="1d"):
    flow_str   = "\n".join([f"  {k}: {v:+.2f}%" for k, v in list(flows.items())])
    signal_str = "\n".join([f"  [{s['severity'].upper()}] {s['message']}" for s in signals])
    leader_str = ", ".join([str(l) for l in leaders[:5]])

    prompt = f"""根據以下完整市場數據，生成今日市場結構總覽報告。

=== 市場數據 ===
時間週期：{timeframe}
市場 Regime：{regime}
領漲股票：{leader_str}
市場聚類結構：{cluster_info}

板塊資金流向：
{flow_str}

風險信號：
{signal_str}

恐慌偵測：{risk_info}

=== 你的報告格式 ===

1. **市場結構總覽**
今天市場是什麼類型（Risk-On/Off/分化/恐慌），用數字說明

2. **資金流向分析**
資金從哪裡來、流向哪裡，板塊強弱排名

3. **領漲核心**
誰在帶動市場，傳導鏈是什麼

4. **風險提示**
今天最值得警惕的風險是什麼，概率多高

5. **今日完整操作計劃**
做多清單（股票+理由+入場條件）
做空/對沖清單（如有）
絕對不碰清單（股票+理由）

✅ 今日交易結論

**一句話總結：**"""

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
