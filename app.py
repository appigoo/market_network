# app.py
# ============================================================
# 美股股票關聯 Force Graph 分析系統
# 機構級市場網絡分析 | Bloomberg Terminal 風格
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
import pytz

# ── 模組導入 ──────────────────────────────────────────────
from config.settings import (
    DEFAULT_WATCHLIST, ALL_DEFAULT_TICKERS,
    TIMEFRAMES, CORR_WINDOWS, UI_COLORS, SECTOR_COLORS,
)
from modules.data_engine      import fetch_price_data, fetch_quote_data, compute_returns, compute_volatility
from modules.correlation_engine import (
    get_corr_matrix, edge_list_from_corr, lead_lag_analysis,
    leader_score, detect_corr_spikes, rolling_corr_series,
)
from modules.cluster_engine   import (
    build_corr_graph, louvain_communities,
    assign_cluster_colors, compute_cluster_stats, compute_centrality,
)
from modules.risk_engine      import (
    detect_market_regime, detect_capital_flow,
    generate_smart_signals, detect_market_panic, detect_vol_explosion,
)
from modules.graph_engine     import (
    build_force_graph, build_heatmap, build_rolling_corr_chart,
)
from modules.ai_engine        import generate_market_summary, generate_telegram_alert
from modules.telegram_module  import send_deduped_alert


# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Market Network | 市場關聯分析",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════
# CSS INJECTION  —  Terminal Dark Style
# ═══════════════════════════════════════════════════════════
def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    /* ── Base ─────────────────────────────────────────── */
    html, body, [class*="css"] {{
        font-family: 'Space Grotesk', sans-serif;
        background-color: {UI_COLORS['bg_primary']} !important;
        color: {UI_COLORS['text_primary']};
    }}
    .stApp {{
        background-color: {UI_COLORS['bg_primary']};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {UI_COLORS['bg_secondary']} !important;
        border-right: 1px solid {UI_COLORS['border']};
    }}

    /* ── Metrics ──────────────────────────────────────── */
    [data-testid="metric-container"] {{
        background: {UI_COLORS['bg_card']};
        border: 1px solid {UI_COLORS['border']};
        border-radius: 6px;
        padding: 12px 16px;
    }}
    [data-testid="metric-container"] label {{
        color: {UI_COLORS['text_secondary']} !important;
        font-size: 11px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.5rem !important;
        font-weight: 600;
    }}
    [data-testid="stMetricDelta"] {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
    }}

    /* ── Cards ────────────────────────────────────────── */
    .terminal-card {{
        background: {UI_COLORS['bg_card']};
        border: 1px solid {UI_COLORS['border']};
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }}
    .terminal-card-title {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: {UI_COLORS['text_secondary']};
        margin-bottom: 10px;
        border-bottom: 1px solid {UI_COLORS['border']};
        padding-bottom: 8px;
    }}

    /* ── Header ───────────────────────────────────────── */
    .main-header {{
        background: linear-gradient(135deg, {UI_COLORS['bg_secondary']} 0%, #0A1628 100%);
        border: 1px solid {UI_COLORS['border']};
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    .main-title {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.4rem;
        font-weight: 600;
        color: {UI_COLORS['accent_blue']};
        letter-spacing: 0.05em;
    }}
    .main-subtitle {{
        font-size: 0.75rem;
        color: {UI_COLORS['text_secondary']};
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.1em;
    }}

    /* ── Regime Badge ─────────────────────────────────── */
    .regime-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.1em;
    }}
    .regime-risk-on  {{ background: rgba(0,255,159,0.15); color: #00FF9F; border: 1px solid rgba(0,255,159,0.4); }}
    .regime-risk-off {{ background: rgba(255,51,102,0.15); color: #FF3366; border: 1px solid rgba(255,51,102,0.4); }}
    .regime-volatile {{ background: rgba(255,184,0,0.15);  color: #FFB800; border: 1px solid rgba(255,184,0,0.4); }}
    .regime-neutral  {{ background: rgba(148,163,184,0.15);color: #94A3B8; border: 1px solid rgba(148,163,184,0.4); }}

    /* ── Signal Cards ─────────────────────────────────── */
    .signal-high {{
        background: rgba(255,51,102,0.1);
        border-left: 3px solid {UI_COLORS['accent_red']};
        padding: 8px 12px;
        border-radius: 0 6px 6px 0;
        margin-bottom: 8px;
        font-size: 12px;
        font-family: 'IBM Plex Mono', monospace;
    }}
    .signal-med {{
        background: rgba(255,184,0,0.1);
        border-left: 3px solid {UI_COLORS['accent_amber']};
        padding: 8px 12px;
        border-radius: 0 6px 6px 0;
        margin-bottom: 8px;
        font-size: 12px;
        font-family: 'IBM Plex Mono', monospace;
    }}
    .signal-low {{
        background: rgba(0,212,255,0.07);
        border-left: 3px solid {UI_COLORS['accent_blue']};
        padding: 8px 12px;
        border-radius: 0 6px 6px 0;
        margin-bottom: 8px;
        font-size: 12px;
        font-family: 'IBM Plex Mono', monospace;
    }}

    /* ── Ticker Table ─────────────────────────────────── */
    .ticker-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid {UI_COLORS['grid']};
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
    }}
    .ticker-name {{ color: {UI_COLORS['text_primary']}; font-weight: 500; }}
    .ticker-price {{ color: {UI_COLORS['text_secondary']}; }}
    .ticker-pos {{ color: {UI_COLORS['accent_green']}; }}
    .ticker-neg {{ color: {UI_COLORS['accent_red']}; }}

    /* ── Plotly tweaks ────────────────────────────────── */
    .js-plotly-plot .plotly .modebar {{
        background: {UI_COLORS['bg_card']} !important;
    }}
    .js-plotly-plot .plotly .modebar-btn path {{
        fill: {UI_COLORS['text_secondary']} !important;
    }}

    /* ── Scrollbar ────────────────────────────────────── */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: {UI_COLORS['bg_secondary']}; }}
    ::-webkit-scrollbar-thumb {{ background: {UI_COLORS['border']}; border-radius: 4px; }}

    /* ── Selectbox, inputs ────────────────────────────── */
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {{
        background: {UI_COLORS['bg_card']} !important;
        border-color: {UI_COLORS['border']} !important;
        color: {UI_COLORS['text_primary']} !important;
    }}
    .stSlider [data-baseweb="slider"] {{
        background: {UI_COLORS['grid']};
    }}
    .stTextInput input {{
        background: {UI_COLORS['bg_card']} !important;
        border-color: {UI_COLORS['border']} !important;
        color: {UI_COLORS['text_primary']} !important;
        font-family: 'IBM Plex Mono', monospace;
    }}
    div[data-testid="stTabs"] button {{
        color: {UI_COLORS['text_secondary']};
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.08em;
    }}
    div[data-testid="stTabs"] button[aria-selected="true"] {{
        color: {UI_COLORS['accent_blue']};
        border-bottom-color: {UI_COLORS['accent_blue']};
    }}
    /* Sidebar text */
    .sidebar-section-title {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: {UI_COLORS['text_secondary']};
        margin: 16px 0 8px 0;
        border-bottom: 1px solid {UI_COLORS['border']};
        padding-bottom: 4px;
    }}
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# SESSION STATE 初始化
# ═══════════════════════════════════════════════════════════
def init_session():
    defaults = {
        "tickers":          ALL_DEFAULT_TICKERS.copy(),
        "timeframe":        "1d",
        "corr_method":      "rolling",
        "corr_window":      60,
        "corr_threshold":   0.3,
        "show_negative":    True,
        "auto_refresh":     False,
        "refresh_interval": 60,
        "last_refresh":     0,
        "ai_summary":       "",
        "groq_api_key":     "",
        "telegram_token":   "",
        "telegram_chat_id": "",
        "selected_pair":    ("SPY", "QQQ"),
        "data_loaded":      False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-section-title">▸ 系統配置</div>', unsafe_allow_html=True)

        # ── 時間週期 ──
        tf_labels = list(TIMEFRAMES.keys())
        tf_idx    = tf_labels.index(st.session_state["timeframe"]) if st.session_state["timeframe"] in tf_labels else 4
        tf = st.selectbox("時間週期", tf_labels, index=tf_idx,
                          format_func=lambda x: f"{x}  ({TIMEFRAMES[x]['label']})")
        st.session_state["timeframe"] = tf

        # ── 相關性方法 ──
        st.markdown('<div class="sidebar-section-title">▸ 相關性引擎</div>', unsafe_allow_html=True)
        method_map = {"Pearson": "pearson", "Rolling": "rolling", "Volatility": "volatility"}
        method_label = st.selectbox("計算方法", list(method_map.keys()), index=1)
        st.session_state["corr_method"] = method_map[method_label]

        window_map = {"短期(20)": 20, "中期(60)": 60, "長期(120)": 120}
        win_label = st.selectbox("時間窗口", list(window_map.keys()), index=1)
        st.session_state["corr_window"] = window_map[win_label]

        threshold = st.slider("最低相關閾值", 0.0, 0.9, st.session_state["corr_threshold"], 0.05)
        st.session_state["corr_threshold"] = threshold

        show_neg = st.checkbox("顯示負相關邊", value=st.session_state["show_negative"])
        st.session_state["show_negative"] = show_neg

        # ── 股票池 ──
        st.markdown('<div class="sidebar-section-title">▸ 股票池管理</div>', unsafe_allow_html=True)

        st.caption("自定義新增股票（逗號分隔）")
        custom_input = st.text_input("自定股票", placeholder="AAPL, META, AMD", key="custom_ticker_input")
        if st.button("＋ 新增", use_container_width=True):
            custom_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
            current = st.session_state["tickers"]
            added   = [t for t in custom_tickers if t not in current]
            if added:
                st.session_state["tickers"] = current + added
                st.success(f"已新增: {', '.join(added)}")
                st.session_state["data_loaded"] = False

        if st.button("↺ 重置為預設", use_container_width=True):
            st.session_state["tickers"] = ALL_DEFAULT_TICKERS.copy()
            st.session_state["data_loaded"] = False

        st.caption(f"當前股票池: {len(st.session_state['tickers'])} 隻")

        # 板塊過濾
        st.markdown('<div class="sidebar-section-title">▸ 板塊過濾</div>', unsafe_allow_html=True)
        sector_select = st.multiselect(
            "僅顯示板塊",
            list(DEFAULT_WATCHLIST.keys()),
            default=[],
        )
        if sector_select:
            filtered = []
            for sector in sector_select:
                filtered.extend(DEFAULT_WATCHLIST.get(sector, []))
            # 保留 custom tickers
            for t in st.session_state["tickers"]:
                all_default_in_sectors = filtered
                if t not in [tt for s in DEFAULT_WATCHLIST.values() for tt in s]:
                    filtered.append(t)
            st.session_state["active_tickers"] = list(dict.fromkeys(filtered))
        else:
            st.session_state["active_tickers"] = st.session_state["tickers"]

        # ── API 設定 ──
        st.markdown('<div class="sidebar-section-title">▸ API 設定</div>', unsafe_allow_html=True)
        with st.expander("🔑 API Keys"):
            groq_key = st.text_input("Groq API Key", type="password",
                                     value=st.session_state.get("groq_api_key", ""))
            st.session_state["groq_api_key"] = groq_key

            tg_token = st.text_input("Telegram Token", type="password",
                                     value=st.session_state.get("telegram_token", ""))
            st.session_state["telegram_token"] = tg_token

            tg_chat = st.text_input("Telegram Chat ID",
                                    value=st.session_state.get("telegram_chat_id", ""))
            st.session_state["telegram_chat_id"] = tg_chat

        # ── 自動刷新 ──
        st.markdown('<div class="sidebar-section-title">▸ 自動刷新</div>', unsafe_allow_html=True)
        auto = st.checkbox("啟用自動刷新", value=st.session_state["auto_refresh"])
        st.session_state["auto_refresh"] = auto
        if auto:
            interval = st.selectbox("刷新間隔", [30, 60, 120, 300], index=1,
                                    format_func=lambda x: f"{x} 秒")
            st.session_state["refresh_interval"] = interval


# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
def render_header(regime: str, last_update: str):
    regime_lower = regime.lower()
    if "risk-on" in regime_lower:
        badge_class = "regime-risk-on"
    elif "risk-off" in regime_lower:
        badge_class = "regime-risk-off"
    elif "volatile" in regime_lower:
        badge_class = "regime-volatile"
    else:
        badge_class = "regime-neutral"

    st.markdown(f"""
    <div class="main-header">
        <div>
            <div class="main-title">📡 MARKET NETWORK ANALYZER</div>
            <div class="main-subtitle">美股關聯 Force Graph 系統 | 機構級量化分析</div>
        </div>
        <div style="text-align:right;">
            <span class="regime-badge {badge_class}">{regime}</span><br>
            <span style="font-size:10px;color:{UI_COLORS['text_secondary']};font-family:'IBM Plex Mono',monospace;margin-top:4px;display:block;">
                {last_update}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TICKER STRIP (上方價格滾動條)
# ═══════════════════════════════════════════════════════════
def render_ticker_strip(quotes_df: pd.DataFrame):
    if quotes_df.empty:
        return
    key_tickers = [t for t in ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "VXX", "GLD", "TLT"] if t in quotes_df.index]
    cols = st.columns(len(key_tickers))
    for i, t in enumerate(key_tickers):
        row = quotes_df.loc[t]
        chg = row["chg_pct"]
        price = row["price"]
        delta_color = "normal"
        cols[i].metric(
            label=t,
            value=f"${price:.2f}" if price > 0 else "N/A",
            delta=f"{chg:+.2f}%" if price > 0 else None,
            delta_color=delta_color,
        )


# ═══════════════════════════════════════════════════════════
# SIGNAL PANEL
# ═══════════════════════════════════════════════════════════
def render_signals(signals: list):
    if not signals:
        st.markdown('<div style="color:#94A3B8;font-family:IBM Plex Mono;font-size:12px;">暫無信號</div>',
                    unsafe_allow_html=True)
        return

    for s in signals:
        sev = s.get("severity", "low")
        cls = f"signal-{sev}"
        icon = {"high": "🔴", "med": "🟡", "low": "🔵"}.get(sev, "⚪")
        st.markdown(f'<div class="{cls}">{icon} {s["message"]}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# QUOTE TABLE
# ═══════════════════════════════════════════════════════════
def render_quote_table(quotes_df: pd.DataFrame, tickers: list):
    if quotes_df.empty:
        return
    available = [t for t in tickers if t in quotes_df.index]
    html = []
    for t in available:
        row  = quotes_df.loc[t]
        chg  = row["chg_pct"]
        price = row["price"]
        chg_class = "ticker-pos" if chg >= 0 else "ticker-neg"
        chg_sign  = "+" if chg >= 0 else ""
        html.append(f"""
        <div class="ticker-row">
            <span class="ticker-name">{t}</span>
            <span class="ticker-price">${price:.2f}</span>
            <span class="{chg_class}">{chg_sign}{chg:.2f}%</span>
        </div>""")
    st.markdown("".join(html), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    inject_css()
    init_session()
    render_sidebar()

    # ── 自動刷新邏輯 ──────────────────────────────────────
    if st.session_state["auto_refresh"]:
        now = time.time()
        if now - st.session_state["last_refresh"] > st.session_state["refresh_interval"]:
            st.session_state["last_refresh"] = now
            st.session_state["data_loaded"]  = False
            st.cache_data.clear()

    # ── 活躍 Tickers ──────────────────────────────────────
    tickers = st.session_state.get("active_tickers", st.session_state["tickers"])
    tickers = list(dict.fromkeys(tickers))  # 去重

    # ── 數據加載 ──────────────────────────────────────────
    with st.spinner("📡 正在連接市場數據..."):
        prices    = fetch_price_data(tickers, st.session_state["timeframe"])
        quotes_df = fetch_quote_data(tickers)

    if prices.empty:
        st.error("⚠️ 無法獲取市場數據，請檢查網絡連接或稍後重試。")
        return

    # 計算收益率
    returns = compute_returns(prices)

    # 計算相關矩陣
    corr_matrix = get_corr_matrix(
        returns,
        method=st.session_state["corr_method"],
        window=st.session_state["corr_window"],
    )

    # ── Market Regime ─────────────────────────────────────
    regime = detect_market_regime(quotes_df, returns)

    # ── 時間戳 ───────────────────────────────────────────
    london_tz  = pytz.timezone("Europe/London")
    ny_tz      = pytz.timezone("America/New_York")
    now_london = datetime.now(london_tz)
    now_ny     = datetime.now(ny_tz)
    last_update = (
        f"London {now_london.strftime('%H:%M:%S')} | "
        f"NY {now_ny.strftime('%H:%M:%S')}"
    )

    # ── HEADER ───────────────────────────────────────────
    render_header(regime, last_update)

    # ── TICKER STRIP ─────────────────────────────────────
    render_ticker_strip(quotes_df)
    st.divider()

    # ── 聚類分析 ─────────────────────────────────────────
    G = build_corr_graph(corr_matrix, threshold=st.session_state["corr_threshold"])
    partition    = louvain_communities(G)
    node_colors  = assign_cluster_colors(partition, tickers)
    centrality   = compute_centrality(G)

    # 系統性重要節點（高 centrality）
    risk_nodes = [t for t, c in centrality.items() if c.get("composite", 0) > 0.5]

    # Smart Signals
    signals   = generate_smart_signals(returns, quotes_df, corr_matrix)
    flows     = detect_capital_flow(quotes_df)
    ll_df     = lead_lag_analysis(returns, max_lag=3)
    ll_scores = leader_score(ll_df, tickers)

    # ── 發送 Telegram 高級警報 ───────────────────────────
    tg_msg = generate_telegram_alert(signals, regime)
    if tg_msg and st.session_state.get("telegram_token"):
        send_deduped_alert(tg_msg, cooldown_minutes=30)

    # ══════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌐 Force Graph",
        "🔥 Heatmap",
        "📊 Cluster",
        "⏱ Lead-Lag",
        "⚠️ Risk",
        "🤖 AI 分析",
    ])

    # ══════════════════════════════════════════════════════
    # TAB 1: FORCE GRAPH
    # ══════════════════════════════════════════════════════
    with tab1:
        col_g, col_r = st.columns([3, 1])

        with col_g:
            st.markdown('<div class="terminal-card-title">市場關聯網絡 Force Graph</div>', unsafe_allow_html=True)
            fig = build_force_graph(
                corr_matrix,
                quotes_df,
                node_colors,
                corr_threshold=st.session_state["corr_threshold"],
                show_negative=st.session_state["show_negative"],
                highlight_nodes=risk_nodes,
            )
            st.plotly_chart(fig, use_container_width=True, config={
                "scrollZoom": True,
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "displaylogo": False,
            })

        with col_r:
            # 板塊圖例
            st.markdown('<div class="terminal-card-title">板塊圖例</div>', unsafe_allow_html=True)
            for sector, color in SECTOR_COLORS.items():
                tks_in_sector = [t for t in tickers if any(t in v for k, v in DEFAULT_WATCHLIST.items() if k == sector)]
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                    f'<div style="width:10px;height:10px;border-radius:50%;background:{color};flex-shrink:0;"></div>'
                    f'<span style="font-size:11px;font-family:IBM Plex Mono;color:{UI_COLORS["text_secondary"]};">{sector}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.divider()
            st.markdown('<div class="terminal-card-title">Smart Signals</div>', unsafe_allow_html=True)
            render_signals(signals[:6])

            st.divider()
            st.markdown('<div class="terminal-card-title">報價</div>', unsafe_allow_html=True)
            render_quote_table(quotes_df, tickers)

    # ══════════════════════════════════════════════════════
    # TAB 2: HEATMAP
    # ══════════════════════════════════════════════════════
    with tab2:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown('<div class="terminal-card-title">相關性熱力圖</div>', unsafe_allow_html=True)
            hm_fig = build_heatmap(corr_matrix)
            st.plotly_chart(hm_fig, use_container_width=True, config={"displaylogo": False})

        with c2:
            st.markdown('<div class="terminal-card-title">滾動相關分析</div>', unsafe_allow_html=True)

            avail_tickers = [t for t in tickers if t in returns.columns]
            if len(avail_tickers) >= 2:
                pair_a = st.selectbox("股票 A", avail_tickers,
                                      index=0, key="pair_a")
                pair_b = st.selectbox("股票 B", avail_tickers,
                                      index=min(1, len(avail_tickers)-1), key="pair_b")
                roll_window = st.slider("滾動窗口", 5, 60,
                                        st.session_state["corr_window"] // 3, key="roll_win")

                if pair_a != pair_b:
                    roll_series = rolling_corr_series(returns, pair_a, pair_b, roll_window)
                    rc_fig = build_rolling_corr_chart(roll_series, pair_a, pair_b)
                    st.plotly_chart(rc_fig, use_container_width=True, config={"displaylogo": False})
                    if not roll_series.empty:
                        current_c = roll_series.iloc[-1]
                        st.metric("當前相關", f"{current_c:.3f}",
                                  delta=f"{(roll_series.iloc[-1] - roll_series.iloc[-5]):.3f} vs 5期前"
                                  if len(roll_series) >= 5 else None)

            # 相關性爆升偵測
            st.markdown('<div class="terminal-card-title" style="margin-top:16px;">相關性爆升偵測</div>', unsafe_allow_html=True)
            spikes = detect_corr_spikes(returns)
            if spikes:
                spike_df = pd.DataFrame([
                    {"配對": f"{a}/{b}", "爆升強度": v}
                    for (a, b), v in list(spikes.items())[:8]
                ])
                st.dataframe(
                    spike_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "爆升強度": st.column_config.ProgressColumn(
                            "爆升強度", min_value=0, max_value=0.5, format="%.3f"
                        )
                    }
                )
            else:
                st.caption("暫無相關性爆升信號")

    # ══════════════════════════════════════════════════════
    # TAB 3: CLUSTER
    # ══════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="terminal-card-title">市場聚類分析 — Louvain Community Detection</div>', unsafe_allow_html=True)

        cluster_stats = compute_cluster_stats(returns, partition)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('<div class="terminal-card-title">聚類成員</div>', unsafe_allow_html=True)
            communities = {}
            for t, cid in partition.items():
                communities.setdefault(cid, []).append(t)

            for cid, members in sorted(communities.items()):
                member_str = "  ".join(members)
                color = node_colors.get(members[0], UI_COLORS["accent_blue"]) if members else UI_COLORS["accent_blue"]
                st.markdown(
                    f'<div class="terminal-card" style="border-left:3px solid {color};">'
                    f'<div class="terminal-card-title">Cluster {cid}</div>'
                    f'<div style="font-family:IBM Plex Mono;font-size:12px;color:{UI_COLORS["text_primary"]};'
                    f'line-height:1.8;">{member_str}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with c2:
            st.markdown('<div class="terminal-card-title">聚類統計</div>', unsafe_allow_html=True)
            if not cluster_stats.empty:
                display_df = cluster_stats[["cluster_id", "size", "avg_vol", "intra_corr"]].copy()
                display_df.columns = ["Cluster", "成員數", "平均波動率%", "內部相關"]
                st.dataframe(
                    display_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "內部相關": st.column_config.ProgressColumn(
                            "內部相關", min_value=-1, max_value=1, format="%.3f"
                        ),
                        "平均波動率%": st.column_config.NumberColumn(format="%.1f%%"),
                    }
                )

            st.markdown('<div class="terminal-card-title" style="margin-top:16px;">中心性分析（系統重要節點）</div>', unsafe_allow_html=True)
            if centrality:
                cent_df = pd.DataFrame([
                    {"股票": t, "中介中心性": v["betweenness"],
                     "度中心性": v["degree"], "綜合": v["composite"]}
                    for t, v in centrality.items()
                ]).sort_values("綜合", ascending=False).head(10)
                st.dataframe(
                    cent_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "綜合": st.column_config.ProgressColumn(
                            "系統重要性", min_value=0, max_value=1, format="%.3f"
                        )
                    }
                )

    # ══════════════════════════════════════════════════════
    # TAB 4: LEAD-LAG
    # ══════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="terminal-card-title">Lead-Lag 領漲分析</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('<div class="terminal-card-title">Leader Score 排名</div>', unsafe_allow_html=True)
            if not ll_scores.empty:
                st.dataframe(
                    ll_scores.head(15),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "leader_score": st.column_config.ProgressColumn(
                            "Leader Score", min_value=0, max_value=float(ll_scores["leader_score"].max() or 1),
                            format="%.2f"
                        ),
                        "ticker":       st.column_config.TextColumn("股票"),
                        "lead_count":   st.column_config.NumberColumn("領先次數"),
                    }
                )
            else:
                st.caption("數據不足，無法計算 Lead-Lag")

        with c2:
            st.markdown('<div class="terminal-card-title">Lead-Lag 配對詳情</div>', unsafe_allow_html=True)
            if not ll_df.empty:
                display_ll = ll_df[ll_df["lag"] > 0].head(20).copy()
                display_ll["corr"] = display_ll["corr"].abs().round(3)
                st.dataframe(
                    display_ll,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "leader":   st.column_config.TextColumn("領先"),
                        "follower": st.column_config.TextColumn("跟隨"),
                        "lag":      st.column_config.NumberColumn("滯後期"),
                        "corr":     st.column_config.ProgressColumn(
                            "相關強度", min_value=0, max_value=1, format="%.3f"
                        ),
                    }
                )

        # 資金流向
        st.divider()
        st.markdown('<div class="terminal-card-title">板塊資金流向</div>', unsafe_allow_html=True)
        if flows:
            flow_df = pd.DataFrame([
                {"板塊": k, "平均漲跌%": v, "方向": "流入 ▲" if v > 0 else "流出 ▼"}
                for k, v in flows.items()
            ])
            st.dataframe(
                flow_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "平均漲跌%": st.column_config.NumberColumn(format="%+.2f%%"),
                    "方向":      st.column_config.TextColumn("方向"),
                }
            )

    # ══════════════════════════════════════════════════════
    # TAB 5: RISK
    # ══════════════════════════════════════════════════════
    with tab5:
        st.markdown('<div class="terminal-card-title">市場風險儀表板</div>', unsafe_allow_html=True)

        # 整體風險分數
        high_count = len([s for s in signals if s["severity"] == "high"])
        med_count  = len([s for s in signals if s["severity"] == "med"])
        risk_score = min(100, high_count * 25 + med_count * 10)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("風險分數", f"{risk_score}/100",
                  delta=f"{'⚠️ 高風險' if risk_score >= 50 else '✅ 正常'}")
        m2.metric("高級警報", str(high_count))
        m3.metric("中級警報", str(med_count))
        m4.metric("系統重要節點", str(len(risk_nodes)))

        st.divider()

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('<div class="terminal-card-title">所有警報信號</div>', unsafe_allow_html=True)
            render_signals(signals)

        with c2:
            st.markdown('<div class="terminal-card-title">波動率爆炸偵測</div>', unsafe_allow_html=True)
            vol_alerts = detect_vol_explosion(returns)
            if vol_alerts:
                vol_df = pd.DataFrame([
                    {"股票": t, "波動率倍數": v} for t, v in vol_alerts.items()
                ])
                st.dataframe(
                    vol_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "波動率倍數": st.column_config.ProgressColumn(
                            "波動率倍數", min_value=1, max_value=5, format="%.1fx"
                        )
                    }
                )
            else:
                st.success("✅ 暫無波動率異常")

            st.markdown('<div class="terminal-card-title" style="margin-top:16px;">市場恐慌指標</div>', unsafe_allow_html=True)
            panic = detect_market_panic(quotes_df)
            p1, p2, p3 = st.columns(3)
            p1.metric("VXX", f"{panic.get('vxx_chg', 0):+.2f}%")
            p2.metric("SPY",  f"{panic.get('spy_chg', 0):+.2f}%")
            p3.metric("恐慌", "🔴 是" if panic.get("is_panic") else "🟢 否")

            st.markdown('<div class="terminal-card-title" style="margin-top:16px;">Telegram 測試</div>', unsafe_allow_html=True)
            if st.button("📲 發送測試通知", use_container_width=True):
                from modules.telegram_module import send_telegram
                test_msg = f"🧪 *測試通知*\nRegime: `{regime}`\n時間: {last_update}"
                ok = send_telegram(test_msg)
                st.success("✅ 已發送") if ok else st.error("❌ 發送失敗（請檢查 API 設定）")

    # ══════════════════════════════════════════════════════
    # TAB 6: AI 分析
    # ══════════════════════════════════════════════════════
    with tab6:
        st.markdown('<div class="terminal-card-title">🤖 AI 市場結構分析</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([3, 1])
        with c2:
            if st.button("🔄 生成 AI 分析", use_container_width=True, type="primary"):
                leaders_list = ll_scores["ticker"].tolist()[:5] if not ll_scores.empty else []
                communities_str = " | ".join([
                    f"C{cid}: {', '.join(members[:3])}"
                    for cid, members in sorted(communities.items())
                ][:4])

                with st.spinner("AI 分析中..."):
                    summary = generate_market_summary(
                        regime=regime,
                        flows=flows,
                        signals=signals,
                        leaders=leaders_list,
                        risk_info={"panic": detect_market_panic(quotes_df), "risk_score": risk_score},
                        cluster_info=communities_str,
                        timeframe=st.session_state["timeframe"],
                    )
                    st.session_state["ai_summary"] = summary

        with c1:
            summary_text = st.session_state.get("ai_summary", "")
            if summary_text:
                st.markdown(f"""
                <div class="terminal-card" style="border-color:{UI_COLORS['accent_purple']}40;">
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;line-height:1.8;color:{UI_COLORS['text_primary']};">
                        {summary_text.replace(chr(10), '<br>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="terminal-card" style="text-align:center;padding:40px;">
                    <div style="color:{UI_COLORS['text_secondary']};font-family:IBM Plex Mono;font-size:13px;">
                        點擊右側「生成 AI 分析」按鈕<br>
                        <span style="font-size:11px;margin-top:8px;display:block;">
                            需要 Groq API Key（免費）<br>
                            若無 API Key，系統將使用本地摘要
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # 數據摘要
        st.divider()
        st.markdown('<div class="terminal-card-title">數據快照</div>', unsafe_allow_html=True)
        snap1, snap2, snap3, snap4 = st.columns(4)
        snap1.metric("分析股票", len([t for t in tickers if t in returns.columns]))
        snap2.metric("數據點", len(returns))
        snap3.metric("相關邊數", G.number_of_edges())
        snap4.metric("聚類數", len(set(partition.values())))

    # ── 自動刷新計數器 ────────────────────────────────────
    if st.session_state["auto_refresh"]:
        interval = st.session_state["refresh_interval"]
        st.markdown(
            f'<div style="text-align:center;color:{UI_COLORS["text_secondary"]};'
            f'font-family:IBM Plex Mono;font-size:10px;margin-top:20px;">'
            f'自動刷新: 每 {interval} 秒 | 下次刷新倒數中...</div>',
            unsafe_allow_html=True,
        )
        time.sleep(interval)
        st.rerun()


# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
