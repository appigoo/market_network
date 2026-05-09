# modules/graph_engine.py
# ============================================================
# Force Graph 視覺化引擎 - Plotly 交互式網絡圖
# ============================================================

import networkx as nx
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from config.settings import UI_COLORS


def _spring_layout_weighted(G: nx.Graph, iterations: int = 100) -> dict:
    """加權 Spring Layout（相關性越高距離越近）"""
    if G.number_of_nodes() == 0:
        return {}
    if G.number_of_nodes() == 1:
        return {list(G.nodes())[0]: np.array([0.0, 0.0])}

    # 使用 spring_layout，correlation 越高 → weight 越高 → 距離越近
    try:
        pos = nx.spring_layout(
            G,
            weight="weight",
            k=2.0 / np.sqrt(G.number_of_nodes()),
            iterations=iterations,
            seed=42,
        )
    except Exception:
        pos = nx.circular_layout(G)
    return pos


def build_force_graph(
    corr_matrix: pd.DataFrame,
    quotes_df: pd.DataFrame,
    node_colors: dict,
    corr_threshold: float = 0.3,
    show_negative: bool = True,
    highlight_nodes: list = None,
) -> go.Figure:
    """
    生成完整的 Force Graph Figure
    """
    if corr_matrix.empty:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=UI_COLORS["bg_primary"],
            plot_bgcolor=UI_COLORS["bg_primary"],
            title=dict(text="⚠️ 暫無數據", font=dict(color=UI_COLORS["text_secondary"])),
        )
        return fig

    # ── 1. 建立圖 ─────────────────────────────────────────
    G = nx.Graph()
    tickers = corr_matrix.columns.tolist()

    for t in tickers:
        G.add_node(t)

    # 邊
    edge_pos_list = []   # 正相關
    edge_neg_list = []   # 負相關

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            c = corr_matrix.iloc[i, j]
            if pd.isna(c) or abs(c) < corr_threshold:
                continue
            if c < 0 and not show_negative:
                continue
            G.add_edge(tickers[i], tickers[j], weight=float(abs(c)), raw_corr=float(c))
            if c >= 0:
                edge_pos_list.append((tickers[i], tickers[j], c))
            else:
                edge_neg_list.append((tickers[i], tickers[j], c))

    # ── 2. 佈局 ───────────────────────────────────────────
    pos = _spring_layout_weighted(G, iterations=150)

    # ── 3. 準備節點屬性 ───────────────────────────────────
    node_x, node_y, node_text, node_sizes, node_clrs = [], [], [], [], []
    node_hover = []
    highlight_nodes = highlight_nodes or []

    for t in G.nodes():
        if t not in pos:
            continue
        x, y = pos[t]
        node_x.append(x)
        node_y.append(y)

        # 大小 = market cap
        mcap = quotes_df.loc[t, "mcap_b"] if t in quotes_df.index else 50
        size = max(20, min(60, np.log1p(mcap) * 7))
        node_sizes.append(size)

        # 顏色 = 板塊
        node_clrs.append(node_colors.get(t, UI_COLORS["text_secondary"]))

        # 標籤
        node_text.append(t)

        # Hover
        chg  = quotes_df.loc[t, "chg_pct"] if t in quotes_df.index else 0
        price = quotes_df.loc[t, "price"]   if t in quotes_df.index else 0
        node_hover.append(
            f"<b>{t}</b><br>"
            f"Price: ${price:.2f}<br>"
            f"Change: {chg:+.2f}%<br>"
            f"Market Cap: ${mcap:.0f}B"
        )

    # ── 4. 繪製邊 ─────────────────────────────────────────
    traces = []

    def _edge_traces(edge_list, color_pos, color_neg, name_prefix):
        result = []
        for t1, t2, c in edge_list:
            if t1 not in pos or t2 not in pos:
                continue
            x0, y0 = pos[t1]
            x1, y1 = pos[t2]
            width = max(0.5, abs(c) * 4)
            col   = color_pos if c >= 0 else color_neg
            opacity = max(0.15, abs(c) * 0.8)
            result.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color=col),
                opacity=opacity,
                hoverinfo="skip",
                showlegend=False,
            ))
        return result

    # 正相關邊（青藍）
    traces.extend(_edge_traces(
        edge_pos_list,
        UI_COLORS["accent_blue"],
        UI_COLORS["accent_blue"],
        "pos",
    ))

    # 負相關邊（紅色）
    traces.extend(_edge_traces(
        edge_neg_list,
        UI_COLORS["accent_red"],
        UI_COLORS["accent_red"],
        "neg",
    ))

    # ── 5. 節點 Trace ─────────────────────────────────────
    # 節點顏色由 chg_pct 驅動（漲跌著色）
    node_chg = []
    for t in G.nodes():
        if t not in pos:
            continue
        node_chg.append(quotes_df.loc[t, "chg_pct"] if t in quotes_df.index else 0)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_chg,
            colorscale=[
                [0.0,  "#FF3366"],
                [0.3,  "#FF6B35"],
                [0.5,  "#1E293B"],
                [0.7,  "#00D4FF"],
                [1.0,  "#00FF9F"],
            ],
            cmin=-5,
            cmax=5,
            colorbar=dict(
                title=dict(text="漲跌 %", font=dict(color=UI_COLORS["text_secondary"], size=11)),
                thickness=12,
                len=0.6,
                tickfont=dict(color=UI_COLORS["text_secondary"], size=10),
            ),
            line=dict(width=1.5, color="rgba(255,255,255,0.2)"),
            showscale=True,
        ),
        text=node_text,
        textposition="middle center",
        textfont=dict(
            size=10,
            color=UI_COLORS["text_primary"],
            family="IBM Plex Mono, monospace",
        ),
        hovertext=node_hover,
        hoverinfo="text",
        hoverlabel=dict(
            bgcolor=UI_COLORS["bg_card"],
            bordercolor=UI_COLORS["border"],
            font=dict(color=UI_COLORS["text_primary"], size=12),
        ),
        name="Stocks",
        showlegend=False,
    )
    traces.append(node_trace)

    # ── 6. 高亮節點（風險高亮）───────────────────────────
    if highlight_nodes:
        hl_x, hl_y = [], []
        for t in highlight_nodes:
            if t in pos:
                x, y = pos[t]
                hl_x.append(x)
                hl_y.append(y)
        if hl_x:
            traces.append(go.Scatter(
                x=hl_x, y=hl_y,
                mode="markers",
                marker=dict(
                    size=70,
                    color="rgba(0,0,0,0)",
                    line=dict(width=2.5, color=UI_COLORS["accent_red"]),
                    symbol="circle",
                ),
                hoverinfo="skip",
                showlegend=False,
                name="Risk Alert",
            ))

    # ── 7. 佈局 ───────────────────────────────────────────
    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor=UI_COLORS["bg_primary"],
        plot_bgcolor=UI_COLORS["bg_primary"],
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(
            showgrid=False, zeroline=False,
            showticklabels=False,
            range=[-1.5, 1.5],
        ),
        yaxis=dict(
            showgrid=False, zeroline=False,
            showticklabels=False,
            range=[-1.5, 1.5],
        ),
        dragmode="pan",
        hovermode="closest",
        uirevision="stable",
        height=650,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=UI_COLORS["text_secondary"]),
        ),
        annotations=[
            dict(
                x=0.01, y=0.99, xref="paper", yref="paper",
                text="● 正相關  ● 負相關  大小=市值  顏色=漲跌%",
                showarrow=False,
                font=dict(size=10, color=UI_COLORS["text_secondary"],
                          family="IBM Plex Mono"),
                bgcolor="rgba(10,15,30,0.8)",
                bordercolor=UI_COLORS["border"],
                borderwidth=1,
                borderpad=6,
            )
        ],
    )
    return fig


def build_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """相關性熱力圖"""
    if corr_matrix.empty:
        return go.Figure()

    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale=[
            [0.0,  "#FF3366"],
            [0.25, "#FF6B35"],
            [0.5,  "#0D1424"],
            [0.75, "#00D4FF"],
            [1.0,  "#00FF9F"],
        ],
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=9, color="white", family="IBM Plex Mono"),
        hovertemplate="%{y} / %{x}<br>相關: %{z:.3f}<extra></extra>",
        colorbar=dict(
            title=dict(text="相關係數", font=dict(color=UI_COLORS["text_secondary"])),
            thickness=14,
            tickfont=dict(color=UI_COLORS["text_secondary"]),
        ),
    ))

    fig.update_layout(
        paper_bgcolor=UI_COLORS["bg_primary"],
        plot_bgcolor=UI_COLORS["bg_primary"],
        margin=dict(l=10, r=10, t=30, b=10),
        height=500,
        xaxis=dict(
            tickfont=dict(color=UI_COLORS["text_secondary"], size=9,
                          family="IBM Plex Mono"),
            gridcolor=UI_COLORS["grid"],
        ),
        yaxis=dict(
            tickfont=dict(color=UI_COLORS["text_secondary"], size=9,
                          family="IBM Plex Mono"),
            gridcolor=UI_COLORS["grid"],
        ),
    )
    return fig


def build_rolling_corr_chart(series: pd.Series, t1: str, t2: str) -> go.Figure:
    """滾動相關序列圖"""
    if series.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        line=dict(color=UI_COLORS["accent_blue"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.08)",
        name=f"{t1}/{t2}",
    ))
    fig.add_hline(y=0.7,  line=dict(dash="dash", color=UI_COLORS["accent_green"], width=1),
                  annotation_text="強正相關", annotation_font_color=UI_COLORS["accent_green"])
    fig.add_hline(y=-0.7, line=dict(dash="dash", color=UI_COLORS["accent_red"], width=1),
                  annotation_text="強負相關", annotation_font_color=UI_COLORS["accent_red"])
    fig.add_hline(y=0,    line=dict(color=UI_COLORS["grid"], width=1))

    fig.update_layout(
        paper_bgcolor=UI_COLORS["bg_primary"],
        plot_bgcolor=UI_COLORS["bg_primary"],
        margin=dict(l=10, r=10, t=30, b=10),
        height=280,
        yaxis=dict(range=[-1.1, 1.1], gridcolor=UI_COLORS["grid"],
                   tickfont=dict(color=UI_COLORS["text_secondary"], size=10),
                   zerolinecolor=UI_COLORS["grid"]),
        xaxis=dict(gridcolor=UI_COLORS["grid"],
                   tickfont=dict(color=UI_COLORS["text_secondary"], size=10)),
        legend=dict(font=dict(color=UI_COLORS["text_secondary"])),
    )
    return fig
