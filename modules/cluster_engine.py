# modules/cluster_engine.py
# ============================================================
# 市場聚類引擎 - Louvain Community Detection
# ============================================================

import networkx as nx
import community as community_louvain  # python-louvain
import numpy as np
import pandas as pd
from config.settings import DEFAULT_WATCHLIST, SECTOR_COLORS


# ── 預定義板塊映射 ────────────────────────────────────────
KNOWN_SECTOR_MAP = {}
for sector, tickers in DEFAULT_WATCHLIST.items():
    for t in tickers:
        KNOWN_SECTOR_MAP[t] = sector

# 板塊標籤（AI 驅動的語義標籤）
CLUSTER_LABELS = {
    0: "AI Core",
    1: "Macro Hedge",
    2: "Risk Assets",
    3: "Defensive",
    4: "Liquidity",
    5: "Growth Tech",
    6: "Commodities",
    7: "Mixed",
}


def build_corr_graph(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.3,
) -> nx.Graph:
    """
    從相關矩陣建立 NetworkX 加權圖
    只添加 |corr| >= threshold 的邊
    """
    G = nx.Graph()
    tickers = corr_matrix.columns.tolist()

    # 添加節點
    for t in tickers:
        G.add_node(t)

    # 添加邊
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            c = corr_matrix.iloc[i, j]
            if not np.isnan(c) and abs(c) >= threshold:
                G.add_edge(tickers[i], tickers[j], weight=float(abs(c)), raw_corr=float(c))

    return G


def louvain_communities(G: nx.Graph, resolution: float = 1.0) -> dict:
    """
    Louvain 社群偵測
    返回: {ticker: community_id}
    """
    if G.number_of_edges() == 0:
        return {node: 0 for node in G.nodes()}

    try:
        partition = community_louvain.best_partition(G, resolution=resolution, random_state=42)
        return partition
    except Exception:
        # 回退：使用已知板塊映射
        return {t: list(DEFAULT_WATCHLIST.keys()).index(
            KNOWN_SECTOR_MAP.get(t, "Custom")) % 8
            for t in G.nodes()}


def assign_cluster_colors(partition: dict, tickers: list) -> dict:
    """
    為每個 ticker 分配聚類顏色
    優先使用已知板塊顏色，否則用聚類顏色
    """
    community_palette = [
        "#00D4FF", "#7C3AED", "#00FF9F", "#FF3366",
        "#FFB800", "#FF6B35", "#A855F7", "#06B6D4",
    ]

    colors = {}
    for t in tickers:
        # 優先使用已知板塊顏色
        sector = KNOWN_SECTOR_MAP.get(t)
        if sector and sector in SECTOR_COLORS:
            colors[t] = SECTOR_COLORS[sector]
        else:
            cid = partition.get(t, 0)
            colors[t] = community_palette[cid % len(community_palette)]

    return colors


def compute_cluster_stats(
    returns: pd.DataFrame,
    partition: dict,
) -> pd.DataFrame:
    """
    計算每個聚類的統計信息
    """
    records = []
    if returns.empty:
        return pd.DataFrame()

    communities = set(partition.values())
    for cid in communities:
        members = [t for t, c in partition.items() if c == cid and t in returns.columns]
        if not members:
            continue

        subset = returns[members]
        avg_ret = subset.mean().mean() * 100
        avg_vol = subset.std().mean() * np.sqrt(252) * 100
        if len(members) > 1:
            intra_corr = subset.corr().values.copy()
            np.fill_diagonal(intra_corr, np.nan)
            avg_corr = float(np.nanmean(intra_corr))
        else:
            avg_corr = 1.0

        records.append({
            "cluster_id": cid,
            "members":    members,
            "size":       len(members),
            "avg_return": round(avg_ret, 4),
            "avg_vol":    round(avg_vol, 2),
            "intra_corr": round(avg_corr, 3),
        })

    return pd.DataFrame(records).sort_values("intra_corr", ascending=False)


def get_sector_for_ticker(ticker: str) -> str:
    """返回 ticker 的已知板塊，否則返回 Custom"""
    return KNOWN_SECTOR_MAP.get(ticker, "Custom")


def compute_centrality(G: nx.Graph) -> dict:
    """計算圖中心性（系統性重要節點）"""
    if G.number_of_nodes() == 0:
        return {}
    try:
        bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
        dc = nx.degree_centrality(G)
        result = {}
        for n in G.nodes():
            result[n] = {
                "betweenness": round(bc.get(n, 0), 4),
                "degree":      round(dc.get(n, 0), 4),
                "composite":   round((bc.get(n, 0) + dc.get(n, 0)) / 2, 4),
            }
        return result
    except Exception:
        return {}
