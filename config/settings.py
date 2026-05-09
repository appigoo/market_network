# config/settings.py
# ============================================================
# 市場網絡分析系統 - 全局配置
# ============================================================

# ── 預設股票池 ──────────────────────────────────────────────
DEFAULT_WATCHLIST = {
    "Market Core": ["SPY", "QQQ", "IWM"],
    "AI Engine": ["NVDA", "TSLA", "MSFT", "AVGO", "SMH"],
    "Liquidity & Risk": ["TLT", "VXX", "HYG", "UUP"],
    "Sector Rotation": ["XLF", "XLE", "XLV", "XLY"],
    "Macro Risk": ["GLD", "IBIT", "FXI"],
}

# 所有預設股票合并
ALL_DEFAULT_TICKERS = []
for group in DEFAULT_WATCHLIST.values():
    ALL_DEFAULT_TICKERS.extend(group)

# ── 板塊顏色 ──────────────────────────────────────────────
SECTOR_COLORS = {
    "Market Core":      "#00D4FF",   # 青藍
    "AI Engine":        "#7C3AED",   # 紫色
    "Liquidity & Risk": "#EF4444",   # 紅色
    "Sector Rotation":  "#10B981",   # 綠色
    "Macro Risk":       "#F59E0B",   # 琥珀
    "Custom":           "#94A3B8",   # 灰色
}

# ── 時間週期 ──────────────────────────────────────────────
TIMEFRAMES = {
    "1m":  {"period": "1d",  "interval": "1m",  "label": "1分鐘"},
    "5m":  {"period": "5d",  "interval": "5m",  "label": "5分鐘"},
    "15m": {"period": "5d",  "interval": "15m", "label": "15分鐘"},
    "1h":  {"period": "1mo", "interval": "1h",  "label": "1小時"},
    "1d":  {"period": "1y",  "interval": "1d",  "label": "日線"},
}

# ── 相關性設定 ──────────────────────────────────────────────
CORR_WINDOWS = {
    "短期 (20)":  20,
    "中期 (60)":  60,
    "長期 (120)": 120,
}

CORR_THRESHOLDS = {
    "strong_pos":  0.70,
    "weak_pos":    0.30,
    "weak_neg":   -0.30,
    "strong_neg": -0.70,
}

# ── 市場 Cap 估算（用於 Node 大小）──────────────────────────
MARKET_CAP_B = {
    "NVDA": 2900, "MSFT": 3100, "AVGO": 780,
    "TSLA": 780,  "SMH":  20,
    "SPY":  500,  "QQQ":  250,  "IWM": 80,
    "TLT":  40,   "VXX":  5,    "HYG": 20,   "UUP": 5,
    "XLF":  40,   "XLE":  30,   "XLV": 35,   "XLY": 20,
    "GLD":  65,   "IBIT": 40,   "FXI": 8,
}

# 預設市值（未知股票）
DEFAULT_MARKET_CAP_B = 50

# ── 風險偵測閾值 ──────────────────────────────────────────
RISK_THRESHOLDS = {
    "corr_spike":     0.85,   # 相關性爆升
    "vol_explosion":  0.04,   # 波動率爆炸 (4% daily move)
    "panic_vxx_pct":  0.10,   # VXX 上漲 10% = 恐慌
}

# ── UI 顏色系統 ──────────────────────────────────────────
UI_COLORS = {
    "bg_primary":    "#050810",
    "bg_secondary":  "#0A0F1E",
    "bg_card":       "#0D1424",
    "accent_blue":   "#00D4FF",
    "accent_purple": "#7C3AED",
    "accent_green":  "#00FF9F",
    "accent_red":    "#FF3366",
    "accent_amber":  "#FFB800",
    "text_primary":  "#E2E8F0",
    "text_secondary":"#94A3B8",
    "grid":          "#1E293B",
    "border":        "#1E2D4A",
}

# ── Groq AI ──────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MAX_TOKENS = 1500
