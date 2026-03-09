"""
個人化股票回測與選股分析 Web App
技術棧: Streamlit + yfinance + Plotly
指標: 純 pandas/numpy 實作 (無第三方 TA 套件依賴)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 頁面設定
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="股票回測分析平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

/* ── 全域背景：乾淨白底 ── */
.stApp {
    background: #f8fafc;
    color: #1e293b;
    font-family: 'Inter', sans-serif;
}

/* ── 側邊欄：淺灰底帶左邊框 ── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 2px solid #e2e8f0;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #0369a1;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

/* ── 主標題 ── */
.main-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.03em;
    margin-bottom: 0.1rem;
}
.main-title span {
    color: #0369a1;
}
.subtitle {
    color: #94a3b8;
    font-size: 0.85rem;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.04em;
    margin-bottom: 1.5rem;
}

/* ── Metric 卡片 ── */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-top: 3px solid #0369a1;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 8px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s, border-top-color 0.2s;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 4px 20px rgba(3,105,161,0.12);
    border-top-color: #0ea5e9;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.45rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── 策略標籤 ── */
.strategy-badge {
    display: inline-block;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1d4ed8;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    padding: 3px 12px;
    border-radius: 20px;
    letter-spacing: 0.1em;
    margin-bottom: 0.8rem;
    font-weight: 600;
}

/* ── VCP 通過/未通過 ── */
.vcp-pass {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #15803d;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
}
.vcp-fail {
    background: #fff7ed;
    border: 1px solid #fdba74;
    border-left: 4px solid #f97316;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #c2410c;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
}

/* ── 執行按鈕 ── */
.stButton > button {
    background: #0369a1;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 0.65rem 1.5rem;
    width: 100%;
    transition: background 0.2s, box-shadow 0.2s;
    letter-spacing: 0.02em;
}
.stButton > button:hover {
    background: #0284c7;
    box-shadow: 0 4px 14px rgba(3,105,161,0.3);
}

/* ── Tab ── */
[data-testid="stTab"] button {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    color: #64748b;
}
[data-testid="stTab"] button[aria-selected="true"] {
    color: #0369a1;
    border-bottom-color: #0369a1;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    overflow: hidden;
}

/* ── Info box ── */
.stAlert {
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
}

/* ── Input / Select ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    font-family: 'IBM Plex Mono', monospace;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    color: #0f172a;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# 指標計算（純 pandas/numpy，無第三方 TA 套件）
# ═══════════════════════════════════════════════════════

def sma(series: pd.Series, period: int) -> pd.Series:
    """簡單移動平均"""
    return series.rolling(window=period).mean()

def ema(series: pd.Series, period: int) -> pd.Series:
    """指數移動平均"""
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI 相對強弱指標"""
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_bbands(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    """布林通道：上軌、中軌、下軌"""
    mid   = sma(series, period)
    sigma = series.rolling(period).std()
    return mid + std_mult * sigma, mid, mid - std_mult * sigma

def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    """MACD 指標"""
    macd_line   = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist        = macd_line - signal_line
    return macd_line, signal_line, hist


# ═══════════════════════════════════════════════════════
# 數據與策略函數
# ═══════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """從 Yahoo Finance 下載數據並快取 1 小時"""
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as e:
        st.error(f"數據下載失敗: {e}")
        return pd.DataFrame()


def compute_indicators(df: pd.DataFrame, strategy: str, params: dict) -> pd.DataFrame:
    """根據策略計算指標，並統一計算 MA150、MA200"""
    df = df.copy()
    c  = df["Close"].squeeze()

    if strategy == "MA 交叉策略":
        df[f"MA{params['ma_fast']}"] = sma(c, params["ma_fast"])
        df[f"MA{params['ma_slow']}"] = sma(c, params["ma_slow"])

    elif strategy == "RSI 動能策略":
        df["RSI"] = calc_rsi(c, params["rsi_period"])

    elif strategy == "布林通道策略":
        df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = calc_bbands(
            c, params["bb_period"], params["bb_std"])

    elif strategy == "MACD 趨勢策略":
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = calc_macd(
            c, params["macd_fast"], params["macd_slow"], params["macd_signal"])

    # VCP 用指標
    df["MA150"] = sma(c, 150)
    df["MA200"] = sma(c, 200)

    return df.dropna()


def generate_signals(df: pd.DataFrame, strategy: str, params: dict) -> pd.DataFrame:
    """產生買(1)/賣(-1)/持有(0) 信號"""
    df = df.copy()
    df["Signal"] = 0
    c = df["Close"].squeeze()

    if strategy == "MA 交叉策略":
        f, s = f"MA{params['ma_fast']}", f"MA{params['ma_slow']}"
        df["Signal"] = np.where(
            (df[f] > df[s]) & (df[f].shift(1) <= df[s].shift(1)), 1,
            np.where((df[f] < df[s]) & (df[f].shift(1) >= df[s].shift(1)), -1, 0))

    elif strategy == "RSI 動能策略":
        df["Signal"] = np.where(
            (df["RSI"] < params["rsi_buy"]) & (df["RSI"].shift(1) >= params["rsi_buy"]), 1,
            np.where((df["RSI"] > params["rsi_sell"]) & (df["RSI"].shift(1) <= params["rsi_sell"]), -1, 0))

    elif strategy == "布林通道策略":
        df["Signal"] = np.where(
            (c <= df["BB_Lower"]) & (c.shift(1) > df["BB_Lower"].shift(1)), 1,
            np.where((c >= df["BB_Upper"]) & (c.shift(1) < df["BB_Upper"].shift(1)), -1, 0))

    elif strategy == "MACD 趨勢策略":
        df["Signal"] = np.where(
            (df["MACD"] > df["MACD_Signal"]) & (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1)), 1,
            np.where((df["MACD"] < df["MACD_Signal"]) & (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1)), -1, 0))

    return df


def check_vcp(df: pd.DataFrame) -> dict:
    """VCP 四條件檢查"""
    c = df["Close"].squeeze()
    last = df.iloc[-1]
    c_a = float(c.iloc[-1]) > float(last["MA200"])
    c_b = float(last["MA150"]) > float(last["MA200"])
    c_c = float(c.iloc[-1]) >= float(c.tail(252).min()) * 1.25
    r4  = float(c.tail(20).max() - c.tail(20).min())
    rp  = float(c.tail(40).head(20).max() - c.tail(40).head(20).min())
    c_d = r4 < rp * 0.5 if rp > 0 else False
    return {"close_above_ma200": c_a, "ma150_above_ma200": c_b,
            "above_52w_low_25pct": c_c, "volatility_contraction": c_d,
            "passed": all([c_a, c_b, c_c, c_d])}


def run_backtest(df: pd.DataFrame, inv_cfg: dict) -> dict:
    """
    支援多種投入方式的回測引擎：
    - inv_cfg["initial"]      : 初始資金
    - inv_cfg["mode"]         : "lump_sum" | "dca"
    - inv_cfg["dca_amount"]   : 每期定投金額（DCA 模式）
    - inv_cfg["dca_day"]      : 每月幾日投入（DCA 模式）
    - inv_cfg["buy_mode"]     : "all_in" | "fixed_amount" | "fixed_pct"
    - inv_cfg["buy_amount"]   : 每次買入固定金額
    - inv_cfg["buy_pct"]      : 每次買入比例 (0~1)
    - inv_cfg["sell_mode"]    : "all_out" | "fixed_amount" | "fixed_pct"
    - inv_cfg["sell_amount"]  : 每次賣出固定金額（市值）
    - inv_cfg["sell_pct"]     : 每次賣出比例 (0~1)
    """
    c              = df["Close"].squeeze()
    initial        = inv_cfg["initial"]
    mode           = inv_cfg.get("mode", "lump_sum")
    dca_amount     = inv_cfg.get("dca_amount", 0)
    dca_freq       = inv_cfg.get("dca_freq", 1)   # 每月幾次（1/2/4）
    buy_mode       = inv_cfg.get("buy_mode", "all_in")
    buy_amount     = inv_cfg.get("buy_amount", initial)
    buy_pct        = inv_cfg.get("buy_pct", 1.0)
    sell_mode      = inv_cfg.get("sell_mode", "all_out")
    sell_amount    = inv_cfg.get("sell_amount", 0)
    sell_pct       = inv_cfg.get("sell_pct", 1.0)

    capital        = float(initial)
    position       = 0.0
    in_market      = False
    total_invested = float(initial)
    dca_invested   = 0.0
    portfolio_values   = []
    invested_values    = []
    buy_dates, sell_dates   = [], []
    buy_prices, sell_prices = [], []
    buy_amounts_list        = []

    # 根據頻率計算每期間隔天數：1次≈30天, 2次≈15天, 4次≈7天
    dca_interval = max(1, 30 // dca_freq)
    last_dca_date = None   # 上次實際投入日期

    for i, (idx, row) in enumerate(df.iterrows()):
        price  = float(c.iloc[i])
        signal = int(row["Signal"])

        # ── 定投：按間隔天數注入資金（不依賴特定日期，確保每期都執行）──
        if mode == "dca" and dca_amount > 0:
            do_dca = (last_dca_date is None) or ((idx - last_dca_date).days >= dca_interval)
            if do_dca:
                capital        += dca_amount
                total_invested += dca_amount
                dca_invested   += dca_amount
                last_dca_date   = idx

        # ── 買入邏輯 ──
        if signal == 1:
            if buy_mode == "all_in":
                use_cash = capital
            elif buy_mode == "fixed_amount":
                use_cash = min(buy_amount, capital)
            else:  # fixed_pct
                use_cash = capital * buy_pct

            if use_cash > 1.0 and not in_market:
                shares    = use_cash / price
                position += shares
                capital  -= use_cash
                in_market = True
                buy_dates.append(idx)
                buy_prices.append(price)
                buy_amounts_list.append(use_cash)

        # ── 賣出邏輯 ──
        elif signal == -1 and in_market:
            if sell_mode == "all_out":
                sell_shares = position
            elif sell_mode == "fixed_amount":
                sell_shares = min(sell_amount / price, position)
            else:  # fixed_pct
                sell_shares = position * sell_pct

            proceeds  = sell_shares * price
            position -= sell_shares
            capital  += proceeds
            if position < 1e-6:
                in_market = False
            sell_dates.append(idx)
            sell_prices.append(price)

        portfolio_values.append(capital + position * price)
        invested_values.append(total_invested)

    final_value   = capital + position * float(c.iloc[-1])
    ps            = pd.Series(portfolio_values, index=df.index)
    inv_series    = pd.Series(invested_values,  index=df.index)
    years         = (df.index[-1] - df.index[0]).days / 365.25
    total_return  = (final_value - total_invested) / total_invested
    cagr          = (final_value / total_invested) ** (1 / max(years, 0.01)) - 1
    drawdown      = (ps - ps.cummax()) / ps.cummax()

    return {
        "portfolio_series": ps,
        "invested_series":  inv_series,
        "final_value":      final_value,
        "total_invested":   total_invested,
        "dca_invested":     dca_invested,
        "total_return":     total_return,
        "cagr":             cagr,
        "max_drawdown":     float(drawdown.min()),
        "drawdown_series":  drawdown,
        "buy_dates":        buy_dates,
        "sell_dates":       sell_dates,
        "buy_prices":       buy_prices,
        "sell_prices":      sell_prices,
    }


def compute_benchmark(df: pd.DataFrame, inv_cfg: dict) -> tuple:
    """
    計算買入持有基準線（支援 DCA）。
    回傳 (portfolio_series, total_invested)
    """
    c          = df["Close"].squeeze()
    initial    = inv_cfg["initial"]
    mode       = inv_cfg.get("mode", "lump_sum")
    dca_amount = inv_cfg.get("dca_amount", 0)
    dca_freq   = inv_cfg.get("dca_freq", 1)

    # 初始全倉買入
    total_invested = float(initial)
    shares         = float(initial) / float(c.iloc[0])
    dca_interval   = max(1, 30 // dca_freq)
    last_dca_date  = None
    values         = []

    for i, idx in enumerate(df.index):
        price = float(c.iloc[i])
        if mode == "dca" and dca_amount > 0:
            do_dca = (last_dca_date is None) or ((idx - last_dca_date).days >= dca_interval)
            if do_dca:
                shares         += dca_amount / price
                total_invested += dca_amount
                last_dca_date   = idx
        values.append(shares * price)

    return pd.Series(values, index=df.index), total_invested


# ═══════════════════════════════════════════════════════
# 圖表
# ═══════════════════════════════════════════════════════
CHART = dict(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
             font=dict(color="#475569", family="IBM Plex Mono, monospace", size=11),
             gridcolor="#e2e8f0")

def plot_equity(portfolio, benchmark, strategy_name, total_invested, invested_series=None):
    fig = go.Figure()
    # 累計投入成本線（DCA 模式為爬升線，一次性為水平線）
    if invested_series is not None:
        fig.add_trace(go.Scatter(x=invested_series.index, y=invested_series.values,
            name="累計投入成本", line=dict(color="#f59e0b", width=1.5, dash="longdash"),
            opacity=0.8))
    else:
        fig.add_hline(y=total_invested, line_dash="dash", line_color="#f59e0b",
                      line_width=1.2, annotation_text="投入成本",
                      annotation_font_color="#f59e0b")
    # 買入持有基準
    fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values,
        name="買入持有 (B&H)", line=dict(color="#94a3b8", width=1.5, dash="dot")))
    # 策略資金曲線
    fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio.values,
        name=strategy_name, line=dict(color="#0369a1", width=2.5),
        fill="tozeroy", fillcolor="rgba(3,105,161,0.07)"))
    fig.update_layout(title=dict(text="📊 資產增長曲線對比（藍=策略 / 灰虛=B&H / 橘虛=累計投入）",
                                 font=dict(size=13, color="#0f172a")),
        xaxis=dict(showgrid=True, gridcolor=CHART["gridcolor"]),
        yaxis=dict(showgrid=True, gridcolor=CHART["gridcolor"], tickprefix="$", tickformat=",.0f"),
        paper_bgcolor=CHART["paper_bgcolor"], plot_bgcolor=CHART["plot_bgcolor"],
        font=CHART["font"], hovermode="x unified", height=420,
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#e2e8f0", borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10,r=10,t=55,b=10))
    return fig


def plot_candlestick(df, strategy, params, buy_dates, sell_dates, buy_prices, sell_prices):
    need_sub = strategy in ["RSI 動能策略", "MACD 趨勢策略"]
    rows = 3 if need_sub else 2
    row_heights = [0.55, 0.2, 0.25] if need_sub else [0.65, 0.35]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=row_heights)
    c = df["Close"].squeeze()

    fig.add_trace(go.Candlestick(x=df.index,
        open=df["Open"].squeeze(), high=df["High"].squeeze(),
        low=df["Low"].squeeze(), close=c, name="K線",
        increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
        increasing_fillcolor="#bbf7d0", decreasing_fillcolor="#fecaca"), row=1, col=1)

    if strategy == "MA 交叉策略":
        fig.add_trace(go.Scatter(x=df.index, y=df[f"MA{params['ma_fast']}"],
            name=f"MA{params['ma_fast']}", line=dict(color="#0369a1", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[f"MA{params['ma_slow']}"],
            name=f"MA{params['ma_slow']}", line=dict(color="#7c3aed", width=1.5)), row=1, col=1)
    elif strategy == "布林通道策略":
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB上軌",
            line=dict(color="#6366f1", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Mid"], name="BB中軌",
            line=dict(color="#94a3b8", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB下軌",
            line=dict(color="#6366f1", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(99,102,241,0.05)"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], name="MA200",
        line=dict(color="#f59e0b", width=1.2, dash="longdash"), opacity=0.9), row=1, col=1)

    if buy_dates:
        fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode="markers", name="買入",
            marker=dict(symbol="triangle-up", color="#16a34a", size=12,
                        line=dict(color="#fff", width=1))), row=1, col=1)
    if sell_dates:
        fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode="markers", name="賣出",
            marker=dict(symbol="triangle-down", color="#dc2626", size=12,
                        line=dict(color="#fff", width=1))), row=1, col=1)

    vol_colors = ["#16a34a" if float(c.iloc[i]) >= float(df["Open"].squeeze().iloc[i])
                  else "#dc2626" for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"].squeeze(),
        name="成交量", marker_color=vol_colors, opacity=0.5), row=2, col=1)

    if strategy == "RSI 動能策略" and need_sub:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
            line=dict(color="#7c3aed", width=1.5)), row=3, col=1)
        fig.add_hline(y=params["rsi_buy"], line_dash="dash", line_color="#16a34a", row=3, col=1)
        fig.add_hline(y=params["rsi_sell"], line_dash="dash", line_color="#dc2626", row=3, col=1)
    elif strategy == "MACD 趨勢策略" and need_sub:
        hist_colors = ["#16a34a" if float(v) >= 0 else "#dc2626" for v in df["MACD_Hist"].values]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Hist",
            marker_color=hist_colors, opacity=0.6), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
            line=dict(color="#0369a1", width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
            line=dict(color="#f59e0b", width=1.2)), row=3, col=1)

    fig.update_layout(paper_bgcolor=CHART["paper_bgcolor"], plot_bgcolor=CHART["plot_bgcolor"],
        font=CHART["font"], xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0", borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hovermode="x unified", height=600 if need_sub else 500,
        margin=dict(l=10,r=10,t=40,b=10))
    for i in range(1, rows+1):
        fig.update_xaxes(showgrid=True, gridcolor=CHART["gridcolor"], zeroline=False, row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=CHART["gridcolor"], zeroline=False, row=i, col=1)
    return fig


# ═══════════════════════════════════════════════════════
# 主程式
# ═══════════════════════════════════════════════════════

def main():
    st.markdown('<div class="main-title">📈 股票<span>回測</span>分析平台</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Multi-Strategy Backtesting · VCP Screening · Performance Analytics</div>',
                unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🎯 數據設定")
        ticker = st.text_input("股票代號", value="VOO", help="支援 VOO, QQQ, AAPL, COST 等").upper().strip()
        col_y1, col_y2 = st.columns(2)
        years_back = col_y1.selectbox("回測年數", [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100], index=3)
        end_date   = col_y2.date_input("截止日期", value=datetime.today())
        end_dt     = datetime.combine(end_date, datetime.min.time())
        try:
            start_dt = end_dt.replace(year=end_dt.year - years_back)
        except ValueError:
            start_dt = end_dt.replace(year=end_dt.year - years_back, day=28)
        start_date = start_dt.strftime("%Y-%m-%d")
        end_str    = end_date.strftime("%Y-%m-%d")
        # ── 初始資金 ──
        initial_capital = st.number_input("初始資金 (USD)", min_value=0,
            max_value=10_000_000, value=100_000, step=10_000, format="%d")

        # ── 投入方式 ──
        st.markdown("### 💰 投入方式")
        inv_mode = st.radio("選擇投入模式", ["一次性投入", "定期定額 (DCA)"],
            horizontal=True,
            help="一次性：回測起始日全額買入。定期定額：每月固定日注入資金，模擬長期積累。")

        dca_amount, dca_freq = 0, 1
        if inv_mode == "定期定額 (DCA)":
            dc1, dc2 = st.columns(2)
            dca_amount = dc1.number_input("每次投入 ($)", min_value=100,
                max_value=1_000_000, value=3_000, step=500, format="%d")
            dca_freq = dc2.selectbox("每月投入次數", [1, 2, 4],
                format_func=lambda x: {1: "1次（月投）", 2: "2次（雙週投）", 4: "4次（週投）"}[x],
                help="1次=每月初投入一次；2次=每兩週投入一次；4次=每週投入一次")
            # 估算總投入
            total_est = initial_capital + dca_amount * dca_freq * 12 * years_back
            st.info(f"💡 預估總投入：${total_est:,.0f}　（初始 ${initial_capital:,} + ${dca_amount:,} × {dca_freq}次/月 × {years_back}年）")

        # ── 每次買入倉位 ──
        st.markdown("### 🎚️ 倉位控制")
        buy_mode_label = st.selectbox("買入方式",
            ["全倉買入", "固定金額", "固定比例 %"],
            help="每次觸發買入信號時，要用多少資金進場。")
        buy_amount, buy_pct = initial_capital, 1.0
        if buy_mode_label == "固定金額":
            buy_amount = st.number_input("每次買入金額 ($)", min_value=100,
                max_value=10_000_000, value=min(10_000, initial_capital), step=1_000, format="%d")
        elif buy_mode_label == "固定比例 %":
            buy_pct = st.slider("買入比例", 5, 100, 100, step=5,
                format="%d%%", help="佔當前可用資金的百分比") / 100

        sell_mode_label = st.selectbox("賣出方式",
            ["全倉賣出", "固定金額", "固定比例 %"],
            help="每次觸發賣出信號時，要賣出多少持倉。")
        sell_amount, sell_pct = 0, 1.0
        if sell_mode_label == "固定金額":
            sell_amount = st.number_input("每次賣出金額 ($)", min_value=100,
                max_value=10_000_000, value=10_000, step=1_000, format="%d")
        elif sell_mode_label == "固定比例 %":
            sell_pct = st.slider("賣出比例", 5, 100, 100, step=5,
                format="%d%%") / 100

        # 彙整投資設定
        inv_cfg = {
            "initial":     initial_capital,
            "mode":        "dca" if inv_mode == "定期定額 (DCA)" else "lump_sum",
            "dca_amount":  dca_amount,
            "dca_freq":    dca_freq,
            "buy_mode":    {"全倉買入": "all_in", "固定金額": "fixed_amount", "固定比例 %": "fixed_pct"}[buy_mode_label],
            "buy_amount":  buy_amount,
            "buy_pct":     buy_pct,
            "sell_mode":   {"全倉賣出": "all_out", "固定金額": "fixed_amount", "固定比例 %": "fixed_pct"}[sell_mode_label],
            "sell_amount": sell_amount,
            "sell_pct":    sell_pct,
        }

        st.markdown("---")
        st.markdown("### 📐 策略選擇")

        STRATEGY_HELP = {
            "MA 交叉策略":
                "移動平均線交叉策略：當短期均線（快線）向上穿越長期均線（慢線）時買入（黃金交叉），"
                "反之向下穿越時賣出（死亡交叉）。適合趨勢明顯的市場，震盪盤容易產生假信號。",
            "RSI 動能策略":
                "相對強弱指標：RSI 低於超賣門檻（預設 30）時買入，高於超買門檻（預設 70）時賣出。"
                "衡量價格動能強弱，適合震盪行情，強趨勢中可能過早賣出。",
            "布林通道策略":
                "布林帶均值回歸：股價觸碰下軌（均值 - 2σ）時視為超跌買入，觸碰上軌時視為超漲賣出。"
                "以統計標準差衡量波動率，適合區間震盪股，突破行情時效果較差。",
            "MACD 趨勢策略":
                "MACD 指標：由快慢 EMA 差值構成，當 MACD 線上穿信號線時買入，下穿時賣出。"
                "兼顧趨勢方向與動能變化，是最常用的趨勢追蹤指標之一。",
        }
        strategy = st.selectbox("選擇回測策略",
            list(STRATEGY_HELP.keys()),
            help=STRATEGY_HELP.get("MA 交叉策略"))  # 預設說明
        # 在策略說明框中顯示當前策略詳細說明
        with st.expander("📖 策略說明", expanded=False):
            st.caption(STRATEGY_HELP[strategy])

        params = {}
        with st.expander("⚙️ 策略參數設定", expanded=True):
            if strategy == "MA 交叉策略":
                params["ma_fast"] = st.slider("快線 MA", 5, 100, 50)
                params["ma_slow"] = st.slider("慢線 MA", 50, 300, 200)
            elif strategy == "RSI 動能策略":
                params["rsi_period"] = st.slider("RSI 週期", 5, 30, 14)
                params["rsi_buy"]    = st.slider("超賣門檻（買入）", 10, 40, 30)
                params["rsi_sell"]   = st.slider("超買門檻（賣出）", 60, 90, 70)
            elif strategy == "布林通道策略":
                params["bb_period"] = st.slider("布林週期", 5, 50, 20)
                params["bb_std"]    = st.select_slider("標準差倍數", [1.5, 2.0, 2.5, 3.0], value=2.0)
            elif strategy == "MACD 趨勢策略":
                params["macd_fast"]   = st.slider("MACD 快線", 5, 20, 12)
                params["macd_slow"]   = st.slider("MACD 慢線", 15, 50, 26)
                params["macd_signal"] = st.slider("MACD 信號線", 5, 20, 9)

        st.markdown("---")
        st.markdown("### 🔬 VCP 篩選器")
        enable_vcp = st.toggle("開啟 VCP 趨勢檢查", value=False)

        st.markdown("---")
        run_btn = st.button("🚀 執行回測分析", type="primary")

    if not run_btn:
        st.info("👈 請在左側設定參數後，點擊「執行回測分析」開始分析。", icon="💡")
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown("**MA 交叉策略**\n\n快慢均線黃金/死亡交叉，捕捉趨勢轉換。")
        col2.markdown("**RSI 動能策略**\n\n超賣買入、超買賣出，適合震盪行情。")
        col3.markdown("**布林通道策略**\n\n觸碰下軌買入、上軌賣出，依賴均值回歸。")
        col4.markdown("**MACD 趨勢策略**\n\nMACD 交叉追蹤動能方向。")
        return

    with st.spinner(f"正在下載 {ticker} 數據..."):
        df_raw = fetch_data(ticker, start_date, end_str)

    if df_raw.empty:
        st.error(f"❌ 無法取得 **{ticker}** 的數據，請確認股票代號是否正確。")
        return

    with st.spinner("計算技術指標中..."):
        df = compute_indicators(df_raw, strategy, params)

    if df.empty or len(df) < 50:
        st.error("數據不足，請增加回測年數或更換股票。")
        return

    df = generate_signals(df, strategy, params)

    if enable_vcp:
        vcp = check_vcp(df)
        st.markdown("#### 🔬 VCP 波動收縮模式檢查")
        v1, v2, v3, v4 = st.columns(4)
        v1.markdown(f"{'✅' if vcp['close_above_ma200'] else '❌'} **收盤 > MA200**")
        v2.markdown(f"{'✅' if vcp['ma150_above_ma200'] else '❌'} **MA150 > MA200**")
        v3.markdown(f"{'✅' if vcp['above_52w_low_25pct'] else '❌'} **高於52週低 +25%**")
        v4.markdown(f"{'✅' if vcp['volatility_contraction'] else '❌'} **波動率收縮**")
        if vcp["passed"]:
            st.markdown('<div class="vcp-pass">✅ VCP 條件全部通過 — 策略信號有效啟用</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="vcp-fail">⛔ VCP 條件未完全通過 — 策略信號已被過濾</div>', unsafe_allow_html=True)
            df["Signal"] = 0
        st.markdown("---")

    result              = run_backtest(df, inv_cfg)
    benchmark, bh_total = compute_benchmark(df, inv_cfg)
    years               = (df.index[-1] - df.index[0]).days / 365.25
    bh_return           = (float(benchmark.iloc[-1]) - bh_total) / bh_total
    bh_cagr             = (float(benchmark.iloc[-1]) / bh_total) ** (1 / max(years, 0.01)) - 1
    bh_dd               = float((benchmark / benchmark.cummax() - 1).min())

    c = df["Close"].squeeze()
    i1, i2, i3, i4, i5 = st.columns(5)
    i1.metric("股票代號", ticker)
    i2.metric("最新收盤價", f"${float(c.iloc[-1]):,.2f}")
    i3.metric("回測期間漲跌", f"{(float(c.iloc[-1])-float(c.iloc[0]))/float(c.iloc[0]):+.1%}")
    i4.metric("數據起始", df.index[0].strftime("%Y-%m-%d"))
    i5.metric("數據截止", df.index[-1].strftime("%Y-%m-%d"))
    st.markdown("---")

    # 投入摘要列
    inv_mode_label = "定期定額 (DCA)" if inv_cfg["mode"] == "dca" else "一次性投入"
    st.markdown(f'<div class="strategy-badge">STRATEGY: {strategy.upper()} ・ {inv_mode_label}</div>',
                unsafe_allow_html=True)

    # 投入金額摘要
    ci1, ci2, ci3 = st.columns(3)
    ci1.metric("初始投入", f"${inv_cfg['initial']:,.0f}")
    ci2.metric("定投累計" if inv_cfg["mode"] == "dca" else "一次性投入",
               f"${result['dca_invested']:,.0f}" if inv_cfg["mode"] == "dca" else f"${inv_cfg['initial']:,.0f}")
    ci3.metric("總投入資金", f"${result['total_invested']:,.0f}",
               delta=f"B&H 總投入 ${bh_total:,.0f}")
    st.markdown("---")

    st.markdown("#### 📊 績效指標對比")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("策略總報酬 (on投入)", f"{result['total_return']:+.2%}", delta=f"vs B&H {bh_return:+.2%}")
    m2.metric("策略 CAGR",  f"{result['cagr']:+.2%}",         delta=f"vs B&H {bh_cagr:+.2%}")
    m3.metric("策略最大回撤", f"{result['max_drawdown']:.2%}", delta=f"vs B&H {bh_dd:.2%}", delta_color="inverse")
    m4.metric("最終資產價值", f"${result['final_value']:,.0f}")
    m5.metric("交易次數", f"{len(result['buy_dates'])} 買 / {len(result['sell_dates'])} 賣")
    m6.metric("B&H 最終價值", f"${float(benchmark.iloc[-1]):,.0f}")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 資產增長曲線", "🕯️ K線圖與買賣標記", "📉 回撤分析"])

    with tab1:
        fig_eq = plot_equity(result["portfolio_series"], benchmark, strategy,
                             result["total_invested"], result.get("invested_series"))
        st.plotly_chart(fig_eq, use_container_width=True)
    with tab2:
        st.plotly_chart(plot_candlestick(df, strategy, params,
            result["buy_dates"], result["sell_dates"],
            result["buy_prices"], result["sell_prices"]), use_container_width=True)
    with tab3:
        bh_dd_series = benchmark / benchmark.cummax() - 1
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=result["drawdown_series"].index,
            y=result["drawdown_series"].values * 100, name=strategy,
            line=dict(color="#dc2626", width=1.5),
            fill="tozeroy", fillcolor="rgba(220,38,38,0.08)"))
        fig_dd.add_trace(go.Scatter(x=bh_dd_series.index,
            y=bh_dd_series.values * 100, name="買入持有",
            line=dict(color="#94a3b8", width=1.2, dash="dot")))
        fig_dd.update_layout(
            title=dict(text="📉 策略 vs 買入持有 回撤對比", font=dict(size=14, color="#0f172a")),
            xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
            yaxis=dict(showgrid=True, gridcolor="#e2e8f0", ticksuffix="%"),
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", font=CHART["font"],
            legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0", borderwidth=1),
            hovermode="x unified", height=380, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig_dd, use_container_width=True)

    with st.expander("📋 查看所有交易明細"):
        trades = []
        for i, (bd, bp) in enumerate(zip(result["buy_dates"], result["buy_prices"])):
            sd = result["sell_dates"][i] if i < len(result["sell_dates"]) else "持倉中"
            sp = result["sell_prices"][i] if i < len(result["sell_prices"]) else float(c.iloc[-1])
            trades.append({
                "買入日期": bd.strftime("%Y-%m-%d") if hasattr(bd, "strftime") else str(bd),
                "買入價格": f"${bp:,.2f}",
                "賣出日期": sd.strftime("%Y-%m-%d") if hasattr(sd, "strftime") else str(sd),
                "賣出價格": f"${sp:,.2f}",
                "單筆報酬": f"{(sp-bp)/bp:+.2%}",
            })
        if trades:
            st.dataframe(pd.DataFrame(trades), use_container_width=True, hide_index=True)
        else:
            st.info("此策略在回測期間未產生任何交易信號。")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#cbd5e1;font-family:IBM Plex Mono;font-size:0.7rem;'>"
        "數據來源: Yahoo Finance · 本工具僅供學習研究，不構成投資建議</div>",
        unsafe_allow_html=True)

if __name__ == "__main__":
    main()
