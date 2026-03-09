"""
個人化股票回測與選股分析 Web App
作者: Claude (Anthropic)
技術棧: Streamlit + yfinance + pandas-ta + Plotly
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 頁面基本設定
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="股票回測分析平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 全域 CSS 美化（深色量化交易風格）
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* 全域背景 */
.stApp {
    background: #0a0e17;
    color: #e2e8f0;
    font-family: 'DM Sans', sans-serif;
}

/* 側邊欄 */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1321 0%, #111827 100%);
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #38bdf8;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* 主標題 */
.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #e879f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #64748b;
    font-size: 0.9rem;
    font-family: 'DM Sans', sans-serif;
    margin-bottom: 1.5rem;
}

/* Metric 卡片 */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827 0%, #1e293b 100%);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: #38bdf8;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'Space Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.5rem !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* 分隔線 */
.section-divider {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 1.5rem 0;
}

/* 策略說明標籤 */
.strategy-badge {
    display: inline-block;
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #38bdf8;
    color: #38bdf8;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.1em;
    margin-bottom: 0.8rem;
}

/* VCP 狀態提示 */
.vcp-pass {
    background: linear-gradient(135deg, #052e16, #064e3b);
    border: 1px solid #10b981;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #34d399;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
}
.vcp-fail {
    background: linear-gradient(135deg, #1c0a0a, #2d1515);
    border: 1px solid #ef4444;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #f87171;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
}

/* 按鈕 */
.stButton > button {
    background: linear-gradient(135deg, #0369a1, #7c3aed);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    padding: 0.6rem 1.5rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover {
    opacity: 0.85;
}

/* Tab 樣式 */
[data-testid="stTab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
}

/* 警告/資訊框 */
.stAlert {
    background: #111827;
    border-radius: 8px;
}

/* 圖表容器 */
.chart-container {
    background: #0d1321;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# 工具函數
# ═══════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    從 Yahoo Finance 下載股票數據，並快取 1 小時。
    Returns OHLCV DataFrame，index 為日期。
    """
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        # 攤平多層欄位（若有）
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception as e:
        st.error(f"數據下載失敗: {e}")
        return pd.DataFrame()


def compute_indicators(df: pd.DataFrame, strategy: str, params: dict) -> pd.DataFrame:
    """
    根據選擇的策略計算對應技術指標。
    """
    df = df.copy()
    close = df["Close"].squeeze()

    if strategy == "MA 交叉策略":
        fast = params["ma_fast"]
        slow = params["ma_slow"]
        df[f"MA{fast}"] = ta.sma(close, length=fast)
        df[f"MA{slow}"] = ta.sma(close, length=slow)

    elif strategy == "RSI 動能策略":
        period = params["rsi_period"]
        df["RSI"] = ta.rsi(close, length=period)

    elif strategy == "布林通道策略":
        period = params["bb_period"]
        std = params["bb_std"]
        bb = ta.bbands(close, length=period, std=std)
        if bb is not None:
            df["BB_Upper"] = bb[f"BBU_{period}_{float(std)}"]
            df["BB_Mid"]   = bb[f"BBM_{period}_{float(std)}"]
            df["BB_Lower"] = bb[f"BBL_{period}_{float(std)}"]

    elif strategy == "MACD 趨勢策略":
        fast = params["macd_fast"]
        slow = params["macd_slow"]
        sig  = params["macd_signal"]
        macd = ta.macd(close, fast=fast, slow=slow, signal=sig)
        if macd is not None:
            df["MACD"]        = macd[f"MACD_{fast}_{slow}_{sig}"]
            df["MACD_Signal"] = macd[f"MACDs_{fast}_{slow}_{sig}"]
            df["MACD_Hist"]   = macd[f"MACDh_{fast}_{slow}_{sig}"]

    # 通用指標：MA150、MA200（用於 VCP 篩選）
    df["MA150"] = ta.sma(close, length=150)
    df["MA200"] = ta.sma(close, length=200)

    return df.dropna()


def generate_signals(df: pd.DataFrame, strategy: str, params: dict) -> pd.DataFrame:
    """
    根據策略規則產生買(1)/賣(-1)/持有(0) 信號。
    """
    df = df.copy()
    df["Signal"] = 0

    close = df["Close"].squeeze()

    if strategy == "MA 交叉策略":
        fast_col = f"MA{params['ma_fast']}"
        slow_col = f"MA{params['ma_slow']}"
        # 黃金交叉 → 買入；死亡交叉 → 賣出
        df["Signal"] = np.where(
            (df[fast_col] > df[slow_col]) & (df[fast_col].shift(1) <= df[slow_col].shift(1)), 1,
            np.where(
                (df[fast_col] < df[slow_col]) & (df[fast_col].shift(1) >= df[slow_col].shift(1)), -1, 0
            )
        )

    elif strategy == "RSI 動能策略":
        buy_thresh  = params["rsi_buy"]
        sell_thresh = params["rsi_sell"]
        df["Signal"] = np.where(
            (df["RSI"] < buy_thresh) & (df["RSI"].shift(1) >= buy_thresh), 1,
            np.where(
                (df["RSI"] > sell_thresh) & (df["RSI"].shift(1) <= sell_thresh), -1, 0
            )
        )

    elif strategy == "布林通道策略":
        df["Signal"] = np.where(
            (close <= df["BB_Lower"]) & (close.shift(1) > df["BB_Lower"].shift(1)), 1,
            np.where(
                (close >= df["BB_Upper"]) & (close.shift(1) < df["BB_Upper"].shift(1)), -1, 0
            )
        )

    elif strategy == "MACD 趨勢策略":
        df["Signal"] = np.where(
            (df["MACD"] > df["MACD_Signal"]) & (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1)), 1,
            np.where(
                (df["MACD"] < df["MACD_Signal"]) & (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1)), -1, 0
            )
        )

    return df


def check_vcp(df: pd.DataFrame) -> dict:
    """
    VCP（波動收縮模式）篩選條件檢查。
    回傳各條件是否通過，及整體是否通過。
    """
    close = df["Close"].squeeze()
    latest = df.iloc[-1]

    # 條件 a: 收盤價 > 200MA
    c_a = float(close.iloc[-1]) > float(latest["MA200"])

    # 條件 b: 150MA > 200MA
    c_b = float(latest["MA150"]) > float(latest["MA200"])

    # 條件 c: 股價高於 52 週低點的 25% 以上
    low_52 = float(close.tail(252).min())
    c_c = float(close.iloc[-1]) >= low_52 * 1.25

    # 條件 d: 波動率收縮（近 4 週區間 vs 前期區間）
    recent_4w  = close.tail(20)
    prior_4w   = close.tail(40).head(20)
    range_recent = float(recent_4w.max() - recent_4w.min())
    range_prior  = float(prior_4w.max() - prior_4w.min())
    c_d = range_recent < range_prior * 0.5 if range_prior > 0 else False

    return {
        "close_above_ma200": c_a,
        "ma150_above_ma200": c_b,
        "above_52w_low_25pct": c_c,
        "volatility_contraction": c_d,
        "passed": all([c_a, c_b, c_c, c_d]),
    }


def run_backtest(df: pd.DataFrame, initial_capital: float) -> dict:
    """
    根據 Signal 欄執行回測，計算資金曲線、總報酬、CAGR、最大回撤。
    """
    df = df.copy()
    close = df["Close"].squeeze()

    capital    = initial_capital
    position   = 0.0   # 持有股數
    in_market  = False
    portfolio_values = []
    buy_dates, sell_dates = [], []
    buy_prices, sell_prices = [], []

    for i, (idx, row) in enumerate(df.iterrows()):
        price  = float(close.iloc[i])
        signal = int(row["Signal"])

        if signal == 1 and not in_market:
            # 買入：全倉
            position  = capital / price
            capital   = 0.0
            in_market = True
            buy_dates.append(idx)
            buy_prices.append(price)

        elif signal == -1 and in_market:
            # 賣出：全倉
            capital   = position * price
            position  = 0.0
            in_market = False
            sell_dates.append(idx)
            sell_prices.append(price)

        # 當日組合價值
        portfolio_values.append(capital + position * price)

    # 若結束時仍持倉，以最後收盤價計算
    if in_market:
        last_price = float(close.iloc[-1])
        capital    = position * last_price

    final_value = capital + position * float(close.iloc[-1]) if not in_market else capital

    # 計算資金曲線
    portfolio_series = pd.Series(portfolio_values, index=df.index)

    # 年化報酬率 (CAGR)
    years = (df.index[-1] - df.index[0]).days / 365.25
    total_return = (final_value - initial_capital) / initial_capital
    cagr = (1 + total_return) ** (1 / max(years, 0.01)) - 1

    # 最大回撤
    rolling_max   = portfolio_series.cummax()
    drawdown      = (portfolio_series - rolling_max) / rolling_max
    max_drawdown  = float(drawdown.min())

    return {
        "portfolio_series": portfolio_series,
        "final_value": final_value,
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "buy_dates": buy_dates,
        "sell_dates": sell_dates,
        "buy_prices": buy_prices,
        "sell_prices": sell_prices,
        "drawdown_series": drawdown,
    }


def compute_benchmark(df: pd.DataFrame, initial_capital: float) -> pd.Series:
    """
    計算買入持有基準線（Buy & Hold）的資金曲線。
    """
    close = df["Close"].squeeze()
    shares = initial_capital / float(close.iloc[0])
    return close * shares


# ═══════════════════════════════════════════════════════
# 圖表繪製函數
# ═══════════════════════════════════════════════════════

CHART_THEME = dict(
    paper_bgcolor="#0a0e17",
    plot_bgcolor="#0d1321",
    font=dict(color="#94a3b8", family="Space Mono, monospace", size=11),
    gridcolor="#1e293b",
    zerolinecolor="#1e293b",
)


def plot_equity_curve(portfolio: pd.Series, benchmark: pd.Series,
                      strategy_name: str, initial_capital: float) -> go.Figure:
    """
    資產增長曲線對比圖（策略 vs 買入持有）。
    """
    fig = go.Figure()

    # 買入持有基準線
    fig.add_trace(go.Scatter(
        x=benchmark.index, y=benchmark.values,
        name="買入持有 (B&H)",
        line=dict(color="#64748b", width=1.5, dash="dot"),
        fill=None,
    ))

    # 策略資金曲線
    fig.add_trace(go.Scatter(
        x=portfolio.index, y=portfolio.values,
        name=strategy_name,
        line=dict(color="#38bdf8", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.05)",
    ))

    # 初始資金基準線
    fig.add_hline(y=initial_capital, line_dash="dash",
                  line_color="#374151", line_width=1,
                  annotation_text="初始資金", annotation_font_color="#64748b")

    fig.update_layout(
        title=dict(text="📊 資產增長曲線對比", font=dict(size=14, color="#e2e8f0")),
        xaxis=dict(showgrid=True, gridcolor=CHART_THEME["gridcolor"], zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=CHART_THEME["gridcolor"],
                   tickprefix="$", tickformat=",.0f"),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e293b",
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def plot_drawdown(drawdown: pd.Series) -> go.Figure:
    """
    最大回撤曲線圖。
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values * 100,
        name="回撤 %",
        line=dict(color="#f87171", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(248, 113, 113, 0.12)",
    ))
    fig.update_layout(
        title=dict(text="📉 回撤曲線", font=dict(size=14, color="#e2e8f0")),
        xaxis=dict(showgrid=True, gridcolor=CHART_THEME["gridcolor"]),
        yaxis=dict(showgrid=True, gridcolor=CHART_THEME["gridcolor"],
                   ticksuffix="%", tickformat=".1f"),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        height=220,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def plot_candlestick(df: pd.DataFrame, strategy: str, params: dict,
                     buy_dates: list, sell_dates: list,
                     buy_prices: list, sell_prices: list) -> go.Figure:
    """
    K線圖 + 指標層 + 買賣標記。
    根據策略動態決定子圖數量。
    """
    close = df["Close"].squeeze()

    need_sub = strategy in ["RSI 動能策略", "MACD 趨勢策略"]
    rows = 3 if need_sub else 2
    row_heights = [0.55, 0.2, 0.25] if need_sub else [0.65, 0.35]
    subplot_titles_list = ["K線圖與指標", "成交量"]
    if need_sub:
        subplot_titles_list.append(strategy.split(" ")[0])

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        subplot_titles=subplot_titles_list,
    )

    # ── K 線 ──
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"].squeeze(), high=df["High"].squeeze(),
        low=df["Low"].squeeze(),  close=close,
        name="K線",
        increasing_line_color="#10b981",
        decreasing_line_color="#f87171",
        increasing_fillcolor="#052e16",
        decreasing_fillcolor="#2d1515",
    ), row=1, col=1)

    # ── 指標線（主圖）──
    if strategy == "MA 交叉策略":
        fig.add_trace(go.Scatter(
            x=df.index, y=df[f"MA{params['ma_fast']}"],
            name=f"MA{params['ma_fast']}", line=dict(color="#38bdf8", width=1.2),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df[f"MA{params['ma_slow']}"],
            name=f"MA{params['ma_slow']}", line=dict(color="#e879f9", width=1.2),
        ), row=1, col=1)

    elif strategy == "布林通道策略":
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"], name="BB 上軌",
            line=dict(color="#818cf8", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Mid"], name="BB 中軌",
            line=dict(color="#94a3b8", width=1),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"], name="BB 下軌",
            line=dict(color="#818cf8", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(129, 140, 248, 0.04)",
        ), row=1, col=1)

    # MA200（所有策略都顯示）
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA200"], name="MA200",
        line=dict(color="#fbbf24", width=1, dash="longdash"), opacity=0.7,
    ), row=1, col=1)

    # ── 買賣標記 ──
    if buy_dates:
        fig.add_trace(go.Scatter(
            x=buy_dates, y=buy_prices,
            mode="markers",
            name="買入",
            marker=dict(symbol="triangle-up", color="#10b981", size=12,
                        line=dict(color="#fff", width=1)),
        ), row=1, col=1)
    if sell_dates:
        fig.add_trace(go.Scatter(
            x=sell_dates, y=sell_prices,
            mode="markers",
            name="賣出",
            marker=dict(symbol="triangle-down", color="#f87171", size=12,
                        line=dict(color="#fff", width=1)),
        ), row=1, col=1)

    # ── 成交量 ──
    colors_vol = ["#10b981" if float(close.iloc[i]) >= float(df["Open"].squeeze().iloc[i])
                  else "#f87171" for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"].squeeze(),
        name="成交量", marker_color=colors_vol, opacity=0.6,
    ), row=2, col=1)

    # ── 子指標圖（RSI / MACD）──
    if strategy == "RSI 動能策略" and need_sub:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"], name="RSI",
            line=dict(color="#a78bfa", width=1.5),
        ), row=3, col=1)
        fig.add_hline(y=params["rsi_buy"],  line_dash="dash", line_color="#10b981",
                      row=3, col=1, annotation_text=f"超賣 {params['rsi_buy']}",
                      annotation_font_color="#10b981")
        fig.add_hline(y=params["rsi_sell"], line_dash="dash", line_color="#f87171",
                      row=3, col=1, annotation_text=f"超買 {params['rsi_sell']}",
                      annotation_font_color="#f87171")

    elif strategy == "MACD 趨勢策略" and need_sub:
        colors_hist = ["#10b981" if float(v) >= 0 else "#f87171"
                       for v in df["MACD_Hist"].values]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_Hist"],
            name="MACD Hist", marker_color=colors_hist, opacity=0.7,
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], name="MACD",
            line=dict(color="#38bdf8", width=1.5),
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_Signal"], name="Signal",
            line=dict(color="#f59e0b", width=1.2),
        ), row=3, col=1)

    # ── 圖表樣式 ──
    fig.update_layout(
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=1.01, xanchor="right", x=1),
        hovermode="x unified",
        height=600 if need_sub else 500,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    for i in range(1, rows + 1):
        fig.update_xaxes(showgrid=True, gridcolor=CHART_THEME["gridcolor"],
                         zeroline=False, row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=CHART_THEME["gridcolor"],
                         zeroline=False, row=i, col=1)

    return fig


# ═══════════════════════════════════════════════════════
# 主應用程式
# ═══════════════════════════════════════════════════════

def main():
    # ── 標題 ──
    st.markdown('<div class="main-title">📈 股票回測分析平台</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Multi-Strategy Backtesting · VCP Screening · Performance Analytics</div>',
                unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # 側邊欄：參數設定
    # ══════════════════════════════════════════
    with st.sidebar:
        st.markdown("### 🎯 數據設定")

        ticker = st.text_input("股票代號", value="VOO",
                               help="支援 VOO, QQQ, AAPL, COST, MSFT 等").upper().strip()

        col_y1, col_y2 = st.columns(2)
        years_back = col_y1.selectbox("回測年數", [1, 2, 3, 5, 7, 10], index=3)
        end_date   = col_y2.date_input("截止日期", value=datetime.today())

        start_date = (datetime.combine(end_date, datetime.min.time())
                      - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")
        end_str    = end_date.strftime("%Y-%m-%d")

        initial_capital = st.number_input(
            "初始投資金額 (USD)", min_value=1000, max_value=10_000_000,
            value=100_000, step=10_000, format="%d",
        )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### 📐 策略選擇")

        strategy = st.selectbox(
            "選擇回測策略",
            ["MA 交叉策略", "RSI 動能策略", "布林通道策略", "MACD 趨勢策略"],
        )

        # ── 策略參數 ──
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

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### 🔬 VCP 篩選器")
        enable_vcp = st.toggle("開啟 VCP 趨勢檢查", value=False,
                               help="波動收縮模式篩選，需滿足 4 個條件才啟動策略")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        run_btn = st.button("🚀 執行回測分析", type="primary")

    # ══════════════════════════════════════════
    # 主內容區
    # ══════════════════════════════════════════
    if not run_btn:
        # 歡迎引導介面
        st.info("👈 請在左側設定參數後，點擊「執行回測分析」開始分析。", icon="💡")

        col1, col2, col3, col4 = st.columns(4)
        col1.markdown("""
        **MA 交叉策略**
        利用快慢移動平均線的黃金/死亡交叉，捕捉趨勢轉換點。
        """)
        col2.markdown("""
        **RSI 動能策略**
        超賣區間買入、超買區間賣出，適合震盪行情。
        """)
        col3.markdown("""
        **布林通道策略**
        價格觸碰下軌買入、上軌賣出，依賴統計回歸均值。
        """)
        col4.markdown("""
        **MACD 趨勢策略**
        MACD 線與信號線交叉，追蹤動能變化與趨勢方向。
        """)
        return

    # ── 載入數據 ──
    with st.spinner(f"正在下載 {ticker} 數據..."):
        df_raw = fetch_data(ticker, start_date, end_str)

    if df_raw.empty:
        st.error(f"❌ 無法取得 **{ticker}** 的數據，請確認股票代號是否正確。")
        return

    # ── 計算指標 ──
    with st.spinner("計算技術指標中..."):
        df = compute_indicators(df_raw, strategy, params)
        if df.empty or len(df) < 50:
            st.error("數據不足，請縮短回測時間或更換股票。")
            return

    # ── 生成信號 ──
    df = generate_signals(df, strategy, params)

    # ── VCP 檢查 ──
    if enable_vcp:
        vcp_result = check_vcp(df)
        st.markdown("#### 🔬 VCP 波動收縮模式檢查")
        vcol1, vcol2, vcol3, vcol4 = st.columns(4)

        def vcp_badge(label, passed):
            icon = "✅" if passed else "❌"
            return f"{icon} {label}"

        vcol1.markdown(f"**{vcp_badge('收盤 > MA200', vcp_result['close_above_ma200'])}**")
        vcol2.markdown(f"**{vcp_badge('MA150 > MA200', vcp_result['ma150_above_ma200'])}**")
        vcol3.markdown(f"**{vcp_badge('高於52週低 +25%', vcp_result['above_52w_low_25pct'])}**")
        vcol4.markdown(f"**{vcp_badge('波動率收縮', vcp_result['volatility_contraction'])}**")

        if vcp_result["passed"]:
            st.markdown('<div class="vcp-pass">✅ VCP 條件全部通過 — 策略信號有效啟用</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="vcp-fail">⛔ VCP 條件未完全通過 — 策略信號已被過濾，僅顯示分析結果</div>',
                        unsafe_allow_html=True)
            # VCP 未通過時清除交易信號
            df["Signal"] = 0

        st.markdown("---")

    # ── 執行回測 ──
    result    = run_backtest(df, initial_capital)
    benchmark = compute_benchmark(df, initial_capital)
    bh_return = (float(benchmark.iloc[-1]) - initial_capital) / initial_capital

    years = (df.index[-1] - df.index[0]).days / 365.25
    bh_cagr = (1 + bh_return) ** (1 / max(years, 0.01)) - 1

    # 基準最大回撤
    bh_dd = (benchmark / benchmark.cummax() - 1).min()

    # ── 股票基本資訊欄 ──
    info_col1, info_col2, info_col3, info_col4, info_col5 = st.columns(5)
    last_close = float(df["Close"].squeeze().iloc[-1])
    first_close = float(df["Close"].squeeze().iloc[0])
    period_chg = (last_close - first_close) / first_close

    info_col1.metric("股票代號", ticker)
    info_col2.metric("最新收盤價", f"${last_close:,.2f}")
    info_col3.metric("回測期間漲跌", f"{period_chg:+.1%}")
    info_col4.metric("數據起始", df.index[0].strftime("%Y-%m-%d"))
    info_col5.metric("數據截止", df.index[-1].strftime("%Y-%m-%d"))

    st.markdown("---")

    # ── 績效指標區 ──
    st.markdown(f'<div class="strategy-badge">STRATEGY: {strategy.upper()}</div>',
                unsafe_allow_html=True)
    st.markdown("#### 📊 績效指標對比")

    m1, m2, m3, m4, m5, m6 = st.columns(6)

    m1.metric(
        "策略總報酬",
        f"{result['total_return']:+.2%}",
        delta=f"vs B&H {bh_return:+.2%}",
    )
    m2.metric(
        "策略 CAGR",
        f"{result['cagr']:+.2%}",
        delta=f"vs B&H {bh_cagr:+.2%}",
    )
    m3.metric(
        "策略最大回撤",
        f"{result['max_drawdown']:.2%}",
        delta=f"vs B&H {bh_dd:.2%}",
        delta_color="inverse",
    )
    m4.metric("最終資產價值", f"${result['final_value']:,.0f}")
    m5.metric("交易次數",
              f"{len(result['buy_dates'])} 買 / {len(result['sell_dates'])} 賣")
    m6.metric("B&H 最終價值", f"${float(benchmark.iloc[-1]):,.0f}")

    st.markdown("---")

    # ── 圖表 Tab ──
    tab1, tab2, tab3 = st.tabs(["📈 資產增長曲線", "🕯️ K線圖與買賣標記", "📉 回撤分析"])

    with tab1:
        st.plotly_chart(
            plot_equity_curve(result["portfolio_series"], benchmark, strategy, initial_capital),
            use_container_width=True,
        )

    with tab2:
        st.plotly_chart(
            plot_candlestick(
                df, strategy, params,
                result["buy_dates"], result["sell_dates"],
                result["buy_prices"], result["sell_prices"],
            ),
            use_container_width=True,
        )

    with tab3:
        # 策略回撤 + 基準回撤
        bh_drawdown_series = (benchmark / benchmark.cummax() - 1)
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=result["drawdown_series"].index,
            y=result["drawdown_series"].values * 100,
            name=strategy, line=dict(color="#f87171", width=1.5),
            fill="tozeroy", fillcolor="rgba(248,113,113,0.1)",
        ))
        fig_dd.add_trace(go.Scatter(
            x=bh_drawdown_series.index,
            y=bh_drawdown_series.values * 100,
            name="買入持有", line=dict(color="#64748b", width=1.2, dash="dot"),
        ))
        fig_dd.update_layout(
            title=dict(text="📉 策略 vs 買入持有 回撤對比", font=dict(size=14, color="#e2e8f0")),
            xaxis=dict(showgrid=True, gridcolor="#1e293b"),
            yaxis=dict(showgrid=True, gridcolor="#1e293b",
                       ticksuffix="%", tickformat=".1f"),
            paper_bgcolor="#0a0e17", plot_bgcolor="#0d1321",
            font=CHART_THEME["font"],
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified", height=380,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    # ── 交易明細 ──
    with st.expander("📋 查看所有交易明細"):
        trades = []
        for i, (bd, bp) in enumerate(zip(result["buy_dates"], result["buy_prices"])):
            sd = result["sell_dates"][i] if i < len(result["sell_dates"]) else "持倉中"
            sp = result["sell_prices"][i] if i < len(result["sell_prices"]) else float(df["Close"].squeeze().iloc[-1])
            ret = (sp - bp) / bp if bp > 0 else 0
            trades.append({
                "買入日期": bd.strftime("%Y-%m-%d") if hasattr(bd, "strftime") else str(bd),
                "買入價格": f"${bp:,.2f}",
                "賣出日期": sd.strftime("%Y-%m-%d") if hasattr(sd, "strftime") else str(sd),
                "賣出價格": f"${sp:,.2f}",
                "單筆報酬": f"{ret:+.2%}",
            })
        if trades:
            st.dataframe(
                pd.DataFrame(trades),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("此策略在回測期間未產生任何交易信號。")

    # ── Footer ──
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#374151; font-family:Space Mono; font-size:0.7rem;'>"
        "數據來源: Yahoo Finance (yfinance) · 本工具僅供學習研究，不構成投資建議"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
