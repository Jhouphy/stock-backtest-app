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


def run_backtest(df: pd.DataFrame, initial_capital: float) -> dict:
    """全倉模式回測"""
    c = df["Close"].squeeze()
    capital, position, in_market = initial_capital, 0.0, False
    portfolio_values = []
    buy_dates, sell_dates, buy_prices, sell_prices = [], [], [], []

    for i, (idx, row) in enumerate(df.iterrows()):
        price  = float(c.iloc[i])
        signal = int(row["Signal"])
        if signal == 1 and not in_market:
            position, capital, in_market = capital / price, 0.0, True
            buy_dates.append(idx); buy_prices.append(price)
        elif signal == -1 and in_market:
            capital, position, in_market = position * price, 0.0, False
            sell_dates.append(idx); sell_prices.append(price)
        portfolio_values.append(capital + position * price)

    final_value = capital + position * float(c.iloc[-1])
    ps = pd.Series(portfolio_values, index=df.index)
    years = (df.index[-1] - df.index[0]).days / 365.25
    total_return = (final_value - initial_capital) / initial_capital
    cagr = (1 + total_return) ** (1 / max(years, 0.01)) - 1
    drawdown = (ps - ps.cummax()) / ps.cummax()

    return {"portfolio_series": ps, "final_value": final_value,
            "total_return": total_return, "cagr": cagr,
            "max_drawdown": float(drawdown.min()), "drawdown_series": drawdown,
            "buy_dates": buy_dates, "sell_dates": sell_dates,
            "buy_prices": buy_prices, "sell_prices": sell_prices}


def compute_benchmark(df: pd.DataFrame, initial_capital: float) -> pd.Series:
    """買入持有基準線"""
    c = df["Close"].squeeze()
    return c * (initial_capital / float(c.iloc[0]))


# ═══════════════════════════════════════════════════════
# 圖表
# ═══════════════════════════════════════════════════════
CHART = dict(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
             font=dict(color="#475569", family="IBM Plex Mono, monospace", size=11),
             gridcolor="#e2e8f0")

def plot_equity(portfolio, benchmark, strategy_name, initial_capital):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values,
        name="買入持有 (B&H)", line=dict(color="#94a3b8", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio.values,
        name=strategy_name, line=dict(color="#0369a1", width=2.5),
        fill="tozeroy", fillcolor="rgba(3,105,161,0.07)"))
    fig.add_hline(y=initial_capital, line_dash="dash", line_color="#cbd5e1",
                  line_width=1, annotation_text="初始資金", annotation_font_color="#94a3b8")
    fig.update_layout(title=dict(text="📊 資產增長曲線對比", font=dict(size=14, color="#0f172a")),
        xaxis=dict(showgrid=True, gridcolor=CHART["gridcolor"]),
        yaxis=dict(showgrid=True, gridcolor=CHART["gridcolor"], tickprefix="$", tickformat=",.0f"),
        paper_bgcolor=CHART["paper_bgcolor"], plot_bgcolor=CHART["plot_bgcolor"],
        font=CHART["font"], hovermode="x unified", height=400,
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#e2e8f0", borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10,r=10,t=50,b=10))
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
        initial_capital = st.number_input("初始投資金額 (USD)", min_value=1000,
            max_value=10_000_000, value=100_000, step=10_000, format="%d")

        st.markdown("---")
        st.markdown("### 📐 策略選擇")
        strategy = st.selectbox("選擇回測策略",
            ["MA 交叉策略", "RSI 動能策略", "布林通道策略", "MACD 趨勢策略"])

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

    result    = run_backtest(df, initial_capital)
    benchmark = compute_benchmark(df, initial_capital)
    bh_return = (float(benchmark.iloc[-1]) - initial_capital) / initial_capital
    years     = (df.index[-1] - df.index[0]).days / 365.25
    bh_cagr   = (1 + bh_return) ** (1 / max(years, 0.01)) - 1
    bh_dd     = float((benchmark / benchmark.cummax() - 1).min())

    c = df["Close"].squeeze()
    i1, i2, i3, i4, i5 = st.columns(5)
    i1.metric("股票代號", ticker)
    i2.metric("最新收盤價", f"${float(c.iloc[-1]):,.2f}")
    i3.metric("回測期間漲跌", f"{(float(c.iloc[-1])-float(c.iloc[0]))/float(c.iloc[0]):+.1%}")
    i4.metric("數據起始", df.index[0].strftime("%Y-%m-%d"))
    i5.metric("數據截止", df.index[-1].strftime("%Y-%m-%d"))
    st.markdown("---")

    st.markdown(f'<div class="strategy-badge">STRATEGY: {strategy.upper()}</div>', unsafe_allow_html=True)
    st.markdown("#### 📊 績效指標對比")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("策略總報酬", f"{result['total_return']:+.2%}", delta=f"vs B&H {bh_return:+.2%}")
    m2.metric("策略 CAGR",  f"{result['cagr']:+.2%}",         delta=f"vs B&H {bh_cagr:+.2%}")
    m3.metric("策略最大回撤", f"{result['max_drawdown']:.2%}", delta=f"vs B&H {bh_dd:.2%}", delta_color="inverse")
    m4.metric("最終資產價值", f"${result['final_value']:,.0f}")
    m5.metric("交易次數", f"{len(result['buy_dates'])} 買 / {len(result['sell_dates'])} 賣")
    m6.metric("B&H 最終價值", f"${float(benchmark.iloc[-1]):,.0f}")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 資產增長曲線", "🕯️ K線圖與買賣標記", "📉 回撤分析"])

    with tab1:
        st.plotly_chart(plot_equity(result["portfolio_series"], benchmark, strategy, initial_capital),
                        use_container_width=True)
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
