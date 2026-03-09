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

    elif strategy == "MA均線偏離策略":
        # 計算買入均線、賣出均線，及與收盤價的偏離百分比
        bp = params["dev_buy_period"]
        sp = params["dev_sell_period"]
        df[f"BuyMA{bp}"]  = sma(c, bp)
        df[f"SellMA{sp}"] = sma(c, sp)
        # 偏離率：(收盤 - MA) / MA * 100
        df["Dev_Buy"]  = (c - df[f"BuyMA{bp}"])  / df[f"BuyMA{bp}"]  * 100
        df["Dev_Sell"] = (c - df[f"SellMA{sp}"]) / df[f"SellMA{sp}"] * 100

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

    elif strategy == "MA均線偏離策略":
        buy_th  = params["dev_buy_pct"]   # 低於均線 N%（負值或0）才買，例如 0 = 跌破均線即買
        sell_th = params["dev_sell_pct"]  # 高於均線 N%（正值）才賣，例如 10 = 漲超 10% 賣

        # ── State-based 狀態機（持倉狀態決定邏輯，不是瞬間穿越）──
        # 原本的 crossover 寫法只在「穿越那一天」才有訊號，
        # 若均線一直沒被穿越則永遠沒訊號，且 acc1/acc2 初始持現金也進不了場。
        # 改成：空倉時只要條件符合就買入；持倉時只要條件符合就賣出。
        dev_buy_col  = df["Dev_Buy"].values
        dev_sell_col = df["Dev_Sell"].values
        ma_buy_col   = df[f"BuyMA{params['dev_buy_period']}"].values
        signals_arr  = np.zeros(len(df), dtype=int)
        in_pos = False

        for idx_i in range(len(df)):
            # 跳過均線尚未形成（NaN）的期間
            if np.isnan(dev_buy_col[idx_i]) or np.isnan(dev_sell_col[idx_i]):
                continue
            if not in_pos:
                # 空倉：收盤偏離率 <= 買入門檻 → 買入
                if dev_buy_col[idx_i] <= buy_th:
                    signals_arr[idx_i] = 1
                    in_pos = True
            else:
                # 持倉：收盤偏離率 >= 賣出門檻 → 賣出
                if dev_sell_col[idx_i] >= sell_th:
                    signals_arr[idx_i] = -1
                    in_pos = False

        df["Signal"] = signals_arr

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
    4 帳戶完全獨立回測。
    acc1: 初始資金 + 策略信號（無 DCA）
    acc2: 初始資金 + DCA + 策略信號
    acc3/acc4 由 compute_benchmark 計算。
    """
    c          = df["Close"].squeeze()
    initial    = float(inv_cfg["initial"])
    mode       = inv_cfg.get("mode", "lump_sum")
    dca_amount = float(inv_cfg.get("dca_amount", 0))
    dca_freq   = inv_cfg.get("dca_freq", 1)
    buy_mode   = inv_cfg.get("buy_mode", "all_in")
    buy_amount = float(inv_cfg.get("buy_amount", initial))
    buy_pct    = float(inv_cfg.get("buy_pct", 1.0))
    sell_mode  = inv_cfg.get("sell_mode", "all_out")
    sell_amount= float(inv_cfg.get("sell_amount", 0))
    sell_pct   = float(inv_cfg.get("sell_pct", 1.0))

    # ── 預先計算 DCA 日期 ──
    trading_days = df.index
    dca_dates    = set()
    if mode == "dca" and dca_amount > 0:
        if dca_freq == 1:
            seen = set()
            for d in trading_days:
                ym = (d.year, d.month)
                if ym not in seen: dca_dates.add(d); seen.add(ym)
        elif dca_freq == 2:
            fm, sm = {}, {}
            for d in trading_days:
                ym = (d.year, d.month)
                if ym not in fm: fm[ym] = d
                if ym not in sm and d.day >= 15: sm[ym] = d
            dca_dates = set(fm.values()) | set(sm.values())
        elif dca_freq == 4:
            dca_dates = set(trading_days[::5])

    # ── acc1：初始資金 + 策略（無 DCA） ──
    a1_cash, a1_shares = initial, 0.0
    a1_invested        = initial

    # ── acc2：初始資金 + DCA + 策略 ──
    a2_cash, a2_shares = initial, 0.0
    a2_invested        = initial
    a2_dca_total       = 0.0

    # 追蹤用
    buy_dates,  sell_dates  = [], []   # acc1 策略信號（代表純策略）
    buy_prices, sell_prices = [], []
    dca_buy_dates  = []
    dca_buy_prices = []
    a1_vals, a2_vals = [], []
    a1_inv_vals, a2_inv_vals = [], []

    def do_buy(cash, shares, price):
        if buy_mode == "all_in":
            use = cash
        elif buy_mode == "fixed_amount":
            use = min(buy_amount, cash)
        else:
            use = cash * buy_pct
        if use > 1.0:
            shares += use / price
            cash   -= use
        return cash, shares

    def do_sell(cash, shares, price):
        if sell_mode == "all_out":
            sell_sh = shares
        elif sell_mode == "fixed_amount":
            sell_sh = min(sell_amount / price, shares)
        else:
            sell_sh = shares * sell_pct
        if sell_sh > 1e-6:
            cash   += sell_sh * price
            shares -= sell_sh
        return cash, shares

    for i, (idx, row) in enumerate(df.iterrows()):
        price  = float(c.iloc[i])
        signal = int(row["Signal"])

        # DCA：只影響 acc2
        if mode == "dca" and dca_amount > 0 and idx in dca_dates:
            a2_cash     += dca_amount
            a2_invested += dca_amount
            a2_dca_total+= dca_amount
            dca_buy_dates.append(idx)
            dca_buy_prices.append(price)

        # 賣出信號
        if signal == -1:
            if a1_shares > 1e-6:
                a1_cash, a1_shares = do_sell(a1_cash, a1_shares, price)
                sell_dates.append(idx); sell_prices.append(price)
            if a2_shares > 1e-6:
                a2_cash, a2_shares = do_sell(a2_cash, a2_shares, price)

        # 買入信號
        elif signal == 1:
            if a1_cash > 1.0:
                prev_shares = a1_shares
                a1_cash, a1_shares = do_buy(a1_cash, a1_shares, price)
                if a1_shares > prev_shares:
                    buy_dates.append(idx); buy_prices.append(price)
            if a2_cash > 1.0:
                a2_cash, a2_shares = do_buy(a2_cash, a2_shares, price)

        a1_vals.append(a1_cash + a1_shares * price)
        a2_vals.append(a2_cash + a2_shares * price)
        a1_inv_vals.append(a1_invested)
        a2_inv_vals.append(a2_invested)

    a1_final = a1_cash + a1_shares * float(c.iloc[-1])
    a2_final = a2_cash + a2_shares * float(c.iloc[-1])
    years    = (df.index[-1] - df.index[0]).days / 365.25

    def perf(final, inv):
        tr   = (final - inv) / inv
        cagr = (final / inv) ** (1 / max(years, 0.01)) - 1
        return tr, cagr

    a1_ps = pd.Series(a1_vals, index=df.index)
    a2_ps = pd.Series(a2_vals, index=df.index)
    a1_tr, a1_cagr = perf(a1_final, a1_invested)
    a2_tr, a2_cagr = perf(a2_final, a2_invested)

    return {
        # acc1 策略（無 DCA）
        "acc1_series":    a1_ps,
        "acc1_final":     a1_final,
        "acc1_invested":  a1_invested,
        "acc1_return":    a1_tr,
        "acc1_cagr":      a1_cagr,
        "acc1_drawdown":  float((a1_ps / a1_ps.cummax() - 1).min()),
        "acc1_dd_series": a1_ps / a1_ps.cummax() - 1,
        # acc2 策略＋DCA
        "acc2_series":    a2_ps,
        "acc2_final":     a2_final,
        "acc2_invested":  a2_invested,
        "acc2_return":    a2_tr,
        "acc2_cagr":      a2_cagr,
        "acc2_drawdown":  float((a2_ps / a2_ps.cummax() - 1).min()),
        "acc2_dd_series": a2_ps / a2_ps.cummax() - 1,
        "dca_invested":   a2_dca_total,
        # 交易記錄
        "buy_dates":      buy_dates,
        "sell_dates":     sell_dates,
        "buy_prices":     buy_prices,
        "sell_prices":    sell_prices,
        "dca_buy_dates":  dca_buy_dates,
        "dca_buy_prices": dca_buy_prices,
        # 投入成本序列（累計）
        "acc1_inv_series": pd.Series(a1_inv_vals, index=df.index),
        "acc2_inv_series": pd.Series(a2_inv_vals, index=df.index),
    }


def compute_benchmark(df: pd.DataFrame, inv_cfg: dict) -> dict:
    """
    acc3: 初始資金 + DCA + 買入持有（永不賣）
    acc4: 初始資金 + 買入持有（永不賣）
    """
    c          = df["Close"].squeeze()
    initial    = float(inv_cfg["initial"])
    mode       = inv_cfg.get("mode", "lump_sum")
    dca_amount = float(inv_cfg.get("dca_amount", 0))
    dca_freq   = inv_cfg.get("dca_freq", 1)

    trading_days = df.index
    dca_dates    = set()
    if mode == "dca" and dca_amount > 0:
        if dca_freq == 1:
            seen = set()
            for d in trading_days:
                ym = (d.year, d.month)
                if ym not in seen: dca_dates.add(d); seen.add(ym)
        elif dca_freq == 2:
            fm, sm = {}, {}
            for d in trading_days:
                ym = (d.year, d.month)
                if ym not in fm: fm[ym] = d
                if ym not in sm and d.day >= 15: sm[ym] = d
            dca_dates = set(fm.values()) | set(sm.values())
        elif dca_freq == 4:
            dca_dates = set(trading_days[::5])

    # acc4：純 B&H
    a4_shares   = initial / float(c.iloc[0])
    a4_invested = initial

    # acc3：B&H + DCA
    a3_shares   = initial / float(c.iloc[0])
    a3_invested = initial

    a3_vals, a4_vals = [], []
    a3_inv_vals, a4_inv_vals = [], []

    for i, idx in enumerate(df.index):
        price = float(c.iloc[i])
        if mode == "dca" and dca_amount > 0 and idx in dca_dates:
            a3_shares   += dca_amount / price
            a3_invested += dca_amount
        a3_vals.append(a3_shares * price)
        a4_vals.append(a4_shares * price)
        a3_inv_vals.append(a3_invested)
        a4_inv_vals.append(a4_invested)

    return {
        "acc3_series":    pd.Series(a3_vals, index=df.index),
        "acc3_invested":  a3_invested,
        "acc3_inv_series":pd.Series(a3_inv_vals, index=df.index),
        "acc4_series":    pd.Series(a4_vals, index=df.index),
        "acc4_invested":  a4_invested,
        "acc4_inv_series":pd.Series(a4_inv_vals, index=df.index),
    }


# ═══════════════════════════════════════════════════════
# 圖表
# ═══════════════════════════════════════════════════════
CHART = dict(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
             font=dict(color="#475569", family="IBM Plex Mono, monospace", size=11),
             gridcolor="#e2e8f0")

def plot_equity(result: dict, bm: dict, strategy_name: str, is_dca: bool):
    """
    4 條線：
    藍實  = 策略（有DCA則為策略+DCA，否則純策略）
    藍虛  = 策略無DCA（DCA模式才顯示，用來對比）
    綠虛  = B&H + DCA（DCA模式才顯示）
    灰虛  = B&H 純初始
    """
    fig = go.Figure()

    # ── 灰虛：B&H 純初始（永遠顯示）──
    fig.add_trace(go.Scatter(
        x=bm["acc4_series"].index, y=bm["acc4_series"].values,
        name=f"B&H 純持有 (初始${bm['acc4_invested']:,.0f})",
        line=dict(color="#94a3b8", width=1.5, dash="dot")))

    if is_dca:
        # ── 綠虛：B&H + DCA ──
        fig.add_trace(go.Scatter(
            x=bm["acc3_series"].index, y=bm["acc3_series"].values,
            name=f"B&H 含DCA (總投${bm['acc3_invested']:,.0f})",
            line=dict(color="#16a34a", width=1.5, dash="dash")))

        # ── 藍細虛：純策略（無DCA）對照 ──
        fig.add_trace(go.Scatter(
            x=result["acc1_series"].index, y=result["acc1_series"].values,
            name=f"策略（無DCA）",
            line=dict(color="#93c5fd", width=1.5, dash="longdash")))

        # ── 藍實粗：策略 + DCA（主線）──
        fig.add_trace(go.Scatter(
            x=result["acc2_series"].index, y=result["acc2_series"].values,
            name=f"{strategy_name} ＋DCA",
            line=dict(color="#0369a1", width=2.5),
            fill="tozeroy", fillcolor="rgba(3,105,161,0.05)"))
    else:
        # 一次性模式：只顯示純策略主線
        fig.add_trace(go.Scatter(
            x=result["acc1_series"].index, y=result["acc1_series"].values,
            name=strategy_name,
            line=dict(color="#0369a1", width=2.5),
            fill="tozeroy", fillcolor="rgba(3,105,161,0.05)"))

    fig.update_layout(
        title=dict(
            text=("📊 資產增長曲線（藍=策略+DCA / 藍虛=純策略 / 綠虛=B&H+DCA / 灰虛=純B&H）"
                  if is_dca else
                  "📊 資產增長曲線（藍=策略 / 灰虛=B&H 純持有）"),
            font=dict(size=12, color="#0f172a")),
        xaxis=dict(showgrid=True, gridcolor=CHART["gridcolor"]),
        yaxis=dict(showgrid=True, gridcolor=CHART["gridcolor"],
                   tickprefix="$", tickformat=",.0f"),
        paper_bgcolor=CHART["paper_bgcolor"], plot_bgcolor=CHART["plot_bgcolor"],
        font=CHART["font"], hovermode="x unified", height=430,
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#e2e8f0", borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=65, b=10))
    return fig


def plot_candlestick(df, strategy, params, buy_dates, sell_dates, buy_prices, sell_prices,
                     dca_buy_dates=None, dca_buy_prices=None):
    need_sub = strategy in ["RSI 動能策略", "MACD 趨勢策略", "MA均線偏離策略"]
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
    elif strategy == "MA均線偏離策略":
        bp = params["dev_buy_period"]
        sp = params["dev_sell_period"]
        fig.add_trace(go.Scatter(x=df.index, y=df[f"BuyMA{bp}"],
            name=f"買入均線 MA{bp}", line=dict(color="#0369a1", width=1.5)), row=1, col=1)
        if sp != bp:
            fig.add_trace(go.Scatter(x=df.index, y=df[f"SellMA{sp}"],
                name=f"賣出均線 MA{sp}", line=dict(color="#dc2626", width=1.5, dash="dot")), row=1, col=1)
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
        fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode="markers", name="信號買入",
            marker=dict(symbol="triangle-up", color="#16a34a", size=13,
                        line=dict(color="#fff", width=1))), row=1, col=1)
    if sell_dates:
        fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode="markers", name="信號賣出",
            marker=dict(symbol="triangle-down", color="#dc2626", size=13,
                        line=dict(color="#fff", width=1))), row=1, col=1)
    # DCA 定投標記：小圓點，不搶奪視覺焦點
    if dca_buy_dates:
        fig.add_trace(go.Scatter(x=dca_buy_dates, y=dca_buy_prices, mode="markers", name="DCA 定投",
            marker=dict(symbol="circle", color="#f59e0b", size=5, opacity=0.6,
                        line=dict(color="#fff", width=0.5))), row=1, col=1)

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
    elif strategy == "MA均線偏離策略" and need_sub:
        # 買入偏離率（藍）與賣出偏離率（紅）
        fig.add_trace(go.Scatter(x=df.index, y=df["Dev_Buy"],
            name=f"買入偏離率(MA{params['dev_buy_period']})",
            line=dict(color="#0369a1", width=1.3)), row=3, col=1)
        if params["dev_sell_period"] != params["dev_buy_period"]:
            fig.add_trace(go.Scatter(x=df.index, y=df["Dev_Sell"],
                name=f"賣出偏離率(MA{params['dev_sell_period']})",
                line=dict(color="#dc2626", width=1.3, dash="dot")), row=3, col=1)
        # 買入/賣出門檻水平線
        fig.add_hline(y=params["dev_buy_pct"], line_dash="dash",
                      line_color="#0369a1", row=3, col=1,
                      annotation_text=f"買入 {params['dev_buy_pct']:+.0f}%",
                      annotation_font_color="#0369a1")
        fig.add_hline(y=params["dev_sell_pct"], line_dash="dash",
                      line_color="#dc2626", row=3, col=1,
                      annotation_text=f"賣出 +{params['dev_sell_pct']:.0f}%",
                      annotation_font_color="#dc2626")
        fig.add_hline(y=0, line_color="#94a3b8", line_width=0.8, row=3, col=1)

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
            "MA均線偏離策略":
                "均線偏離回歸策略：當收盤價低於買入均線達設定百分比時買入，"
                "高於賣出均線達設定百分比時賣出。"
                "例如：低於60日均線 0% 買入（即跌破均線即買），高於5日均線 20% 時賣出。"
                "買入/賣出可使用不同週期的均線，靈活度高。",
        }
        strategy = st.selectbox("選擇回測策略",
            list(STRATEGY_HELP.keys()),
            help=STRATEGY_HELP.get("MA 交叉策略"))
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
            elif strategy == "MA均線偏離策略":
                st.markdown("**📉 買入條件**")
                bc1, bc2 = st.columns(2)
                params["dev_buy_period"] = bc1.number_input(
                    "買入均線週期", min_value=5, max_value=500, value=60, step=5,
                    help="計算買入基準的移動平均線週期，例如 60 表示 60 日均線")
                params["dev_buy_pct"] = bc2.number_input(
                    "低於均線 % 買入", min_value=-50.0, max_value=0.0, value=0.0, step=1.0,
                    help="收盤價低於均線此百分比時觸發買入。0% = 跌破均線即買；-5% = 跌破均線再多跌 5% 才買")
                st.markdown("**📈 賣出條件**")
                sc1, sc2 = st.columns(2)
                params["dev_sell_period"] = sc1.number_input(
                    "賣出均線週期", min_value=5, max_value=500, value=5, step=5,
                    help="計算賣出基準的移動平均線週期，例如 5 表示 5 日均線")
                params["dev_sell_pct"] = sc2.number_input(
                    "高於均線 % 賣出", min_value=0.0, max_value=100.0, value=20.0, step=1.0,
                    help="收盤價高於均線此百分比時觸發賣出。20% = 漲超均線 20% 才賣")
                # 即時預覽說明
                st.caption(
                    f"目前設定：收盤 ≤ {params['dev_buy_period']}日均線"
                    f"{params['dev_buy_pct']:+.0f}% 買入 ／ "
                    f"收盤 ≥ {params['dev_sell_period']}日均線"
                    f"+{params['dev_sell_pct']:.0f}% 賣出"
                )

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

    result = run_backtest(df, inv_cfg)
    bm     = compute_benchmark(df, inv_cfg)
    is_dca = inv_cfg["mode"] == "dca"
    years  = (df.index[-1] - df.index[0]).days / 365.25

    # 取主策略帳戶（DCA模式用acc2，一次性用acc1）
    main_series   = result["acc2_series"]   if is_dca else result["acc1_series"]
    main_final    = result["acc2_final"]    if is_dca else result["acc1_final"]
    main_invested = result["acc2_invested"] if is_dca else result["acc1_invested"]
    main_return   = result["acc2_return"]   if is_dca else result["acc1_return"]
    main_cagr     = result["acc2_cagr"]     if is_dca else result["acc1_cagr"]
    main_drawdown = result["acc2_drawdown"] if is_dca else result["acc1_drawdown"]
    main_dd_series= result["acc2_dd_series"]if is_dca else result["acc1_dd_series"]

    # 比較基準（DCA模式對比acc3，一次性對比acc4）
    bh_series   = bm["acc3_series"]   if is_dca else bm["acc4_series"]
    bh_invested = bm["acc3_invested"] if is_dca else bm["acc4_invested"]
    bh_final    = float(bh_series.iloc[-1])
    bh_return   = (bh_final - bh_invested) / bh_invested
    bh_cagr     = (bh_final / bh_invested) ** (1 / max(years, 0.01)) - 1
    bh_dd       = float((bh_series / bh_series.cummax() - 1).min())

    c = df["Close"].squeeze()
    i1, i2, i3, i4, i5 = st.columns(5)
    i1.metric("股票代號", ticker)
    i2.metric("最新收盤價", f"${float(c.iloc[-1]):,.2f}")
    i3.metric("回測期間漲跌", f"{(float(c.iloc[-1])-float(c.iloc[0]))/float(c.iloc[0]):+.1%}")
    i4.metric("數據起始", df.index[0].strftime("%Y-%m-%d"))
    i5.metric("數據截止", df.index[-1].strftime("%Y-%m-%d"))
    st.markdown("---")

    inv_mode_label = "定期定額 (DCA)" if is_dca else "一次性投入"
    st.markdown(f'<div class="strategy-badge">STRATEGY: {strategy.upper()} ・ {inv_mode_label}</div>',
                unsafe_allow_html=True)

    ci1, ci2, ci3 = st.columns(3)
    ci1.metric("初始投入", f"${inv_cfg['initial']:,.0f}")
    ci2.metric("DCA 累計投入" if is_dca else "投入方式",
               f"${result['dca_invested']:,.0f}" if is_dca else "一次性")
    ci3.metric("主帳戶總投入", f"${main_invested:,.0f}",
               delta=f"基準總投入 ${bh_invested:,.0f}")
    st.markdown("---")

    st.markdown("#### 📊 績效指標對比（主策略帳戶 vs 買入持有基準）")
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("策略總報酬", f"{main_return:+.2%}",   delta=f"vs B&H {bh_return:+.2%}")
    m2.metric("策略 CAGR",  f"{main_cagr:+.2%}",    delta=f"vs B&H {bh_cagr:+.2%}")
    m3.metric("策略最大回撤", f"{main_drawdown:.2%}", delta=f"vs B&H {bh_dd:.2%}", delta_color="inverse")
    m4.metric("策略最終資產", f"${main_final:,.0f}")
    m5.metric("B&H 最終資產", f"${bh_final:,.0f}")
    n_dca  = len(result["dca_buy_dates"])
    n_buy  = len(result["buy_dates"])
    n_sell = len(result["sell_dates"])
    m6.metric("DCA 定投次數", f"{n_dca} 次" if is_dca else "—")
    m7.metric("信號買 / 賣",  f"{n_buy} / {n_sell}")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 資產增長曲線", "🕯️ K線圖與買賣標記", "📉 回撤分析"])

    with tab1:
        st.plotly_chart(plot_equity(result, bm, strategy, is_dca),
                        use_container_width=True)

    with tab2:
        st.plotly_chart(plot_candlestick(df, strategy, params,
            result["buy_dates"],     result["sell_dates"],
            result["buy_prices"],    result["sell_prices"],
            result["dca_buy_dates"], result["dca_buy_prices"]),
            use_container_width=True)

    with tab3:
        fig_dd = go.Figure()
        # 主策略回撤
        fig_dd.add_trace(go.Scatter(
            x=main_dd_series.index, y=main_dd_series.values * 100,
            name=strategy + (" ＋DCA" if is_dca else ""),
            line=dict(color="#dc2626", width=1.5),
            fill="tozeroy", fillcolor="rgba(220,38,38,0.08)"))
        # acc1 純策略回撤（DCA模式才顯示）
        if is_dca:
            fig_dd.add_trace(go.Scatter(
                x=result["acc1_dd_series"].index, y=result["acc1_dd_series"].values * 100,
                name=strategy + "（無DCA）",
                line=dict(color="#f87171", width=1.2, dash="longdash")))
        # B&H 基準回撤
        bh_dd_s = bh_series / bh_series.cummax() - 1
        fig_dd.add_trace(go.Scatter(
            x=bh_dd_s.index, y=bh_dd_s.values * 100,
            name="B&H 基準",
            line=dict(color="#94a3b8", width=1.2, dash="dot")))
        fig_dd.update_layout(
            title=dict(text="📉 回撤對比", font=dict(size=14, color="#0f172a")),
            xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
            yaxis=dict(showgrid=True, gridcolor="#e2e8f0", ticksuffix="%"),
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", font=CHART["font"],
            legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0", borderwidth=1),
            hovermode="x unified", height=380, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig_dd, use_container_width=True)

    with st.expander("📋 查看所有交易明細（信號交易）"):
        trades = []
        sell_dates_list  = result["sell_dates"]
        sell_prices_list = result["sell_prices"]
        for i, (bd, bp) in enumerate(zip(result["buy_dates"], result["buy_prices"])):
            sd = sell_dates_list[i]  if i < len(sell_dates_list)  else "持倉中"
            sp = sell_prices_list[i] if i < len(sell_prices_list) else float(c.iloc[-1])
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
            st.info("此策略在回測期間未產生任何信號交易。")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#cbd5e1;font-family:IBM Plex Mono;font-size:0.7rem;'>"
        "數據來源: Yahoo Finance · 本工具僅供學習研究，不構成投資建議</div>",
        unsafe_allow_html=True)

if __name__ == "__main__":
    main()
