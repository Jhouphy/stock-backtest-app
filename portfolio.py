"""
portfolio.py
資產配置分析模組
功能：相關係數熱力圖、個別資產績效、組合回測（含再平衡）、基準貨幣轉換
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

# ─── 共用圖表樣式（與 app.py 一致）───
CHART = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafc",
    font=dict(color="#475569", family="IBM Plex Mono, monospace", size=11),
    gridcolor="#e2e8f0",
)

# ─── 匯率代號對照表 ───
FX_TICKERS = {
    "USD": None,          # 基準，不需轉換
    "TWD": "TWD=X",       # yfinance：USD/TWD
    "JPY": "JPY=X",
    "EUR": "EURUSD=X",    # 注意：EUR 是反向報價
    "HKD": "HKD=X",
}

# EUR 是反向報價（1 EUR = x USD），其他都是 1 USD = x 外幣
FX_INVERSE = {"EUR"}


@st.cache_data(ttl=3600)
def fetch_fx_rate(base_currency: str, start: str, end: str) -> pd.Series | None:
    """
    取得外幣對 USD 的匯率序列（base_currency 每單位等於多少 USD）。
    回傳 Series，index 為日期，值為「1 外幣 = ? USD」。
    若 base_currency == "USD" 回傳 None（不需轉換）。
    """
    if base_currency == "USD":
        return None
    fx_ticker = FX_TICKERS.get(base_currency)
    if not fx_ticker:
        return None
    try:
        df = yf.download(fx_ticker, start=start, end=end,
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        close = df["Close"].squeeze()
        if base_currency in FX_INVERSE:
            # EURUSD=X 報價是 1 EUR = x USD，已是正向
            return close
        else:
            # TWD=X 報價是 1 USD = x TWD，要取倒數得到 1 TWD = ? USD
            return 1.0 / close
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_portfolio_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    同時下載多個標的的收盤價。
    美股與台股交易日不同，改用各自下載再 ffill 對齊，避免 dropna 刪掉大量資料。
    """
    if not tickers:
        return pd.DataFrame()

    frames = {}
    for t in tickers:
        try:
            raw = yf.download(t, start=start, end=end,
                              progress=False, auto_adjust=True)
            if raw.empty:
                continue
            s = raw["Close"].squeeze()
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s.name = t
            frames[t] = s
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    # 以聯集日期對齊，前向填充補休市缺口，再刪掉初始全空列
    df = pd.DataFrame(frames)
    df = df.ffill().bfill()
    df = df.dropna(how="all")
    return df




@st.cache_data(ttl=3600)
def fetch_dividends(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    下載各標的的股息資料（ex-dividend date 與金額）。
    回傳 DataFrame，index=日期，columns=ticker，值為每股股息金額（0 代表無配息）。
    只對美股有效；台股、現金不配息跳過。
    """
    if not tickers:
        return pd.DataFrame()
    result = {}
    for ticker in tickers:
        if ticker == "CASH" or ticker.endswith(".TW") or ticker.endswith(".T"):
            continue
        try:
            tk   = yf.Ticker(ticker)
            divs = tk.dividends  # Series，index=ex-date，value=每股金額
            if divs.empty:
                continue
            # 過濾到回測區間
            divs = divs[(divs.index >= pd.Timestamp(start)) &
                        (divs.index <= pd.Timestamp(end))]
            if not divs.empty:
                result[ticker] = divs
        except Exception:
            continue
    if not result:
        return pd.DataFrame()
    df = pd.DataFrame(result).fillna(0.0)
    return df

def detect_currency(ticker: str) -> str:
    """根據代號後綴判斷計價貨幣。"""
    t = ticker.upper()
    if t.endswith(".TW") or t.endswith(".TWO"):
        return "TWD"
    if t.endswith(".T"):
        return "JPY"
    if t.endswith(".HK"):
        return "HKD"
    if t.endswith(".DE") or t.endswith(".PA") or t.endswith(".AS"):
        return "EUR"
    return "USD"


def convert_prices_to_base(
    prices: pd.DataFrame,
    tickers: list[str],
    base_currency: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    將各欄的股價統一換算為基準貨幣。
    對每個 ticker 判斷其原始貨幣，若與 base_currency 不同則乘以匯率。
    """
    result = prices.copy()
    fx_cache: dict[str, pd.Series | None] = {}

    for ticker in tickers:
        src_currency = detect_currency(ticker)
        if src_currency == base_currency:
            continue   # 已是基準貨幣，不需轉換

        # 目標：把 src_currency 計價的股價 → base_currency
        # 需要「1 src_currency = ? base_currency」的匯率
        if base_currency == "USD":
            # src→USD：取 src 的 USD 匯率
            key = f"{src_currency}_to_USD"
            if key not in fx_cache:
                fx_cache[key] = fetch_fx_rate(src_currency, start, end)
            fx = fx_cache[key]
        else:
            # 先轉 USD，再轉 base
            key_src = f"{src_currency}_to_USD"
            key_base = f"{base_currency}_to_USD"
            if key_src not in fx_cache:
                fx_cache[key_src] = fetch_fx_rate(src_currency, start, end)
            if key_base not in fx_cache:
                fx_cache[key_base] = fetch_fx_rate(base_currency, start, end)
            fx_src  = fx_cache[key_src]
            fx_base = fx_cache[key_base]
            if fx_src is not None and fx_base is not None:
                # price_base = price_src * (src→USD) / (base→USD)
                fx_src_al  = fx_src.reindex(prices.index).ffill().bfill()
                fx_base_al = fx_base.reindex(prices.index).ffill().bfill()
                result[ticker] = prices[ticker] * fx_src_al / fx_base_al
            continue

        if fx is not None:
            fx_aligned = fx.reindex(prices.index).ffill().bfill()
            result[ticker] = prices[ticker] * fx_aligned

    return result


def run_portfolio_backtest(
    prices: pd.DataFrame,
    weights: dict[str, float],
    initial: float,
    rebalance_freq: str,        # "none" / "monthly" / "quarterly" / "yearly"
    cash_return: float,         # 現金年化報酬率（0.0 ~ 0.05）
    commission: float = 0.0,    # 單邊手續費比例（買入+賣出各扣一次）
    div_tax: float = 0.0,       # 股息預扣稅率（台灣居民投資美股：0.30）
    dividends: pd.DataFrame | None = None,  # fetch_dividends 取得的股息資料
    dca_amount: float = 0.0,    # 每次定期定額投入金額（0 = 不啟用）
    dca_dates: set | None = None,  # 預先計算好的 DCA 日期集合
) -> dict:
    """
    組合回測引擎。
    - 第一天依權重買入（扣手續費；initial=0 時跳過）
    - 每個 DCA 日依權重追加買入（扣手續費）
    - 依 rebalance_freq 定期再平衡（買賣各扣手續費）
    - "CASH" 作為虛擬標的，每日以 cash_return/252 複利增長
    - 配息日：股息金額 × (1 - div_tax) 加回現金池，超過的稅金從組合扣除
    """
    tickers = list(prices.columns)
    dates   = prices.index
    n       = len(dates)

    # 處理現金（虛擬資產）
    if "CASH" in tickers:
        daily_cash_r = (1 + cash_return) ** (1 / 252) - 1
        base_price   = float(prices["CASH"].iloc[0]) if "CASH" in prices.columns else 1.0
        cash_series  = pd.Series(
            [base_price * (1 + daily_cash_r) ** i for i in range(n)],
            index=dates, name="CASH"
        )
        prices = prices.copy()
        prices["CASH"] = cash_series

    # 確保權重正規化
    total_w = sum(weights.values())
    w = {t: weights[t] / total_w for t in tickers if t in weights}

    # 初始持股（第一天收盤價買入；initial=0 時從零開始靠 DCA 累積）
    first_prices = prices.iloc[0]
    if initial > 0:
        shares = {
            t: initial * w.get(t, 0) / (float(first_prices[t]) * (1 + commission))
            for t in tickers
        }
    else:
        shares = {t: 0.0 for t in tickers}
    total_invested = initial

    # 再平衡日期計算
    rebal_dates = set()
    if rebalance_freq != "none":
        prev_period = None
        for d in dates:
            if rebalance_freq == "monthly":
                period = (d.year, d.month)
            elif rebalance_freq == "quarterly":
                period = (d.year, (d.month - 1) // 3)
            else:   # yearly
                period = (d.year,)
            if prev_period is not None and period != prev_period:
                rebal_dates.add(d)
            prev_period = period

    # 逐日計算組合價值
    portfolio_vals  = []
    rebal_dates_hit = []

    _dca_dates = dca_dates or set()

    for i, (date, row) in enumerate(prices.iterrows()):
        # ── DCA 定期定額買入 ──
        if dca_amount > 0 and date in _dca_dates and i > 0:
            for t in tickers:
                amt = dca_amount * w.get(t, 0)
                eff_price = float(row[t]) * (1 + commission)
                if eff_price > 0:
                    shares[t] += amt / eff_price
            total_invested += dca_amount

        # 再平衡
        if date in rebal_dates and i > 0:
            total_val = sum(shares[t] * float(row[t]) for t in tickers)
            for t in tickers:
                target_val  = total_val * w.get(t, 0)
                current_val = shares[t] * float(row[t])
                diff = target_val - current_val
                if diff > 0:
                    # 買入：扣手續費（有效價格較高）
                    eff_price   = float(row[t]) * (1 + commission)
                    shares[t]   = target_val / eff_price
                    total_val  -= diff * commission   # 手續費從池子扣
                else:
                    # 賣出：扣手續費（有效價格較低）
                    shares[t]   = target_val / (float(row[t]) * (1 - commission)) if float(row[t]) > 0 else 0
                    total_val  += diff * commission   # diff 為負，手續費扣損
            rebal_dates_hit.append(date)

        # ── 股息稅務處理 ──
        if dividends is not None and not dividends.empty and date in dividends.index:
            div_row = dividends.loc[date]
            for t in tickers:
                if t in div_row and div_row[t] > 0:
                    gross_div     = shares[t] * div_row[t]          # 稅前股息
                    tax_drag      = gross_div * div_tax              # 預扣稅
                    # 稅後股息已反映在 adjusted price 中，這裡扣除「額外損失」
                    # yfinance 的 adjusted close 已含股息再投入假設，
                    # 所以只扣稅率部分的損失（避免雙重計算）
                    shares[t]    -= tax_drag / float(row[t]) if float(row[t]) > 0 else 0

        val = sum(shares[t] * float(row[t]) for t in tickers)
        portfolio_vals.append(val)

    port_series = pd.Series(portfolio_vals, index=dates)
    final_val   = float(port_series.iloc[-1])
    years       = (dates[-1] - dates[0]).days / 365.25
    base        = total_invested if total_invested > 0 else max(final_val, 1.0)
    total_ret   = (final_val - total_invested) / base
    cagr        = (final_val / base) ** (1 / max(years, 0.1)) - 1
    dd_series   = port_series / port_series.cummax() - 1
    max_dd      = float(dd_series.min())
    # 只取 portfolio>0 的部分計算波動率（initial=0 時前期為零，排除）
    valid_series = port_series[port_series > 0]
    daily_r  = valid_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    ann_vol  = float(daily_r.std() * np.sqrt(252)) if len(daily_r) > 5 else 0.0
    sharpe   = (cagr - 0.04) / ann_vol if ann_vol > 0.001 else 0

    return {
        "series":         port_series,
        "final":          final_val,
        "total_invested": total_invested,
        "total_ret":      total_ret,
        "cagr":           cagr,
        "max_dd":         max_dd,
        "ann_vol":        ann_vol,
        "sharpe":         sharpe,
        "dd_series":      dd_series,
        "rebal_dates":    rebal_dates_hit,
        "dca_amount":     dca_amount,
        "dca_dates":      _dca_dates,
        "commission":     commission,
    }


def calc_asset_stats(prices: pd.DataFrame, initial: float) -> pd.DataFrame:
    """計算各資產的個別績效指標。"""
    rows = []
    for col in prices.columns:
        s     = prices[col].dropna()
        if len(s) < 10:
            continue
        years = (s.index[-1] - s.index[0]).days / 365.25
        ret   = (s.iloc[-1] - s.iloc[0]) / s.iloc[0]
        cagr  = (s.iloc[-1] / s.iloc[0]) ** (1 / max(years, 0.1)) - 1
        dd    = (s / s.cummax() - 1).min()
        daily_r = s.pct_change().dropna()
        vol   = float(daily_r.std() * np.sqrt(252))
        sharpe= (cagr - 0.04) / vol if vol > 0.001 else 0
        rows.append({
            "標的":      col,
            "總報酬":    ret,
            "CAGR":     cagr,
            "最大回撤":  dd,
            "年化波動":  vol,
            "夏普比率":  sharpe,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════
# 圖表函數
# ═══════════════════════════════════════════════

def plot_correlation_heatmap(prices: pd.DataFrame) -> go.Figure:
    """相關係數熱力圖。"""
    daily_r = prices.pct_change().dropna()
    corr    = daily_r.corr().round(3)
    labels  = list(corr.columns)
    z       = corr.values

    # 顏色：-1 紅 → 0 白 → 1 藍
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale=[
            [0.0,  "#dc2626"],
            [0.5,  "#ffffff"],
            [1.0,  "#1d4ed8"],
        ],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=12),
        hoverongaps=False,
    ))
    fig.update_layout(
        title=dict(text="📊 資產相關係數矩陣", font=dict(size=14, color="#0f172a")),
        paper_bgcolor=CHART["paper_bgcolor"],
        plot_bgcolor=CHART["plot_bgcolor"],
        font=CHART["font"],
        height=max(300, len(labels) * 60 + 100),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def plot_portfolio_equity(
    port_result: dict,
    asset_prices: pd.DataFrame,
    weights: dict,
    initial: float,
    base_currency: str,
    currency_symbol: str,
) -> go.Figure:
    """組合資產曲線 + 各資產實際持倉市值曲線。"""
    fig = go.Figure()

    # 各資產虛線：假設把全部 DCA 金額只買該資產的走勢
    # 從 port_result 取得 DCA 相關資訊
    colors = px.colors.qualitative.Set2
    dca_amount = port_result.get("dca_amount", 0)
    dca_dates  = port_result.get("dca_dates",  set())
    commission = port_result.get("commission", 0.0)

    for i, col in enumerate(asset_prices.columns):
        if col == "CASH":
            continue
        price_s = asset_prices[col].dropna()
        if price_s.empty:
            continue

        # 模擬「全部 X 只買此資產」
        _shares = initial / float(price_s.iloc[0]) if initial > 0 else 0.0
        _vals   = []
        for date, price in price_s.items():
            if dca_amount > 0 and date in dca_dates:
                eff = float(price) * (1 + commission)
                if eff > 0:
                    _shares += dca_amount / eff
            _vals.append(_shares * float(price))

        s = pd.Series(_vals, index=price_s.index)
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=f"{col}（全押）",
            line=dict(color=colors[i % len(colors)], width=1.2, dash="dot"),
            opacity=0.7,
            hovertemplate=f"{col}（全押）: {currency_symbol}%{{y:,.0f}}<extra></extra>",
        ))

    # 組合主線
    fig.add_trace(go.Scatter(
        x=port_result["series"].index,
        y=port_result["series"].values,
        name="📦 組合總值",
        line=dict(color="#0369a1", width=2.5),
    ))

    # 再平衡標記
    for rd in port_result["rebal_dates"]:
        fig.add_vline(x=rd, line_color="#f59e0b",
                      line_width=0.8, line_dash="dot", opacity=0.5)

    fig.update_layout(
        title=dict(text="📈 組合資產曲線（點線=個別資產，實線=組合）",
                   font=dict(size=13, color="#0f172a")),
        xaxis=dict(showgrid=True, gridcolor=CHART["gridcolor"]),
        yaxis=dict(showgrid=True, gridcolor=CHART["gridcolor"],
                   tickprefix=currency_symbol,
                   tickformat=",.0f"),
        paper_bgcolor=CHART["paper_bgcolor"],
        plot_bgcolor=CHART["plot_bgcolor"],
        font=CHART["font"],
        hovermode="x unified", height=450,
        dragmode="pan",
        legend=dict(bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e2e8f0", borderwidth=1),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def plot_portfolio_drawdown(port_result: dict) -> go.Figure:
    """組合回撤曲線。"""
    dd = port_result["dd_series"] * 100
    fig = go.Figure(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy", fillcolor="rgba(220,38,38,0.08)",
        line=dict(color="#dc2626", width=1.5),
        name="組合回撤",
    ))
    fig.update_layout(
        dragmode="pan",
        title=dict(text="📉 組合回撤曲線", font=dict(size=13, color="#0f172a")),
        xaxis=dict(showgrid=True, gridcolor=CHART["gridcolor"]),
        yaxis=dict(showgrid=True, gridcolor=CHART["gridcolor"],
                   ticksuffix="%"),
        paper_bgcolor=CHART["paper_bgcolor"],
        plot_bgcolor=CHART["plot_bgcolor"],
        font=CHART["font"],
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def plot_weight_pie(weights: dict) -> go.Figure:
    """權重分配圓餅圖。"""
    labels = list(weights.keys())
    values = [weights[k] * 100 for k in labels]
    colors = px.colors.qualitative.Set2[:len(labels)]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.45,
        marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
        textinfo="label+percent",
        textfont=dict(size=12),
    ))
    fig.update_layout(
        title=dict(text="🍰 資產配置比例", font=dict(size=13, color="#0f172a")),
        paper_bgcolor=CHART["paper_bgcolor"],
        font=CHART["font"],
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════
# 主頁面渲染函數（由 app.py 呼叫）
# ═══════════════════════════════════════════════

CURRENCY_SYMBOLS = {"USD": "$", "TWD": "NT$", "JPY": "¥", "EUR": "€", "HKD": "HK$"}

def render_portfolio_tab():
    """資產配置分析頁面，完整 UI + 計算邏輯。"""
    st.markdown("### 💼 多標的資產配置分析")
    st.caption("輸入 2~10 個標的與配置比例，分析組合績效、相關性與再平衡效益。")

    # ──────────────────────────────────────────
    # 設定區（inline，不用側邊欄）
    # ──────────────────────────────────────────
    with st.expander("⚙️ 組合設定", expanded=True):

        # 基準貨幣 + 年數
        cfg1, cfg2 = st.columns(2)
        base_currency = cfg1.selectbox(
            "基準貨幣",
            ["USD", "TWD", "JPY", "EUR", "HKD"],
            help="所有標的的價格與組合價值都將換算為此貨幣後計算。"
        )
        currency_symbol = CURRENCY_SYMBOLS.get(base_currency, "$")
        years_back = cfg2.selectbox("回測年數", [1, 2, 3, 5, 7, 10, 15, 20], index=4)

        # 初始資金（可為 0）
        initial = st.number_input(
            f"初始資金 ({base_currency})　（可設為 0，純靠定期定額累積）",
            min_value=0, max_value=100_000_000,
            value=1_000_000, step=100_000, format="%d"
        )

        # DCA 定期定額設定
        st.markdown("**📅 定期定額設定（可選）**")
        dca1, dca2, dca3 = st.columns(3)
        port_dca_enable = dca1.toggle(
            "啟用定期定額",
            value=False,
            help="每期依設定金額，按當前權重比例追加買入各標的。"
        )
        if port_dca_enable:
            port_dca_amount = dca2.number_input(
                f"每次投入金額 ({base_currency})",
                min_value=100, max_value=10_000_000,
                value=30_000, step=1_000, format="%d"
            )
            port_dca_freq = dca3.selectbox(
                "每月投入次數",
                [1, 2, 4],
                format_func=lambda x: {1:"1次（月投）", 2:"2次（雙週投）", 4:"4次（週投）"}[x],
                help="1次=每月初，2次=每兩週，4次=每週"
            )
            total_dca_est = initial + port_dca_amount * port_dca_freq * 12 * years_back
            st.caption(
                f"💡 預估總投入：{currency_symbol}{total_dca_est:,.0f}"
                f"（初始 {currency_symbol}{initial:,} ＋ 每月 {currency_symbol}{port_dca_amount*port_dca_freq:,} × {years_back}年）"
            )
        else:
            port_dca_amount = 0
            port_dca_freq   = 1
        end_dt     = datetime.today()
        start_dt   = end_dt.replace(year=end_dt.year - years_back)
        start_str  = start_dt.strftime("%Y-%m-%d")
        end_str    = end_dt.strftime("%Y-%m-%d")

        st.markdown("---")

        # 再平衡設定 + 現金利率
        rb1, rb2 = st.columns(2)
        rebalance_freq = rb1.selectbox(
            "再平衡頻率",
            ["none", "monthly", "quarterly", "yearly"],
            index=2,
            format_func=lambda x: {
                "none": "不再平衡", "monthly": "每月",
                "quarterly": "每季", "yearly": "每年"
            }[x],
            help="定期將各資產比例調回設定權重。不再平衡則讓各資產自由漲跌。"
        )
        cash_return_pct = rb2.slider(
            "現金年化利率（若配置現金）", 0.0, 6.0, 4.0, 0.5,
            format="%.1f%%",
            help="將 CASH 加入組合時，此利率模擬貨幣基金或短期債券的無風險報酬。"
        )

        st.markdown("---")
        st.markdown("**💸 交易成本 & 稅務設定**")
        tc1, tc2, tc3 = st.columns(3)
        commission_pct = tc1.select_slider(
            "單邊手續費／滑價",
            options=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0],
            value=0.1,
            format_func=lambda x: f"{x:.2f}%",
            help="每筆買入或賣出各扣一次。美股ETF建議 0.05~0.1%；台股含證交稅建議 0.1~0.2%。"
        )
        enable_div_tax = tc2.toggle(
            "美股股息預扣稅 30%",
            value=False,
            help="台灣居民投資美股，股息依條約預扣 30% 稅（非 ETF 配息稅）。開啟後每次配息日自動扣除稅額，影響長期複利約 0.5~1%/年。"
        )
        if enable_div_tax:
            div_tax_rate = tc3.slider(
                "預扣稅率", 0, 30, 30, 1,
                format="%d%%",
                help="美台條約標準為 30%；若持有符合條件的退休帳戶（如 IRA）可設為 0%。"
            ) / 100
            tc2.caption("📌 需下載股息資料，初次執行較慢。")
        else:
            div_tax_rate = 0.0

        st.markdown("---")
        st.markdown("**📋 標的與權重設定**")
        st.caption("每行輸入一個標的代號與配置比例（%）。台股格式：2330.TW；現金請輸入 CASH。")

        # 預設 2 個標的
        defaults = [
            ("VOO", 50),
            ("QQQ", 50),
        ]
        n_assets = st.number_input("標的數量", min_value=2, max_value=10,
                                   value=2, step=1)

        tickers_input = []
        weights_input = []
        cols_per_row  = 2
        asset_rows    = [
            st.columns([2, 1, 2, 1]) if i + 1 < int(n_assets)
            else st.columns([2, 1, 2, 1])
            for i in range(0, int(n_assets), cols_per_row)
        ]

        flat_cols = []
        for row in asset_rows:
            flat_cols.extend(row)

        for idx in range(int(n_assets)):
            col_t = flat_cols[idx * 2]
            col_w = flat_cols[idx * 2 + 1]
            default_t = defaults[idx][0] if idx < len(defaults) else ""
            default_w = defaults[idx][1] if idx < len(defaults) else 10
            t = col_t.text_input(f"標的 {idx+1}", value=default_t,
                                  key=f"pt_{idx}").upper().strip()
            w = col_w.number_input(f"比例 {idx+1} %", min_value=0,
                                    max_value=100, value=default_w,
                                    key=f"pw_{idx}")
            if t:
                tickers_input.append(t)
                weights_input.append(w)

        total_pct = sum(weights_input)
        if total_pct == 100:
            st.success(f"✅ 合計 {total_pct}%，配置有效")
        else:
            st.error(f"⚠️ 合計 {total_pct}%，請調整至 100%")

    run_port = st.button("🚀 執行組合分析", type="primary",
                         disabled=(total_pct != 100 or len(tickers_input) < 2))
    if not run_port:
        st.info("👆 設定好標的與權重後點擊「執行組合分析」。", icon="💡")
        return

    # ──────────────────────────────────────────
    # 資料下載與換算
    # ──────────────────────────────────────────
    real_tickers = [t for t in tickers_input if t != "CASH"]
    weights_raw  = {t: w/100 for t, w in zip(tickers_input, weights_input)}

    with st.spinner(f"下載 {len(real_tickers)} 個標的的歷史數據..."):
        if real_tickers:
            prices_raw = fetch_portfolio_data(real_tickers, start_str, end_str)
        else:
            prices_raw = pd.DataFrame()

    if prices_raw.empty and real_tickers:
        st.error("❌ 無法取得數據，請確認標的代號是否正確。")
        return

    # 找到所有標的共同的最早日期
    if not prices_raw.empty:
        missing = [t for t in real_tickers if t not in prices_raw.columns]
        if missing:
            st.warning(f"⚠️ 以下標的無法取得數據，已略過：{', '.join(missing)}")
            tickers_input = [t for t in tickers_input if t not in missing]
            weights_raw   = {t: w for t, w in weights_raw.items() if t not in missing}
            real_tickers  = [t for t in real_tickers if t not in missing]
            if not real_tickers:
                st.error("所有標的均無效，請重新設定。")
                return

    # 貨幣換算
    with st.spinner(f"換算為基準貨幣 {base_currency}..."):
        if not prices_raw.empty:
            prices_converted = convert_prices_to_base(
                prices_raw, real_tickers, base_currency, start_str, end_str)
        else:
            prices_converted = pd.DataFrame()

    # 加入現金（虛擬欄，初始值 1）
    if "CASH" in tickers_input and not prices_converted.empty:
        prices_converted["CASH"] = 1.0   # 固定 1，在 run_portfolio_backtest 中處理複利

    # 最終 tickers（含 CASH）
    final_tickers = [t for t in tickers_input if t in prices_converted.columns]
    if len(final_tickers) < 2:
        st.error("有效標的不足，至少需要 2 個，請重新設定。")
        return

    prices_final = prices_converted[final_tickers]
    weights_final = {t: weights_raw[t] for t in final_tickers}
    # 重新正規化權重（去掉無效標的後）
    w_sum = sum(weights_final.values())
    weights_final = {t: v / w_sum for t, v in weights_final.items()}

    # 顯示共同回測區間
    common_start = prices_final.index[0].strftime("%Y-%m-%d")
    common_end   = prices_final.index[-1].strftime("%Y-%m-%d")
    st.info(f"📅 共同回測區間：{common_start} ～ {common_end}（{len(prices_final)} 個交易日）")

    # ──────────────────────────────────────────
    # 執行回測
    # ──────────────────────────────────────────
    # 若開啟股息稅，下載各標的股息資料
    dividends_df = None
    if enable_div_tax and div_tax_rate > 0:
        us_tickers = [t for t in final_tickers
                      if not t.endswith(".TW") and not t.endswith(".T") and t != "CASH"]
        if us_tickers:
            with st.spinner("下載股息資料..."):
                dividends_df = fetch_dividends(us_tickers, common_start, common_end)

    # 建立 DCA 日期集合
    port_dca_date_set = set()
    if port_dca_enable and port_dca_amount > 0:
        freq_map = {1: "MS", 2: "SMS", 4: "W-MON"}
        freq_str = freq_map.get(port_dca_freq, "MS")
        try:
            port_dca_date_set = set(
                pd.date_range(common_start, common_end, freq=freq_str).normalize()
            )
        except Exception:
            port_dca_date_set = set()

    with st.spinner("計算組合績效..."):
        port_result = run_portfolio_backtest(
            prices_final, weights_final, initial,
            rebalance_freq, cash_return_pct / 100,
            commission   = commission_pct / 100,
            div_tax      = div_tax_rate,
            dividends    = dividends_df,
            dca_amount   = port_dca_amount if port_dca_enable else 0.0,
            dca_dates    = port_dca_date_set,
        )
        asset_stats = calc_asset_stats(prices_final, initial)

    # ──────────────────────────────────────────
    # Metrics
    # ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 組合績效摘要")
    _final    = port_result["final"]
    _invested = port_result.get("total_invested", initial)
    _gain     = _final - _invested
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("總報酬",
              f"{port_result['total_ret']:+.2%}",
              delta=f"{currency_symbol}{_gain:+,.0f}")
    m2.metric("年化報酬 CAGR", f"{port_result['cagr']:+.2%}")
    m3.metric("最大回撤",      f"{port_result['max_dd']:.2%}")
    m4.metric("年化波動率",    f"{port_result['ann_vol']:.2%}"
                               if not np.isnan(port_result['ann_vol']) else "—")
    m5.metric("夏普比率",      f"{port_result['sharpe']:.3f}"
                               if not np.isnan(port_result['sharpe']) else "—")
    st.caption(
        f"📌 總投入：{currency_symbol}{_invested:,.0f}　"
        f"最終資產：{currency_symbol}{_final:,.0f}　"
        f"損益：{currency_symbol}{_gain:+,.0f}"
    )
    if port_result["rebal_dates"]:
        st.caption(f"🔄 共執行 {len(port_result['rebal_dates'])} 次再平衡（圖中橘色虛線標示）")
    cost_notes = []
    if commission_pct > 0:
        cost_notes.append(f"手續費 {commission_pct:.2f}%/單邊")
    if enable_div_tax and div_tax_rate > 0:
        cost_notes.append(f"股息預扣稅 {div_tax_rate*100:.0f}%")
    if port_dca_enable and port_dca_amount > 0:
        cost_notes.append(f"DCA {currency_symbol}{port_dca_amount:,}/次 × {port_dca_freq}次/月")
    if cost_notes:
        st.caption(f"💸 已套用設定：{' ｜ '.join(cost_notes)}")

    # ──────────────────────────────────────────
    # 圖表分頁
    # ──────────────────────────────────────────
    st.markdown("---")
    pt1, pt2, pt3, pt4 = st.tabs([
        "📈 資產曲線", "🌡️ 相關係數", "📉 回撤", "📋 個別資產績效"
    ])

    with pt1:
        left, right = st.columns([3, 1])
        with left:
            st.plotly_chart(
                plot_portfolio_equity(port_result, prices_final,
                                      weights_final, initial,
                                      base_currency, currency_symbol),
                use_container_width=True,
                config=dict(
                    locale="zh-TW", displayModeBar=True,
                    modeBarButtonsToRemove=["select2d","lasso2d","autoScale2d"],
                    toImageButtonOptions={"format":"png","filename":"portfolio_equity","scale":2},
                    displaylogo=False, scrollZoom=False,
                )
            )
        with right:
            st.plotly_chart(plot_weight_pie(weights_final),
                            use_container_width=True,
                            config=dict(locale="zh-TW", displayModeBar=True,
                                toImageButtonOptions={"format":"png","filename":"portfolio_weights","scale":2},
                                displaylogo=False, scrollZoom=False))

    with pt2:
        real_prices = prices_final[[t for t in final_tickers if t != "CASH"]]
        if len(real_prices.columns) >= 2:
            st.plotly_chart(plot_correlation_heatmap(real_prices),
                            use_container_width=True,
                            config=dict(locale="zh-TW", displayModeBar=True,
                                toImageButtonOptions={"format":"png","filename":"correlation_heatmap","scale":2},
                                displaylogo=False, scrollZoom=False))
            # 解讀提示
            daily_r = real_prices.pct_change().dropna()
            corr    = daily_r.corr()
            pairs   = []
            cols_list = list(corr.columns)
            for i in range(len(cols_list)):
                for j in range(i + 1, len(cols_list)):
                    pairs.append((cols_list[i], cols_list[j],
                                  corr.iloc[i, j]))
            pairs.sort(key=lambda x: x[2])
            if pairs:
                lo = pairs[0]
                hi = pairs[-1]
                st.caption(
                    f"📌 相關性最低：{lo[0]} ↔ {lo[1]} ({lo[2]:+.2f})　｜　"
                    f"相關性最高：{hi[0]} ↔ {hi[1]} ({hi[2]:+.2f})"
                )
        else:
            st.info("至少需要 2 個非現金標的才能計算相關係數。")

    with pt3:
        st.plotly_chart(plot_portfolio_drawdown(port_result),
                        use_container_width=True,
                        config=dict(locale="zh-TW", displayModeBar=True,
                            modeBarButtonsToRemove=["select2d","lasso2d","autoScale2d"],
                            toImageButtonOptions={"format":"png","filename":"portfolio_drawdown","scale":2},
                            displaylogo=False, scrollZoom=False))

    with pt4:
        if not asset_stats.empty:
            display_stats = asset_stats.copy()
            display_stats["總報酬"]   = display_stats["總報酬"].map("{:+.2%}".format)
            display_stats["CAGR"]    = display_stats["CAGR"].map("{:+.2%}".format)
            display_stats["最大回撤"] = display_stats["最大回撤"].map("{:.2%}".format)
            display_stats["年化波動"] = display_stats["年化波動"].map("{:.2%}".format)
            display_stats["夏普比率"] = display_stats["夏普比率"].map("{:.3f}".format)
            st.dataframe(display_stats, use_container_width=True, hide_index=True)

            # 再平衡效益說明
            if rebalance_freq != "none":
                with st.spinner("計算無再平衡基準..."):
                    no_rebal = run_portfolio_backtest(
                        prices_final, weights_final, initial,
                        "none", cash_return_pct / 100,
                        commission = commission_pct / 100,
                        div_tax    = div_tax_rate,
                        dividends  = dividends_df,
                        dca_amount = port_dca_amount if port_dca_enable else 0.0,
                        dca_dates  = port_dca_date_set,
                    )
                delta_cagr = port_result["cagr"] - no_rebal["cagr"]
                delta_dd   = port_result["max_dd"] - no_rebal["max_dd"]
                freq_label = {"monthly":"每月","quarterly":"每季","yearly":"每年"}[rebalance_freq]
                st.markdown(f"#### 🔄 再平衡效益（{freq_label}）")
                rb1, rb2, rb3, rb4 = st.columns(4)
                rb1.metric("再平衡 CAGR",  f"{port_result['cagr']:+.2%}")
                rb2.metric("無再平衡 CAGR", f"{no_rebal['cagr']:+.2%}",
                           delta=f"{delta_cagr:+.2%}")
                rb3.metric("再平衡最大回撤",  f"{port_result['max_dd']:.2%}")
                rb4.metric("無再平衡最大回撤", f"{no_rebal['max_dd']:.2%}",
                           delta=f"{delta_dd:+.2%}", delta_color="inverse")
