"""
個人化股票回測與選股分析 Web App
架構：app.py（UI） + engine.py（回測引擎） + charts.py（圖表） + portfolio.py（資產配置）
"""

import streamlit as st
from engine import (
    fetch_data, compute_indicators, generate_signals,
    run_backtest, compute_benchmark, run_grid_search,
    call_claude_analysis, check_vcp, CHART,
    plot_equity, plot_candlestick,
)
from portfolio import render_portfolio_tab
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

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


def apply_opt_result(r: dict):
    """
    一鍵套用：只存入 _pending_apply，觸發 rerun。
    實際寫入 widget session_state 在 _flush_pending_apply() 完成，
    必須在所有 widget 渲染之前呼叫，才不會觸發 StreamlitAPIException。
    """
    st.session_state["_pending_apply"] = r
    st.session_state["_auto_run"] = True
    st.rerun()


def _flush_pending_apply():
    """在任何 widget 渲染前呼叫，把 pending 資料寫入 widget key。"""
    r = st.session_state.pop("_pending_apply", None)
    if r is None:
        return
    cfg      = r.get("inv_cfg", {})
    params   = r.get("params",  {})
    strategy = r.get("strategy", "")
    strats   = ["MA 交叉策略", "RSI 動能策略", "布林通道策略", "MACD 趨勢策略", "MA均線偏離策略"]

    st.session_state["w_inv_mode"] = "定期定額 (DCA)" if cfg.get("mode") == "dca" else "一次性投入"

    bm = cfg.get("buy_mode", "all_in")
    st.session_state["w_buy_mode"] = {"all_in": "全倉買入", "fixed_amount": "固定金額",
                                       "fixed_pct": "固定比例 %"}.get(bm, "全倉買入")
    if bm == "fixed_amount":
        st.session_state["w_buy_amount"] = int(cfg.get("buy_amount", 10000))
    elif bm == "fixed_pct":
        st.session_state["w_buy_pct"] = int(cfg.get("buy_pct", 1.0) * 100)

    sm = cfg.get("sell_mode", "all_out")
    st.session_state["w_sell_mode"] = {"all_out": "全倉賣出", "fixed_amount": "固定金額",
                                        "fixed_pct": "固定比例 %"}.get(sm, "全倉賣出")
    if sm == "fixed_amount":
        st.session_state["w_sell_amount"] = int(cfg.get("sell_amount", 10000))
    elif sm == "fixed_pct":
        st.session_state["w_sell_pct"] = int(cfg.get("sell_pct", 1.0) * 100)

    if strategy in strats:
        st.session_state["w_strategy"] = strategy

    for k, v in params.items():
        key = f"w_{k}"
        if k == "bb_std":
            st.session_state[key] = float(v)
        elif isinstance(v, (int, float)):
            st.session_state[key] = int(v) if isinstance(v, float) and v == int(v) else v

def main():
    # ── 必須在所有 widget 渲染前執行，寫入套用設定 ──
    _flush_pending_apply()
    # _auto_run 在此就取出，避免之後 widget rerun 時被丟失
    _auto_run = st.session_state.pop("_auto_run", False)
    if _auto_run:
        # 滾回頁面頂部，讓使用者看到回測執行過程
        st.components.v1.html(
            "<script>window.parent.scrollTo({top: 0, behavior: 'smooth'});</script>",
            height=0,
        )

    st.markdown('<div class="main-title">📈 投資<span>研究</span>工作站</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Strategy Backtesting · Portfolio Analysis · VCP Screening</div>',
                unsafe_allow_html=True)

    # ── 側邊欄（全域，兩個 Tab 共用標題）──
    with st.sidebar:
        st.markdown("### 🎯 數據設定")
        ticker = st.text_input(
            "股票代號",
            value="VOO",
            key="w_ticker",
            help="美股：直接輸入代號，如 VOO、QQQ、AAPL\n台股：代號後加 .TW，如 2330.TW（台積電）、0050.TW（元大台灣50）、2317.TW（鴻海）"
        ).upper().strip()
        # 台股自動補 .TW 後綴提示
        if ticker.isdigit() and not ticker.endswith(".TW"):
            st.caption(f"💡 看起來像台股代號，請改輸入 **{ticker}.TW**")
        col_y1, col_y2 = st.columns(2)
        years_back = col_y1.selectbox("回測年數", [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100], index=3, key="w_years")
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
            max_value=10_000_000, value=100_000, step=10_000, format="%d", key="w_initial")

        # ── 投入方式 ──
        st.markdown("### 💰 投入方式")
        inv_mode = st.radio("選擇投入模式", ["一次性投入", "定期定額 (DCA)"],
            horizontal=True, key="w_inv_mode",
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
            key="w_buy_mode",
            help="每次觸發買入信號時，要用多少資金進場。")
        buy_amount, buy_pct = initial_capital, 1.0
        if buy_mode_label == "固定金額":
            buy_amount = st.number_input("每次買入金額 ($)", min_value=100,
                max_value=10_000_000, value=min(10_000, initial_capital), step=1_000, format="%d", key="w_buy_amount")
        elif buy_mode_label == "固定比例 %":
            buy_pct = st.slider("買入比例", 5, 100, 100, step=5,
                format="%d%%", key="w_buy_pct", help="佔當前可用資金的百分比") / 100

        sell_mode_label = st.selectbox("賣出方式",
            ["全倉賣出", "固定金額", "固定比例 %"],
            key="w_sell_mode",
            help="每次觸發賣出信號時，要賣出多少持倉。")
        sell_amount, sell_pct = 0, 1.0
        if sell_mode_label == "固定金額":
            sell_amount = st.number_input("每次賣出金額 ($)", min_value=100,
                max_value=10_000_000, value=10_000, step=1_000, format="%d", key="w_sell_amount")
        elif sell_mode_label == "固定比例 %":
            sell_pct = st.slider("賣出比例", 5, 100, 100, step=5,
                format="%d%%", key="w_sell_pct") / 100

        # ── 交易成本設定 ──
        st.markdown("### 💸 交易成本")
        commission_pct = st.select_slider(
            "手續費 + 滑點（單邊）",
            options=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0],
            value=0.1,
            format_func=lambda x: f"{x:.2f}%" if x > 0 else "0%（無成本）",
            help="每次買入或賣出時扣除的摩擦成本。\n美股 ETF 建議 0.05~0.1%，台股建議 0.1~0.2%（含證交稅），頻繁交易可設更高。"
        ) / 100   # 轉成小數

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
            "commission":  commission_pct,
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
                "均線偏離策略：當收盤價跌破買入均線時買入，"
                "當收盤價高於賣出均線達設定百分比時賣出。"
                "例如：跌破 30 日均線買入，高於 5 日均線 +10% 時賣出。"
                "買入/賣出使用不同週期均線，可捕捉回調後的反彈行情。",
        }
        strategy = st.selectbox("選擇回測策略",
            list(STRATEGY_HELP.keys()),
            key="w_strategy",
            help=STRATEGY_HELP.get("MA 交叉策略"))
        with st.expander("📖 策略說明", expanded=False):
            st.caption(STRATEGY_HELP[strategy])

        params = {}
        with st.expander("⚙️ 策略參數設定", expanded=True):
            if strategy == "MA 交叉策略":
                params["ma_fast"] = st.slider("快線 MA", 5, 100, 50, key="w_ma_fast")
                params["ma_slow"] = st.slider("慢線 MA", 50, 300, 200, key="w_ma_slow")
            elif strategy == "RSI 動能策略":
                params["rsi_period"] = st.slider("RSI 週期", 5, 30, 14, key="w_rsi_period")
                params["rsi_buy"]    = st.slider("超賣門檻（買入）", 10, 40, 30, key="w_rsi_buy")
                params["rsi_sell"]   = st.slider("超買門檻（賣出）", 60, 90, 70, key="w_rsi_sell")
            elif strategy == "布林通道策略":
                params["bb_period"] = st.slider("布林週期", 5, 50, 20, key="w_bb_period")
                params["bb_std"]    = st.select_slider("標準差倍數", [1.5, 2.0, 2.5, 3.0], value=2.0, key="w_bb_std")
            elif strategy == "MACD 趨勢策略":
                params["macd_fast"]   = st.slider("MACD 快線", 5, 20, 12, key="w_macd_fast")
                params["macd_slow"]   = st.slider("MACD 慢線", 15, 50, 26, key="w_macd_slow")
                params["macd_signal"] = st.slider("MACD 信號線", 5, 20, 9, key="w_macd_signal")
            elif strategy == "MA均線偏離策略":
                st.markdown("**📉 買入條件**")
                bc1, bc2, bc3 = st.columns(3)
                params["dev_buy_period"] = bc1.number_input(
                    "買入均線週期", min_value=5, max_value=500, value=30, step=5,
                    help="當收盤價跌破此均線時買入。例如 30 = 跌破 30 日均線即買入")
                params["dev_buy_pct"] = 0.0
                params["dev_buy_cooldown"] = int(bc2.number_input(
                    "買入冷卻（交易日）", min_value=1, max_value=250, value=20, step=5,
                    help="買入觸發後，至少等幾個交易日才能再次買入。20≈1月，60≈1季"))
                st.caption(
                    f"買入：跌破 {params['dev_buy_period']} 日均線，"
                    f"冷卻 {params['dev_buy_cooldown']} 交易日"
                )

                st.markdown("**📈 賣出條件**")
                sc1, sc2, sc3 = st.columns(3)
                params["dev_sell_period"] = sc1.number_input(
                    "賣出均線週期", min_value=5, max_value=500, value=5, step=5,
                    help="計算賣出基準的移動平均線週期")
                params["dev_sell_pct"] = sc2.number_input(
                    "高於均線 % 賣出", min_value=0.1, max_value=100.0, value=20.0, step=0.5,
                    help="收盤價高於此均線多少 % 時賣出")
                params["dev_sell_cooldown"] = int(sc3.number_input(
                    "賣出冷卻（交易日）", min_value=1, max_value=250, value=20, step=5,
                    help="賣出觸發後，至少等幾個交易日才能再次賣出。20≈1月，60≈1季"))
                st.caption(
                    f"賣出：高於 {params['dev_sell_period']} 日均線 "
                    f"+{params['dev_sell_pct']:.1f}%，"
                    f"冷卻 {params['dev_sell_cooldown']} 交易日"
                )

        st.markdown("---")
        st.markdown("### 🛡️ 移動停利（Trailing Stop）")
        enable_ts = st.toggle("開啟移動停利", value=False,
                              help="持倉期間追蹤最高價，若回撤超過設定幅度則強制賣出，避免獲利回吐或長期套牢")
        ts_cfg = {"enabled": False}
        if enable_ts:
            ts_mode = st.radio("觸發方式", ["回撤百分比", "固定價差"],
                               horizontal=True,
                               help="回撤百分比：從持倉高點下跌 N% 觸發；固定價差：從持倉高點下跌固定金額觸發")
            if ts_mode == "回撤百分比":
                ts_pct = st.slider("回撤觸發比例", 1, 50, 10, step=1,
                                   format="%d%%",
                                   help="例如 10% = 股價從持倉期間最高點跌超過 10% 就賣出")
                ts_cfg = {"enabled": True, "mode": "pct", "value": ts_pct / 100}
                st.caption(f"持倉高點回落 {ts_pct}% 時強制賣出")
            else:
                ts_gap = st.number_input("固定價差觸發（$）", min_value=0.1,
                                         max_value=10000.0, value=10.0, step=0.5,
                                         help="例如 10 = 股價從持倉期間最高點下跌 $10 就賣出")
                ts_cfg = {"enabled": True, "mode": "fixed", "value": ts_gap}
                st.caption(f"持倉高點下跌 ${ts_gap:.2f} 時強制賣出")

        st.markdown("---")
        st.markdown("### 🔬 VCP 篩選器")
        enable_vcp = st.toggle("開啟 VCP 趨勢檢查", value=False)

        st.markdown("---")
        run_btn = st.button("🚀 執行回測分析", type="primary")
        # 套用最佳化結果後自動觸發回測（_auto_run 已在 main() 頂部取出）
        if _auto_run:
            run_btn = True


    # ── 主分頁（sidebar 在兩個 Tab 共用）──
    main_tab1, main_tab2 = st.tabs(["🎯 單一標的策略回測", "💼 資產配置組合分析"])

    with main_tab2:
        render_portfolio_tab()

    with main_tab1:
        if not run_btn:
            st.info("👈 請在左側設定參數後，點擊「執行回測分析」開始分析。", icon="💡")
            col1, col2, col3, col4 = st.columns(4)
            col1.markdown("**MA 交叉策略**\n\n快慢均線黃金/死亡交叉，捕捉趨勢轉換。")
            col2.markdown("**RSI 動能策略**\n\n超賣買入、超買賣出，適合震盪行情。")
            col3.markdown("**布林通道策略**\n\n觸碰下軌買入、上軌賣出，依賴均值回歸。")
            col4.markdown("**MACD 趨勢策略**\n\nMACD 交叉追蹤動能方向。")
        else:
            if _auto_run:
                st.success("✅ 已套用最佳化設定，正在執行回測...", icon="🚀")

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
    
            result = run_backtest(df, inv_cfg, ts_cfg)
            bm     = compute_benchmark(df, inv_cfg)
            is_dca = inv_cfg["mode"] == "dca"
            years  = (df.index[-1] - df.index[0]).days / 365.25
    
            # 把回測所需資料存入 session_state，讓最佳化按鈕跨次重新執行時能取用
            st.session_state["_opt_df_raw"]  = df_raw
            st.session_state["_opt_inv_cfg"] = inv_cfg
            st.session_state["_opt_ticker"]  = ticker
    
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
    
            tab1, tab2, tab3, tab4 = st.tabs(["📈 資產增長曲線", "🕯️ K線圖與買賣標記", "📉 回撤分析", "💰 現金比例診斷"])
    
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
    
            with tab4:
                # ── 現金比例 = 現金池 / 總資產 ──
                a1_cash_s = result["acc1_cash_series"]
                a2_cash_s = result["acc2_cash_series"]
                a1_total_s = result["acc1_series"]
                a2_total_s = result["acc2_series"]
    
                # 避免除以零
                a1_cash_pct = (a1_cash_s / a1_total_s.replace(0, np.nan) * 100).fillna(0)
                a2_cash_pct = (a2_cash_s / a2_total_s.replace(0, np.nan) * 100).fillna(0)
    
                fig_cash = go.Figure()
    
                # acc2（主線）現金比例
                fig_cash.add_trace(go.Scatter(
                    x=a2_cash_pct.index, y=a2_cash_pct.values,
                    name="現金比例（策略+DCA）" if is_dca else "現金比例（策略）",
                    line=dict(color="#0369a1", width=2),
                    fill="tozeroy", fillcolor="rgba(3,105,161,0.08)"))
    
                # acc1 現金比例（DCA 模式時才顯示對比）
                if is_dca:
                    fig_cash.add_trace(go.Scatter(
                        x=a1_cash_pct.index, y=a1_cash_pct.values,
                        name="現金比例（純策略無DCA）",
                        line=dict(color="#93c5fd", width=1.5, dash="dash")))
    
                # 加入買賣信號標記線（方便對照何時現金變化）
                for bd in result["buy_dates"]:
                    fig_cash.add_vline(x=bd, line_color="#16a34a",
                                       line_width=0.8, line_dash="dot", opacity=0.4)
                for sd in result["sell_dates"]:
                    fig_cash.add_vline(x=sd, line_color="#dc2626",
                                       line_width=0.8, line_dash="dot", opacity=0.4)
    
                fig_cash.update_layout(
                    title=dict(
                        text="💰 現金池比例（綠虛線=買入信號 / 紅虛線=賣出信號）",
                        font=dict(size=13, color="#0f172a")),
                    xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
                    yaxis=dict(showgrid=True, gridcolor="#e2e8f0",
                               ticksuffix="%", range=[-2, 102]),
                    paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                    font=CHART["font"], hovermode="x unified", height=400,
                    legend=dict(bgcolor="rgba(255,255,255,0.9)",
                                bordercolor="#e2e8f0", borderwidth=1),
                    margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_cash, use_container_width=True)
    
                # 診斷摘要
                avg_cash = float(a2_cash_pct.mean())
                max_cash = float(a2_cash_pct.max())
                zero_days = int((a2_cash_pct < 1).sum())
                total_days = len(a2_cash_pct)
                pct_zero = zero_days / total_days * 100
    
                d1, d2, d3 = st.columns(3)
                d1.metric("平均現金比例", f"{avg_cash:.1f}%",
                          help="越低代表資金越積極運用，但也意味著較少備用資金")
                d2.metric("最高現金比例", f"{max_cash:.1f}%",
                          help="通常出現在賣出後、等待下次買入信號期間")
                d3.metric("現金近乎 0 的天數", f"{zero_days} 天 ({pct_zero:.0f}%)",
                          help="這段期間即使出現買入信號也無法加碼，可能代表需要增加初始資金或調整賣出策略")
    
                if pct_zero > 60:
                    st.warning("⚠️ 超過 60% 的時間現金池接近 0，若此期間出現買入信號將無法執行。"
                               "建議：增加初始資金、降低買入倉位比例、或調整賣出策略讓資金更早釋放。")
                elif avg_cash > 40:
                    st.info("💡 平均現金比例偏高，資金有較多閒置時間。"
                            "若希望更積極運用，可嘗試降低買入均線週期或縮短冷卻期。")
    
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

        # ── 策略最佳化 ──
        st.markdown("## 🔍 策略最佳化")
        st.caption("對此標的進行 Grid Search，依自訂指標加權找出最佳策略，再由 Claude AI 解讀結果。")

        # ── 加權設定 ──
        with st.expander("⚖️ 自訂指標加權（合計須為 100%）", expanded=True):
            st.caption("調整各指標的重要程度，系統會將每個指標標準化後依比重計算綜合分數。")
            wc1, wc2, wc3, wc4 = st.columns(4)
            w_cagr  = wc1.slider("📈 年化報酬 CAGR",    0, 100, 30, 5,
                                 help="越高代表越重視絕對報酬率")
            w_shar  = wc2.slider("📐 夏普比率",          0, 100, 30, 5,
                                 help="越高代表越重視風險調整後報酬")
            w_dd    = wc3.slider("📉 最大回撤（越小越好）", 0, 100, 25, 5,
                                 help="越高代表越重視減少最大虧損幅度")
            w_stab  = wc4.slider("🏔️ 穩定性（Calmar）",  0, 100, 15, 5,
                                 help="CAGR/最大回撤，衡量每單位風險獲得的報酬是否穩定")
            total_w = w_cagr + w_shar + w_dd + w_stab
            if total_w == 100:
                st.success(f"✅ 合計 {total_w}%，配置有效")
            else:
                st.error(f"⚠️ 合計 {total_w}%，請調整至 100% 才能執行最佳化")

        weights = {
            "cagr":      w_cagr  / 100,
            "sharpe":    w_shar  / 100,
            "max_dd":    w_dd    / 100,
            "stability": w_stab  / 100,
        }

        opt_c1, opt_c2 = st.columns([2, 1])
        with opt_c1:
            api_key = st.text_input("Anthropic API Key", type="password",
                                    placeholder="sk-ant-...",
                                    help="用於 Claude AI 解讀。Key 不會被儲存。")
        with opt_c2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_opt = st.button("🚀 開始最佳化", type="secondary",
                                disabled=(total_w != 100))

        # ── 執行 Grid Search（只在按下按鈕時跑）──
        if run_opt:
            _df_raw  = st.session_state.get("_opt_df_raw")
            _inv_cfg = st.session_state.get("_opt_inv_cfg")
            if _df_raw is None or _inv_cfg is None:
                st.error("⚠️ 請先點擊「執行回測分析」完成回測，再執行最佳化。")
            else:
                weight_desc = (f"CAGR {w_cagr}% ／ 夏普 {w_shar}% ／ "
                               f"回撤 {w_dd}% ／ 穩定性 {w_stab}%")
                with st.spinner("Grid Search 執行中，約需 10~30 秒..."):
                    opt_results = run_grid_search(_df_raw, _inv_cfg, weights)
                if opt_results:
                    st.session_state["_opt_results"]     = opt_results
                    st.session_state["_opt_weight_desc"] = weight_desc
                    st.session_state["_opt_ticker_val"]  = st.session_state.get("_opt_ticker", "")
                else:
                    st.error("最佳化失敗，請確認數據是否充足（建議至少 3 年）。")

        # ── 顯示結果（永遠從 session_state 讀，與 run_opt 無關）──
        if "_opt_results" in st.session_state:
            _prev       = st.session_state["_opt_results"][:10]
            _prev_desc  = st.session_state.get("_opt_weight_desc", "")
            _prev_ticker= st.session_state.get("_opt_ticker_val", ticker)

            st.markdown("### 📊 Top 10 策略排名")
            if _prev_desc:
                st.caption(f"加權方式：{_prev_desc}")

            rows = []
            for rank, r in enumerate(_prev, 1):
                rows.append({
                    "排名":    rank,       "策略":    r["strategy"],
                    "策略參數": r["param_summary"],
                    "倉位設定": r.get("pos_summary", "—"),
                    "綜合分數": f"{r['score']:.4f}",
                    "夏普比率": f"{r['sharpe']:+.3f}",
                    "CAGR":   f"{r['cagr']:+.2%}",
                    "最大回撤": f"{r['max_dd']:.2%}",
                    "穩定性":  f"{r['stability']:.3f}",
                    "交易次數": r["n_trades"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # ── 前三名一鍵套用（永遠顯示，點擊後立即生效）──
            st.markdown("#### 🏅 前三名一鍵套用")
            st.caption("點擊後側邊欄設定自動更新，並立即重新執行回測。")
            medals = ["🥇", "🥈", "🥉"]
            top3_cols = st.columns(3)
            for col_idx, (col, r) in enumerate(zip(top3_cols, _prev[:3])):
                cfg      = r.get("inv_cfg", {})
                buy_lbl  = {"all_in":"全倉買","fixed_pct":f"{cfg.get('buy_pct',1)*100:.0f}%買","fixed_amount":f"固定${cfg.get('buy_amount',0):,.0f}買"}.get(cfg.get("buy_mode","all_in"),"全倉買")
                sell_lbl = {"all_out":"全倉賣","fixed_pct":f"{cfg.get('sell_pct',1)*100:.0f}%賣","fixed_amount":f"固定${cfg.get('sell_amount',0):,.0f}賣"}.get(cfg.get("sell_mode","all_out"),"全倉賣")
                dca_lbl  = "DCA" if cfg.get("mode") == "dca" else "一次性"
                with col:
                    st.markdown(
                        f"**{medals[col_idx]} 第{col_idx+1}名**\n\n"
                        f"`{r['strategy']}`\n\n"
                        f"📐 {r['param_summary']}\n\n"
                        f"💰 {dca_lbl} ／ {buy_lbl} ／ {sell_lbl}\n\n"
                        f"📈 CAGR {r['cagr']:+.2%}　回撤 {r['max_dd']:.2%}　分數 {r['score']:.3f}"
                    )
                    if st.button(f"套用第{col_idx+1}名設定",
                                 key=f"apply_btn_{col_idx}",
                                 type="primary" if col_idx == 0 else "secondary",
                                 use_container_width=True):
                        apply_opt_result(r)

            # ── Claude AI（只在剛跑完時）──
            if run_opt:
                if api_key.startswith("sk-ant-"):
                    with st.spinner("Claude AI 分析中..."):
                        ai_analysis = call_claude_analysis(
                            api_key, _prev_ticker, _prev,
                            {"cagr": w_cagr/100, "sharpe": w_shar/100,
                             "max_dd": w_dd/100, "stability": w_stab/100})
                    st.markdown("### 🤖 Claude AI 策略解讀")
                    st.markdown(ai_analysis)
                elif api_key:
                    st.warning("API Key 格式不正確，請確認是否以 sk-ant- 開頭。")
                else:
                    st.info("輸入 Anthropic API Key 可獲得 Claude AI 的策略解讀與建議。")
        else:
            st.info("點擊「開始最佳化」執行 Grid Search，完成後可在此查看結果並一鍵套用。", icon="💡")
            # ── 前三名套用按鈕（restore 區塊也要有）──
            st.markdown("#### 🏅 前三名一鍵套用")
            medals = ["🥇", "🥈", "🥉"]
            top3_cols = st.columns(3)
            for col_idx, (col, r) in enumerate(zip(top3_cols, _prev[:3])):
                cfg = r.get("inv_cfg", {})
                buy_lbl  = {"all_in":"全倉買","fixed_pct":f"{cfg.get('buy_pct',1)*100:.0f}%買","fixed_amount":f"固定${cfg.get('buy_amount',0):,.0f}買"}.get(cfg.get("buy_mode","all_in"),"全倉買")
                sell_lbl = {"all_out":"全倉賣","fixed_pct":f"{cfg.get('sell_pct',1)*100:.0f}%賣","fixed_amount":f"固定${cfg.get('sell_amount',0):,.0f}賣"}.get(cfg.get("sell_mode","all_out"),"全倉賣")
                dca_lbl  = "DCA" if cfg.get("mode") == "dca" else "一次性"
                with col:
                    st.markdown(
                        f"**{medals[col_idx]} 第{col_idx+1}名**\n\n"
                        f"`{r['strategy']}`\n\n"
                        f"📐 {r['param_summary']}\n\n"
                        f"💰 {dca_lbl} ／ {buy_lbl} ／ {sell_lbl}\n\n"
                        f"📈 CAGR {r['cagr']:+.2%}　回撤 {r['max_dd']:.2%}"
                    )
                    if st.button(f"套用第{col_idx+1}名設定", key=f"restore_btn_{col_idx}",
                                 type="primary" if col_idx == 0 else "secondary",
                                 use_container_width=True):
                        apply_opt_result(r)
            st.caption("💡 調整加權比例後點擊「開始最佳化」重新計算排名。")

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center;color:#cbd5e1;font-family:IBM Plex Mono;font-size:0.7rem;'>"
            "數據來源: Yahoo Finance · 本工具僅供學習研究，不構成投資建議</div>",
            unsafe_allow_html=True)

if __name__ == "__main__":
    main()
