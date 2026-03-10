"""
charts.py
圖表渲染模組
包含：資產曲線、K線圖、回撤圖（使用 Plotly）
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine import CHART

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

