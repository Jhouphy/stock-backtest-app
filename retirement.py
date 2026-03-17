"""
retirement.py
退休規劃計算器 — 財務自由試算
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 圖表主題（與 app.py / portfolio.py 一致）──
CHART = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafc",
    gridcolor="#e2e8f0",
    font=dict(family="Inter, sans-serif", color="#475569", size=11),
)


# ═══════════════════════════════════════════════
# 核心計算引擎
# ═══════════════════════════════════════════════

def calc_retirement(
    age_start: int,
    year_start: int,
    initial: float,
    monthly_contrib: float,
    annual_return: float,       # 年化報酬率（小數）
    inflation: float,           # 通膨率（小數）
    withdrawal_rate: float,     # 提領率（小數）
    monthly_expense: float,     # 月生活開銷（名目，第一年）
    years: int = 60,            # 試算年數
    contrib_stop_age: int = 0,  # 幾歲停止投入（0=持續到財務自由）
) -> pd.DataFrame:
    """
    逐年計算資產變化，直到達成財務自由或試算結束。

    財務自由條件：投資回報（資產 × 提領率）≥ 當年生活開銷
    """
    rows = []
    asset = float(initial)
    monthly_r = (1 + annual_return) ** (1 / 12) - 1
    fi_year = None
    fi_age  = None

    for yr in range(years):
        age  = age_start + yr
        year = year_start + yr

        # 當年生活開銷（含通膨）
        expense_annual = monthly_expense * 12 * (1 + inflation) ** yr

        # 當年投入（月複利累積）
        stop = contrib_stop_age if contrib_stop_age > 0 else 9999
        if age < stop:
            contrib_annual = monthly_contrib * 12
            # 月複利：每月初投入，年底結算
            contrib_fv = monthly_contrib * (
                ((1 + monthly_r) ** 12 - 1) / monthly_r
            ) if monthly_r > 0 else monthly_contrib * 12
        else:
            contrib_annual = 0.0
            contrib_fv     = 0.0

        # 期初資產成長 + 當年投入
        asset_begin = asset
        asset = asset * (1 + annual_return) + contrib_fv
        invest_return = asset_begin * annual_return

        # 財務自由判斷
        fi_income = asset * withdrawal_rate   # 年可提領金額
        is_fi     = fi_income >= expense_annual

        if is_fi and fi_year is None:
            fi_year = year
            fi_age  = age

        rows.append({
            "年齡":      age,
            "年度":      yr,
            "年份":      year,
            "生活開銷":  expense_annual,
            "投資價值":  asset,
            "投資回報":  invest_return,
            "可提領金額": fi_income,
            "年投入":    contrib_annual,
            "財務自由了嗎": "YES!!" if is_fi else "No",
            "_fi":       is_fi,
        })

    df = pd.DataFrame(rows)
    df.attrs["fi_year"] = fi_year
    df.attrs["fi_age"]  = fi_age
    return df


# ═══════════════════════════════════════════════
# 圖表
# ═══════════════════════════════════════════════

def plot_retirement(df: pd.DataFrame, currency_symbol: str = "$") -> go.Figure:
    """資產成長 vs 生活開銷折線圖，標記財務自由點。"""
    fig = go.Figure()

    # 投資價值（主線）
    fig.add_trace(go.Scatter(
        x=df["年份"], y=df["投資價值"],
        name="💼 投資資產",
        line=dict(color="#7c3aed", width=2.5),
        hovertemplate=f"年份 %{{x}}：{currency_symbol}%{{y:,.0f}}<extra>投資資產</extra>",
    ))

    # 可提領金額
    fig.add_trace(go.Scatter(
        x=df["年份"], y=df["可提領金額"],
        name="💰 年可提領金額",
        line=dict(color="#0369a1", width=2, dash="dash"),
        hovertemplate=f"年份 %{{x}}：{currency_symbol}%{{y:,.0f}}<extra>年可提領</extra>",
    ))

    # 生活開銷（含通膨）
    fig.add_trace(go.Scatter(
        x=df["年份"], y=df["生活開銷"],
        name="🏠 年生活開銷",
        line=dict(color="#dc2626", width=2),
        hovertemplate=f"年份 %{{x}}：{currency_symbol}%{{y:,.0f}}<extra>年生活開銷</extra>",
    ))

    # 財務自由交叉點
    fi_year = df.attrs.get("fi_year")
    if fi_year:
        fi_row  = df[df["年份"] == fi_year].iloc[0]
        fig.add_vline(
            x=fi_year,
            line_color="#16a34a", line_width=2, line_dash="dot",
            annotation_text=f"🎉 財務自由 {fi_year}",
            annotation_font=dict(color="#16a34a", size=12),
            annotation_position="top right",
        )
        fig.add_trace(go.Scatter(
            x=[fi_year], y=[fi_row["投資價值"]],
            mode="markers",
            marker=dict(color="#16a34a", size=12, symbol="star"),
            name="🎉 財務自由點",
            hovertemplate=f"財務自由！資產 {currency_symbol}%{{y:,.0f}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="📈 財務自由試算曲線", font=dict(size=14, color="#0f172a")),
        xaxis=dict(title="年份", showgrid=True, gridcolor=CHART["gridcolor"]),
        yaxis=dict(
            title=f"金額 ({currency_symbol})",
            showgrid=True, gridcolor=CHART["gridcolor"],
            tickprefix=currency_symbol, tickformat=",.0f",
        ),
        paper_bgcolor=CHART["paper_bgcolor"],
        plot_bgcolor=CHART["plot_bgcolor"],
        font=CHART["font"],
        hovermode="x unified",
        height=450,
        dragmode="pan",
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e2e8f0", borderwidth=1,
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
        margin=dict(l=10, r=10, t=65, b=10),
    )
    return fig


def plot_asset_breakdown(df: pd.DataFrame, currency_symbol: str = "$") -> go.Figure:
    """資產累積 vs 累積投入（顯示投資增值部分）。"""
    cum_contrib = df["年投入"].cumsum() + df["投資價值"].iloc[0]  # 累積投入（含初始）

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["年份"], y=df["投資價值"],
        name="💼 總資產",
        fill="tozeroy",
        fillcolor="rgba(124,58,237,0.08)",
        line=dict(color="#7c3aed", width=2),
        hovertemplate=f"{currency_symbol}%{{y:,.0f}}<extra>總資產</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["年份"], y=cum_contrib,
        name="📥 累積投入本金",
        fill="tozeroy",
        fillcolor="rgba(148,163,184,0.25)",
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
        hovertemplate=f"{currency_symbol}%{{y:,.0f}}<extra>累積投入</extra>",
    ))
    fig.update_layout(
        title=dict(text="📊 總資產 vs 累積投入（差額即投資增值）",
                   font=dict(size=13, color="#0f172a")),
        xaxis=dict(showgrid=True, gridcolor=CHART["gridcolor"]),
        yaxis=dict(showgrid=True, gridcolor=CHART["gridcolor"],
                   tickprefix=currency_symbol, tickformat=",.0f"),
        paper_bgcolor=CHART["paper_bgcolor"],
        plot_bgcolor=CHART["plot_bgcolor"],
        font=CHART["font"],
        hovermode="x unified", height=350, dragmode="pan",
        legend=dict(bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e2e8f0", borderwidth=1),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ═══════════════════════════════════════════════
# 主渲染函數
# ═══════════════════════════════════════════════

def render_retirement_tab():
    st.markdown("### 🏖️ 退休規劃計算器")
    st.caption("輸入個人財務狀況，試算達成財務自由的時間與資產軌跡。")

    with st.expander("⚙️ 基本設定", expanded=True):
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        age_start   = r1c1.number_input("目前年齡", 18, 80, 25, 1)
        year_start  = r1c2.number_input("起始年份", 2000, 2100, 2026, 1)
        currency    = r1c3.selectbox("計價貨幣", ["USD ($)", "TWD (NT$)", "JPY (¥)", "EUR (€)"])
        years       = r1c4.selectbox("試算年數", [30, 40, 50, 60, 70], index=2)
        sym_map     = {"USD ($)": "$", "TWD (NT$)": "NT$", "JPY (¥)": "¥", "EUR (€)": "€"}
        csym        = sym_map[currency]

        st.markdown("---")
        r2c1, r2c2, r2c3 = st.columns(3)
        initial       = r2c1.number_input(
            f"初始投資 ({csym})", 0, 1_000_000_000, 5_900_000, 100_000, format="%d")
        monthly_contrib = r2c2.number_input(
            f"每月投入 ({csym})", 0, 10_000_000, 50_000, 1_000, format="%d")
        contrib_stop_age = r2c3.number_input(
            "幾歲停止投入（0=持續至財務自由）", 0, 100, 0, 1,
            help="設 0 表示持續投入直到達到財務自由為止")

        st.markdown("---")
        r3c1, r3c2, r3c3 = st.columns(3)
        annual_return_pct = r3c1.slider(
            "預估年化報酬率", 0.0, 20.0, 8.0, 0.5, format="%.1f%%")
        inflation_pct = r3c2.slider(
            "通貨膨脹率 / 年", 0.0, 10.0, 2.0, 0.5, format="%.1f%%")
        withdrawal_pct = r3c3.slider(
            "提領率（4% 法則）", 1.0, 10.0, 4.0, 0.5, format="%.1f%%",
            help="財務自由後每年從資產中提領的比例。4% 為常見標準（三十年不耗盡）。")

        st.markdown("---")
        st.markdown("**🏠 生活開銷設定**")
        exp_c1, exp_c2 = st.columns(2)
        monthly_expense = exp_c1.number_input(
            f"月生活開銷 ({csym})（現值）", 0, 10_000_000, 40_000, 1_000, format="%d",
            help="目前每月的生活開銷，計算時會按通膨率逐年調整。")

        # 連動：計算財務自由所需資產（月支出反推目標資產）
        annual_expense_now = monthly_expense * 12
        fi_target = annual_expense_now / withdrawal_pct * 100 if withdrawal_pct > 0 else 0
        exp_c2.markdown(f"""
        <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:10px;
                    padding:14px 18px;margin-top:4px">
            <div style="font-size:0.78rem;color:#166534;font-weight:600;margin-bottom:4px">
                💡 財務自由目標資產（現值）
            </div>
            <div style="font-size:1.2rem;font-weight:700;color:#15803d">
                {csym}{fi_target:,.0f}
            </div>
            <div style="font-size:0.72rem;color:#4ade80;margin-top:2px">
                月支出 {csym}{monthly_expense:,} × 12 ÷ {withdrawal_pct}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── 執行計算 ──
    df = calc_retirement(
        age_start       = age_start,
        year_start      = year_start,
        initial         = initial,
        monthly_contrib = monthly_contrib,
        annual_return   = annual_return_pct / 100,
        inflation       = inflation_pct / 100,
        withdrawal_rate = withdrawal_pct / 100,
        monthly_expense = monthly_expense,
        years           = years,
        contrib_stop_age= contrib_stop_age,
    )

    fi_year = df.attrs.get("fi_year")
    fi_age  = df.attrs.get("fi_age")

    # ── 結果摘要 ──
    st.markdown("---")
    if fi_year:
        years_to_fi = fi_year - year_start
        fi_row = df[df["年份"] == fi_year].iloc[0]
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);
                    border:2px solid #16a34a;border-radius:16px;
                    padding:20px 28px;margin-bottom:16px;text-align:center">
            <div style="font-size:2rem;margin-bottom:4px">🎉</div>
            <div style="font-size:1.2rem;font-weight:700;color:#15803d">
                預計 <span style="font-size:1.6rem">{fi_year}</span> 年達成財務自由
            </div>
            <div style="color:#166534;margin-top:6px">
                年齡 {fi_age} 歲 ／ 從現在起 {years_to_fi} 年後
                ／ 屆時資產 {csym}{fi_row['投資價值']:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ 在 {years} 年試算期間內尚未達成財務自由，建議提高月投入或降低目標支出。")

    # ── Metrics ──
    last_row = df.iloc[-1]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("最終資產", f"{csym}{last_row['投資價值']:,.0f}")
    m2.metric("最終年可提領", f"{csym}{last_row['可提領金額']:,.0f}")
    m3.metric("最終生活開銷/年", f"{csym}{last_row['生活開銷']:,.0f}")
    total_contrib = initial + monthly_contrib * 12 * min(
        years, (fi_age - age_start) if fi_age else years)
    m4.metric("累積總投入（估）", f"{csym}{total_contrib:,.0f}")
    m5.metric("財務自由年齡", f"{fi_age} 歲" if fi_age else "未達成")

    # ── 圖表 ──
    st.markdown("---")
    rt1, rt2 = st.tabs(["📈 財務自由曲線", "📊 資產 vs 投入"])

    _chart_cfg = dict(
        locale="zh-TW", displayModeBar=True,
        modeBarButtonsToRemove=["select2d", "lasso2d", "autoScale2d"],
        toImageButtonOptions={"format": "png", "filename": "retirement_chart", "scale": 2},
        displaylogo=False, scrollZoom=False,
    )

    with rt1:
        st.plotly_chart(plot_retirement(df, csym),
                        use_container_width=True, config=_chart_cfg)

    with rt2:
        st.plotly_chart(plot_asset_breakdown(df, csym),
                        use_container_width=True, config=_chart_cfg)

    # ── 逐年明細表 ──
    st.markdown("---")
    st.markdown("#### 📋 逐年明細")

    display_df = df[[
        "年齡", "年度", "年份", "生活開銷", "投資價值", "投資回報", "可提領金額", "財務自由了嗎"
    ]].copy()

    # 格式化
    for col in ["生活開銷", "投資價值", "投資回報", "可提領金額"]:
        display_df[col] = display_df[col].apply(lambda x: f"{csym}{x:,.0f}")

    # 高亮財務自由後的列
    def highlight_fi(row):
        if row["財務自由了嗎"] == "YES!!":
            return ["background-color: #f0fdf4; color: #15803d"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df.style.apply(highlight_fi, axis=1),
        use_container_width=True,
        hide_index=True,
        height=400,
    )
