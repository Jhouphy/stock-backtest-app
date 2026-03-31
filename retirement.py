"""
retirement.py
退休規劃計算器 — 財務自由試算
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from settings import init_session, save_settings

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
    annual_return: float,
    inflation: float,
    withdrawal_rate: float,
    monthly_expense: float,
    years: int = 50,
    contrib_stop_age: int = 0,
    withdrawal_start_age: int = 0,
    # ── 三階段新增參數 ──
    reduce_contrib_age: int = 0,       # 幾歲開始減少投入（0=不啟用）
    reduced_monthly_contrib: float = 0, # 減少後的每月投入金額
) -> pd.DataFrame:
    """
    三階段模型：
    1. 全額投入期：age_start → reduce_contrib_age（monthly_contrib）
    2. 減少投入期：reduce_contrib_age → contrib_stop_age（reduced_monthly_contrib）
       若未啟用減少，則直接從全額投入跳至停止
    3. 提領期：contrib_stop_age 後按提領率提領（含通膨調整）
    財務自由條件：可提領金額 ≥ 當年生活開銷
    """
    rows = []
    asset = float(initial)
    monthly_r = (1 + annual_return) ** (1 / 12) - 1
    fi_year = None
    fi_age  = None

    # 決定各階段門檻
    _reduce = reduce_contrib_age if reduce_contrib_age > 0 else 9999
    _stop   = contrib_stop_age   if contrib_stop_age   > 0 else 9999
    # 提領起始：若未設定則緊接停止投入
    _wd_start = withdrawal_start_age if withdrawal_start_age > 0 else _stop

    for yr in range(years):
        age  = age_start + yr
        year = year_start + yr

        # 當年生活開銷（含通膨）
        expense_annual = monthly_expense * 12 * (1 + inflation) ** yr

        # ── 判斷當前投入金額（三階段）──
        if age >= _stop:
            _contrib = 0.0          # 階段 3：停止投入
            phase    = "stop"
        elif age >= _reduce:
            _contrib = reduced_monthly_contrib  # 階段 2：減少投入
            phase    = "reduce"
        else:
            _contrib = monthly_contrib          # 階段 1：全額投入
            phase    = "full"

        if _contrib > 0:
            contrib_fv = _contrib * (
                ((1 + monthly_r) ** 12 - 1) / monthly_r
            ) if monthly_r > 0 else _contrib * 12
            contrib_annual = _contrib * 12
        else:
            contrib_fv     = 0.0
            contrib_annual = 0.0

        # ── 資產成長 ──
        asset_begin   = asset
        asset_growth  = asset_begin * (1 + annual_return) + contrib_fv
        invest_return = asset_begin * annual_return

        # ── 提領期：從資產中扣除提領金額 ──
        in_withdrawal_phase = age >= _wd_start
        withdrawal_amount   = 0.0
        asset_after_withdrawal = asset_growth

        if in_withdrawal_phase:
            withdrawal_amount      = asset_growth * withdrawal_rate
            asset_after_withdrawal = asset_growth - withdrawal_amount

        asset = asset_after_withdrawal

        # 可提領金額（用於財務自由判斷）
        fi_income = asset_growth * withdrawal_rate

        # 財務自由判斷
        is_fi = fi_income >= expense_annual
        if is_fi and fi_year is None:
            fi_year = year
            fi_age  = age

        rows.append({
            "年齡":           age,
            "年度":           yr,
            "年份":           year,
            "生活開銷":       expense_annual,
            "投資價值":       asset,
            "投資價值_提領前": asset_growth,
            "投資回報":       invest_return,
            "可提領金額":     fi_income,
            "實際提領金額":   withdrawal_amount,
            "年投入":         contrib_annual,
            "投入階段":       phase,
            "財務自由了嗎":   "YES!!" if is_fi else "No",
            "提領中":         in_withdrawal_phase,
            "_fi":            is_fi,
        })

    df = pd.DataFrame(rows)
    df.attrs["fi_year"]      = fi_year
    df.attrs["fi_age"]       = fi_age
    df.attrs["wd_start"]     = _wd_start
    df.attrs["reduce_age"]   = reduce_contrib_age if reduce_contrib_age > 0 else None
    df.attrs["stop_age"]     = contrib_stop_age   if contrib_stop_age   > 0 else None
    return df


# ═══════════════════════════════════════════════
# 圖表
# ═══════════════════════════════════════════════

def plot_retirement(df: pd.DataFrame, currency_symbol: str = "$",
                    withdrawal_start_age: int = 0) -> go.Figure:
    fig = go.Figure()

    wd_start = df.attrs.get("wd_start", 9999)
    wd_df    = df[df["提領中"]]
    acc_df   = df[~df["提領中"]]

    # ── 累積期資產（實線，紫）──
    fig.add_trace(go.Scatter(
        x=acc_df["年份"], y=acc_df["投資價值"],
        name="💼 資產（累積期）",
        line=dict(color="#7c3aed", width=2.5),
        hovertemplate=f"%{{x}}年：{currency_symbol}%{{y:,.0f}}<extra>資產（累積）</extra>",
    ))

    # ── 提領期資產（實線，橘，扣除提領後）──
    if not wd_df.empty:
        # 連接累積期最後一點到提領期
        connect = pd.concat([acc_df.iloc[[-1]], wd_df]) if not acc_df.empty else wd_df
        fig.add_trace(go.Scatter(
            x=connect["年份"], y=connect["投資價值"],
            name="💸 資產（提領期，扣提領後）",
            line=dict(color="#ea580c", width=2.5),
            hovertemplate=f"%{{x}}年：{currency_symbol}%{{y:,.0f}}<extra>資產（提領後）</extra>",
        ))

        # ── 每年實際提領金額（虛線，藍）──
        fig.add_trace(go.Scatter(
            x=wd_df["年份"], y=wd_df["實際提領金額"],
            name="💰 每年提領金額",
            line=dict(color="#0369a1", width=1.8, dash="dash"),
            hovertemplate=f"%{{x}}年：{currency_symbol}%{{y:,.0f}}<extra>年提領額</extra>",
        ))

    # ── 年生活開銷（虛線，紅）──
    fig.add_trace(go.Scatter(
        x=df["年份"], y=df["生活開銷"],
        name="🏠 年生活開銷（含通膨）",
        line=dict(color="#dc2626", width=1.8, dash="dot"),
        hovertemplate=f"%{{x}}年：{currency_symbol}%{{y:,.0f}}<extra>年生活開銷</extra>",
    ))

    # ── 提領開始標記 ──
    if not wd_df.empty:
        wd_year = int(wd_df["年份"].iloc[0])
        fig.add_vline(
            x=wd_year,
            line_color="#ea580c", line_width=1.5, line_dash="dash",
            annotation_text=f"🏖️ 開始提領 {wd_year}",
            annotation_font=dict(color="#ea580c", size=11),
            annotation_position="top left",
        )

    # ── 財務自由標記 ──
    fi_year = df.attrs.get("fi_year")
    if fi_year:
        fi_row = df[df["年份"] == fi_year].iloc[0]
        fig.add_vline(
            x=fi_year,
            line_color="#16a34a", line_width=1.5, line_dash="dot",
            annotation_text=f"🎉 財務自由 {fi_year}",
            annotation_font=dict(color="#16a34a", size=11),
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
        title=dict(text="📈 財務自由試算曲線（實線=資產，虛線=提領額/生活開銷）",
                   font=dict(size=13, color="#0f172a")),
        xaxis=dict(title="年份", showgrid=True, gridcolor=CHART["gridcolor"]),
        yaxis=dict(showgrid=True, gridcolor=CHART["gridcolor"],
                   tickprefix=currency_symbol, tickformat=",.0f"),
        paper_bgcolor=CHART["paper_bgcolor"],
        plot_bgcolor=CHART["plot_bgcolor"],
        font=CHART["font"],
        hovermode="x unified", height=480, dragmode="pan",
        legend=dict(bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e2e8f0", borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        margin=dict(l=10, r=10, t=75, b=10),
    )
    return fig


def plot_asset_breakdown(df: pd.DataFrame, currency_symbol: str = "$") -> go.Figure:
    cum_contrib = df["年投入"].cumsum() + (df["投資價值"].iloc[0] if len(df) > 0 else 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["年份"], y=df["投資價值"],
        name="💼 總資產（提領後）",
        fill="tozeroy", fillcolor="rgba(124,58,237,0.08)",
        line=dict(color="#7c3aed", width=2),
        hovertemplate=f"{currency_symbol}%{{y:,.0f}}<extra>總資產</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["年份"], y=cum_contrib,
        name="📥 累積投入本金",
        fill="tozeroy", fillcolor="rgba(148,163,184,0.2)",
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
        hovertemplate=f"{currency_symbol}%{{y:,.0f}}<extra>累積投入</extra>",
    ))
    fig.update_layout(
        title=dict(text="📊 總資產 vs 累積投入（差額即投資增值）",
                   font=dict(size=13, color="#0f172a")),
        xaxis=dict(showgrid=True, gridcolor=CHART["gridcolor"]),
        yaxis=dict(showgrid=True, gridcolor=CHART["gridcolor"],
                   tickprefix=currency_symbol, tickformat=",.0f"),
        paper_bgcolor=CHART["paper_bgcolor"], plot_bgcolor=CHART["plot_bgcolor"],
        font=CHART["font"], hovermode="x unified", height=350, dragmode="pan",
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0", borderwidth=1),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ═══════════════════════════════════════════════
# 主渲染函數
# ═══════════════════════════════════════════════

def render_retirement_tab():
    # ── 初始化：第一次載入時從 JSON 注入 session_state ──
    init_session("retirement", st.session_state)

    st.markdown("### 🏖️ 退休規劃計算器")
    st.caption("輸入個人財務狀況，試算達成財務自由的時間與資產軌跡。")

    with st.expander("⚙️ 基本設定", expanded=True):

        # ── 儲存按鈕列 ──
        save_col, status_col = st.columns([1, 3])
        if save_col.button("💾 儲存目前設定", key="ret_save_btn",
                           help="將目前所有設定存至本機 app_settings.json，下次開啟自動載入。"):
            _data = {k: st.session_state.get(k)
                     for k in [
                         "ret_age_start", "ret_year_start", "ret_currency",
                         "ret_years", "ret_initial", "ret_monthly_contrib",
                         "ret_contrib_change_mode", "ret_contrib_stop_age",
                         "ret_reduce_contrib_age", "ret_reduced_monthly_contrib",
                         "ret_annual_return_pct", "ret_inflation_pct",
                         "ret_withdrawal_pct", "ret_monthly_expense",
                         "ret_withdrawal_start_age",
                     ]}
            if save_settings("retirement", _data):
                st.session_state["_ret_save_ok"] = True
            else:
                st.session_state["_ret_save_ok"] = False

        if st.session_state.get("_ret_save_ok") is True:
            status_col.success("✅ 設定已儲存！下次開啟 App 將自動載入。", icon="💾")
        elif st.session_state.get("_ret_save_ok") is False:
            status_col.error("❌ 儲存失敗，請確認目錄有寫入權限。")

        st.markdown("---")

        # ── 第一行：基本資訊 ──
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        age_start  = r1c1.number_input("目前年齡", 18, 80, step=1,
                                        key="ret_age_start")
        year_start = r1c2.number_input("起始年份", 2000, 2100, step=1,
                                        key="ret_year_start")
        currency   = r1c3.selectbox("計價貨幣",
                        ["USD ($)", "TWD (NT$)", "JPY (¥)", "EUR (€)"],
                        key="ret_currency")
        sym_map    = {"USD ($)": "$", "TWD (NT$)": "NT$",
                      "JPY (¥)": "¥", "EUR (€)": "€"}
        csym       = sym_map[currency]

        # 試算年數：selectbox 預設選項 + 可手動輸入
        _preset_years = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70]
        _years_choice = r1c4.selectbox("試算年數（快選）", _preset_years, index=7)
        years = st.number_input(
            "或直接輸入試算年數", min_value=1, max_value=100,
            step=1, key="ret_years",
            help="可直接輸入任意年數，或用上方快選")

        st.markdown("---")

        # ── 第二行：資金設定 ──
        r2c1, r2c2 = st.columns(2)
        initial         = r2c1.number_input(
            f"初始投資 ({csym})", 0, 1_000_000_000, step=100_000, format="%d",
            key="ret_initial")
        monthly_contrib = r2c2.number_input(
            f"每月投入 ({csym})（第一階段）", 0, 10_000_000, step=1_000, format="%d",
            key="ret_monthly_contrib")

        # ── 投入變化設定（三階段）──
        st.markdown("**📉 投入變化設定（可選）**")
        ph1, ph2 = st.columns([1, 3])
        contrib_change_mode = ph1.radio(
            "變化方式",
            ["不變化", "停止投入", "減少投入"],
            key="ret_contrib_change_mode",
            help="「停止投入」：到達設定年齡後直接停止。「減少投入」：先降低投入額，之後再完全停止。",
        )

        # 依模式顯示對應欄位
        reduce_contrib_age      = 0
        reduced_monthly_contrib = 0
        contrib_stop_age        = 0

        if contrib_change_mode == "停止投入":
            with ph2:
                contrib_stop_age = st.number_input(
                    "幾歲停止投入（0=持續至財務自由）", 0, 100, step=1,
                    key="ret_contrib_stop_age",
                    help="設 0 表示持續投入直到達成財務自由為止")
        elif contrib_change_mode == "減少投入":
            with ph2:
                rc1, rc2, rc3 = st.columns(3)
                reduce_contrib_age = rc1.number_input(
                    "幾歲開始減少投入", 1, 100, step=1,
                    key="ret_reduce_contrib_age",
                    help="從此年齡起，每月投入改為下方設定的較低金額。")
                reduced_monthly_contrib = rc2.number_input(
                    f"減少後每月投入 ({csym})", 0, 10_000_000, step=1_000, format="%d",
                    key="ret_reduced_monthly_contrib",
                    help="第二階段的每月投入金額，需小於第一階段。")
                contrib_stop_age = rc3.number_input(
                    "幾歲完全停止投入（0=持續）", 0, 100, step=1,
                    key="ret_contrib_stop_age",
                    help="從此年齡起完全停止投入，設 0 表示第二階段持續至模擬結束。")

            # 邏輯警告
            if reduce_contrib_age > 0 and contrib_stop_age > 0 and reduce_contrib_age >= contrib_stop_age:
                st.warning("⚠️ 「開始減少」年齡需早於「完全停止」年齡。")
            if reduced_monthly_contrib >= monthly_contrib and monthly_contrib > 0:
                st.caption("💡 提示：減少後的金額應小於第一階段才有意義。")

        st.markdown("---")

        # ── 第三行：報酬率設定 ──
        r3c1, r3c2, r3c3 = st.columns(3)
        annual_return_pct = r3c1.slider(
            "預估年化報酬率", 0.0, 20.0, step=0.5, format="%.1f%%",
            key="ret_annual_return_pct")
        inflation_pct = r3c2.slider(
            "通貨膨脹率 / 年", 0.0, 10.0, step=0.5, format="%.1f%%",
            key="ret_inflation_pct")
        withdrawal_pct = r3c3.slider(
            "提領率（4% 法則）", 1.0, 10.0, step=0.5, format="%.1f%%",
            key="ret_withdrawal_pct",
            help="財務自由後每年從資產中提領的比例。4% 為常見標準。")

        st.markdown("---")

        # ── 第四行：生活開銷 + 提領起始年齡 ──
        st.markdown("**🏠 生活開銷 & 提領設定**")
        exp_c1, exp_c2, exp_c3 = st.columns(3)
        monthly_expense = exp_c1.number_input(
            f"月生活開銷 ({csym})（現值）", 0, 10_000_000, step=1_000, format="%d",
            key="ret_monthly_expense",
            help="目前每月的生活開銷，計算時會按通膨率逐年調整。")

        # 提領起始年齡：預設為「停止投入年齡」
        _default_wd = contrib_stop_age if contrib_stop_age > 0 else 0
        withdrawal_start_age = exp_c2.number_input(
            "幾歲開始提領（0=停止投入後立即）", 0, 100, step=1,
            key="ret_withdrawal_start_age",
            help="設為 0 或與「停止投入年齡」相同，表示停止投入後立即開始提領。\n"
                 "也可以設為晚幾年，讓資產繼續累積一段時間再提領。")

        # 連動顯示財務自由目標資產
        annual_expense_now = monthly_expense * 12
        fi_target = annual_expense_now / (withdrawal_pct / 100) if withdrawal_pct > 0 else 0
        exp_c3.markdown(f"""
        <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:10px;
                    padding:14px 16px;margin-top:4px">
            <div style="font-size:0.75rem;color:#166534;font-weight:600;margin-bottom:4px">
                💡 財務自由目標資產（現值）
            </div>
            <div style="font-size:1.15rem;font-weight:700;color:#15803d">
                {csym}{fi_target:,.0f}
            </div>
            <div style="font-size:0.7rem;color:#4ade80;margin-top:2px">
                月支出 {csym}{monthly_expense:,} × 12 ÷ {withdrawal_pct}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── 執行計算 ──
    df = calc_retirement(
        age_start               = age_start,
        year_start              = year_start,
        initial                 = initial,
        monthly_contrib         = monthly_contrib,
        annual_return           = annual_return_pct / 100,
        inflation               = inflation_pct / 100,
        withdrawal_rate         = withdrawal_pct / 100,
        monthly_expense         = monthly_expense,
        years                   = int(years),
        contrib_stop_age        = contrib_stop_age,
        withdrawal_start_age    = withdrawal_start_age,
        reduce_contrib_age      = reduce_contrib_age,
        reduced_monthly_contrib = reduced_monthly_contrib,
    )

    fi_year = df.attrs.get("fi_year")
    fi_age  = df.attrs.get("fi_age")

    # ── 結果橫幅 ──
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
        st.warning(f"⚠️ 在 {int(years)} 年試算期間內尚未達成財務自由，建議提高月投入或降低目標支出。")

    # ── Metrics ──
    last_row      = df.iloc[-1]
    wd_rows       = df[df["提領中"]]
    total_contrib = float(df["年投入"].sum()) + initial
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("最終資產",       f"{csym}{last_row['投資價值']:,.0f}")
    m2.metric("最終年可提領",   f"{csym}{last_row['可提領金額']:,.0f}")
    m3.metric("最終生活開銷/年", f"{csym}{last_row['生活開銷']:,.0f}")
    m4.metric("累積總投入",     f"{csym}{total_contrib:,.0f}")
    m5.metric("財務自由年齡",   f"{fi_age} 歲" if fi_age else "未達成")

    # ── 圖表 ──
    st.markdown("---")
    rt1, rt2 = st.tabs(["📈 財務自由曲線", "📊 資產 vs 投入"])

    _cfg = dict(
        locale="zh-TW", displayModeBar=True,
        modeBarButtonsToRemove=["select2d", "lasso2d", "autoScale2d"],
        toImageButtonOptions={"format": "png", "filename": "retirement", "scale": 2},
        displaylogo=False, scrollZoom=False,
    )
    with rt1:
        st.plotly_chart(
            plot_retirement(df, csym, withdrawal_start_age),
            use_container_width=True, config=_cfg)
    with rt2:
        st.plotly_chart(
            plot_asset_breakdown(df, csym),
            use_container_width=True, config=_cfg)

    # ── 逐年明細 ──
    st.markdown("---")
    st.markdown("#### 📋 逐年明細")
    display_df = df[[
        "年齡", "年度", "年份", "生活開銷",
        "投資價值", "投資回報", "實際提領金額", "可提領金額", "財務自由了嗎"
    ]].copy()
    for col in ["生活開銷", "投資價值", "投資回報", "實際提領金額", "可提領金額"]:
        display_df[col] = display_df[col].apply(lambda x: f"{csym}{x:,.0f}")

    def highlight_fi(row):
        if row["財務自由了嗎"] == "YES!!":
            return ["background-color:#f0fdf4;color:#15803d"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df.style.apply(highlight_fi, axis=1),
        use_container_width=True, hide_index=True, height=400,
    )
