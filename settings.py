"""
settings.py
設定持久化模組 — 零外部依賴版
儲存策略：優先嘗試瀏覽器 Cookie（需安裝 extra-streamlit-components），
          若套件不存在則自動降級為 JSON 檔案儲存。
"""

import json
import datetime
from pathlib import Path

# JSON 備用路徑
_SETTINGS_FILE = Path(__file__).parent / "app_settings.json"

# ──────────────────────────────────────────────
# 預設值
# ──────────────────────────────────────────────
DEFAULTS: dict = {
    "backtest": {
        "w_ticker":           "VOO",
        "w_years":            5,
        "w_display_currency": "USD",
        "w_initial":          100_000,
        "w_inv_mode":         "一次性投入",
        "w_dca_amount":       3_000,
        "w_strategy_amount":  2_000,
        "w_buy_mode":         "全倉買入",
        "w_buy_amount":       10_000,
        "w_buy_pct":          100,
        "w_sell_mode":        "全倉賣出",
        "w_sell_amount":      10_000,
        "w_sell_pct":         100,
        "w_strategy":         "MA 交叉策略",
        "w_ma_fast":          50,
        "w_ma_slow":          200,
        "w_rsi_period":       14,
        "w_rsi_buy":          30,
        "w_rsi_sell":         70,
        "w_bb_period":        20,
        "w_bb_std":           2.0,
        "w_macd_fast":        12,
        "w_macd_slow":        26,
        "w_macd_signal":      9,
        "w_dev_buy_period":   30,
        "w_dev_sell_period":  5,
        "w_dev_sell_pct":     10,
        "w_cool_period":      5,
    },
    "portfolio": {
        "port_base_currency":   "USD",
        "port_years_back":      7,
        "port_initial":         1_000_000,
        "port_dca_enable":      False,
        "port_dca_amount":      30_000,
        "port_dca_freq":        1,
        "port_rebalance_freq":  "quarterly",
        "port_cash_return_pct": 4.0,
        "port_commission_pct":  0.1,
        "port_enable_div_tax":  False,
        "port_div_tax_rate":    30,
        "port_n_assets":        2,
        "port_tickers":         ["VOO", "QQQ"],
        "port_weights":         [50, 50],
    },
    "retirement": {
        "ret_age_start":               30,
        "ret_year_start":              2026,
        "ret_currency":                "TWD (NT$)",
        "ret_years":                   50,
        "ret_initial":                 0,
        "ret_monthly_contrib":         0,
        "ret_contrib_change_mode":     "不變化",
        "ret_contrib_stop_age":        0,
        "ret_reduce_contrib_age":      0,
        "ret_reduced_monthly_contrib": 0,
        "ret_annual_return_pct":       7.0,
        "ret_inflation_pct":           2.0,
        "ret_withdrawal_pct":          4.0,
        "ret_monthly_expense":         0,
        "ret_withdrawal_start_age":    0,
    },
}


# ── Cookie 管理器（懶加載，失敗自動降級）──
def _get_cookie_manager():
    try:
        import streamlit as st
        if "_cookie_manager" not in st.session_state:
            import extra_streamlit_components as stx
            st.session_state["_cookie_manager"] = stx.CookieManager(key="_cm")
        return st.session_state["_cookie_manager"]
    except Exception:
        return None


def _cookie_name(namespace: str) -> str:
    return f"stockapp_{namespace}"


# ── 讀取設定（Cookie → JSON → 預設值，依序 fallback）──
def load_settings(namespace: str) -> dict:
    defaults = DEFAULTS.get(namespace, {})

    # 1. 嘗試從 Cookie 讀取
    cm = _get_cookie_manager()
    if cm is not None:
        try:
            raw = cm.get(cookie=_cookie_name(namespace))
            if raw:
                saved = json.loads(raw) if isinstance(raw, str) else raw
                merged = defaults.copy()
                merged.update(saved)
                return merged
        except Exception:
            pass

    # 2. 降級：從 JSON 檔案讀取
    if _SETTINGS_FILE.exists():
        try:
            with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                all_settings = json.load(f)
            saved = all_settings.get(namespace, {})
            merged = defaults.copy()
            merged.update(saved)
            return merged
        except Exception:
            pass

    # 3. 最終 fallback：預設值
    return defaults.copy()


# ── 儲存設定（Cookie → JSON，依序嘗試）──
def save_settings(namespace: str, data: dict) -> bool:
    saved_cookie = False

    # 1. 嘗試寫入 Cookie
    cm = _get_cookie_manager()
    if cm is not None:
        try:
            expire_date = datetime.datetime.now() + datetime.timedelta(days=365)
            cm.set(
                cookie=_cookie_name(namespace),
                val=json.dumps(data, ensure_ascii=False),
                expires_at=expire_date,
            )
            saved_cookie = True
        except Exception:
            pass

    # 2. 同時寫入 JSON（備份）
    try:
        all_settings: dict = {}
        if _SETTINGS_FILE.exists():
            with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                all_settings = json.load(f)
        all_settings[namespace] = data
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_settings, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        pass

    return saved_cookie


# ── 注入 session_state（每個 namespace 只執行一次）──
def init_session(namespace: str, st_session) -> None:
    flag = f"_settings_loaded_{namespace}"
    if st_session.get(flag):
        return
    saved = load_settings(namespace)
    for k, v in saved.items():
        if k not in st_session:
            st_session[k] = v
    st_session[flag] = True
