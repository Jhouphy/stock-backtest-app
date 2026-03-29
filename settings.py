"""
settings.py
本地設定持久化模組 — 儲存／載入各頁面的使用者設定至 app_settings.json
"""

import json
from pathlib import Path

# 設定檔路徑：與 app.py 同目錄
SETTINGS_FILE = Path(__file__).parent / "app_settings.json"

# ──────────────────────────────────────────────
# 各命名空間的預設值（新裝置或設定檔損毀時使用）
# ──────────────────────────────────────────────
DEFAULTS: dict[str, dict] = {
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
        "ret_age_start":          30,
        "ret_year_start":         2026,
        "ret_currency":           "TWD (NT$)",
        "ret_years":              50,
        "ret_initial":            0,
        "ret_monthly_contrib":    0,
        "ret_contrib_stop_age":   0,
        "ret_annual_return_pct":  7.0,
        "ret_inflation_pct":      2.0,
        "ret_withdrawal_pct":     4.0,
        "ret_monthly_expense":    0,
        "ret_withdrawal_start_age": 0,
    },
}


def load_settings(namespace: str) -> dict:
    """載入指定命名空間的設定，找不到時回傳預設值。"""
    defaults = DEFAULTS.get(namespace, {})
    if not SETTINGS_FILE.exists():
        return defaults.copy()
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            all_settings = json.load(f)
        saved = all_settings.get(namespace, {})
        # 以預設值為底，覆蓋已儲存的值（保證所有 key 都存在）
        merged = defaults.copy()
        merged.update(saved)
        return merged
    except Exception:
        return defaults.copy()


def save_settings(namespace: str, data: dict) -> bool:
    """將設定寫入 JSON 檔案，回傳是否成功。"""
    all_settings: dict = {}
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                all_settings = json.load(f)
        except Exception:
            pass
    all_settings[namespace] = data
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_settings, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def init_session(namespace: str, st_session) -> None:
    """
    第一次進入頁面時，把儲存的設定注入 st.session_state。
    已存在的 key 不覆蓋（尊重使用者在本次 session 中的修改）。
    """
    flag = f"_settings_loaded_{namespace}"
    if st_session.get(flag):
        return
    saved = load_settings(namespace)
    for k, v in saved.items():
        if k not in st_session:
            st_session[k] = v
    st_session[flag] = True
