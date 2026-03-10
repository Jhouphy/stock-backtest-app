"""
engine.py
回測引擎模組
包含：技術指標計算、交易信號生成、回測執行、效能評估、Grid Search 最佳化
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import urllib.request, json as _json
warnings.filterwarnings("ignore")

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


def _state_machine_signals(buy_cond: np.ndarray, sell_cond: np.ndarray,
                            valid_mask: np.ndarray) -> np.ndarray:
    """
    通用狀態機：將買入/賣出條件陣列轉換為信號陣列。
    規則：
      - 空倉中：buy_cond 為 True → 產生買入信號(1)，進入持倉
      - 持倉中：sell_cond 為 True → 產生賣出信號(-1)，回到空倉
      - valid_mask 為 False 的位置跳過（用於跳過 NaN 區間）
    保證買賣嚴格交替，不會出現重複信號。
    """
    n = len(buy_cond)
    signals = np.zeros(n, dtype=int)
    in_pos  = False
    for i in range(n):
        if not valid_mask[i]:
            continue
        if not in_pos:
            if buy_cond[i]:
                signals[i] = 1
                in_pos = True
        else:
            if sell_cond[i]:
                signals[i] = -1
                in_pos = False
    return signals


def generate_signals(df: pd.DataFrame, strategy: str, params: dict) -> pd.DataFrame:
    """
    產生買(1)/賣(-1)/持有(0) 信號。
    所有策略統一使用狀態機，確保買賣嚴格交替、信號次數對稱。
    """
    df   = df.copy()
    c    = df["Close"].squeeze()
    nans = np.zeros(len(df), dtype=bool)   # 全部視為有效（個別策略再覆寫）

    if strategy == "MA 交叉策略":
        f_col = df[f"MA{params['ma_fast']}"].values
        s_col = df[f"MA{params['ma_slow']}"].values
        valid = ~(np.isnan(f_col) | np.isnan(s_col))
        # 黃金交叉買入，死亡交叉賣出
        buy_cond  = f_col > s_col
        sell_cond = f_col < s_col
        df["Signal"] = _state_machine_signals(buy_cond, sell_cond, valid)

    elif strategy == "RSI 動能策略":
        rsi   = df["RSI"].values
        valid = ~np.isnan(rsi)
        buy_cond  = rsi < params["rsi_buy"]
        sell_cond = rsi > params["rsi_sell"]
        df["Signal"] = _state_machine_signals(buy_cond, sell_cond, valid)

    elif strategy == "布林通道策略":
        cl    = c.values
        lower = df["BB_Lower"].values
        upper = df["BB_Upper"].values
        valid = ~(np.isnan(lower) | np.isnan(upper))
        buy_cond  = cl <= lower
        sell_cond = cl >= upper
        df["Signal"] = _state_machine_signals(buy_cond, sell_cond, valid)

    elif strategy == "MACD 趨勢策略":
        macd   = df["MACD"].values
        signal = df["MACD_Signal"].values
        valid  = ~(np.isnan(macd) | np.isnan(signal))
        buy_cond  = macd > signal
        sell_cond = macd < signal
        df["Signal"] = _state_machine_signals(buy_cond, sell_cond, valid)

    elif strategy == "MA均線偏離策略":
        dev_buy   = df["Dev_Buy"].values
        dev_sell  = df["Dev_Sell"].values
        valid     = ~(np.isnan(dev_buy) | np.isnan(dev_sell))
        buy_th    = params["dev_buy_pct"]
        sell_th   = params["dev_sell_pct"]
        buy_cd    = int(params.get("dev_buy_cooldown",  20))
        sell_cd   = int(params.get("dev_sell_cooldown", 20))

        n           = len(df)
        signals_arr = np.zeros(n, dtype=int)
        in_pos      = False
        last_buy_i  = -buy_cd  - 1   # 初始值確保第一天可觸發
        last_sell_i = -sell_cd - 1

        for idx_i in range(n):
            if not valid[idx_i]:
                continue
            if not in_pos:
                # 空倉：條件符合 且 距上次買入超過買入冷卻期
                if (dev_buy[idx_i] <= buy_th and
                        (idx_i - last_buy_i) > buy_cd):
                    signals_arr[idx_i] = 1
                    in_pos     = True
                    last_buy_i = idx_i
            else:
                # 持倉：條件符合 且 距上次賣出超過賣出冷卻期
                if (dev_sell[idx_i] >= sell_th and
                        (idx_i - last_sell_i) > sell_cd):
                    signals_arr[idx_i] = -1
                    in_pos      = False
                    last_sell_i = idx_i

        df["Signal"] = signals_arr

    else:
        df["Signal"] = 0

    return df


def run_grid_search(df_raw: pd.DataFrame, inv_cfg: dict,
                    weights: dict | None = None) -> list:
    """
    窮舉策略參數 × 倉位控制組合，依自訂加權綜合分數排序。
    倉位維度：DCA開關 × 買入模式(全倉/50%/固定額) × 賣出模式(全倉/50%/固定額)
    """
    RISK_FREE = 0.04
    weights   = weights or {"cagr": 0.25, "sharpe": 0.25, "max_dd": 0.25, "stability": 0.25}
    results   = []
    initial   = float(inv_cfg["initial"])

    # ── 策略參數網格 ──
    param_grids = {
        "MA 交叉策略": [
            {"ma_fast": f, "ma_slow": s}
            for f in [10, 20, 50]
            for s in [100, 150, 200]
            if f < s
        ],
        "RSI 動能策略": [
            {"rsi_period": p, "rsi_buy": b, "rsi_sell": sl}
            for p  in [10, 14]
            for b  in [25, 30]
            for sl in [65, 70]
        ],
        "布林通道策略": [
            {"bb_period": p, "bb_std": s}
            for p in [10, 20]
            for s in [1.5, 2.0, 2.5]
        ],
        "MACD 趨勢策略": [
            {"macd_fast": f, "macd_slow": s, "macd_signal": sig}
            for f   in [8, 12]
            for s   in [21, 26]
            for sig in [7, 9]
            if f < s
        ],
        "MA均線偏離策略": [
            {"dev_buy_period": b, "dev_buy_pct": 0.0,
             "dev_sell_period": sl, "dev_sell_pct": pct,
             "dev_buy_cooldown": 20, "dev_sell_cooldown": 20}
            for b   in [20, 30, 60]
            for sl  in [5, 10]
            for pct in [5.0, 10.0, 15.0, 20.0]
        ],
    }

    # ── 倉位控制網格 ──
    # 買入模式：全倉 / 50%倉 / 固定金額(初始資金25%)
    buy_grid = [
        {"buy_mode": "all_in",       "buy_amount": initial,      "buy_pct": 1.0,  "buy_label": "全倉買"},
        {"buy_mode": "fixed_pct",    "buy_amount": initial,      "buy_pct": 0.5,  "buy_label": "50%買"},
        {"buy_mode": "fixed_amount", "buy_amount": initial*0.25, "buy_pct": 1.0,  "buy_label": f"固定${initial*0.25:,.0f}買"},
    ]
    # 賣出模式：全倉 / 50%倉 / 固定金額(初始資金25%)
    sell_grid = [
        {"sell_mode": "all_out",      "sell_amount": initial,      "sell_pct": 1.0, "sell_label": "全倉賣"},
        {"sell_mode": "fixed_pct",    "sell_amount": initial,      "sell_pct": 0.5, "sell_label": "50%賣"},
        {"sell_mode": "fixed_amount", "sell_amount": initial*0.25, "sell_pct": 1.0, "sell_label": f"固定${initial*0.25:,.0f}賣"},
    ]
    # DCA 開關（沿用 inv_cfg 的 dca_amount；關閉時改成 lump_sum）
    dca_grid = [
        {"mode": "lump_sum", "dca_label": "無DCA"},
        {"mode": inv_cfg.get("mode", "lump_sum"),
         "dca_label": "有DCA" if inv_cfg.get("mode") == "dca" else "無DCA"},
    ]
    # 去重（若 inv_cfg 本來就是 lump_sum，dca_grid 只有一種）
    seen_modes = set()
    dca_grid_dedup = []
    for d in dca_grid:
        if d["mode"] not in seen_modes:
            dca_grid_dedup.append(d); seen_modes.add(d["mode"])

    for strategy, grid in param_grids.items():
        for params in grid:
            try:
                df_ind = compute_indicators(df_raw, strategy, params)
                if df_ind.empty or len(df_ind) < 60:
                    continue
                df_sig = generate_signals(df_ind, strategy, params)
            except Exception:
                continue

            for dca_cfg in dca_grid_dedup:
                for b_cfg in buy_grid:
                    for s_cfg in sell_grid:
                        try:
                            # 組合 inv_cfg
                            cfg = {**inv_cfg,
                                   "mode":        dca_cfg["mode"],
                                   "buy_mode":    b_cfg["buy_mode"],
                                   "buy_amount":  b_cfg["buy_amount"],
                                   "buy_pct":     b_cfg["buy_pct"],
                                   "sell_mode":   s_cfg["sell_mode"],
                                   "sell_amount": s_cfg["sell_amount"],
                                   "sell_pct":    s_cfg["sell_pct"],
                            }
                            res = run_backtest(df_sig, cfg)

                            ps      = res["acc1_series"]
                            daily_r = ps.pct_change().dropna()
                            if len(daily_r) < 20:
                                continue
                            ann_vol   = float(daily_r.std() * (252 ** 0.5))
                            cagr      = res["acc1_cagr"]
                            sharpe    = (cagr - RISK_FREE) / ann_vol if ann_vol > 0.001 else -99
                            max_dd    = res["acc1_drawdown"]
                            n_trades  = len(res["buy_dates"])
                            stability = cagr / abs(max_dd) if abs(max_dd) > 0.001 else 0

                            # 策略摘要
                            if strategy == "MA 交叉策略":
                                s_sum = f"MA{params['ma_fast']}/MA{params['ma_slow']}"
                            elif strategy == "RSI 動能策略":
                                s_sum = f"RSI{params['rsi_period']} 買<{params['rsi_buy']} 賣>{params['rsi_sell']}"
                            elif strategy == "布林通道策略":
                                s_sum = f"BB{params['bb_period']} ±{params['bb_std']}σ"
                            elif strategy == "MACD 趨勢策略":
                                s_sum = f"MACD {params['macd_fast']}/{params['macd_slow']}/{params['macd_signal']}"
                            else:
                                s_sum = (f"買MA{params['dev_buy_period']} "
                                         f"賣MA{params['dev_sell_period']}+{params['dev_sell_pct']:.0f}%")

                            # 倉位摘要（可直接複製到 App 設定）
                            pos_sum  = f"{dca_cfg['dca_label']} ／ {b_cfg['buy_label']} ／ {s_cfg['sell_label']}"
                            full_sum = f"{s_sum}｜{pos_sum}"

                            results.append({
                                "strategy":      strategy,
                                "params":        params,
                                "inv_cfg":       cfg,          # 完整設定，可直接套用
                                "param_summary": s_sum,
                                "pos_summary":   pos_sum,
                                "full_summary":  full_sum,
                                "sharpe":        round(sharpe, 4),
                                "cagr":          cagr,
                                "max_dd":        max_dd,
                                "stability":     round(stability, 4),
                                "n_trades":      n_trades,
                                "ann_vol":       round(ann_vol, 4),
                            })
                        except Exception:
                            continue

    if not results:
        return []

    # ── 標準化各指標到 [0,1]，再加權計算綜合分數 ──
    import pandas as _pd
    df_r = _pd.DataFrame(results)

    def norm_col(col, higher_better=True):
        mn, mx = col.min(), col.max()
        if mx == mn:
            return _pd.Series([0.5] * len(col))
        normed = (col - mn) / (mx - mn)
        return normed if higher_better else 1 - normed

    df_r["_n_cagr"]      = norm_col(df_r["cagr"],      higher_better=True)
    df_r["_n_sharpe"]    = norm_col(df_r["sharpe"],     higher_better=True)
    df_r["_n_max_dd"]    = norm_col(df_r["max_dd"],     higher_better=False)  # 回撤越小越好
    df_r["_n_stability"] = norm_col(df_r["stability"],  higher_better=True)

    w_cagr  = weights.get("cagr",      0.25)
    w_shar  = weights.get("sharpe",    0.25)
    w_dd    = weights.get("max_dd",    0.25)
    w_stab  = weights.get("stability", 0.25)

    df_r["score"] = (
        df_r["_n_cagr"]      * w_cagr +
        df_r["_n_sharpe"]    * w_shar +
        df_r["_n_max_dd"]    * w_dd   +
        df_r["_n_stability"] * w_stab
    )
    df_r = df_r.sort_values("score", ascending=False).reset_index(drop=True)
    for col in ["_n_cagr","_n_sharpe","_n_max_dd","_n_stability"]:
        df_r.drop(columns=col, inplace=True)

    return df_r.to_dict("records")


def call_claude_analysis(api_key: str, ticker: str, top_results: list,
                         weights: dict | None = None) -> str:
    """呼叫 Claude API，解讀 Grid Search 最佳化結果。"""
    import urllib.request, json as _json

    weights   = weights or {"cagr": 0.25, "sharpe": 0.25, "max_dd": 0.25, "stability": 0.25}
    w_desc    = (f"CAGR {weights['cagr']*100:.0f}% ／ "
                 f"夏普比率 {weights['sharpe']*100:.0f}% ／ "
                 f"最大回撤控制 {weights['max_dd']*100:.0f}% ／ "
                 f"穩定性 {weights['stability']*100:.0f}%")

    top_summary = "\n".join([
        f"{i+1}. {r['strategy']} ({r['param_summary']}) — "
        f"綜合分數={r['score']:.4f}, 夏普={r['sharpe']:.3f}, "
        f"CAGR={r['cagr']:+.2%}, 最大回撤={r['max_dd']:.2%}, "
        f"穩定性={r['stability']:.3f}, 交易{r['n_trades']}次"
        for i, r in enumerate(top_results)
    ])

    prompt = f"""你是一位量化投資分析師。以下是對股票 {ticker} 進行回測最佳化的結果。

使用者的指標加權偏好：{w_desc}
（綜合分數 = 各指標標準化後依以上比重加權）

排名結果（已按綜合分數排序）：
{top_summary}

請用繁體中文回答以下問題：
1. 排名第一的策略為何符合使用者的加權偏好？請結合加權設定解釋。
2. 前五名有哪些共同特徵？這反映了 {ticker} 的什麼市場特性？
3. 若使用者想進一步優化，針對其偏好（加權方向）有什麼參數調整建議？
4. 有沒有哪個策略雖然排名靠前但有潛在風險（例如交易次數極少、過擬合跡象）？
5. 一句話總結：依照此加權偏好，{ticker} 最適合什麼風格的交易策略？

回答請簡潔，每點 2-3 句話即可。"""

    payload = _json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read())
        return data["content"][0]["text"]
    except Exception as e:
        return f"❌ Claude API 呼叫失敗：{e}"


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


def _build_dca_dates(trading_days, dca_amount, dca_freq, mode):
    """共用：計算 DCA 觸發日期 set"""
    dca_dates = set()
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
    return dca_dates


def run_backtest(df: pd.DataFrame, inv_cfg: dict, ts_cfg: dict | None = None) -> dict:
    """
    ┌──────────────────────────────────────────────────────────────────┐
    │  4 帳戶設計（每月定投金額 = X）                                    │
    │                                                                  │
    │  acc1  初始全買股票 │ 每月 X → 現金池 │ 信號才進出現金池            │
    │  acc2  初始全買股票 │ 每月 X → 直接買股（DCA）                      │
    │              │ 每月 X → 現金池  │ 信號才進出現金池                  │
    │  acc3  初始全買股票 │ 每月 2X → 直接買股（B&H，永不賣）             │
    │  acc4  初始全買股票 │ 永不動                                       │
    │                                                                  │
    │  比較邏輯：                                                        │
    │   acc1 vs acc4 → 信號現金池策略 vs 純持有                         │
    │   acc2 vs acc3 → DCA+信號策略 vs 純DCA持有（投入金額相同）         │
    └──────────────────────────────────────────────────────────────────┘
    """
    c           = df["Close"].squeeze()
    initial     = float(inv_cfg["initial"])
    mode        = inv_cfg.get("mode", "lump_sum")
    dca_amount  = float(inv_cfg.get("dca_amount", 0))
    dca_freq    = inv_cfg.get("dca_freq", 1)
    buy_mode    = inv_cfg.get("buy_mode", "all_in")
    buy_amount  = float(inv_cfg.get("buy_amount", initial))
    buy_pct     = float(inv_cfg.get("buy_pct", 1.0))
    sell_mode   = inv_cfg.get("sell_mode", "all_out")
    sell_amount = float(inv_cfg.get("sell_amount", 0))
    sell_pct    = float(inv_cfg.get("sell_pct", 1.0))
    commission  = float(inv_cfg.get("commission", 0.0))   # 單邊交易成本比例

    dca_dates   = _build_dca_dates(df.index, dca_amount, dca_freq, mode)
    first_price = float(c.iloc[0])
    ts_enabled  = bool((ts_cfg or {}).get("enabled", False))
    ts_mode     = (ts_cfg or {}).get("mode", "pct")
    ts_value    = float((ts_cfg or {}).get("value", 0.1))

    # ── acc1：初始全買 + 每月X→現金池 + 信號換手 ──
    a1_shares   = initial / first_price
    a1_cash     = 0.0        # 信號現金池（每月X注入，信號買/賣）
    a1_invested = initial    # 追蹤總投入（含月注入）
    a1_signal_invested = 0.0 # 現金池累計注入

    # ── acc2：初始全買 + 每月X→直接買股 + 每月X→現金池 + 信號換手 ──
    a2_shares   = initial / first_price
    a2_cash     = 0.0        # 信號現金池（每月X注入，信號買/賣）
    a2_invested = initial
    a2_dca_shares_invested = 0.0   # DCA 直接買股的累計金額
    a2_signal_invested     = 0.0   # 現金池累計注入

    # Trailing Stop 追蹤（各帳戶獨立）
    a1_trail_high = 0.0   # acc1 持倉期間最高價
    a2_trail_high = 0.0   # acc2 持倉期間最高價

    # 記錄
    buy_dates,  sell_dates  = [], []   # acc1 信號記錄（代表純策略）
    buy_prices, sell_prices = [], []
    dca_buy_dates, dca_buy_prices = [], []
    a1_vals, a2_vals           = [], []
    a1_cash_vals, a2_cash_vals = [], []   # 現金池逐日數值
    a1_inv_vals, a2_inv_vals   = [], []

    def do_buy(cash, shares, price):
        if buy_mode == "all_in":   use = cash
        elif buy_mode == "fixed_amount": use = min(buy_amount, cash)
        else:                      use = cash * buy_pct
        if use > 1.0:
            # 扣除買入手續費：實際買到的股數較少
            effective_price = price * (1 + commission)
            shares += use / effective_price
            cash   -= use
        return cash, shares

    def do_sell(cash, shares, price):
        if sell_mode == "all_out":        sell_sh = shares
        elif sell_mode == "fixed_amount": sell_sh = min(sell_amount / price, shares)
        else:                             sell_sh = shares * sell_pct
        if sell_sh > 1e-6:
            # 扣除賣出手續費：實際收到的現金較少
            effective_price = price * (1 - commission)
            cash   += sell_sh * effective_price
            shares -= sell_sh
        return cash, shares

    for i, (idx, row) in enumerate(df.iterrows()):
        price  = float(c.iloc[i])
        signal = int(row["Signal"])

        if mode == "dca" and dca_amount > 0 and idx in dca_dates:
            # acc1：每月 X 注入現金池（不買股，等信號）
            a1_cash     += dca_amount
            a1_invested += dca_amount
            a1_signal_invested += dca_amount

            # acc2：每月 X 直接買股（DCA），另外 X 注入現金池
            a2_shares   += dca_amount / (price * (1 + commission))   # DCA 直接買（含手續費）
            a2_cash     += dca_amount                  # 現金池也注入 X
            a2_invested += dca_amount * 2              # 兩筆都計入投入
            a2_dca_shares_invested += dca_amount
            a2_signal_invested     += dca_amount

            dca_buy_dates.append(idx)
            dca_buy_prices.append(price)

        # ── Trailing Stop：更新持倉高點，觸發則強制賣出 ──
        if ts_enabled:
            # 更新各帳戶的持倉最高價
            if a1_shares > 1e-6:
                a1_trail_high = max(a1_trail_high, price)
            else:
                a1_trail_high = 0.0   # 空倉時重置
            if a2_shares > 1e-6:
                a2_trail_high = max(a2_trail_high, price)
            else:
                a2_trail_high = 0.0

            # 判斷是否觸發停利
            def ts_triggered(trail_high, cur_price):
                if trail_high <= 0: return False
                if ts_mode == "pct":
                    return cur_price <= trail_high * (1 - ts_value)
                else:  # fixed
                    return cur_price <= trail_high - ts_value

            # acc1 強制賣出
            if ts_triggered(a1_trail_high, price) and a1_shares > 1e-6:
                a1_cash, a1_shares = do_sell(a1_cash, a1_shares, price)
                sell_dates.append(idx); sell_prices.append(price)
                a1_trail_high = 0.0
                signal = 0   # 避免下方再次觸發一般賣出邏輯重複記錄

            # acc2 強制賣出
            if ts_triggered(a2_trail_high, price) and a2_shares > 1e-6:
                a2_cash, a2_shares = do_sell(a2_cash, a2_shares, price)
                a2_trail_high = 0.0
                signal = 0

        # 買入後重置 trailing high（讓新持倉從買入當天開始追蹤）
        # （在買入後立即設為當天價格，下方買入邏輯之後處理）

        # 賣出信號：從 shares 賣出，所得放入各自現金池
        if signal == -1:
            if a1_shares > 1e-6:
                a1_cash, a1_shares = do_sell(a1_cash, a1_shares, price)
                sell_dates.append(idx); sell_prices.append(price)
            if a2_shares > 1e-6:
                a2_cash, a2_shares = do_sell(a2_cash, a2_shares, price)

        # 買入信號：從各自現金池買入股票
        elif signal == 1:
            if a1_cash > 1.0:
                prev = a1_shares
                a1_cash, a1_shares = do_buy(a1_cash, a1_shares, price)
                if a1_shares > prev:
                    buy_dates.append(idx); buy_prices.append(price)
                    a1_trail_high = price   # 新買入，從當天開始追蹤高點
            if a2_cash > 1.0:
                prev2 = a2_shares
                a2_cash, a2_shares = do_buy(a2_cash, a2_shares, price)
                if a2_shares > prev2:
                    a2_trail_high = price

        a1_total = a1_cash + a1_shares * price
        a2_total = a2_cash + a2_shares * price
        a1_vals.append(a1_total)
        a2_vals.append(a2_total)
        a1_cash_vals.append(a1_cash)
        a2_cash_vals.append(a2_cash)
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
        "acc1_series":    a1_ps,
        "acc1_final":     a1_final,
        "acc1_invested":  a1_invested,
        "acc1_return":    a1_tr,
        "acc1_cagr":      a1_cagr,
        "acc1_drawdown":  float((a1_ps / a1_ps.cummax() - 1).min()),
        "acc1_dd_series": a1_ps / a1_ps.cummax() - 1,
        "acc2_series":    a2_ps,
        "acc2_final":     a2_final,
        "acc2_invested":  a2_invested,
        "acc2_return":    a2_tr,
        "acc2_cagr":      a2_cagr,
        "acc2_drawdown":  float((a2_ps / a2_ps.cummax() - 1).min()),
        "acc2_dd_series": a2_ps / a2_ps.cummax() - 1,
        "dca_invested":   a2_dca_shares_invested,
        "buy_dates":      buy_dates,
        "sell_dates":     sell_dates,
        "buy_prices":     buy_prices,
        "sell_prices":    sell_prices,
        "dca_buy_dates":  dca_buy_dates,
        "dca_buy_prices": dca_buy_prices,
        "acc1_inv_series":  pd.Series(a1_inv_vals,   index=df.index),
        "acc2_inv_series":  pd.Series(a2_inv_vals,   index=df.index),
        # 現金池序列（用於現金比例圖診斷）
        "acc1_cash_series": pd.Series(a1_cash_vals,  index=df.index),
        "acc2_cash_series": pd.Series(a2_cash_vals,  index=df.index),
    }


def compute_benchmark(df: pd.DataFrame, inv_cfg: dict) -> dict:
    """
    acc3: 初始全買 + 每月 2X 直接買股（B&H，永不賣）
          對應 acc2 的總投入（DCA部分X + 信號池部分X = 2X）
    acc4: 初始全買，永不動（純B&H基準）
    """
    c          = df["Close"].squeeze()
    initial    = float(inv_cfg["initial"])
    mode       = inv_cfg.get("mode", "lump_sum")
    dca_amount = float(inv_cfg.get("dca_amount", 0))
    dca_freq   = inv_cfg.get("dca_freq", 1)

    dca_dates = _build_dca_dates(df.index, dca_amount, dca_freq, mode)

    # acc4：純 B&H（永遠持有初始股票）
    a4_shares   = initial / float(c.iloc[0])
    a4_invested = initial

    # acc3：B&H + 每月 2X（對應 acc2 的相同總資金）
    a3_shares   = initial / float(c.iloc[0])
    a3_invested = initial

    a3_vals, a4_vals     = [], []
    a3_inv_vals, a4_inv_vals = [], []

    for i, idx in enumerate(df.index):
        price = float(c.iloc[i])
        if mode == "dca" and dca_amount > 0 and idx in dca_dates:
            comm = float(inv_cfg.get("commission", 0.0))
            a3_shares   += (dca_amount * 2) / (price * (1 + comm))
            a3_invested += dca_amount * 2
        a3_vals.append(a3_shares * price)
        a4_vals.append(a4_shares * price)
        a3_inv_vals.append(a3_invested)
        a4_inv_vals.append(a4_invested)

    return {
        "acc3_series":     pd.Series(a3_vals,     index=df.index),
        "acc3_invested":   a3_invested,
        "acc3_inv_series": pd.Series(a3_inv_vals, index=df.index),
        "acc4_series":     pd.Series(a4_vals,     index=df.index),
        "acc4_invested":   a4_invested,
        "acc4_inv_series": pd.Series(a4_inv_vals, index=df.index),
    }


# ═══════════════════════════════════════════════════════
# 圖表
# ═══════════════════════════════════════════════════════
CHART = dict(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
             font=dict(color="#475569", family="IBM Plex Mono, monospace", size=11),
             gridcolor="#e2e8f0")

