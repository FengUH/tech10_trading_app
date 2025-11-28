# ============================================================
# Tech10 Trading Strategy Lab
# Product-style Streamlit app for interview demo
# ============================================================

import datetime as dt
import os
import pandas as pd
import numpy as np
import streamlit as st
import snowflake.connector
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# 0. Page config & styling
# ============================================================
st.set_page_config(
    page_title="Tech10 Strategy Lab",
    page_icon="📈",
    layout="wide",
)

# Sidebar & main area CSS
st.markdown(
    """
    <style>
    /* Sidebar: light background + dark text */
    [data-testid="stSidebar"] {
        background-color: #eef2ff;  /* 淡淡的蓝紫色 */
        color: #111827;
    }
    [data-testid="stSidebar"] * {
        color: #111827 !important;
    }

    .main-header {
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #e0e4ec;
        margin-bottom: 0.6rem;
    }

    .chart-box {
        border: 1px solid #e1e4ee;
        border-radius: 0.75rem;
        padding: 0.75rem;
        background-color: #ffffff;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 1. Snowflake config —— 读取自 Streamlit secrets 或环境变量
#    （不要在代码里写死账号和密码）
# ============================================================
def get_snowflake_config():
    # 优先从 Streamlit 的 secrets 里读取（本地和云端都支持）
    if "SNOWFLAKE_ACCOUNT" in st.secrets:
        cfg = {
            "SNOWFLAKE_ACCOUNT":   st.secrets["SNOWFLAKE_ACCOUNT"],
            "SNOWFLAKE_USER":      st.secrets["SNOWFLAKE_USER"],
            "SNOWFLAKE_PASSWORD":  st.secrets["SNOWFLAKE_PASSWORD"],
            "SNOWFLAKE_ROLE":      st.secrets["SNOWFLAKE_ROLE"],
            "SNOWFLAKE_WAREHOUSE": st.secrets["SNOWFLAKE_WAREHOUSE"],
            "SNOWFLAKE_DATABASE":  st.secrets["SNOWFLAKE_DATABASE"],
            "SNOWFLAKE_SCHEMA":    st.secrets["SNOWFLAKE_SCHEMA"],
            "TABLE_NAME":          st.secrets.get("TABLE_NAME", "TECH10_DAILY_PRICES"),
        }
    else:
        # 如果本地没有 secrets（比如只用环境变量调试），就从环境变量里读
        cfg = {
            "SNOWFLAKE_ACCOUNT":   os.environ["SNOWFLAKE_ACCOUNT"],
            "SNOWFLAKE_USER":      os.environ["SNOWFLAKE_USER"],
            "SNOWFLAKE_PASSWORD":  os.environ["SNOWFLAKE_PASSWORD"],
            "SNOWFLAKE_ROLE":      os.environ.get("SNOWFLAKE_ROLE", "SYSADMIN"),
            "SNOWFLAKE_WAREHOUSE": os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
            "SNOWFLAKE_DATABASE":  os.environ.get("SNOWFLAKE_DATABASE", "FENG_DB"),
            "SNOWFLAKE_SCHEMA":    os.environ.get("SNOWFLAKE_SCHEMA", "TRADING"),
            "TABLE_NAME":          os.environ.get("TABLE_NAME", "TECH10_DAILY_PRICES"),
        }
    return cfg


_cfg = get_snowflake_config()
SNOWFLAKE_ACCOUNT   = _cfg["SNOWFLAKE_ACCOUNT"]
SNOWFLAKE_USER      = _cfg["SNOWFLAKE_USER"]
SNOWFLAKE_PASSWORD  = _cfg["SNOWFLAKE_PASSWORD"]
SNOWFLAKE_ROLE      = _cfg["SNOWFLAKE_ROLE"]
SNOWFLAKE_WAREHOUSE = _cfg["SNOWFLAKE_WAREHOUSE"]
SNOWFLAKE_DATABASE  = _cfg["SNOWFLAKE_DATABASE"]
SNOWFLAKE_SCHEMA    = _cfg["SNOWFLAKE_SCHEMA"]
TABLE_NAME          = _cfg["TABLE_NAME"]


# ============================================================
# 2. Load prices from Snowflake
# ============================================================
@st.cache_data
def load_prices() -> pd.DataFrame:
    """
    Pull the Tech10 daily price history from Snowflake
    and return as a pandas DataFrame.
    """
    conn = snowflake.connector.connect(
        account=SNOWFLAKE_ACCOUNT,
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        role=SNOWFLAKE_ROLE,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA,
    )
    cs = conn.cursor()

    sql = f"""
        SELECT *
        FROM {TABLE_NAME}
        ORDER BY TICKER, TRADE_DATE
    """
    cs.execute(sql)
    df = cs.fetch_pandas_all()

    cs.close()
    conn.close()

    df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"])
    return df


# ============================================================
# 3. Strategy functions: MA, MACD, Buy&Hold
# ============================================================
def apply_ma_strategy(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """Simple moving-average crossover on a single ticker."""
    df = df.copy()
    df["RET"] = df["ADJ_CLOSE"].pct_change()

    df["MA_FAST"] = df["ADJ_CLOSE"].rolling(fast).mean()
    df["MA_SLOW"] = df["ADJ_CLOSE"].rolling(slow).mean()

    df["SIGNAL"] = np.where(df["MA_FAST"] > df["MA_SLOW"], 1, 0)
    df["STRAT_RET"] = df["SIGNAL"].shift(1) * df["RET"]

    df["EQ_STRAT"] = (1 + df["STRAT_RET"]).cumprod()
    df["EQ_BH"] = (1 + df["RET"]).cumprod()
    return df


def apply_macd_strategy(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_span: int = 9
) -> pd.DataFrame:
    """Classic MACD strategy on a single ticker."""
    df = df.copy()
    df["RET"] = df["ADJ_CLOSE"].pct_change()

    ema_fast = df["ADJ_CLOSE"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["ADJ_CLOSE"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=signal_span, adjust=False).mean()

    df["SIGNAL"] = np.where(df["MACD"] > df["MACD_SIGNAL"], 1, 0)
    df["STRAT_RET"] = df["SIGNAL"].shift(1) * df["RET"]

    df["EQ_STRAT"] = (1 + df["STRAT_RET"]).cumprod()
    df["EQ_BH"] = (1 + df["RET"]).cumprod()
    return df


def apply_buyhold(df: pd.DataFrame) -> pd.DataFrame:
    """Buy & hold benchmark for a single ticker."""
    df = df.copy()
    df["RET"] = df["ADJ_CLOSE"].pct_change()
    df["EQ_BH"] = (1 + df["RET"]).cumprod()
    df["EQ_STRAT"] = df["EQ_BH"]          # strategy = benchmark
    df["SIGNAL"] = 1                      # always in the market
    return df


def generate_signal_summary(df_strat: pd.DataFrame, strategy: str, name: str) -> str:
    """
    Short, product-style summary of the current position and what
    the rule is doing in this window.
    """
    if df_strat.empty:
        return "There is not enough data in the selected window to generate trading suggestions."

    start_date = df_strat["TRADE_DATE"].min().date()
    end_date = df_strat["TRADE_DATE"].max().date()

    # Buy & Hold summary
    if strategy == "Buy & Hold":
        start_price = df_strat["ADJ_CLOSE"].iloc[0]
        end_price = df_strat["ADJ_CLOSE"].iloc[-1]
        total_ret = end_price / start_price - 1.0

        text = (
            f"**Current rule: Buy & Hold — {name}**  \n\n"
            f"- Over **{start_date} → {end_date}**, a simple buy-and-hold approach "
            f"for {name} would have delivered a cumulative return of "
            f"approximately **{total_ret:.2%}**.  \n"
            f"- The position stays fully invested throughout the window and does not react "
            f"to short-term price fluctuations.  \n"
            f"- This serves as a benchmark to compare against more active trading rules."
        )
        return text

    # MA / MACD summary
    sig = df_strat["SIGNAL"].dropna()
    if sig.empty:
        return "Signals are very sparse in this window, so it is difficult to form a clear trading view."

    last_signal = int(sig.iloc[-1])
    current_pos = "LONG (invested)" if last_signal == 1 else "FLAT (out of the market)"

    text = (
        f"**Current rule: {strategy} — {name}**  \n\n"
        f"- In **{start_date} → {end_date}**, the rule switches between being *in the market* and *on the sidelines* "
        f"based on price trends.  \n"
        f"- The latest signal in this window suggests the portfolio is currently **{current_pos}**.  \n"
        f"- This rule is meant to be a simple, transparent example of how you could overlay systematic signals "
        f"on top of a single stock."
    )
    return text


# ============================================================
# Helper: candlestick + signals
# ============================================================
def plot_candlestick_with_signals(
    df_recent: pd.DataFrame,
    signal_col: str,
    title: str = "Price & trade signals (selected window)",
):
    """
    df_recent: single-ticker slice with
        TRADE_DATE, OPEN, HIGH, LOW, CLOSE, SIGNAL
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    dates = pd.to_datetime(df_recent["TRADE_DATE"])
    x = mdates.date2num(dates)

    o = df_recent["OPEN"].values
    h = df_recent["HIGH"].values
    l = df_recent["LOW"].values
    c = df_recent["CLOSE"].values

    width = 0.6
    line_width = 0.8

    candle_up_color = "#0B8043"   # dark green
    candle_down_color = "#B71C1C" # dark red

    # Candles
    for i in range(len(df_recent)):
        color = candle_up_color if c[i] >= o[i] else candle_down_color
        # Wick
        ax.vlines(x[i], l[i], h[i], linewidth=line_width, color=color)
        # Body
        rect_bottom = min(o[i], c[i])
        rect_height = abs(c[i] - o[i]) if abs(c[i] - o[i]) > 1e-6 else 0.5
        ax.add_patch(
            plt.Rectangle(
                (x[i] - width / 2.0, rect_bottom),
                width,
                rect_height,
                edgecolor=color,
                facecolor=color,
                alpha=0.7,
            )
        )

    ax.set_title(title)
    ax.xaxis_date()
    fig.autofmt_xdate()
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)

    # Trade signals (lighter colors)
    sig = df_recent[signal_col].fillna(0).astype(int)
    prev = sig.shift(1).fillna(0).astype(int)

    buy_mask = (sig == 1) & (prev == 0)
    sell_mask = (sig == 0) & (prev == 1)

    buy_color = "#81C784"   # light green
    sell_color = "#EF9A9A"  # light red

    if buy_mask.any():
        ax.plot(
            dates[buy_mask],
            l[buy_mask] * 0.995,
            marker="^",
            linestyle="None",
            markersize=9,
            color=buy_color,
            alpha=0.9,
            label="Buy signal",
        )

    if sell_mask.any():
        ax.plot(
            dates[sell_mask],
            h[sell_mask] * 1.005,
            marker="v",
            linestyle="None",
            markersize=9,
            color=sell_color,
            alpha=0.9,
            label="Sell signal",
        )

    # Keep legend entries even if no buy/sell in this window
    if not buy_mask.any():
        ax.plot(
            [], [],
            marker="^",
            linestyle="None",
            markersize=9,
            color=buy_color,
            alpha=0.9,
            label="Buy signal",
        )

    if not sell_mask.any():
        ax.plot(
            [], [],
            marker="v",
            linestyle="None",
            markersize=9,
            color=sell_color,
            alpha=0.9,
            label="Sell signal",
        )

    ax.legend(loc="upper left")

    return fig


# ============================================================
# Helper: backtest figure & helpers
# ============================================================
def build_backtest_figure(df_strat_window: pd.DataFrame,
                          strategy: str,
                          title_suffix: str):
    """
    df_strat_window: time-window slice with EQ_STRAT / EQ_BH already normalized.
    """
    if df_strat_window.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No data for backtest window", ha="center", va="center")
        ax.set_axis_off()
        return fig

    if strategy == "MACD":
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Price
        axes[0].plot(df_strat_window["TRADE_DATE"], df_strat_window["ADJ_CLOSE"], label="Price")
        axes[0].set_title("Price")
        axes[0].legend()

        # Equity curves
        axes[1].plot(df_strat_window["TRADE_DATE"], df_strat_window["EQ_STRAT"], label="Strategy")
        axes[1].plot(df_strat_window["TRADE_DATE"], df_strat_window["EQ_BH"], label="Buy & Hold")
        axes[1].set_title("Equity curve (normalized to 1.0 at window start)")
        axes[1].legend()

        # MACD
        axes[2].plot(df_strat_window["TRADE_DATE"], df_strat_window["MACD"], label="MACD")
        axes[2].plot(df_strat_window["TRADE_DATE"], df_strat_window["MACD_SIGNAL"], label="Signal")
        axes[2].axhline(0, linewidth=0.8)
        axes[2].set_title("MACD indicator")
        axes[2].legend()

        plt.tight_layout()
        return fig

    else:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # Price + MA
        axes[0].plot(df_strat_window["TRADE_DATE"], df_strat_window["ADJ_CLOSE"], label="Price")
        if strategy == "MA Crossover":
            axes[0].plot(df_strat_window["TRADE_DATE"], df_strat_window["MA_FAST"], label="Fast MA")
            axes[0].plot(df_strat_window["TRADE_DATE"], df_strat_window["MA_SLOW"], label="Slow MA")

        axes[0].set_title(
            "Price with moving averages" if strategy == "MA Crossover" else "Price"
        )
        axes[0].legend()

        # Equity curves
        axes[1].plot(df_strat_window["TRADE_DATE"], df_strat_window["EQ_STRAT"], label="Strategy")
        axes[1].plot(df_strat_window["TRADE_DATE"], df_strat_window["EQ_BH"], label="Buy & Hold")
        axes[1].set_title("Equity curve (normalized to 1.0 at window start)")
        axes[1].legend()

        plt.tight_layout()
        return fig


def render_backtest_box(fig_backtest):
    """Show backtest figure inside a bordered card."""
    st.markdown(
        """
        <div class="chart-box" style="margin-top: 0.6rem;">
          <b>Historical performance in this window</b><br>
          <span style="font-size: 12px; color: #6b7280;">
          Strategy vs. Buy & Hold, using the same time window as the candlestick chart.
          </span>
        """,
        unsafe_allow_html=True,
    )
    st.pyplot(fig_backtest)
    st.markdown("</div>", unsafe_allow_html=True)


def normalize_equity_in_window(df_window: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize EQ_STRAT / EQ_BH to 1.0 at the start of the window
    so the curves remain well-scaled even with NaNs.
    """
    df_window = df_window.copy()
    for col in ["EQ_STRAT", "EQ_BH"]:
        if col in df_window.columns:
            series = df_window[col]
            if series.notna().any():
                first_valid = series.dropna().iloc[0]
                if first_valid != 0 and not np.isnan(first_valid):
                    df_window[col] = series / first_valid
    return df_window


# ============================================================
# 4. Streamlit UI  (single-ticker view)
# ============================================================
df_all = load_prices()
tickers = sorted(df_all["TICKER"].unique())

# ---- Header & intro ---------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
      <h2 style="margin-bottom: 0.3rem;">📈 Tech10 Strategy Lab</h2>
      <p style="margin: 0; font-size: 0.9rem; color: #4b5563;">
        A single-stock strategy sandbox built on top of Snowflake.  
        Pick a Tech10 name, choose a simple rule (MA crossover, MACD, or Buy & Hold),
        and explore how it would have behaved over different time windows.
      </p>
      <p style="margin: 0.25rem 0 0; font-size: 0.8rem; color: #9ca3af;">
        Data source: Yahoo Finance (via yfinance), stored in Snowflake and streamed into this app.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("⚙️ Strategy setup")

# --- Ticker selection ---
ticker = st.sidebar.selectbox("Select a ticker", tickers)

df_ticker_all = df_all[df_all["TICKER"] == ticker].copy().sort_values("TRADE_DATE")

if df_ticker_all.empty:
    st.warning("This ticker has no data.")
    st.stop()

first_date = df_ticker_all["TRADE_DATE"].min().date()
last_date = df_ticker_all["TRADE_DATE"].max().date()

# --- Time window selection ---
date_range_options = ["1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "All"]
date_range_label = st.sidebar.selectbox(
    "Backtest window",
    date_range_options,
    index=3,  # default 6M
)


def compute_start_date(label: str, last_date_: dt.date, first_date_: dt.date) -> dt.date:
    if label == "1D":
        delta = dt.timedelta(days=1)
        start = last_date_ - delta
    elif label == "5D":
        delta = dt.timedelta(days=5)
        start = last_date_ - delta
    elif label == "1M":
        delta = dt.timedelta(days=30)
        start = last_date_ - delta
    elif label == "6M":
        delta = dt.timedelta(days=182)
        start = last_date_ - delta
    elif label == "YTD":
        start = dt.date(last_date_.year, 1, 1)
    elif label == "1Y":
        delta = dt.timedelta(days=365)
        start = last_date_ - delta
    elif label == "5Y":
        delta = dt.timedelta(days=365 * 5)
        start = last_date_ - delta
    else:  # "All"
        start = first_date_

    if start < first_date_:
        start = first_date_
    return start


start_date = compute_start_date(date_range_label, last_date, first_date)
end_date = last_date

mask_window = (df_ticker_all["TRADE_DATE"].dt.date >= start_date) & (
    df_ticker_all["TRADE_DATE"].dt.date <= end_date
)
df_window_raw = df_ticker_all[mask_window].copy()

if df_window_raw.empty:
    st.warning("No data in the selected date range.")
    st.stop()

# --- Strategy selection ---
strategy = st.sidebar.selectbox(
    "Trading rule",
    ["MA Crossover", "MACD", "Buy & Hold"],
)

# Strategy parameters
if strategy == "MA Crossover":
    st.sidebar.markdown("**MA crossover parameters**")
    fast_ma = st.sidebar.slider(
        "Fast MA (short-term trend)",
        5, 50, 20,
        help="Window size for the short-term moving average."
    )
    slow_ma = st.sidebar.slider(
        "Slow MA (long-term trend)",
        20, 200, 50,
        help="Window size for the long-term moving average."
    )
elif strategy == "MACD":
    fast_ma_macd, slow_ma_macd, signal_macd = 12, 26, 9
    st.sidebar.caption("MACD parameters fixed at 12 / 26 / 9 for this demo.")
else:
    st.sidebar.caption("Buy & Hold benchmark: stay fully invested in this ticker.")

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ For illustration only — not investment advice.")

# ============================================================
# 5. Single-ticker backtest on selected window
# ============================================================
# Apply strategy on full history, then slice window
if strategy == "MA Crossover":
    df_strat_full = apply_ma_strategy(df_ticker_all, fast_ma, slow_ma)
    title_suffix = f"MA crossover ({fast_ma}, {slow_ma})"
elif strategy == "MACD":
    df_strat_full = apply_macd_strategy(df_ticker_all, fast_ma_macd, slow_ma_macd, signal_macd)
    title_suffix = "MACD (12, 26, 9)"
else:
    df_strat_full = apply_buyhold(df_ticker_all)
    title_suffix = "Buy & Hold benchmark"

mask_window_strat = (df_strat_full["TRADE_DATE"].dt.date >= start_date) & (
    df_strat_full["TRADE_DATE"].dt.date <= end_date
)
df_strat = df_strat_full[mask_window_strat].copy()

if df_strat.empty:
    st.warning("Not enough data for this trading rule in the selected window.")
    st.stop()

window_label = f"{start_date} → {end_date}"

# ---- KPI metrics（用 st.metric，去掉原来的“胶囊框”） -----------------------
if len(df_strat) >= 2:
    px_start = df_strat["ADJ_CLOSE"].iloc[0]
    px_end = df_strat["ADJ_CLOSE"].iloc[-1]
    ret_px = px_end / px_start - 1.0

    eq_strat = df_strat["EQ_STRAT"].dropna()
    eq_bh = df_strat["EQ_BH"].dropna()
    if eq_strat.size > 0:
        strat_ret = eq_strat.iloc[-1] / eq_strat.iloc[0] - 1.0
    else:
        strat_ret = np.nan
    if eq_bh.size > 0:
        bh_ret = eq_bh.iloc[-1] / eq_bh.iloc[0] - 1.0
    else:
        bh_ret = np.nan
    excess_ret = strat_ret - bh_ret if (not np.isnan(strat_ret) and not np.isnan(bh_ret)) else np.nan
else:
    ret_px = strat_ret = bh_ret = excess_ret = np.nan

col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
with col_kpi1:
    st.metric("Window price move", f"{ret_px:+.2%}" if not np.isnan(ret_px) else "n/a", window_label)

with col_kpi2:
    st.metric(
        "Strategy return",
        f"{strat_ret:+.2%}" if not np.isnan(strat_ret) else "n/a",
        title_suffix,
    )

with col_kpi3:
    st.metric(
        "Excess vs. Buy & Hold",
        f"{excess_ret:+.2%}" if not np.isnan(excess_ret) else "n/a",
        "Strategy − Buy & Hold",
    )

# ---- Main chart --------------------------------------------------------------
st.markdown(
    f"### {ticker} · {title_suffix}  \n"
    f"<span style='font-size: 0.85rem; color: #6b7280;'>Window: {window_label} · View: {date_range_label}</span>",
    unsafe_allow_html=True,
)

st.markdown('<div class="chart-box">', unsafe_allow_html=True)
fig_candle = plot_candlestick_with_signals(
    df_strat,
    signal_col="SIGNAL",
    title=f"{ticker} — trade signals over {date_range_label} window",
)
st.pyplot(fig_candle)
st.markdown('</div>', unsafe_allow_html=True)

# ---- Backtesting PnL toggle --------------------------------------------------
show_pnl = st.checkbox(
    "Show historical performance (price & equity curves)",
    value=False,
    help="Uses the same time window as the candlestick view.",
)

df_bt = normalize_equity_in_window(df_strat)
fig_backtest = build_backtest_figure(df_bt, strategy, title_suffix)

if show_pnl:
    render_backtest_box(fig_backtest)

# ---- Position status + explanation -------------------------------------------
sig_series = df_strat["SIGNAL"].dropna()
if not sig_series.empty:
    last_sig = int(sig_series.iloc[-1])
    last_date_sig = df_strat["TRADE_DATE"].iloc[-1].date()

    if strategy == "Buy & Hold":
        st.info(
            f"As of **{last_date_sig}**, the Buy & Hold benchmark remains fully **LONG** "
            f"over {window_label}."
        )
    elif last_sig == 1:
        st.success(
            f"As of **{last_date_sig}**, the rule is **LONG / IN THE MARKET** for {ticker}."
        )
    else:
        st.warning(
            f"As of **{last_date_sig}**, the rule is **FLAT / OUT OF THE MARKET** for {ticker}."
        )
else:
    st.info("No valid trading signals were generated within the selected window.")

st.markdown("---")
st.markdown(
    "### How to read this view  \n"
    "<span style='font-size: 0.85rem; color: #6b7280;'>"
    "This page is a lightweight sandbox to illustrate how simple, rule-based trading "
    "logic behaves on a single stock. It is for demonstration only and is not investment advice."
    "</span>",
    unsafe_allow_html=True,
)

summary_text = generate_signal_summary(df_strat, strategy, ticker)
st.markdown(summary_text)
