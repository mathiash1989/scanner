import pandas as pd
import requests
import urllib3
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(layout="wide")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

KRAKEN_OHLC = "https://api.kraken.com/0/public/OHLC"

# -------------------------------
# 50 Most Popular Kraken USD Pairs
# -------------------------------
POPULAR_PAIRS = [
    "BCHUSD", "FARTCOINUSD", "JASMYUSD", "TRXUSD", "BATUSD"
    #"FILUSD", "ICPUSD", "SOLUSD", "WIFUSD", "MINAUSD", "LTCUSD", "DOGEUSD", "ETCUSD"
    # full list omitted for now…
    #"XBTUSD","ETHUSD","SOLUSD","XRPUSD","ADAUSD","DOGEUSD","AVAXUSD","DOTUSD","LINKUSD","ATOMUSD",
    #"LTCUSD","BCHUSD","ETCUSD","UNIUSD","FILUSD","MATICUSD","XLMUSD","ALGOUSD","APTUSD","ARBUSD",
    #"SUIUSD","OPUSD","HBARUSD","MKRUSD","AAVEUSD","CRVUSD","IMXUSD","SNXUSD","INJUSD","FTMUSD",
    #"GRTUSD","TRXUSD","BNBUSD","EGLDUSD","NEARUSD","QNTUSD","FLOWUSD","KSMUSD","DASHUSD","ZECUSD",
    #"COMPUSD","YFIUSD","LDOUSD","RUNEUSD","1INCHUSD","ENJUSD","CHZUSD","ICPUSD","MINAUSD","SOLUSD"
]

# ------------------------------------------
# Download OHLC from Kraken
# ------------------------------------------
def get_ohlcv_kraken(pair, interval=1440, limit=200):
    url = f"{KRAKEN_OHLC}?pair={pair}&interval={interval}"
    try:
        resp = requests.get(url, verify=False).json()
    except Exception:
        return None

    if resp.get("error"):
        return None

    key = next(iter(resp["result"]))
    raw = resp["result"][key]

    df = pd.DataFrame(raw, columns=[
        "time","open","high","low","close","vwap","volume","count"
    ])
    df[["open","high","low","close","vwap","volume"]] = df[
        ["open","high","low","close","vwap","volume"]
    ].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit='s')
    df = df.set_index("time")
    return df.tail(limit)


# ------------------------------------------
# ATR-20 for Turtle N
# ------------------------------------------
def calculate_atr(df, period=20):
    df = df.copy()
    df["prev_close"] = df["close"].shift(1)

    df["TR"] = df.apply(lambda row: max(
        row["high"] - row["low"],
        abs(row["high"] - row["prev_close"]),
        abs(row["low"] - row["prev_close"])
    ), axis=1)

    df["ATR"] = df["TR"].rolling(period).mean()
    return df


# ------------------------------------------
# Turtle System 1 memory rule
# ------------------------------------------
def last_breakout_is_win(df):
    high20_series = df["high"].rolling(20).max()
    close = df["close"]
    prev_close = df["close"].shift(1)

    breakout_up = (prev_close < high20_series.shift(1)) & (close > high20_series)
    breakout_indices = breakout_up[breakout_up].index.tolist()

    if len(breakout_indices) < 2:
        return False

    last_idx = breakout_indices[-1]
    prev_idx = breakout_indices[-2]

    entry_price = df.loc[prev_idx, "close"]
    exit_price = df.loc[last_idx, "close"]

    return exit_price > entry_price


# ------------------------------------------
# Full Turtle Breakout Logic with previous trade info
# ------------------------------------------
def breakout_distances(df, symbol, portfolio_size=10000):
    df = calculate_atr(df)
    N = df["ATR"].iloc[-1]

    stop_distance = 2 * N
    risk_per_unit = portfolio_size * 0.01
    unit_size = risk_per_unit / stop_distance if stop_distance > 0 else 0

    current = df["close"].iloc[-1]
    volume = df["volume"].iloc[-1]

    # 20d / 55d highs
    high20 = df["high"].rolling(20).max().iloc[-1]
    high55 = df["high"].rolling(55).max().iloc[-1]

    dist20 = (high20 - current) / high20 * 100
    dist55 = (high55 - current) / high55 * 100

    idx_20 = df["high"].rolling(20).apply(lambda x: x.argmax()).iloc[-1]
    idx_55 = df["high"].rolling(55).apply(lambda x: x.argmax()).iloc[-1]

    # -------------------------------------------------------
    # 1) Correct Turtle System-1 breakout detection
    # -------------------------------------------------------
    high20_prev = df["high"].rolling(20).max().shift(1)
    breakout_signal = (df["close"] > high20_prev) & (df["close"].shift(1) <= high20_prev)
    breakout_indices = breakout_signal[breakout_signal].index.tolist()

    # -------------------------------------------------------
    # If no previous trade → return default
    # -------------------------------------------------------
    if len(breakout_indices) < 2:
        return {
            "symbol": symbol,
            "current_price": current,
            "20d_high": high20,
            "20d_dist_%": dist20,
            "55d_high": high55,
            "55d_dist_%": dist55,
            "N": N,
            "2N_stop_distance": stop_distance,
            "unit_size_coins": unit_size,

            "system1_allowed": True,
            "system1_previous_trade": "No Trade",
            "system1_prev_entry": None,
            "system1_prev_stop_loss": None,
            "system1_prev_exit": None,
            "system1_prev_exit_type": None,

            "volume": volume,
        }

    # -------------------------------------------------------
    # 2) SIMULATE previous trade
    # -------------------------------------------------------
    entry_idx = breakout_indices[-2]
    entry_price = df.loc[entry_idx, "close"]
    N_entry = df.loc[entry_idx, "ATR"]
    stop_loss = entry_price - 2 * N_entry

    # Forward-scan until stop-loss or 10-day-low exit
    df_fwd = df.loc[df.index > entry_idx]
    exit_price = None
    exit_type = None

    for i in range(len(df_fwd)):
        row = df_fwd.iloc[i]
        ts = row.name

        # STOP LOSS first
        if row["low"] <= stop_loss:
            exit_price = stop_loss
            exit_type = "STOP LOSS"
            outcome = "LOSS"
            break

        # 10-day low exit
        low10 = df.loc[:ts]["low"].tail(10).min()
        if row["close"] < low10:
            exit_price = row["close"]
            exit_type = "10-DAY LOW EXIT"
            outcome = "WIN"
            break

    # No exit found
    if exit_price is None:
        exit_price = df_fwd["close"].iloc[-1]
        exit_type = "OPEN"
        outcome = "UNKNOWN"

    # Memory rule: System-1 allowed only if previous trade was LOSS
    system1_allowed = (outcome == "LOSS")

    return {
        "symbol": symbol,
        "current_price": current,
        "20d_high": high20,
        "20d_dist_%": dist20,
        "55d_high": high55,
        "55d_dist_%": dist55,
        "N": N,
        "2N_stop_distance": stop_distance,
        "unit_size_coins": unit_size,

        "system1_allowed": system1_allowed,
        "system1_previous_trade": outcome,
        "system1_prev_entry": entry_price,
        "system1_prev_stop_loss": stop_loss,
        "system1_prev_exit": exit_price,
        "system1_prev_exit_type": exit_type,

        "volume": volume,
    }



# ------------------------------------------
# Parallel scanner
# ------------------------------------------
def scan_pairs(pairs, workers=10):
    results = []

    def worker(pair):
        df = get_ohlcv_kraken(pair)
        if df is None or len(df) < 60:
            return None
        return breakout_distances(df, pair)

    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(worker, p): p for p in pairs}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                results.append(res)

    return pd.DataFrame(results)


# ------------------------------------------
# STREAMLIT UI
# ------------------------------------------
st.title("🐢 Crypto Turtle Breakout Scanner 🐢")
st.caption("Top strongest breakouts & near-breakouts")

# Run scan
with st.spinner("Scanning Kraken pairs…"):
    df = scan_pairs(POPULAR_PAIRS)

# Sort by 20d dist % (ascending)
ranked = df.sort_values("20d_dist_%").head(20)

# List of numeric columns to format
numeric_cols = [
    "current_price", "20d_high", "20d_dist_%", "55d_high", "55d_dist_%",
    "N", "2N_stop_distance", "unit_size_coins", "volume"
]

# Format numeric columns as strings with commas and 2 decimals
for col in numeric_cols:
    if col in ranked.columns:
        ranked[col] = ranked[col].map(lambda x: f"{x:,.3f}")

# ------------------------------
# Explanations / Documentation
# ------------------------------
st.subheader("About This Table 📝")
with st.expander("Click to see column explanations and Turtle System 1 logic"):
    st.markdown("""
### 🐢 Turtle System 1 Overview  
This dashboard implements the **original Turtle System 1 breakout method**, adapted for crypto.

### How System-1 Works:
1. **A breakout occurs when today's close goes above yesterday’s 20-day high.**  
   This is the official Turtle rule for entering long trades.

2. **Before allowing a new breakout trade, the system checks the outcome of the *previous* breakout trade.**  
   - If the previous System-1 trade ended in a **LOSS**, the next breakout is allowed.  
   - If the previous System-1 trade ended in a **WIN**, the next breakout must be **ignored**  
     (this is called the “Turtle memory rule”).

3. **Each previous trade is simulated historically**:
   - Entry at the breakout close price  
   - Stop-loss at **2 × N** below entry  
   - Exit at first of:  
       - Stop-loss hit → **LOSS**  
       - Close below 10-day low → **WIN** (trend exit)  
       - Still active → **OPEN**  

This table shows your **current breakout levels**, plus **the full evaluation of the last System-1 trade**.


---

## 📘 Column Explanations

### **Market Information**
- **`symbol`** — Kraken trading pair symbol  
- **`current_price`** — Latest daily closing price  
- **`volume`** — Latest daily trading volume

---

### **Breakout Levels**
- **`20d_high`** — Highest *high* of the last 20 days  
- **`20d_dist_%`** — Percentage distance from current price to 20-day breakout  
- **`55d_high`** — Highest *high* of the last 55 days  
- **`55d_dist_%`** — Percentage distance from current price to 55-day breakout  

These show how close each asset is to triggering a Turtle breakout.

---
### **Volatility & Position Sizing**
- **`N`** — ATR-20 volatility (Turtle “N” unit)  
- **`2N_stop_distance`** — Recommended stop distance (2 × N)  
- **`unit_size_coins`** — Position size (in coins) using a **1% risk model**  
  - Based on a default portfolio size of **$10,000**  
  - Formula: `risk_per_trade / (2N)`

---
               
### **Previous System-1 Trade Simulation**
These fields come from a **full historical simulation**:

- **`system1_previous_trade`** — Result of the last actual Turtle break:  
  - **WIN** = Exit at 10-day low (trend exit)  
  - **LOSS** = 2×N stop-loss hit  
  - **OPEN** = No exit triggered  
  - **No Trade** = Not enough history to simulate

- **`system1_prev_entry`** — Entry price of the previous breakout  
- **`system1_prev_stop_loss`** — Stop-loss level for that entry  
- **`system1_prev_exit`** — Exit price  
- **`system1_prev_exit_type`** — `"STOP LOSS"` / `"10-DAY LOW EXIT"` / `"OPEN"`

---

                
### **System-1 Eligibility**
- **`system1_allowed`**  
  - **True** = You are allowed to take the next 20-day breakout  
  - **False** = Previous trade was a **WIN**, so the next breakout must be skipped  
    (per official Turtle rules)

---
""")

# ------------------------------
# Display Table
# ------------------------------
# Rank the results
ranked = ranked.reset_index(drop=True)
ranked.insert(0, "Rank", range(1, len(ranked) + 1))

# ---- CSS-based styling (no matplotlib needed) ----

def color_system1(val):
    if val is True:
        return "color: white; background-color: #3CB371; font-weight: bold;"  # green
    elif val is False:
        return "color: white; background-color: #8B0000; font-weight: bold;"  # red
    return ""

def heatmap_color(v):
    """Manually create a green/red gradient without matplotlib."""
    if pd.isna(v):
        return ""
    
    # Normalize around zero
    if v > 0:
        # Green scale
        intensity = min(int(v * 5), 100)
        return f"background-color: rgba(0, 180, 0, {intensity/100});"
    else:
        # Red scale
        intensity = min(int(abs(v) * 5), 100)
        return f"background-color: rgba(200, 0, 0, {intensity/100});"

def apply_heatmap(df, cols):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for c in cols:
        styles[c] = df[c].apply(heatmap_color)
    return styles


heatmap_cols = ["20d_dist_%", "55d_dist_%"]
base_styles = apply_heatmap(ranked, heatmap_cols)

styled = (
    ranked.style
        .map(color_system1, subset=["system1_allowed"])
        .apply(lambda _: base_styles, axis=None)
)

st.subheader("Top Results")

st.write(
    styled.hide(axis="index")
)

