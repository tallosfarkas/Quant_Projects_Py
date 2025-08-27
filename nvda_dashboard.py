#!/usr/bin/env python3
"""
Streamlit Dashboard: NVIDIA (NVDA) Live Monitoring

Features:
- Live (auto-refresh) intraday price chart with optional indicators (SMA, EMA, RSI)
- Historical price selection (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
- Key fundamentals (Income Statement, Balance Sheet, Cash Flow, Ratios / Margins)
- Valuation & Performance metrics
- Company profile & business summary
- Recent news headlines with source & published time
- Resilient data layer using OpenBB (primary) with yfinance fallback

Run:
  conda activate quant-env
  conda run -n quant-env pip install streamlit yfinance
  streamlit run nvda_dashboard.py
"""

import os
import time
import math
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

import streamlit as st

# Attempt OpenBB import
USE_OPENBB = True
try:
    from openbb import obb
except Exception:
    USE_OPENBB = False

# Fallback: yfinance
try:
    import yfinance as yf
except ImportError:
    yf = None

TICKER = "NVDA"

# ------------------------------ Utility Functions --------------------------- #

def safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return e  # Return exception for later display

@st.cache_data(ttl=60)
def get_intraday(interval: str = "1m", period: str = "1d") -> pd.DataFrame:
    """Fetch intraday data (short TTL)."""
    if USE_OPENBB:
        try:
            data = obb.equity.price.historical(TICKER, period=period, interval=interval)
            df = data.to_df() if hasattr(data, "to_df") else pd.DataFrame(data)
            if not df.empty:
                df.index = pd.to_datetime(df.index)
            return df
        except Exception:
            pass
    # Fallback yfinance
    if yf is not None:
        y = yf.Ticker(TICKER)
        df = y.history(period=period, interval=interval)
        return df
    return pd.DataFrame()

@st.cache_data(ttl=300)
def get_history(period: str = "6mo") -> pd.DataFrame:
    if USE_OPENBB:
        try:
            data = obb.equity.price.historical(TICKER, period=period)
            df = data.to_df() if hasattr(data, "to_df") else pd.DataFrame(data)
            if not df.empty:
                df.index = pd.to_datetime(df.index)
            return df
        except Exception:
            pass
    if yf is not None:
        y = yf.Ticker(TICKER)
        df = y.history(period=period)
        return df
    return pd.DataFrame()

@st.cache_data(ttl=900)
def get_fundamentals() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if USE_OPENBB:
        # Wrap each call individually
        mapping = {
            "income_statement": lambda: obb.equity.fundamental.income_statement(TICKER),
            "balance_sheet": lambda: obb.equity.fundamental.balance_sheet(TICKER),
            "cash_flow": lambda: obb.equity.fundamental.cash_flow(TICKER),
            "profile": lambda: obb.equity.profile(TICKER),
            "ratios": lambda: obb.equity.fundamental.ratios(TICKER),
        }
        for k, fn in mapping.items():
            res = safe_call(fn)
            if hasattr(res, "to_df"):
                try:
                    out[k] = res.to_df()
                except Exception:
                    out[k] = res
            else:
                out[k] = res
    # Fallback augment using yfinance where missing
    if yf is not None:
        y = yf.Ticker(TICKER)
        if "income_statement" not in out or isinstance(out.get("income_statement"), Exception):
            try:
                out["income_statement"] = y.income_stmt.T
            except Exception:
                pass
        if "balance_sheet" not in out or isinstance(out.get("balance_sheet"), Exception):
            try:
                out["balance_sheet"] = y.balance_sheet.T
            except Exception:
                pass
        if "cash_flow" not in out or isinstance(out.get("cash_flow"), Exception):
            try:
                out["cash_flow"] = y.cashflow.T
            except Exception:
                pass
        if "profile" not in out or isinstance(out.get("profile"), Exception):
            try:
                info = y.info
                out["profile"] = pd.DataFrame([info])
            except Exception:
                pass
    return out

@st.cache_data(ttl=300)
def get_news(limit: int = 15) -> pd.DataFrame:
    if USE_OPENBB:
        try:
            news = obb.news.company(TICKER, limit=limit)
            df = news.to_df() if hasattr(news, "to_df") else pd.DataFrame(news)
            return df
        except Exception:
            pass
    # Fallback using yfinance (limited & less structured)
    if yf is not None:
        try:
            y = yf.Ticker(TICKER)
            news_items = y.news[:limit]
            df = pd.DataFrame(news_items)
            return df
        except Exception:
            pass
    return pd.DataFrame()

# ------------------------------ Indicators ---------------------------------- #

def add_indicators(df: pd.DataFrame, sma_windows=(20, 50), ema_windows=(12, 26), rsi_period=14):
    if df is None or df.empty:
        return df
    df = df.copy()
    price_col = None
    for c in ["close", "Close", "adj_close", "Adj Close"]:
        if c in df.columns:
            price_col = c
            break
    if not price_col:
        return df
    for w in sma_windows:
        if w < len(df):
            df[f"SMA_{w}"] = df[price_col].rolling(w).mean()
    for w in ema_windows:
        if w < len(df):
            df[f"EMA_{w}"] = df[price_col].ewm(span=w, adjust=False).mean()
    # RSI
    if rsi_period < len(df):
        delta = df[price_col].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        roll_up = pd.Series(gain).rolling(rsi_period).mean()
        roll_down = pd.Series(loss).rolling(rsi_period).mean()
        rs = roll_up / roll_down
        rsi = 100 - (100 / (1 + rs))
        df["RSI"] = rsi.values
    return df

# ------------------------------ Streamlit UI -------------------------------- #

st.set_page_config(page_title="NVDA Live Dashboard", layout="wide", page_icon="💻")
st.title("NVIDIA (NVDA) – Live Monitoring Dashboard")
st.caption("Data via OpenBB platform (primary) with yfinance fallback. Educational use only.")

# Sidebar controls
st.sidebar.header("Controls")
period_hist = st.sidebar.selectbox("Historical Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=2)
intraday_interval = st.sidebar.selectbox("Intraday Interval", ["1m", "2m", "5m", "15m", "30m", "60m"], index=0)
refresh_seconds = st.sidebar.slider("Auto-refresh seconds", 15, 300, 60, step=15)
show_indicators = st.sidebar.multiselect("Indicators", ["SMA20", "SMA50", "EMA12", "EMA26", "RSI"], default=["SMA20", "EMA12", "RSI"])

# Auto refresh (replace deprecated experimental_rerun usage)
# Initialize last refresh timestamp in session state
if "_last_refresh_ts" not in st.session_state:
    st.session_state._last_refresh_ts = time.time()

elapsed = time.time() - st.session_state._last_refresh_ts
remaining = max(0, int(refresh_seconds - elapsed))
info_box = st.sidebar.empty()
info_box.markdown(f"Auto-refresh every **{refresh_seconds}s** • Next in **{remaining}s**")

# Trigger rerun when interval elapsed
if elapsed >= refresh_seconds:
    st.session_state._last_refresh_ts = time.time()
    st.rerun()

manual = st.sidebar.button("↻ Refresh now")
if manual:
    st.session_state._last_refresh_ts = time.time()
    st.rerun()

# Fetch data
with st.spinner("Fetching data..."):
    intraday_df = get_intraday(interval=intraday_interval, period="1d")
    history_df = get_history(period=period_hist)
    fundamentals = get_fundamentals()
    news_df = get_news(limit=20)

# Add indicators
intraday_df = add_indicators(intraday_df)

# ------------------------------ Layout -------------------------------------- #

col_price, col_metrics = st.columns([3, 1])
with col_price:
    st.subheader("Intraday Price")
    if intraday_df.empty:
        st.warning("No intraday data available.")
    else:
        price_col = [c for c in ["close", "Close", "adj_close", "Adj Close"] if c in intraday_df.columns]
        if price_col:
            base_col = price_col[0]
            plot_df = intraday_df[[base_col]].copy()
            # Append selected indicators if exist
            mapping = {"SMA20": "SMA_20", "SMA50": "SMA_50", "EMA12": "EMA_12", "EMA26": "EMA_26", "RSI": "RSI"}
            for label, col in mapping.items():
                if label in show_indicators and col in intraday_df.columns and label != "RSI":
                    plot_df[col] = intraday_df[col]
            st.line_chart(plot_df, height=400)
            if "RSI" in show_indicators and "RSI" in intraday_df.columns:
                st.line_chart(intraday_df[["RSI"]], height=150)
        else:
            st.dataframe(intraday_df.tail())
    st.markdown("---")
    st.subheader("Historical Price (Adjusted)")
    if history_df.empty:
        st.warning("No historical data available.")
    else:
        hist_col = [c for c in ["adj_close", "Adj Close", "close", "Close"] if c in history_df.columns]
        if hist_col:
            st.line_chart(history_df[hist_col])
        else:
            st.dataframe(history_df.tail())

with col_metrics:
    st.subheader("Snapshot")
    try:
        last_price = intraday_df[[c for c in intraday_df.columns if c.lower().startswith("close")][0]].iloc[-1]
        st.metric("Last Price", f"${last_price:,.2f}")
    except Exception:
        st.metric("Last Price", "—")
    if history_df is not None and not history_df.empty:
        try:
            ret_1m = history_df.iloc[-21:][[c for c in history_df.columns if c.lower().startswith("adj") or c.lower().startswith("close")][0]].pct_change().add(1).prod() - 1
            st.metric("~1M Return", f"{ret_1m*100:,.2f}%")
        except Exception:
            pass
    if "profile" in fundamentals and not isinstance(fundamentals.get("profile"), Exception):
        prof = fundamentals["profile"]
        if isinstance(prof, pd.DataFrame):
            fields = ["longName", "sector", "industry", "country", "website", "marketCap"]
            for f in fields:
                if f in prof.columns:
                    val = prof[f].iloc[0]
                    if f == "marketCap" and pd.notna(val):
                        val = f"${val/1e9:,.2f}B"
                    st.write(f"**{f}:** {val}")
    st.markdown("---")
    st.write("Data source priority: OpenBB > yfinance.")

st.markdown("---")

# Fundamentals Tabs
st.subheader("Fundamentals & Financials")
fin_tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow", "Ratios", "Profile Raw"])

fs_map = [
    ("income_statement", fin_tabs[0]),
    ("balance_sheet", fin_tabs[1]),
    ("cash_flow", fin_tabs[2]),
    ("ratios", fin_tabs[3]),
    ("profile", fin_tabs[4]),
]

for key, tab in fs_map:
    with tab:
        obj = fundamentals.get(key)
        if obj is None:
            st.info("No data available.")
        elif isinstance(obj, Exception):
            st.error(f"Error: {obj}")
        elif isinstance(obj, pd.DataFrame):
            # Limit very wide tables
            st.dataframe(obj.tail(12))
        else:
            st.write(obj)

# News Section
st.subheader("Recent News")
if news_df.empty:
    st.info("No news available.")
else:
    # Normalize columns
    display_cols = []
    rename_map = {}
    for c in news_df.columns:
        lc = c.lower()
        if lc in ["title", "published", "datetime", "provider", "source", "url", "link", "summary", "description"]:
            display_cols.append(c)
        if lc == "datetime":
            try:
                news_df[c] = pd.to_datetime(news_df[c], utc=True)
            except Exception:
                pass
    # Try derive 'published'
    if 'published' not in news_df.columns:
        for alt in ['datetime', 'published_at', 'time']:
            if alt in news_df.columns:
                news_df['published'] = news_df[alt]
                break
    cols_final = [c for c in ["published", "title", "provider", "source", "summary", "url", "link"] if c in news_df.columns]
    st.dataframe(news_df[cols_final].head(20))

st.markdown("---")
st.caption("© OpenBB / Yahoo Finance | For educational & research purposes only.")

