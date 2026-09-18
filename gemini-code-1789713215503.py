
"""
NSE Single-Stock Short Strangle Backtester — Institutional v2
===============================================================

Run:
    pip install streamlit pandas numpy scipy plotly yfinance requests
    streamlit run nse_strangle_v2.py

Data source
-----------
NSE India direct data only. The dashboard does not require file uploads and
does not use Yahoo Finance or Black-Scholes prices for the production path.
Historical equity prices are fetched from NSE's historical equity endpoint;
historical stock-option prices are fetched contract-by-contract from NSE's
historical F&O endpoint. NSE contract metadata (lot size and strike scheme)
is also downloaded directly from NSE.

Important
---------
NSE's current contract specification says individual-security options expire
on Tuesday of the expiry period; if that Tuesday is a trading holiday,
expiry is the previous trading day. The application therefore uses a
holiday-aware Tuesday rule for synthetic/theoretical cycles.

Actual historical option prices should be preferred for execution research.
"""

import io
import math
import calendar
import datetime as dt
import time
import requests

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="NSE Strangle Quant Terminal",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

GREEN = "#00CC96"
RED = "#FF5B61"
BLUE = "#4C9AFF"
YELLOW = "#F6C85F"
PURPLE = "#9B7EDE"
CYAN = "#4CC9F0"
BG = "#080B10"
CARD = "#0D121A"
GRID = "#202733"
TEXT = "#E8EDF3"
MUTED = "#7F8997"


# ============================================================
# UI
# ============================================================

st.markdown(
    f"""
<style>
.stApp {{
    background:
      radial-gradient(circle at 15% 0%, rgba(76,154,255,.09), transparent 30%),
      #080b10;
    color: {TEXT};
}}
.block-container {{
    max-width: 1650px;
    padding: 1rem 1.2rem 3rem;
}}
section[data-testid="stSidebar"] {{
    background:#0b0f15;
    border-right:1px solid {GRID};
}}
.dashboard {{
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:18px;
    padding:20px 24px;
    margin-bottom:14px;
    border:1px solid {GRID};
    border-radius:14px;
    background:linear-gradient(135deg,#111722,#0b0f15);
}}
.title {{
    font-size:1.6rem;
    font-weight:800;
    letter-spacing:-.5px;
}}
.subtitle {{
    color:{MUTED};
    font-size:.8rem;
    margin-top:4px;
}}
.status {{
    border:1px solid rgba(0,204,150,.3);
    background:rgba(0,204,150,.08);
    color:{GREEN};
    border-radius:999px;
    padding:7px 13px;
    font-size:.72rem;
    font-weight:700;
    white-space:nowrap;
}}
.mode {{
    border:1px solid rgba(246,200,95,.3);
    background:rgba(246,200,95,.08);
    color:{YELLOW};
    border-radius:999px;
    padding:7px 13px;
    font-size:.72rem;
    font-weight:700;
    white-space:nowrap;
}}
.kpis {{
    display:grid;
    grid-template-columns:repeat(6,minmax(0,1fr));
    gap:10px;
    margin:12px 0 16px;
}}
.kpi {{
    background:linear-gradient(145deg,#111722,#0d121a);
    border:1px solid {GRID};
    border-radius:11px;
    padding:13px 14px;
    min-height:94px;
}}
.kpi-label {{
    color:{MUTED};
    text-transform:uppercase;
    letter-spacing:.65px;
    font-size:.62rem;
    font-weight:700;
}}
.kpi-value {{
    color:{TEXT};
    font-size:1.22rem;
    font-weight:800;
    margin-top:7px;
}}
.kpi-sub {{
    color:#697482;
    font-size:.65rem;
    margin-top:4px;
}}
.section {{
    color:#e9edf2;
    font-size:1rem;
    font-weight:750;
    margin:12px 0 8px;
}}
.card {{
    background:{CARD};
    border:1px solid {GRID};
    border-radius:11px;
    padding:14px;
    height:100%;
}}
.small {{
    color:{MUTED};
    font-size:.68rem;
    text-transform:uppercase;
    letter-spacing:.5px;
}}
.big {{
    color:{TEXT};
    font-size:1.05rem;
    font-weight:750;
    margin-top:5px;
}}
@media(max-width:1100px){{
    .kpis{{grid-template-columns:repeat(3,minmax(0,1fr));}}
}}
@media(max-width:650px){{
    .block-container{{padding:.6rem .45rem 2rem;}}
    .dashboard{{padding:14px;}}
    .title{{font-size:1.15rem;}}
    .kpis{{grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;}}
    .kpi{{padding:10px;min-height:84px;}}
    .kpi-value{{font-size:1rem;}}
}}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MARKET SPECIFICATIONS
# ============================================================

FNO = {
    "ADANIENT": {"name": "Adani Enterprises", "lot": 1, "step": 20},
    "ADANIPORTS": {"name": "Adani Ports & SEZ", "lot": 1, "step": 20},
    "APOLLOHOSP": {"name": "Apollo Hospitals", "lot": 1, "step": 50},
    "ASIANPAINT": {"name": "Asian Paints", "lot": 1, "step": 20},
    "AXISBANK": {"name": "Axis Bank", "lot": 1, "step": 10},
    "BAJAJ-AUTO": {"name": "Bajaj Auto", "lot": 1, "step": 50},
    "BAJAJFINSV": {"name": "Bajaj Finserv", "lot": 1, "step": 20},
    "BAJFINANCE": {"name": "Bajaj Finance", "lot": 1, "step": 20},
    "BEL": {"name": "Bharat Electronics", "lot": 1, "step": 5},
    "BHARTIARTL": {"name": "Bharti Airtel", "lot": 1, "step": 10},
    "CIPLA": {"name": "Cipla", "lot": 1, "step": 20},
    "COALINDIA": {"name": "Coal India", "lot": 1, "step": 5},
    "DRREDDY": {"name": "Dr Reddy's Laboratories", "lot": 1, "step": 20},
    "EICHERMOT": {"name": "Eicher Motors", "lot": 1, "step": 25},
    "ETERNAL": {"name": "Eternal", "lot": 1, "step": 5},
    "GRASIM": {"name": "Grasim Industries", "lot": 1, "step": 20},
    "HCLTECH": {"name": "HCL Technologies", "lot": 1, "step": 20},
    "HDFCBANK": {"name": "HDFC Bank", "lot": 1, "step": 10},
    "HDFCLIFE": {"name": "HDFC Life Insurance", "lot": 1, "step": 10},
    "HINDALCO": {"name": "Hindalco Industries", "lot": 1, "step": 5},
    "HINDUNILVR": {"name": "Hindustan Unilever", "lot": 1, "step": 20},
    "ICICIBANK": {"name": "ICICI Bank", "lot": 1, "step": 10},
    "INDIGO": {"name": "InterGlobe Aviation", "lot": 1, "step": 50},
    "INFY": {"name": "Infosys", "lot": 1, "step": 20},
    "ITC": {"name": "ITC", "lot": 1, "step": 5},
    "JIOFIN": {"name": "Jio Financial Services", "lot": 1, "step": 5},
    "JSWSTEEL": {"name": "JSW Steel", "lot": 1, "step": 5},
    "KOTAKBANK": {"name": "Kotak Mahindra Bank", "lot": 1, "step": 10},
    "LT": {"name": "Larsen & Toubro", "lot": 1, "step": 25},
    "MARUTI": {"name": "Maruti Suzuki", "lot": 1, "step": 50},
    "MAXHEALTH": {"name": "Max Healthcare", "lot": 1, "step": 10},
    "M&M": {"name": "Mahindra & Mahindra", "lot": 1, "step": 25},
    "NESTLEIND": {"name": "Nestle India", "lot": 1, "step": 20},
    "NTPC": {"name": "NTPC", "lot": 1, "step": 5},
    "ONGC": {"name": "ONGC", "lot": 1, "step": 5},
    "POWERGRID": {"name": "Power Grid", "lot": 1, "step": 5},
    "RELIANCE": {"name": "Reliance Industries", "lot": 1, "step": 20},
    "SBILIFE": {"name": "SBI Life Insurance", "lot": 1, "step": 10},
    "SBIN": {"name": "State Bank of India", "lot": 1, "step": 5},
    "SHRIRAMFIN": {"name": "Shriram Finance", "lot": 1, "step": 10},
    "SUNPHARMA": {"name": "Sun Pharmaceutical", "lot": 1, "step": 20},
    "TATACONSUM": {"name": "Tata Consumer Products", "lot": 1, "step": 10},
    "TATASTEEL": {"name": "Tata Steel", "lot": 1, "step": 2},
    "TCS": {"name": "Tata Consultancy Services", "lot": 1, "step": 50},
    "TECHM": {"name": "Tech Mahindra", "lot": 1, "step": 20},
    "TITAN": {"name": "Titan Company", "lot": 1, "step": 25},
    "TMPV": {"name": "Tata Motors Passenger Vehicles", "lot": 1, "step": 10},
    "TRENT": {"name": "Trent", "lot": 1, "step": 20},
    "ULTRACEMCO": {"name": "UltraTech Cement", "lot": 1, "step": 50},
    "WIPRO": {"name": "Wipro", "lot": 1, "step": 5},
}

# September-2026 transition:
# Wipro remains in the current 13-Sep-2026 constituent list used here.
# BSE is scheduled to replace Wipro effective 30-Sep-2026 after the close
# on 29-Sep-2026. Update this universe after the effective date if desired.



# ============================================================
# QUANT HELPERS
# ============================================================

def bs_price(S, K, T, r, sigma, kind):
    if T <= 1e-6:
        return max(S-K, 0.0) if kind == "C" else max(K-S, 0.0)

    sigma = max(float(sigma), 1e-6)
    d1 = (np.log(S/K) + (r + .5*sigma*sigma)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if kind == "C":
        value = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        value = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

    return max(float(value), 0.0)


def third_tuesday_or_previous_trading_day(year, month, trading_dates):
    """
    Current NSE individual-security rule:
    Tuesday of expiry period; previous trading day if Tuesday is holiday.
    """
    c = calendar.Calendar().monthdatescalendar(year, month)
    tuesdays = [
        d for week in c for d in week
        if d.month == month and d.weekday() == 1
    ]
    if not tuesdays:
        return None

    # Expiry period convention: use the last Tuesday.
    target = tuesdays[-1]
    valid = [d for d in trading_dates if d <= target]
    return valid[-1] if valid else None


@st.cache_data(ttl=3600, show_spinner=False)
def yahoo_data(symbol, years):
    end = dt.date.today()
    start = end - dt.timedelta(days=years*365 + 150)
    df = yf.download(
        f"{symbol}.NS",
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df = df[keep].dropna()
    df.index = pd.to_datetime(df.index).date
    return df


# ============================================================
# NSE CSV PARSER
# ============================================================

ALIASES = {
    "date": [
        "Date", "Trade Date", "TIMESTAMP", "DATE", "trade_date"
    ],
    "symbol": [
        "Symbol", "SYMBOL", "Underlying", "Underlying Symbol"
    ],
    "expiry": [
        "Expiry", "EXPIRY", "Expiry Date", "ExpiryDate"
    ],
    "option_type": [
        "Option Type", "OPTION TYPE", "OptionType", "Type", "OPT_TYPE"
    ],
    "strike": [
        "Strike Price", "STRIKE", "Strike", "StrikePrice"
    ],
    "close": [
        "Close Price", "CLOSE", "Close", "Settlement Price",
        "Settle Price", "SETTLE_PRICE", "Settlement"
    ],
}


def find_column(df, aliases):
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in normalized:
            return normalized[a.lower()]

    for c in df.columns:
        s = str(c).strip().lower()
        for a in aliases:
            if a.lower() in s:
                return c
    return None


def parse_nse_csv(uploaded):
    if uploaded is None:
        return pd.DataFrame()

    raw = pd.read_csv(uploaded)

    cols = {
        k: find_column(raw, v)
        for k, v in ALIASES.items()
    }

    required = ["date", "expiry", "option_type", "strike", "close"]
    if any(cols[k] is None for k in required):
        return pd.DataFrame()

    out = pd.DataFrame({
        "Date": pd.to_datetime(raw[cols["date"]], errors="coerce").dt.date,
        "Expiry": pd.to_datetime(raw[cols["expiry"]], errors="coerce").dt.date,
        "OptionType": raw[cols["option_type"]].astype(str).str.upper().str.strip(),
        "Strike": pd.to_numeric(raw[cols["strike"]], errors="coerce"),
        "Close": pd.to_numeric(raw[cols["close"]], errors="coerce"),
    })

    if cols["symbol"] is not None:
        out["Symbol"] = raw[cols["symbol"]].astype(str).str.upper().str.strip()
    else:
        out["Symbol"] = ""

    out = out.dropna(subset=["Date","Expiry","Strike","Close"])

    out["OptionType"] = (
        out["OptionType"]
        .replace({
            "CALL": "CE",
            "PUT": "PE",
            "C": "CE",
            "P": "PE",
        })
    )

    out = out[out["OptionType"].isin(["CE","PE"])]

    return out.sort_values(["Date","Expiry","Strike"])



# ============================================================
# NSE DIRECT DATA ENGINE
# ============================================================

NSE_BASE = "https://www.nseindia.com"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/136.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-IN,en;q=0.9,en-US;q=0.8",
    "Referer": "https://www.nseindia.com/report-detail/fo_eq_security",
    "X-Requested-With": "XMLHttpRequest",
}


def _nse_session():
    s = requests.Session()
    s.headers.update(NSE_HEADERS)
    # NSE expects a normal page visit before API requests so that session
    # cookies are established.
    s.get(f"{NSE_BASE}/report-detail/fo_eq_security", timeout=20)
    return s


@st.cache_resource(show_spinner=False)
def nse_http_session():
    return _nse_session()


def _nse_get_json(url, params=None, retries=3):
    last = None
    for attempt in range(retries):
        try:
            r = nse_http_session().get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (401, 403, 429):
                # Refresh cookies once and back off. NSE can throttle bursts.
                nse_http_session().get(f"{NSE_BASE}/", timeout=20)
            last = RuntimeError(f"NSE HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise last or RuntimeError("NSE request failed")


@st.cache_data(ttl=86400, show_spinner=False)
def nse_equity_history(symbol, start_date, end_date):
    """Daily NSE cash-market OHLC for one symbol."""
    url = f"{NSE_BASE}/api/historicalOR/generateSecurityWiseHistoricalData"
    params = {
        "from": start_date.strftime("%d-%m-%Y"),
        "to": end_date.strftime("%d-%m-%Y"),
        "symbol": symbol,
        "type": "priceVolumeDeliverable",
        "series": "EQ",
        "csv": "true",
    }
    payload = _nse_get_json(url, params)
    rows = payload.get("data", payload) if isinstance(payload, dict) else payload
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    def col(*names):
        return find_column(df, list(names))

    d = col("Date", "CH_TIMESTAMP", "mTIMESTAMP", "timestamp")
    c = col("Close", "CH_CLOSING_PRICE", "close")
    if d is None or c is None:
        return pd.DataFrame()
    out = pd.DataFrame({
        "Date": pd.to_datetime(df[d], errors="coerce", dayfirst=True).dt.date,
        "Close": pd.to_numeric(df[c], errors="coerce"),
    }).dropna()
    return out.sort_values("Date").drop_duplicates("Date")


@st.cache_data(ttl=86400, show_spinner=False)
def nse_contract_metadata():
    """Download NSE's current permitted lot-size and strike-scheme files."""
    out = {"lots": {}, "strikes": pd.DataFrame(), "status": "OK"}
    try:
        lot_url = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"
        lot_df = pd.read_csv(lot_url)
        if lot_df.shape[1] >= 2:
            # NSE file has symbol and market-lot fields; use robust aliases.
            sc = find_column(lot_df, ["SYMBOL", "Symbol", "Underlying", "CODE"])
            lc = find_column(lot_df, ["MARKET LOT", "Market Lot", "LOT SIZE", "Lot Size"])
            if sc is not None and lc is not None:
                out["lots"] = dict(zip(
                    lot_df[sc].astype(str).str.upper().str.strip(),
                    pd.to_numeric(lot_df[lc], errors="coerce").fillna(1).astype(int)
                ))
    except Exception as e:
        out["status"] = f"Lot file unavailable: {e}"

    try:
        strike_url = "https://archives.nseindia.com/content/fo/NSE_FO_SosScheme.csv"
        strike_df = pd.read_csv(strike_url)
        strike_df.columns = [str(c).strip() for c in strike_df.columns]
        out["strikes"] = strike_df
    except Exception as e:
        if out["status"] == "OK":
            out["status"] = f"Strike scheme unavailable: {e}"
    return out


def _step_from_scheme(symbol, month_type="Near Month", fallback=5.0):
    meta = nse_contract_metadata()
    df = meta.get("strikes", pd.DataFrame())
    if df.empty:
        return fallback
    sc = find_column(df, ["Symbol", "SYMBOL"])
    mc = find_column(df, ["Month type", "Month Type", "MONTH TYPE"])
    vc = find_column(df, ["Step Value", "Step value", "STEP VALUE"])
    if sc is None or vc is None:
        return fallback
    x = df[df[sc].astype(str).str.upper().str.strip().eq(symbol.upper())].copy()
    if mc is not None:
        m = x[x[mc].astype(str).str.strip().str.lower().eq(month_type.lower())]
        if not m.empty:
            x = m
    if x.empty:
        return fallback
    v = pd.to_numeric(x[vc], errors="coerce").dropna()
    return float(v.iloc[0]) if not v.empty else fallback


def _nearest_strike(target, step):
    step = max(float(step), 0.05)
    return round(round(float(target) / step) * step, 2)


def _expiry_for_month(year, month, trading_dates):
    """Last Tuesday of month, shifted back to the previous NSE trading day."""
    last_day = calendar.monthrange(year, month)[1]
    d = dt.date(year, month, last_day)
    while d.weekday() != 1:
        d -= dt.timedelta(days=1)
    dates = set(trading_dates)
    while d not in dates and d >= dt.date(year, month, 1):
        d -= dt.timedelta(days=1)
    return d


@st.cache_data(ttl=86400, show_spinner=False)
def nse_option_history(symbol, expiry, option_type, strike, start_date, end_date):
    """Actual NSE daily contract-wise OHLC/OI for one OPTSTK contract."""
    url = f"{NSE_BASE}/api/historicalOR/foCPV"
    params = {
        "from": start_date.strftime("%d-%m-%Y"),
        "to": end_date.strftime("%d-%m-%Y"),
        "instrumentType": "OPTSTK",
        "symbol": symbol,
        "year": str(expiry.year),
        "expiryDate": expiry.strftime("%d-%b-%Y").upper(),
        "optionType": option_type,
        "strikePrice": f"{strike:g}",
    }
    payload = _nse_get_json(url, params)
    rows = payload.get("data", []) if isinstance(payload, dict) else payload
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    dc = find_column(df, ["Date", "Trade Date", "TIMESTAMP", "CH_TIMESTAMP", "timestamp"])
    cc = find_column(df, ["Close Price", "CLOSE", "Close", "CH_CLOSING_PRICE", "close"])
    if dc is None or cc is None:
        return pd.DataFrame()
    out = pd.DataFrame({
        "Date": pd.to_datetime(df[dc], errors="coerce", dayfirst=True).dt.date,
        "Close": pd.to_numeric(df[cc], errors="coerce"),
    })
    # Preserve observed lot size if NSE returns it in the contract data.
    lc = find_column(df, ["Market Lot", "MARKET LOT", "marketLot", "MarketLot"])
    if lc is not None:
        out["MarketLot"] = pd.to_numeric(df[lc], errors="coerce")
    else:
        out["MarketLot"] = np.nan
    return out.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates("Date")


def _fetch_contract_with_fallback(symbol, expiry, option_type, target_strike, start_date, end_date, step):
    # Historical strike schemes can differ after quarterly reviews. Try the
    # model-selected strike plus nearby NSE intervals and use the first actual
    # contract returned by NSE. This avoids inventing an option price.
    candidates = []
    base = _nearest_strike(target_strike, step)
    for k in range(0, 5):
        if k == 0:
            candidates.append(base)
        else:
            candidates.extend([round(base + k*step, 2), round(base - k*step, 2)])
    seen = set()
    for strike in candidates:
        if strike in seen or strike <= 0:
            continue
        seen.add(strike)
        try:
            df = nse_option_history(symbol, expiry, option_type, strike, start_date, end_date)
            if not df.empty:
                return strike, df
        except Exception:
            continue
    return np.nan, pd.DataFrame()


def nse_actual_backtest(symbol, years, y_pct, stop_mult, friction, margin_pct):
    """Execution-style daily-close backtest using observed NSE option prices."""
    end = dt.date.today()
    start = dt.date(end.year - years, end.month, end.day)
    spot = nse_equity_history(symbol, start, end)
    if spot.empty:
        return pd.DataFrame(), "No NSE equity history"
    spot_map = dict(zip(spot["Date"], spot["Close"]))
    trading_dates = sorted(spot_map)
    if len(trading_dates) < 40:
        return pd.DataFrame(), "Insufficient NSE equity history"

    meta = nse_contract_metadata()
    default_lot = int(meta.get("lots", {}).get(symbol.upper(), FNO.get(symbol, {}).get("lot", 1)))
    step = _step_from_scheme(symbol, fallback=FNO.get(symbol, {}).get("step", 5.0))

    cycles = []
    # Calendar months whose 15th has a later trading date in the sample.
    months = sorted(set((d.year, d.month) for d in trading_dates))
    for y, m in months[:-1]:
        entry_target = dt.date(y, m, 15)
        entry_candidates = [d for d in trading_dates if d >= entry_target and d.year == y and d.month == m]
        if not entry_candidates:
            continue
        entry_date = entry_candidates[0]

        # Next calendar month
        nm = 1 if m == 12 else m + 1
        ny = y + 1 if m == 12 else y
        exit_target = dt.date(ny, nm, 15)
        exit_candidates = [d for d in trading_dates if d >= exit_target and d.year == ny and d.month == nm]
        if not exit_candidates:
            continue
        planned_exit = exit_candidates[0]
        expiry = _expiry_for_month(ny, nm, trading_dates)
        if expiry is None or expiry <= entry_date:
            continue
        exit_date = min(planned_exit, expiry)
        exit_candidates2 = [d for d in trading_dates if entry_date <= d <= exit_date]
        if not exit_candidates2:
            continue

        S = float(spot_map[entry_date])
        put_target = S * (1 - y_pct/100)
        call_target = S * (1 + y_pct/100)
        # Use the applicable near-month step from NSE's current strike scheme;
        # historical contract fallback prevents synthetic prices.
        pstrike, put = _fetch_contract_with_fallback(
            symbol, expiry, "PE", put_target, entry_date, exit_date, step
        )
        cstrike, call = _fetch_contract_with_fallback(
            symbol, expiry, "CE", call_target, entry_date, exit_date, step
        )
        if put.empty or call.empty:
            continue

        p = put.set_index("Date")["Close"].rename("Put")
        c = call.set_index("Date")["Close"].rename("Call")
        path = pd.concat([p, c], axis=1).dropna()
        if path.empty or entry_date not in path.index:
            continue
        path = path[(path.index >= entry_date) & (path.index <= exit_date)]
        if path.empty:
            continue

        entry_credit = float(path.loc[entry_date, "Put"] + path.loc[entry_date, "Call"])
        if entry_credit <= 0:
            continue
        stop_level = entry_credit * stop_mult if stop_mult > 0 else np.inf
        stop_date = None
        for d, r in path.iterrows():
            if d <= entry_date:
                continue
            if float(r["Put"] + r["Call"]) >= stop_level:
                stop_date = d
                break
        if stop_date is not None:
            actual_exit = stop_date
            exit_reason = "Stop Loss"
        else:
            actual_exit = path.index[-1]
            exit_reason = "Scheduled / Expiry"
        exit_credit = float(path.loc[actual_exit, "Put"] + path.loc[actual_exit, "Call"])
        lot = default_lot
        observed_lots = pd.concat([put, call])["MarketLot"].dropna()
        if not observed_lots.empty:
            lot = int(observed_lots.mode().iloc[0])

        gross = (entry_credit - exit_credit) * lot
        turnover = (entry_credit + exit_credit) * lot
        costs = turnover * friction/100
        net = gross - costs
        margin = max(S * lot * margin_pct/100, 1.0)
        rom = net / margin * 100
        spot_exit = float(spot_map.get(actual_exit, np.nan))
        cycles.append({
            "Symbol": symbol,
            "Entry": entry_date,
            "Exit": actual_exit,
            "Entry_Date": entry_date,
            "Exit_Date": actual_exit,
            "Expiry": expiry,
            "Put_Strike": pstrike,
            "Call_Strike": cstrike,
            "Entry_Spot": S,
            "Exit_Spot": spot_exit,
            "Sold_Prem": entry_credit * lot,
            "Exit_Prem": exit_credit * lot,
            "Gross_PnL": gross,
            "Costs": costs,
            "Net_PnL": net,
            "Margin": margin,
            "Allocated_Margin": margin,
            "ROM_%": rom,
            "Exit_Reason": exit_reason,
            "Days_Held": (actual_exit-entry_date).days,
        })
    return pd.DataFrame(cycles), "OK" if cycles else "No complete NSE option cycles"


def run_all_nifty50_nse(symbols, years, y_pct, stop_mult, friction, margin_pct, progress=None):
    rows = []
    for i, sym in enumerate(symbols, 1):
        try:
            ledger, status = nse_actual_backtest(sym, years, y_pct, stop_mult, friction, margin_pct)
            if ledger.empty:
                rows.append({"Symbol": sym, "Status": status, "Cycles": 0})
            else:
                pnl = ledger["Net_PnL"]
                wins = pnl > 0
                rows.append({
                    "Symbol": sym, "Status": "OK", "Cycles": len(ledger),
                    "Win_Rate_%": wins.mean()*100,
                    "Net_PnL": pnl.sum(),
                    "Avg_PnL": pnl.mean(),
                    "Median_PnL": pnl.median(),
                    "Profit_Factor": pnl[pnl>0].sum()/abs(pnl[pnl<0].sum()) if (pnl<0).any() else np.inf,
                    "Avg_ROM_%": ledger["ROM_%"].mean(),
                    "Max_DD": (ledger["Net_PnL"].cumsum()-ledger["Net_PnL"].cumsum().cummax()).min(),
                    "Stop_Rate_%": (ledger["Exit_Reason"].eq("Stop Loss").mean()*100),
                })
        except Exception as e:
            rows.append({"Symbol": sym, "Status": f"ERROR: {str(e)[:120]}", "Cycles": 0})
        if progress:
            progress(i, len(symbols), sym)
    out = pd.DataFrame(rows)
    if not out.empty and "Win_Rate_%" in out:
        out["Win_Rate_Rank"] = out["Win_Rate_%"].rank(method="min", ascending=False).astype("Int64")
        out = out.sort_values(["Status", "Win_Rate_%"], ascending=[True, False])
    return out


# ============================================================
# THEORETICAL BACKTEST
# ============================================================

def theoretical_backtest(
    spot,
    y_pct,
    step,
    lot,
    rf,
    iv_markup,
    stop_mult,
    friction,
    margin_pct,
    years,
    vol_window=30,
):
    df = spot.copy()
    df["ret"] = np.log(df["Close"]/df["Close"].shift(1))
    df["rv"] = df["ret"].rolling(vol_window).std()*np.sqrt(252)
    df["rv"] = df["rv"].bfill().clip(lower=.12)

    dates = sorted(df.index)
    boundary = dt.date.today() - dt.timedelta(days=years*365)
    dates = [d for d in dates if d >= boundary]

    ledger = []

    if len(dates) < vol_window + 10:
        return pd.DataFrame()

    y, m = dates[0].year, dates[0].month

    while True:
        target_entry = dt.date(y, m, 15)
        entries = [d for d in dates if d >= target_entry]
        if not entries:
            break

        entry = entries[0]

        nm = 1 if m == 12 else m + 1
        ny = y + 1 if m == 12 else y

        target_exit = dt.date(ny, nm, 15)
        exits = [d for d in dates if d >= target_exit]
        if not exits:
            break

        scheduled_exit = exits[0]
        expiry = third_tuesday_or_previous_trading_day(
            ny, nm, dates
        )

        if expiry is None:
            break

        S = float(df.loc[entry, "Close"])
        sigma = float(df.loc[entry, "rv"])*iv_markup

        kp = round((S*(1-y_pct/100))/step)*step
        kc = round((S*(1+y_pct/100))/step)*step

        T = max((expiry-entry).days, 1)/365

        p0 = bs_price(S,kp,T,rf,sigma,"P")
        c0 = bs_price(S,kc,T,rf,sigma,"C")
        credit = p0+c0

        actual_exit = scheduled_exit
        exit_reason = "Scheduled"
        S_exit = float(df.loc[scheduled_exit,"Close"])
        exit_value = None

        for d in [x for x in dates if entry < x <= scheduled_exit]:
            Si = float(df.loc[d,"Close"])
            sigi = float(df.loc[d,"rv"])*iv_markup
            Ti = max((expiry-d).days, 0)/365

            pv = bs_price(Si,kp,Ti,rf,sigi,"P")
            cv = bs_price(Si,kc,Ti,rf,sigi,"C")
            value = pv+cv

            if stop_mult > 0 and value >= credit*stop_mult:
                actual_exit = d
                S_exit = Si
                exit_value = value
                exit_reason = f"Stop {stop_mult:.1f}x"
                break

        if exit_value is None:
            sig = float(df.loc[scheduled_exit,"rv"])*iv_markup
            T2 = max((expiry-scheduled_exit).days,0)/365
            exit_value = (
                bs_price(S_exit,kp,T2,rf,sig,"P")
                + bs_price(S_exit,kc,T2,rf,sig,"C")
            )

        gross = (credit-exit_value)*lot
        costs = (credit+exit_value)*lot*friction/100
        net = gross-costs
        margin = S*lot*margin_pct/100
        rom = net/margin*100 if margin else np.nan

        ledger.append({
            "Entry_Date": entry,
            "Exit_Date": actual_exit,
            "Expiry_Date": expiry,
            "Spot_Entry": S,
            "Spot_Exit": S_exit,
            "Spot_Move_%": (S_exit/S-1)*100,
            "Put_Strike": kp,
            "Call_Strike": kc,
            "Sold_Prem": credit,
            "Exit_Prem": exit_value,
            "Gross_PnL": gross,
            "Transaction_Cost": costs,
            "Net_PnL": net,
            "Allocated_Margin": margin,
            "ROM_%": rom,
            "Days_Held": (actual_exit-entry).days,
            "Exit_Reason": exit_reason,
        })

        y, m = ny, nm

    return pd.DataFrame(ledger)



# ============================================================
# ALL-NIFTY-50 BATCH ENGINE
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def batch_spot_data(symbols, years):
    out = {}
    for sym in symbols:
        try:
            df = yahoo_data(sym, years)
            if not df.empty:
                out[sym] = df
        except Exception:
            out[sym] = pd.DataFrame()
    return out


def run_all_nifty50(
    symbols,
    years,
    y_pct,
    rf,
    iv_markup,
    stop_mult,
    friction,
    margin_pct,
):
    """
    Cross-sectional comparison.

    Lot is deliberately normalized to 1 for this comparison:
    win rate is lot-independent, and ROM remains comparable because
    both P&L and modeled margin scale linearly with lot size.
    """
    rows = []

    for sym in symbols:
        try:
            spot_i = yahoo_data(sym, years)

            if spot_i.empty:
                rows.append({
                    "Symbol": sym,
                    "Company": FNO[sym]["name"],
                    "Status": "No data",
                })
                continue

            test = theoretical_backtest(
                spot=spot_i,
                y_pct=y_pct,
                step=FNO[sym]["step"],
                lot=1,
                rf=rf,
                iv_markup=iv_markup,
                stop_mult=stop_mult,
                friction=friction,
                margin_pct=margin_pct,
                years=years,
            )

            if test.empty:
                rows.append({
                    "Symbol": sym,
                    "Company": FNO[sym]["name"],
                    "Status": "No cycles",
                })
                continue

            pnl = test["Net_PnL"]
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]

            equity = pnl.cumsum()
            peak = equity.cummax()
            dd = equity - peak

            cycle_rom = test["ROM_%"]
            cycle_ret = (
                test["Net_PnL"]
                / test["Allocated_Margin"]
            )

            sharpe = (
                cycle_ret.mean()
                / cycle_ret.std()
                * np.sqrt(12)
                if cycle_ret.std() > 0
                else np.nan
            )

            downside = cycle_ret[cycle_ret < 0]
            sortino = (
                cycle_ret.mean()
                / downside.std()
                * np.sqrt(12)
                if len(downside) > 1 and downside.std() > 0
                else np.nan
            )

            total_days = max(
                (
                    pd.to_datetime(test["Exit_Date"]).max()
                    - pd.to_datetime(test["Entry_Date"]).min()
                ).days,
                1,
            )

            years_actual = total_days / 365.25
            capital_proxy = test["Allocated_Margin"].mean()
            normalized_growth = pnl.sum() / max(capital_proxy, 1)

            annualized_rom = (
                ((1 + normalized_growth) ** (1 / years_actual) - 1) * 100
                if normalized_growth > -1
                else -100
            )

            rows.append({
                "Symbol": sym,
                "Company": FNO[sym]["name"],
                "Status": "OK",
                "Cycles": len(test),
                "Win_Rate_%": (pnl > 0).mean() * 100,
                "Net_PnL_Normalized": pnl.sum(),
                "Avg_ROM_%": cycle_rom.mean(),
                "Median_ROM_%": cycle_rom.median(),
                "Annualized_ROM_%": annualized_rom,
                "Profit_Factor": (
                    wins.sum() / abs(losses.sum())
                    if len(losses) else np.nan
                ),
                "Sharpe": sharpe,
                "Sortino": sortino,
                "Max_Drawdown_Normalized": abs(dd.min()),
                "Best_Trade_Normalized": pnl.max(),
                "Worst_Trade_Normalized": pnl.min(),
                "Stop_Rate_%": (
                    test["Exit_Reason"]
                    .str.contains("Stop", na=False)
                    .mean() * 100
                ),
                "Avg_Days_Held": test["Days_Held"].mean(),
            })

        except Exception as exc:
            rows.append({
                "Symbol": sym,
                "Company": FNO[sym]["name"],
                "Status": f"Error: {type(exc).__name__}",
            })

    result = pd.DataFrame(rows)

    if not result.empty and "Win_Rate_%" in result.columns:
        result = result.sort_values(
            "Win_Rate_%",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

        result.insert(
            0,
            "Win_Rate_Rank",
            np.arange(1, len(result) + 1),
        )

    return result


# ============================================================
# METRICS
# ============================================================

def performance_metrics(ledger):
    if ledger.empty:
        return {}

    pnl = ledger["Net_PnL"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    equity = pnl.cumsum()
    peak = equity.cummax()
    dd = equity-peak

    total_days = max(
        (pd.to_datetime(ledger["Exit_Date"]).max()
         - pd.to_datetime(ledger["Entry_Date"]).min()).days,
        1
    )

    years = total_days/365.25
    total_return = equity.iloc[-1]
    avg_margin = ledger["Allocated_Margin"].mean()

    annualized_rom = (
        ((1 + total_return/max(avg_margin,1)) ** (1/years) - 1)*100
        if total_return/max(avg_margin,1) > -1
        else -100
    )

    # Sharpe using cycle returns.
    cycle_returns = ledger["Net_PnL"]/ledger["Allocated_Margin"]
    sharpe = (
        cycle_returns.mean()/cycle_returns.std()*np.sqrt(12)
        if cycle_returns.std() > 0 else np.nan
    )

    downside = cycle_returns[cycle_returns < 0]
    sortino = (
        cycle_returns.mean()/downside.std()*np.sqrt(12)
        if len(downside) and downside.std() > 0 else np.nan
    )

    return {
        "cycles": len(ledger),
        "net_pnl": pnl.sum(),
        "win_rate": (pnl > 0).mean()*100,
        "profit_factor": wins.sum()/abs(losses.sum()) if len(losses) else np.nan,
        "max_dd": abs(dd.min()),
        "avg_rom": ledger["ROM_%"].mean(),
        "median_rom": ledger["ROM_%"].median(),
        "annualized_rom": annualized_rom,
        "sharpe": sharpe,
        "sortino": sortino,
        "best": pnl.max(),
        "worst": pnl.min(),
        "avg_trade": pnl.mean(),
        "stop_rate": ledger["Exit_Reason"].str.contains("Stop",na=False).mean()*100,
        "avg_margin": avg_margin,
        "days": total_days,
    }



# ============================================================
# MONTE CARLO
# ============================================================

def monte_carlo(ledger, simulations=3000, cycles=None):
    if ledger.empty:
        return pd.DataFrame()

    returns = ledger["Net_PnL"].values
    n = cycles or len(returns)

    rng = np.random.default_rng(42)
    sampled = rng.choice(
        returns,
        size=(simulations,n),
        replace=True
    )

    paths = sampled.cumsum(axis=1)
    terminal = paths[:,-1]
    minimum = paths.min(axis=1)

    return pd.DataFrame({
        "Terminal_PnL": terminal,
        "Minimum_Equity": minimum,
    })


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="dashboard">
  <div>
    <div class="title">◈ NSE SINGLE-STOCK STRANGLE QUANT TERMINAL</div>
    <div class="subtitle">
      NSE-direct contract-wise historical prices · risk analytics ·
      sensitivity · Monte Carlo
    </div>
  </div>
  <div class="status">● READY</div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## ◈ Engine Controls")

symbol = st.sidebar.selectbox(
    "Underlying",
    list(FNO),
    format_func=lambda x: f"{x} — {FNO[x]['name']}",
)

spec = FNO[symbol]

data_mode = "NSE DIRECT"

years = st.sidebar.slider(
    "Lookback Years", 1, 5, 3
)

y_pct = st.sidebar.slider(
    "OTM Distance (%)", 1.0, 15.0, 5.0, .5
)

margin_pct = st.sidebar.slider(
    "Margin / Notional (%)",
    10., 35., 22., .5
)

stop_mult = st.sidebar.slider(
    "Stop Loss (× Credit)",
    0., 5., 2.5, .5
)

friction = st.sidebar.slider(
    "Turnover Friction (%)",
    0., 3., 1.2, .1
)

st.sidebar.divider()

st.sidebar.caption(
    "The engine uses NSE contract metadata and observed historical option prices. No file upload is required.\n\nCurrent NSE contract specifications use Tuesday expiry for "
    "individual-security options, adjusted to the previous trading day "
    "when Tuesday is a holiday."
)


# ============================================================
# DATA — NSE DIRECT ONLY
# ============================================================

mode_label = "NSE DIRECT — CONTRACT-WISE HISTORICAL"

with st.spinner("Connecting to NSE and loading historical cash-market data..."):
    ledger, data_status = nse_actual_backtest(
        symbol=symbol,
        years=years,
        y_pct=y_pct,
        stop_mult=stop_mult,
        friction=friction,
        margin_pct=margin_pct,
    )

if not ledger.empty:
    _spot_start = pd.to_datetime(ledger["Entry_Date"]).min().date()
    _spot_end = pd.to_datetime(ledger["Exit_Date"]).max().date()
    spot = nse_equity_history(symbol, _spot_start, _spot_end).copy()
    if not spot.empty:
        spot["Date"] = pd.to_datetime(spot["Date"])
        spot = spot.set_index("Date")

if ledger.empty:
    st.error(
        f"NSE did not return enough complete contract history for {symbol}. "
        f"Status: {data_status}. NSE may limit historical F&O API retention; "
        "try a shorter lookback. No synthetic option prices are substituted."
    )
    st.stop()

metrics = performance_metrics(ledger)

# ============================================================
# EQUITY FIELDS
# ============================================================

ledger["Cumulative_PnL"] = ledger["Net_PnL"].cumsum()
ledger["Peak"] = ledger["Cumulative_PnL"].cummax()
ledger["Drawdown"] = ledger["Cumulative_PnL"]-ledger["Peak"]

ledger["Premium_Capture_%"] = np.where(
    ledger["Sold_Prem"] > 0,
    (ledger["Sold_Prem"]-ledger["Exit_Prem"])
    / ledger["Sold_Prem"]*100,
    np.nan,
)


# ============================================================
# STATUS
# ============================================================

st.markdown(
    f"""
<div style="display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 12px;">
  <div class="mode">{mode_label}</div>
  <div class="status">{symbol} · {len(ledger)} CYCLES</div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# KPI
# ============================================================

def kpi(label, value, sub="", cls=""):
    return f"""
<div class="kpi">
 <div class="kpi-label">{label}</div>
 <div class="kpi-value {cls}">{value}</div>
 <div class="kpi-sub">{sub}</div>
</div>
"""


pnl_cls = "kpi-positive" if metrics["net_pnl"] >= 0 else "kpi-negative"

st.markdown(
    f"""
<div class="kpis">
{kpi("Net P&L",f"₹{metrics['net_pnl']:,.0f}",
     f"{metrics['cycles']} cycles",pnl_cls)}
{kpi("Win Rate",f"{metrics['win_rate']:.1f}%",
     f"{metrics['cycles']} total")}
{kpi("Annualized ROM",f"{metrics['annualized_rom']:+.1f}%",
     "model annualization")}
{kpi("Sharpe",f"{metrics['sharpe']:.2f}",
     "monthly cycle normalization")}
{kpi("Sortino",f"{metrics['sortino']:.2f}",
     "downside-adjusted")}
{kpi("Max Drawdown",f"₹{metrics['max_dd']:,.0f}",
     f"Avg margin ₹{metrics['avg_margin']:,.0f}","kpi-negative")}
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

overview, risk, sensitivity, universe_tab, monte, ledger_tab = st.tabs([
    "◈ Overview",
    "⚠ Risk",
    "◇ Sensitivity",
    "◆ All Nifty 50",
    "∿ Monte Carlo",
    "☷ Audit",
])


# ============================================================
# CHART HELPER
# ============================================================

def layout(fig, height=390):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT),
        margin=dict(l=45,r=20,t=45,b=40),
        height=height,
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


# ============================================================
# OVERVIEW
# ============================================================

with overview:

    st.markdown('<div class="section">Equity Curve & Drawdown</div>',
                unsafe_allow_html=True)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[.72,.28],
        vertical_spacing=.05,
    )

    fig.add_trace(
        go.Scatter(
            x=ledger["Exit_Date"],
            y=ledger["Cumulative_PnL"],
            mode="lines+markers",
            name="Equity",
            line=dict(color=GREEN,width=2.5),
            marker=dict(size=4),
        ),
        row=1,col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ledger["Exit_Date"],
            y=ledger["Drawdown"],
            mode="lines",
            name="Drawdown",
            line=dict(color=RED,width=1.5),
            fill="tozeroy",
        ),
        row=2,col=1,
    )

    layout(fig,500)
    st.plotly_chart(fig,use_container_width=True,
                    config={"displayModeBar":False})

    c1,c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section">Underlying Price</div>',
                    unsafe_allow_html=True)

        figp = go.Figure()

        figp.add_trace(
            go.Scatter(
                x=list(spot.index),
                y=spot["Close"],
                mode="lines",
                name="Spot",
                line=dict(color=BLUE,width=1.7),
            )
        )

        figp.add_trace(
            go.Scatter(
                x=ledger["Entry_Date"],
                y=ledger["Entry_Spot"],
                mode="markers",
                name="Entry",
                marker=dict(
                    color=GREEN,
                    size=8,
                    symbol="triangle-up",
                ),
            )
        )

        figp.add_trace(
            go.Scatter(
                x=ledger["Exit_Date"],
                y=ledger["Exit_Spot"],
                mode="markers",
                name="Exit",
                marker=dict(
                    color=RED,
                    size=8,
                    symbol="triangle-down",
                ),
            )
        )

        layout(figp,360)
        st.plotly_chart(figp,use_container_width=True,
                        config={"displayModeBar":False})

    with c2:
        st.markdown('<div class="section">Cycle ROM</div>',
                    unsafe_allow_html=True)

        colors = [
            GREEN if x >= 0 else RED
            for x in ledger["ROM_%"]
        ]

        figr = go.Figure(
            go.Bar(
                x=ledger["Exit_Date"],
                y=ledger["ROM_%"],
                marker_color=colors,
                hovertemplate="ROM %{y:.2f}%<extra></extra>",
            )
        )

        layout(figr,360)
        figr.update_yaxes(title="ROM %")
        st.plotly_chart(figr,use_container_width=True,
                        config={"displayModeBar":False})


# ============================================================
# RISK
# ============================================================

with risk:

    r1,r2,r3,r4 = st.columns(4)

    for col,label,value in [
        (r1,"PROFIT FACTOR",
         "N/A" if np.isnan(metrics["profit_factor"])
         else f"{metrics['profit_factor']:.2f}"),
        (r2,"BEST CYCLE",f"₹{metrics['best']:,.0f}"),
        (r3,"WORST CYCLE",f"₹{metrics['worst']:,.0f}"),
        (r4,"STOP RATE",f"{metrics['stop_rate']:.1f}%"),
    ]:
        with col:
            st.markdown(
                f"""
<div class="card">
 <div class="small">{label}</div>
 <div class="big">{value}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    c1,c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section">ROM Distribution</div>',
                    unsafe_allow_html=True)

        f = go.Figure(
            go.Histogram(
                x=ledger["ROM_%"],
                nbinsx=25,
                name="ROM",
            )
        )
        f.add_vline(x=0,line_dash="dash")
        f.add_vline(
            x=metrics["avg_rom"],
            line_dash="dot",
            annotation_text=f"Mean {metrics['avg_rom']:.2f}%",
        )
        layout(f,370)
        f.update_xaxes(title="ROM %")
        f.update_yaxes(title="Cycles")
        st.plotly_chart(f,use_container_width=True,
                        config={"displayModeBar":False})

    with c2:
        st.markdown('<div class="section">Spot Move vs P&L</div>',
                    unsafe_allow_html=True)

        f = go.Figure(
            go.Scatter(
                x=ledger["Spot_Move_%"],
                y=ledger["Net_PnL"],
                mode="markers",
                marker=dict(size=8),
                text=ledger["Exit_Reason"],
                hovertemplate=(
                    "Spot: %{x:.2f}%<br>"
                    "P&L: ₹%{y:,.0f}<br>"
                    "%{text}<extra></extra>"
                ),
            )
        )
        f.add_hline(y=0,line_dash="dash")
        f.add_vline(x=0,line_dash="dash")
        layout(f,370)
        f.update_xaxes(title="Spot Move %")
        f.update_yaxes(title="Net P&L ₹")
        st.plotly_chart(f,use_container_width=True,
                        config={"displayModeBar":False})

    st.markdown('<div class="section">Monthly P&L Heatmap</div>',
                unsafe_allow_html=True)

    tmp = ledger.copy()
    tmp["Year"] = pd.to_datetime(tmp["Exit_Date"]).dt.year
    tmp["Month"] = pd.to_datetime(tmp["Exit_Date"]).dt.month

    m = tmp.groupby(["Year","Month"])["Net_PnL"].sum().unstack()
    m = m.reindex(columns=range(1,13))

    f = go.Figure(
        go.Heatmap(
            z=m.values,
            x=["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"],
            y=m.index,
            colorscale=[
                [0,RED],
                [.5,"#202733"],
                [1,GREEN],
            ],
            hovertemplate=(
                "Year %{y}<br>"
                "Month %{x}<br>"
                "P&L ₹%{z:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    layout(f,320)
    st.plotly_chart(f,use_container_width=True,
                    config={"displayModeBar":False})


# ============================================================
# SENSITIVITY — NSE DIRECT
# ============================================================

with sensitivity:

    st.markdown(
        '<div class="section">NSE Contract-Priced Sensitivity</div>',
        unsafe_allow_html=True
    )
    st.caption(
        "Every cell reruns the selected stock against NSE historical option "
        "contracts. There are no model option prices in this matrix."
    )

    y_grid = [3, 5, 7]
    stop_grid = [2.0, 2.5, 3.0]
    run_sens = st.button("▶ Run 3×3 NSE Sensitivity", use_container_width=True)
    if run_sens or "nse_sensitivity" in st.session_state:
        if run_sens:
            matrix = []
            with st.spinner("Fetching NSE contracts for sensitivity grid..."):
                for yy in y_grid:
                    row = []
                    for ss in stop_grid:
                        test, _ = nse_actual_backtest(
                            symbol, years, yy, ss, friction, margin_pct
                        )
                        row.append(test["Net_PnL"].sum() if not test.empty else np.nan)
                    matrix.append(row)
            st.session_state["nse_sensitivity"] = matrix
        matrix = st.session_state["nse_sensitivity"]
        f = go.Figure(go.Heatmap(
            z=matrix, x=stop_grid, y=y_grid,
            colorscale=[[0, RED], [.5, "#202733"], [1, GREEN]],
            colorbar=dict(title="Net P&L ₹"),
            hovertemplate="OTM %{y:.1f}%<br>Stop %{x:.1f}x<br>Net P&L ₹%{z:,.0f}<extra></extra>"
        ))
        layout(f, 420)
        f.update_xaxes(title="Stop-Loss Multiple")
        f.update_yaxes(title="OTM Distance")
        st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})

# ============================================================
# ALL NIFTY 50 COMPARISON
# ============================================================

with universe_tab:

    st.markdown(
        '<div class="section">Nifty 50 Win-Rate Comparison</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Cross-sectional comparison uses the same strategy parameters for "
        "all constituents. Results are normalized to one unit of notional, "
        "so win rate is directly comparable across stocks. This is the "
        "actual NSE contract-wise historical option prices. No upload, Yahoo "
        "prices or Black-Scholes option estimates are used."
    )

    run_batch = st.button(
        "▶ Run All 50 Stocks",
        type="primary",
        use_container_width=True,
    )

    if run_batch or "nifty50_comparison" in st.session_state:

        if run_batch:
            with st.spinner(
                "Running the strategy across all 50 Nifty constituents..."
            ):
                st.session_state["nifty50_comparison"] = run_all_nifty50_nse(
                    symbols=list(FNO.keys()),
                    years=years,
                    y_pct=y_pct,
                    stop_mult=stop_mult,
                    friction=friction,
                    margin_pct=margin_pct,
                )

        comp = st.session_state["nifty50_comparison"].copy()

        ok = comp[
            comp["Status"].eq("OK")
        ].copy()

        if ok.empty:
            st.error(
                "No constituent completed successfully. "
                "Check NSE connectivity, NSE rate limits, and the selected lookback window."
            )
        else:

            c1,c2,c3,c4 = st.columns(4)

            with c1:
                st.markdown(
                    f"""
<div class="card">
 <div class="small">STOCKS COMPLETED</div>
 <div class="big">{len(ok)} / 50</div>
</div>
""",
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    f"""
<div class="card">
 <div class="small">AVERAGE WIN RATE</div>
 <div class="big">{ok["Win_Rate_%"].mean():.1f}%</div>
</div>
""",
                    unsafe_allow_html=True,
                )

            with c3:
                st.markdown(
                    f"""
<div class="card">
 <div class="small">MEDIAN WIN RATE</div>
 <div class="big">{ok["Win_Rate_%"].median():.1f}%</div>
</div>
""",
                    unsafe_allow_html=True,
                )

            with c4:
                st.markdown(
                    f"""
<div class="card">
 <div class="small">STOCKS &gt; 60% WIN RATE</div>
 <div class="big">{(ok["Win_Rate_%"] > 60).sum()}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

            # Bar chart — all available constituents.
            chart_df = ok.sort_values(
                "Win_Rate_%",
                ascending=True,
            )

            fig = go.Figure(
                go.Bar(
                    x=chart_df["Win_Rate_%"],
                    y=chart_df["Symbol"],
                    orientation="h",
                    text=chart_df["Win_Rate_%"].map(
                        lambda x: f"{x:.1f}%"
                    ),
                    textposition="outside",
                    hovertemplate=(
                        "%{y}<br>"
                        "Win Rate: %{x:.1f}%"
                        "<extra></extra>"
                    ),
                )
            )

            layout(fig, max(720, len(chart_df)*24))
            fig.update_xaxes(
                title="Win Rate %",
                range=[0,100],
            )
            fig.update_yaxes(
                title="Constituent",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

            # Secondary comparison: win rate + ROM + Sharpe.
            st.markdown(
                '<div class="section">Cross-Sectional Metrics</div>',
                unsafe_allow_html=True,
            )

            shown = comp[
                [
                    "Win_Rate_Rank",
                    "Symbol",
                    "Company",
                    "Cycles",
                    "Win_Rate_%",
                    "Avg_ROM_%",
                    "Median_ROM_%",
                    "Annualized_ROM_%",
                    "Profit_Factor",
                    "Sharpe",
                    "Sortino",
                    "Max_Drawdown_Normalized",
                    "Stop_Rate_%",
                    "Status",
                ]
            ].copy()

            st.dataframe(
                shown,
                use_container_width=True,
                height=650,
                column_config={
                    "Win_Rate_Rank":
                        st.column_config.NumberColumn(
                            "Win Rank",
                            format="%d",
                        ),
                    "Win_Rate_%":
                        st.column_config.NumberColumn(
                            "Win Rate",
                            format="%.1f%%",
                        ),
                    "Avg_ROM_%":
                        st.column_config.NumberColumn(
                            "Avg ROM",
                            format="%.2f%%",
                        ),
                    "Median_ROM_%":
                        st.column_config.NumberColumn(
                            "Median ROM",
                            format="%.2f%%",
                        ),
                    "Annualized_ROM_%":
                        st.column_config.NumberColumn(
                            "Annualized ROM",
                            format="%.1f%%",
                        ),
                    "Profit_Factor":
                        st.column_config.NumberColumn(
                            "Profit Factor",
                            format="%.2f",
                        ),
                    "Sharpe":
                        st.column_config.NumberColumn(
                            "Sharpe",
                            format="%.2f",
                        ),
                    "Sortino":
                        st.column_config.NumberColumn(
                            "Sortino",
                            format="%.2f",
                        ),
                    "Max_Drawdown_Normalized":
                        st.column_config.NumberColumn(
                            "Max DD / normalized ₹",
                            format="₹%.2f",
                        ),
                    "Stop_Rate_%":
                        st.column_config.NumberColumn(
                            "Stop Rate",
                            format="%.1f%%",
                        ),
                },
            )

            csv = comp.to_csv(index=False).encode("utf-8")

            st.download_button(
                "↓ Download All-50 Comparison CSV",
                csv,
                file_name="nifty50_strangle_comparison.csv",
                mime="text/csv",
            )

            st.info(
                "The comparison is descriptive, not a recommendation. "
                "A higher win rate alone does not imply a better strategy: "
                "loss magnitude, drawdown, premium capture, transaction costs "
                "and tail-risk behavior also matter."
            )

# ============================================================
# MONTE CARLO
# ============================================================

with monte:

    simulations = st.slider(
        "Simulations",
        500,
        10000,
        3000,
        500,
    )

    mc = monte_carlo(
        ledger,
        simulations=simulations
    )

    if not mc.empty:

        q5,q50,q95 = np.percentile(
            mc["Terminal_PnL"],
            [5,50,95]
        )

        c1,c2,c3 = st.columns(3)

        for col,label,value in [
            (c1,"5th PERCENTILE",f"₹{q5:,.0f}"),
            (c2,"MEDIAN",f"₹{q50:,.0f}"),
            (c3,"95th PERCENTILE",f"₹{q95:,.0f}"),
        ]:
            with col:
                st.markdown(
                    f"""
<div class="card">
 <div class="small">{label}</div>
 <div class="big">{value}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

        f = go.Figure()

        f.add_trace(
            go.Histogram(
                x=mc["Terminal_PnL"],
                nbinsx=60,
                name="Terminal P&L",
            )
        )

        f.add_vline(x=0,line_dash="dash")

        layout(f,390)
        f.update_xaxes(title="Simulated Terminal P&L ₹")
        f.update_yaxes(title="Simulations")

        st.plotly_chart(
            f,
            use_container_width=True,
            config={"displayModeBar":False},
        )

        f2 = go.Figure()

        sample_n = min(100, len(mc))
        rng = np.random.default_rng(123)
        sampled_indices = rng.choice(
            len(mc),
            size=sample_n,
            replace=False
        )

        cycle_pnl = ledger["Net_PnL"].values

        for i in sampled_indices:
            path = rng.choice(
                cycle_pnl,
                size=len(cycle_pnl),
                replace=True
            ).cumsum()

            f2.add_trace(
                go.Scatter(
                    y=path,
                    mode="lines",
                    line=dict(
                        width=.8,
                        color="rgba(76,154,255,.16)"
                    ),
                    showlegend=False,
                )
            )

        layout(f2,400)
        f2.update_xaxes(title="Cycle")
        f2.update_yaxes(title="Simulated Equity ₹")

        st.plotly_chart(
            f2,
            use_container_width=True,
            config={"displayModeBar":False},
        )


# ============================================================
# AUDIT
# ============================================================

with ledger_tab:

    st.markdown(
        '<div class="section">Trade Audit Ledger</div>',
        unsafe_allow_html=True
    )

    display = ledger.copy()

    st.dataframe(
        display,
        use_container_width=True,
        height=600,
        column_config={
            "Spot_Entry":
                st.column_config.NumberColumn(
                    "Entry Spot",format="₹%.2f"),
            "Spot_Exit":
                st.column_config.NumberColumn(
                    "Exit Spot",format="₹%.2f"),
            "Spot_Move_%":
                st.column_config.NumberColumn(
                    "Spot Move",format="%.2f%%"),
            "Put_Strike":
                st.column_config.NumberColumn(
                    "Put",format="₹%.0f"),
            "Call_Strike":
                st.column_config.NumberColumn(
                    "Call",format="₹%.0f"),
            "Sold_Prem":
                st.column_config.NumberColumn(
                    "Sold Prem",format="₹%.2f"),
            "Exit_Prem":
                st.column_config.NumberColumn(
                    "Exit Prem",format="₹%.2f"),
            "Net_PnL":
                st.column_config.NumberColumn(
                    "Net P&L",format="₹%.0f"),
            "ROM_%":
                st.column_config.NumberColumn(
                    "ROM",format="%.2f%%"),
            "Premium_Capture_%":
                st.column_config.NumberColumn(
                    "Premium Capture",format="%.1f%%"),
        }
    )

    csv = ledger.to_csv(index=False).encode()

    st.download_button(
        "↓ Download Backtest CSV",
        csv,
        file_name=f"{symbol}_strangle_v2.csv",
        mime="text/csv",
    )


# ============================================================
# MODEL DISCLOSURE
# ============================================================

st.divider()

with st.expander("Data, Pricing & Methodology"):

    st.markdown(
        """
### NSE direct mode

The dashboard now uses NSE directly and requires no upload.

1. NSE historical equity API supplies the underlying close used on entry.
2. NSE contract metadata supplies current permitted lot size and stock-option strike scheme.
3. NSE historical F&O contract API supplies the actual OPTSTK CE/PE contract OHLC/close for the selected expiry and strike.
4. The engine selects the nearest NSE strike to the configured OTM target and tries nearby official strike intervals if the historical strike scheme has changed.
5. Stop-loss checks use the observed daily closing premium of the two short contracts.
6. No Black-Scholes price is substituted when NSE data is missing.

NSE states that individual-security options expire on the last Tuesday of the expiry month, adjusted to the previous trading day if Tuesday is a holiday, and that stock-option strike intervals are reviewed periodically based on underlying volatility.

### Risk analytics

The dashboard includes:

- Win rate
- Profit factor
- Maximum drawdown
- Average and median ROM
- Annualized ROM
- Sharpe ratio
- Sortino ratio
- Stop-loss frequency
- P&L distribution
- Monthly P&L
- Parameter sensitivity
- Bootstrap Monte Carlo

### Important

Parameter sensitivity and Monte Carlo results are diagnostic tools.
They do not establish future profitability or eliminate model risk.
"""
    )
