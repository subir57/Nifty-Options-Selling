"""
NSE Single-Stock Short Strangle Backtester — Institutional V3 Quant Terminal
=============================================================================

Run:
    pip install streamlit pandas numpy scipy plotly yfinance
    streamlit run nse_strangle_v3_ui.py

V3 focuses on presentation/UI while preserving the theoretical backtest engine.
Historical NSE CSV is parsed and displayed/validated, but exact historical option
contract execution is NOT claimed unless a contract-selection/pricing layer is
implemented. Theoretical mode uses Yahoo spot + Black-Scholes + realized vol.

Important:
- The synthetic expiry helper uses the last Tuesday of the expiry month and
  rolls backward to the last available trading date. For historical execution
  research, use actual expiry dates from NSE contract data.
- Sharpe/Sortino are cycle-based monthly statistics.
- Monte Carlo is bootstrap resampling of historical cycle P&L.
"""

import calendar
import datetime as dt
import io
import math

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import yfinance as yf


# ============================================================
# PAGE / THEME
# ============================================================

st.set_page_config(
    page_title="NSE Strangle Quant Terminal",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

C = {
    "bg": "#070A0F",
    "panel": "#0D121A",
    "panel2": "#101722",
    "border": "#202936",
    "text": "#E9EEF5",
    "muted": "#7F8997",
    "green": "#00D09C",
    "red": "#FF5B61",
    "blue": "#4C9AFF",
    "cyan": "#4CC9F0",
    "yellow": "#F6C85F",
    "purple": "#9B7EDE",
}

PLOT = dict(
    paper_bgcolor=C["panel"],
    plot_bgcolor=C["panel"],
    font=dict(color=C["text"], family="Inter, Arial, sans-serif"),
    margin=dict(l=45, r=18, t=42, b=42),
    hoverlabel=dict(bgcolor="#151C27", font_color=C["text"]),
)

st.markdown(f"""
<style>
.stApp {{
    background:
      radial-gradient(circle at 12% -8%, rgba(76,154,255,.12), transparent 28%),
      radial-gradient(circle at 92% 5%, rgba(155,126,222,.09), transparent 24%),
      {C["bg"]};
}}
.block-container {{
    max-width: 1700px;
    padding: .75rem 1rem 3rem;
}}
section[data-testid="stSidebar"] {{
    background: #090D13;
    border-right: 1px solid {C["border"]};
}}
section[data-testid="stSidebar"] > div {{
    padding-top: .8rem;
}}
div[data-testid="stMetric"] {{
    background: transparent;
}}
div[data-testid="stTabs"] button {{
    font-weight: 700;
}}
.topbar {{
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:16px;
    padding:18px 20px;
    margin-bottom:10px;
    border:1px solid {C["border"]};
    border-radius:16px;
    background:linear-gradient(135deg,rgba(17,24,35,.98),rgba(10,14,20,.98));
    box-shadow:0 12px 40px rgba(0,0,0,.18);
}}
.brand {{
    display:flex;
    align-items:center;
    gap:13px;
}}
.logo {{
    width:42px;height:42px;border-radius:12px;
    display:flex;align-items:center;justify-content:center;
    background:linear-gradient(135deg,#17283D,#121925);
    border:1px solid #2B405B;
    color:{C["cyan"]};font-weight:900;font-size:19px;
}}
.title {{
    font-size:1.30rem;font-weight:850;letter-spacing:-.4px;
}}
.sub {{
    color:{C["muted"]};font-size:.72rem;margin-top:3px;
}}
.badges {{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end;}}
.badge {{
    padding:6px 10px;border-radius:999px;font-size:.64rem;
    font-weight:800;letter-spacing:.4px;border:1px solid;
}}
.ready {{color:{C["green"]};background:rgba(0,208,156,.07);border-color:rgba(0,208,156,.28);}}
.theory {{color:{C["yellow"]};background:rgba(246,200,95,.07);border-color:rgba(246,200,95,.25);}}
.csv {{color:{C["cyan"]};background:rgba(76,201,240,.07);border-color:rgba(76,201,240,.25);}}
.ticker {{
    display:flex;gap:8px;overflow-x:auto;padding:2px 0 9px;
    scrollbar-width:none;
}}
.tick {{
    min-width:150px;padding:8px 11px;border:1px solid {C["border"]};
    border-radius:10px;background:#0B1017;
}}
.tick .k {{font-size:.58rem;color:{C["muted"]};font-weight:800;letter-spacing:.5px;}}
.tick .v {{font-size:.85rem;font-weight:800;margin-top:2px;}}
.tick .s {{font-size:.60rem;color:#667181;margin-top:1px;}}
.kpi-grid {{
    display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:9px;margin:5px 0 13px;
}}
.kpi {{
    min-height:95px;padding:12px 13px;border-radius:12px;
    border:1px solid {C["border"]};
    background:linear-gradient(145deg,#111823,#0C1118);
}}
.kpi-label {{font-size:.58rem;color:{C["muted"]};font-weight:800;letter-spacing:.65px;text-transform:uppercase;}}
.kpi-value {{font-size:1.17rem;font-weight:850;margin-top:8px;}}
.kpi-sub {{font-size:.61rem;color:#697483;margin-top:4px;}}
.section {{
    margin:13px 0 8px;font-size:.92rem;font-weight:850;letter-spacing:.1px;
}}
.card {{
    border:1px solid {C["border"]};border-radius:13px;padding:13px;
    background:linear-gradient(145deg,#0F151E,#0B1017);
    height:100%;
}}
.card-title {{font-size:.72rem;font-weight:850;text-transform:uppercase;letter-spacing:.6px;}}
.card-note {{font-size:.60rem;color:{C["muted"]};margin-top:3px;}}
.cockpit {{
    display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:7px;
}}
.cock {{
    padding:9px;border-radius:9px;background:#0A0F16;border:1px solid #1A222E;
}}
.cock .l {{font-size:.55rem;color:{C["muted"]};text-transform:uppercase;font-weight:800;}}
.cock .n {{font-size:.83rem;font-weight:800;margin-top:4px;}}
.risk {{
    display:flex;gap:8px;flex-wrap:wrap;margin:8px 0;
}}
.risk-chip {{
    padding:7px 10px;border-radius:8px;background:#0B1017;
    border:1px solid {C["border"]};font-size:.62rem;font-weight:800;
}}
.good {{color:{C["green"]};}}
.warn {{color:{C["yellow"]};}}
.bad {{color:{C["red"]};}}
.footer {{
    margin-top:18px;padding-top:10px;border-top:1px solid {C["border"]};
    color:#5E6978;font-size:.58rem;line-height:1.5;
}}
@media(max-width:1200px) {{
    .kpi-grid {{grid-template-columns:repeat(3,minmax(0,1fr));}}
    .cockpit {{grid-template-columns:repeat(4,minmax(0,1fr));}}
}}
@media(max-width:650px) {{
    .block-container {{padding:.45rem .4rem 2rem;}}
    .topbar {{padding:13px;align-items:flex-start;}}
    .title {{font-size:1.02rem;}}
    .sub {{font-size:.62rem;}}
    .badges {{display:none;}}
    .kpi-grid {{grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;}}
    .kpi {{min-height:82px;padding:9px;}}
    .kpi-value {{font-size:.94rem;}}
    .cockpit {{grid-template-columns:repeat(2,minmax(0,1fr));}}
}}
</style>
""", unsafe_allow_html=True)


# ============================================================
# F&O SPECIFICATIONS
# ============================================================

FNO = {
    "RELIANCE": {"name": "Reliance Industries", "lot": 250, "step": 20},
    "TCS": {"name": "Tata Consultancy Services", "lot": 175, "step": 50},
    "INFY": {"name": "Infosys", "lot": 400, "step": 20},
    "HDFCBANK": {"name": "HDFC Bank", "lot": 550, "step": 10},
    "ICICIBANK": {"name": "ICICI Bank", "lot": 700, "step": 10},
    "SBIN": {"name": "State Bank of India", "lot": 750, "step": 5},
    "BHARTIARTL": {"name": "Bharti Airtel", "lot": 475, "step": 10},
    "ITC": {"name": "ITC", "lot": 1600, "step": 5},
    "LT": {"name": "Larsen & Toubro", "lot": 150, "step": 25},
    "TATAMOTORS": {"name": "Tata Motors", "lot": 550, "step": 10},
}


# ============================================================
# QUANT ENGINE
# ============================================================

def bs_price(S, K, T, r, sigma, kind):
    if T <= 1e-8:
        return max(S-K, 0.0) if kind == "C" else max(K-S, 0.0)
    sigma = max(float(sigma), 1e-6)
    d1 = (np.log(S/K) + (r + .5*sigma*sigma)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if kind == "C":
        v = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        v = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return max(float(v), 0.0)


def last_tuesday_or_previous_trading_day(year, month, trading_dates):
    cal = calendar.Calendar().monthdatescalendar(year, month)
    tuesdays = [
        d for week in cal for d in week
        if d.month == month and d.weekday() == 1
    ]
    if not tuesdays:
        return None
    target = tuesdays[-1]
    valid = [d for d in trading_dates if d <= target]
    return valid[-1] if valid else None


@st.cache_data(ttl=3600, show_spinner=False)
def yahoo_data(symbol, years):
    end = dt.date.today()
    start = end - dt.timedelta(days=years*365 + 150)
    df = yf.download(
        f"{symbol}.NS", start=start, end=end,
        progress=False, auto_adjust=False
    )
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df = df[keep].dropna()
    df.index = pd.to_datetime(df.index).date
    return df


ALIASES = {
    "date": ["Date","Trade Date","TIMESTAMP","DATE","trade_date"],
    "symbol": ["Symbol","SYMBOL","Underlying","Underlying Symbol"],
    "expiry": ["Expiry","EXPIRY","Expiry Date","ExpiryDate"],
    "option_type": ["Option Type","OPTION TYPE","OptionType","Type","OPT_TYPE"],
    "strike": ["Strike Price","STRIKE","Strike","StrikePrice"],
    "close": ["Close Price","CLOSE","Close","Settlement Price","Settle Price","SETTLE_PRICE","Settlement"],
}


def find_column(df, aliases):
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in normalized:
            return normalized[a.lower()]
    for c in df.columns:
        s = str(c).strip().lower()
        if any(a.lower() in s for a in aliases):
            return c
    return None


def parse_nse_csv(uploaded):
    if uploaded is None:
        return pd.DataFrame()
    raw = pd.read_csv(uploaded)
    cols = {k: find_column(raw, v) for k,v in ALIASES.items()}
    required = ["date","expiry","option_type","strike","close"]
    if any(cols[k] is None for k in required):
        return pd.DataFrame()
    out = pd.DataFrame({
        "Date": pd.to_datetime(raw[cols["date"]], errors="coerce").dt.date,
        "Expiry": pd.to_datetime(raw[cols["expiry"]], errors="coerce").dt.date,
        "OptionType": raw[cols["option_type"]].astype(str).str.upper().str.strip(),
        "Strike": pd.to_numeric(raw[cols["strike"]], errors="coerce"),
        "Close": pd.to_numeric(raw[cols["close"]], errors="coerce"),
    })
    out["Symbol"] = (
        raw[cols["symbol"]].astype(str).str.upper().str.strip()
        if cols["symbol"] is not None else ""
    )
    out = out.dropna(subset=["Date","Expiry","Strike","Close"])
    out["OptionType"] = out["OptionType"].replace(
        {"CALL":"CE","PUT":"PE","C":"CE","P":"PE"}
    )
    return out[out["OptionType"].isin(["CE","PE"])].sort_values(
        ["Date","Expiry","Strike"]
    )


def theoretical_backtest(spot, y_pct, step, lot, rf, iv_markup,
                         stop_mult, friction, margin_pct, years, vol_window=30):
    df = spot.copy()
    df["ret"] = np.log(df["Close"]/df["Close"].shift(1))
    df["rv"] = df["ret"].rolling(vol_window).std()*np.sqrt(252)
    df["rv"] = df["rv"].bfill().clip(lower=.12)
    dates = sorted(df.index)
    boundary = dt.date.today() - dt.timedelta(days=years*365)
    dates = [d for d in dates if d >= boundary]
    if len(dates) < vol_window + 10:
        return pd.DataFrame()

    ledger = []
    y, m = dates[0].year, dates[0].month

    while True:
        target_entry = dt.date(y,m,15)
        entries = [d for d in dates if d >= target_entry]
        if not entries: break
        entry = entries[0]

        nm = 1 if m == 12 else m+1
        ny = y+1 if m == 12 else y
        target_exit = dt.date(ny,nm,15)
        exits = [d for d in dates if d >= target_exit]
        if not exits: break
        scheduled_exit = exits[0]
        expiry = last_tuesday_or_previous_trading_day(ny,nm,dates)
        if expiry is None: break

        S = float(df.loc[entry,"Close"])
        sigma = float(df.loc[entry,"rv"])*iv_markup
        kp = round((S*(1-y_pct/100))/step)*step
        kc = round((S*(1+y_pct/100))/step)*step
        T = max((expiry-entry).days,1)/365
        p0 = bs_price(S,kp,T,rf,sigma,"P")
        c0 = bs_price(S,kc,T,rf,sigma,"C")
        credit = p0+c0

        actual_exit = scheduled_exit
        S_exit = float(df.loc[scheduled_exit,"Close"])
        exit_reason = "Scheduled"
        exit_value = None

        for d in [x for x in dates if entry < x <= scheduled_exit]:
            Si = float(df.loc[d,"Close"])
            sigi = float(df.loc[d,"rv"])*iv_markup
            Ti = max((expiry-d).days,0)/365
            value = bs_price(Si,kp,Ti,rf,sigi,"P") + bs_price(Si,kc,Ti,rf,sigi,"C")
            if stop_mult > 0 and value >= credit*stop_mult:
                actual_exit, S_exit, exit_value = d, Si, value
                exit_reason = f"Stop {stop_mult:.1f}x"
                break

        if exit_value is None:
            sig = float(df.loc[scheduled_exit,"rv"])*iv_markup
            T2 = max((expiry-scheduled_exit).days,0)/365
            exit_value = bs_price(S_exit,kp,T2,rf,sig,"P") + bs_price(S_exit,kc,T2,rf,sig,"C")

        gross = (credit-exit_value)*lot
        costs = (credit+exit_value)*lot*friction/100
        net = gross-costs
        margin = S*lot*margin_pct/100

        ledger.append({
            "Entry_Date":entry,"Exit_Date":actual_exit,"Expiry_Date":expiry,
            "Spot_Entry":S,"Spot_Exit":S_exit,
            "Spot_Move_%":(S_exit/S-1)*100,
            "Put_Strike":kp,"Call_Strike":kc,
            "Sold_Prem":credit,"Exit_Prem":exit_value,
            "Gross_PnL":gross,"Transaction_Cost":costs,"Net_PnL":net,
            "Allocated_Margin":margin,
            "ROM_%":net/margin*100 if margin else np.nan,
            "Days_Held":(actual_exit-entry).days,
            "Exit_Reason":exit_reason
        })
        y,m = ny,nm

    return pd.DataFrame(ledger)


def performance_metrics(ledger):
    pnl = ledger["Net_PnL"]
    equity = pnl.cumsum()
    peak = equity.cummax()
    dd = equity-peak
    wins, losses = pnl[pnl>0], pnl[pnl<0]
    total_days = max(
        (pd.to_datetime(ledger["Exit_Date"]).max() -
         pd.to_datetime(ledger["Entry_Date"]).min()).days, 1
    )
    years = total_days/365.25
    avg_margin = max(ledger["Allocated_Margin"].mean(),1)
    ratio = equity.iloc[-1]/avg_margin
    annualized_rom = ((1+ratio)**(1/years)-1)*100 if ratio > -1 else -100
    cr = ledger["Net_PnL"]/ledger["Allocated_Margin"]
    sharpe = cr.mean()/cr.std()*np.sqrt(12) if cr.std()>0 else np.nan
    downside = cr[cr<0]
    sortino = cr.mean()/downside.std()*np.sqrt(12) if len(downside) and downside.std()>0 else np.nan
    return {
        "cycles":len(ledger),"net_pnl":pnl.sum(),
        "win_rate":(pnl>0).mean()*100,
        "profit_factor":wins.sum()/abs(losses.sum()) if len(losses) else np.nan,
        "max_dd":abs(dd.min()),"avg_rom":ledger["ROM_%"].mean(),
        "median_rom":ledger["ROM_%"].median(),"annualized_rom":annualized_rom,
        "sharpe":sharpe,"sortino":sortino,"best":pnl.max(),
        "worst":pnl.min(),"avg_trade":pnl.mean(),
        "stop_rate":ledger["Exit_Reason"].str.contains("Stop",na=False).mean()*100,
        "avg_margin":avg_margin,"days":total_days
    }


def monte_carlo(ledger, simulations=3000, cycles=None):
    if ledger.empty: return pd.DataFrame()
    returns = ledger["Net_PnL"].to_numpy()
    n = cycles or len(returns)
    rng = np.random.default_rng(42)
    sampled = rng.choice(returns,size=(simulations,n),replace=True)
    paths = sampled.cumsum(axis=1)
    return pd.DataFrame({
        "Terminal_PnL":paths[:,-1],
        "Minimum_Equity":paths.min(axis=1)
    })


# ============================================================
# CHART HELPERS
# ============================================================

def base_fig(title="", height=350):
    fig = go.Figure()
    fig.update_layout(**PLOT, title=dict(text=title, font=dict(size=13)),
                      height=height, xaxis=dict(gridcolor=C["border"]),
                      yaxis=dict(gridcolor=C["border"]))
    return fig


def money(x):
    if pd.isna(x): return "—"
    a = abs(x)
    sign = "-" if x < 0 else ""
    if a >= 1e7: return f"{sign}₹{a/1e7:.2f} Cr"
    if a >= 1e5: return f"{sign}₹{a/1e5:.2f} L"
    return f"{sign}₹{a:,.0f}"


def pct(x, digits=1):
    return "—" if pd.isna(x) else f"{x:.{digits}f}%"


def mini_spark(values, positive=True):
    fig = go.Figure(go.Scatter(
        y=list(values), mode="lines",
        line=dict(color=C["green"] if positive else C["red"], width=1.5),
        hoverinfo="skip"
    ))
    fig.update_layout(
        height=32, margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False)
    )
    return fig


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="topbar">
  <div class="brand">
    <div class="logo">◈</div>
    <div>
      <div class="title">NSE SINGLE-STOCK STRANGLE</div>
      <div class="sub">QUANT RESEARCH TERMINAL · BACKTEST · RISK · SCENARIO ANALYTICS</div>
    </div>
  </div>
  <div class="badges">
    <span class="badge ready">● ENGINE READY</span>
    <span class="badge theory">MODELLED OPTIONS</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.markdown("## ◈ Strategy Cockpit")

symbol = st.sidebar.selectbox(
    "Underlying", list(FNO),
    format_func=lambda x:f"{x} — {FNO[x]['name']}"
)
spec = FNO[symbol]

data_mode = st.sidebar.radio(
    "Pricing Data",
    ["Theoretical / Yahoo","Historical NSE Option CSV"]
)

years = st.sidebar.slider("Lookback Years",1,5,3)
y_pct = st.sidebar.slider("OTM Distance (%)",1.0,15.0,5.0,.5)
lot = st.sidebar.number_input("Lot Size",1,20000,int(spec["lot"]),25)
step = st.sidebar.number_input("Strike Step ₹",.5,500.,float(spec["step"]),1.)
margin_pct = st.sidebar.slider("Margin / Notional (%)",10.,35.,22.,.5)
stop_mult = st.sidebar.slider("Stop Loss (× Credit)",0.,5.,2.5,.5)
rf = st.sidebar.slider("Risk-Free Rate (%)",4.,9.,6.75,.25)/100
iv_markup = st.sidebar.slider("IV / Realized Vol",1.0,1.5,1.15,.01)
friction = st.sidebar.slider("Turnover Friction (%)",0.,3.,1.2,.1)

with st.sidebar.expander("Advanced / Research Notes"):
    vol_window = st.number_input("Realized-vol window",10,120,30,5)
    sims = st.number_input("Monte Carlo simulations",500,10000,3000,500)
    st.caption("Entry: first available trading day on/after the 15th. Exit: first available trading day on/after the following month's 15th.")
    st.caption("Theoretical option values are Black-Scholes estimates. They are not historical bid/ask executions.")


# ============================================================
# DATA LOAD
# ============================================================

nse_options = pd.DataFrame()
mode_label = "THEORETICAL MODEL"

if data_mode == "Historical NSE Option CSV":
    uploaded = st.sidebar.file_uploader(
        "Upload NSE historical option CSV", type=["csv"]
    )
    if uploaded:
        nse_options = parse_nse_csv(uploaded)
        if nse_options.empty:
            st.error("CSV mapping failed. Required: date, expiry, option type, strike and close/settlement price.")
            st.stop()
        mode_label = "NSE CSV LOADED"
        st.sidebar.success(f"{len(nse_options):,} option observations loaded")
    else:
        mode_label = "WAITING FOR CSV"

with st.spinner("Loading spot history…"):
    spot = yahoo_data(symbol, years)

if spot.empty:
    st.error("Yahoo Finance spot data unavailable for the selected underlying.")
    st.stop()

ledger = theoretical_backtest(
    spot, y_pct, step, lot, rf, iv_markup, stop_mult,
    friction, margin_pct, years, vol_window
)

if ledger.empty:
    st.warning("No complete strategy cycles were generated.")
    st.stop()

metrics = performance_metrics(ledger)

ledger["Cumulative_PnL"] = ledger["Net_PnL"].cumsum()
ledger["Peak"] = ledger["Cumulative_PnL"].cummax()
ledger["Drawdown"] = ledger["Cumulative_PnL"]-ledger["Peak"]
ledger["Premium_Capture_%"] = np.where(
    ledger["Sold_Prem"]>0,
    (ledger["Sold_Prem"]-ledger["Exit_Prem"])/ledger["Sold_Prem"]*100,
    np.nan
)


# ============================================================
# TICKER STRIP
# ============================================================

last_spot = float(spot["Close"].iloc[-1])
rv = float(
    np.log(spot["Close"]/spot["Close"].shift(1))
    .rolling(vol_window).std().iloc[-1] * np.sqrt(252)
)
rv = rv if np.isfinite(rv) else np.nan

chips = [
    (symbol, f"₹{last_spot:,.2f}", "LAST"),
    ("REALIZED VOL", pct(rv*100), f"{vol_window}D"),
    ("OTM", f"{y_pct:.1f}%", "PUT / CALL"),
    ("LOT", f"{int(lot):,}", "CONTRACT"),
    ("MARGIN", f"{margin_pct:.1f}%", "NOTIONAL"),
    ("CYCLES", f"{metrics['cycles']}", f"{years}Y"),
    ("STOP", f"{stop_mult:.1f}×", "CREDIT"),
]
ticker = '<div class="ticker">'
for k,v,s in chips:
    ticker += f'<div class="tick"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>'
ticker += "</div>"
st.markdown(ticker, unsafe_allow_html=True)


# ============================================================
# KPI CARDS
# ============================================================

kpis = [
    ("NET P&L", money(metrics["net_pnl"]), f"{metrics['cycles']} cycles"),
    ("WIN RATE", pct(metrics["win_rate"]), "positive cycles"),
    ("PROFIT FACTOR", f"{metrics['profit_factor']:.2f}" if np.isfinite(metrics["profit_factor"]) else "—", "gross wins / losses"),
    ("MAX DRAWDOWN", money(metrics["max_dd"]), "cycle equity"),
    ("AVG ROM", pct(metrics["avg_rom"]), "return on margin"),
    ("SHARPE", f"{metrics['sharpe']:.2f}" if np.isfinite(metrics["sharpe"]) else "—", "monthly-cycle basis"),
]
html = '<div class="kpi-grid">'
for label,val,sub in kpis:
    html += f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div><div class="kpi-sub">{sub}</div></div>'
html += "</div>"
st.markdown(html, unsafe_allow_html=True)


# ============================================================
# STRATEGY COCKPIT
# ============================================================

latest = ledger.iloc[-1]
S = latest["Spot_Entry"]
cock = [
    ("UNDERLYING", symbol),
    ("SPOT @ ENTRY", f"₹{S:,.2f}"),
    ("PUT", f"₹{latest['Put_Strike']:,.0f}"),
    ("CALL", f"₹{latest['Call_Strike']:,.0f}"),
    ("SOLD CREDIT", f"₹{latest['Sold_Prem']:.2f}"),
    ("MARGIN", money(latest["Allocated_Margin"])),
    ("EXPIRY", str(latest["Expiry_Date"])),
    ("LAST EXIT", str(latest["Exit_Date"])),
]
html = '<div class="card"><div class="card-title">Strategy Cockpit · Latest Completed Cycle</div><div class="cockpit" style="margin-top:9px;">'
for l,n in cock:
    html += f'<div class="cock"><div class="l">{l}</div><div class="n">{n}</div></div>'
html += "</div></div>"
st.markdown(html, unsafe_allow_html=True)

stop_class = "bad" if metrics["stop_rate"] > 20 else "warn" if metrics["stop_rate"] > 10 else "good"
st.markdown(
    f'<div class="risk">'
    f'<div class="risk-chip">STOP-LOSS RATE <span class="{stop_class}">{metrics["stop_rate"]:.1f}%</span></div>'
    f'<div class="risk-chip">WORST CYCLE <span class="bad">{money(metrics["worst"])}</span></div>'
    f'<div class="risk-chip">BEST CYCLE <span class="good">{money(metrics["best"])}</span></div>'
    f'<div class="risk-chip">MEDIAN ROM <span>{pct(metrics["median_rom"])}</span></div>'
    f'<div class="risk-chip">AVG HOLD <span>{ledger["Days_Held"].mean():.1f} days</span></div>'
    f'<div class="risk-chip">SORTINO <span>{metrics["sortino"]:.2f}</span></div>'
    f'</div>',
    unsafe_allow_html=True
)


# ============================================================
# NAVIGATION
# ============================================================

tabs = st.tabs([
    "◈ Overview",
    "⚠ Risk Monitor",
    "⌁ Trade Analytics",
    "◎ Monte Carlo",
    "▤ Trade Explorer",
    "ⓘ Methodology"
])


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:
    st.markdown('<div class="section">Portfolio Path</div>', unsafe_allow_html=True)
    c1,c2 = st.columns([2.2,1])

    with c1:
        fig = base_fig("Cumulative P&L vs Drawdown", 410)
        fig.add_trace(go.Scatter(
            x=ledger["Exit_Date"],y=ledger["Cumulative_PnL"],
            mode="lines+markers",name="Cumulative P&L",
            line=dict(color=C["cyan"],width=2.2),
            marker=dict(size=5)
        ))
        fig.add_trace(go.Scatter(
            x=ledger["Exit_Date"],y=ledger["Drawdown"],
            mode="lines",name="Drawdown",
            fill="tozeroy",line=dict(color=C["red"],width=1.2),
            opacity=.55,yaxis="y2"
        ))
        fig.update_layout(
            yaxis=dict(title="P&L ₹",gridcolor=C["border"]),
            yaxis2=dict(title="Drawdown ₹",overlaying="y",side="right",
                        showgrid=False),
            legend=dict(orientation="h",y=1.08,x=0)
        )
        st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

    with c2:
        fig = base_fig("Cycle P&L Distribution", 410)
        fig.add_trace(go.Histogram(
            x=ledger["Net_PnL"],nbinsx=max(8,min(22,len(ledger))),
            marker_line_width=0,opacity=.85,name="P&L"
        ))
        fig.add_vline(x=0,line_dash="dash",line_color=C["muted"])
        fig.update_layout(showlegend=False,xaxis_title="Net P&L ₹",yaxis_title="Cycles")
        st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

    st.markdown('<div class="section">Underlying & Strategy Behaviour</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        fig = base_fig("Underlying Price", 330)
        fig.add_trace(go.Scatter(
            x=spot.index,y=spot["Close"],mode="lines",
            name=symbol,line=dict(color=C["blue"],width=1.8)
        ))
        entries = pd.to_datetime(ledger["Entry_Date"])
        exits = pd.to_datetime(ledger["Exit_Date"])
        entry_y = [spot.loc[d.date(),"Close"] if d.date() in spot.index else np.nan for d in entries]
        exit_y = [spot.loc[d.date(),"Close"] if d.date() in spot.index else np.nan for d in exits]
        fig.add_trace(go.Scatter(x=entries,y=entry_y,mode="markers",name="Entry",
                                 marker=dict(symbol="triangle-up",size=8,color=C["green"])))
        fig.add_trace(go.Scatter(x=exits,y=exit_y,mode="markers",name="Exit",
                                 marker=dict(symbol="x",size=7,color=C["red"])))
        st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

    with c2:
        fig = base_fig("Spot Move vs Cycle P&L", 330)
        wins = ledger["Net_PnL"] >= 0
        fig.add_trace(go.Scatter(
            x=ledger.loc[wins,"Spot_Move_%"],y=ledger.loc[wins,"Net_PnL"],
            mode="markers",name="Winning cycles",marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=ledger.loc[~wins,"Spot_Move_%"],y=ledger.loc[~wins,"Net_PnL"],
            mode="markers",name="Losing cycles",marker=dict(size=8)
        ))
        fig.add_hline(y=0,line_dash="dash",line_color=C["muted"])
        fig.add_vline(x=0,line_dash="dot",line_color=C["muted"])
        fig.update_layout(xaxis_title="Spot Move %",yaxis_title="Net P&L ₹")
        st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})


# ============================================================
# RISK MONITOR
# ============================================================

with tabs[1]:
    st.markdown('<div class="section">Risk Dashboard</div>', unsafe_allow_html=True)
    r1,r2,r3,r4 = st.columns(4)
    r1.metric("Max Drawdown",money(metrics["max_dd"]))
    r2.metric("Worst Cycle",money(metrics["worst"]))
    r3.metric("Stop Rate",pct(metrics["stop_rate"]))
    r4.metric("Sortino",f"{metrics['sortino']:.2f}" if np.isfinite(metrics["sortino"]) else "—")

    c1,c2 = st.columns(2)
    with c1:
        fig = base_fig("Drawdown Profile", 350)
        fig.add_trace(go.Scatter(
            x=ledger["Exit_Date"],y=ledger["Drawdown"],
            fill="tozeroy",mode="lines",line=dict(color=C["red"],width=1.7)
        ))
        st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
    with c2:
        fig = base_fig("ROM by Cycle", 350)
        fig.add_trace(go.Bar(
            x=ledger["Exit_Date"],y=ledger["ROM_%"],
            marker=dict(color=np.where(ledger["ROM_%"]>=0,C["green"],C["red"]))
        ))
        fig.add_hline(y=0,line_dash="dash",line_color=C["muted"])
        st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

    reason = ledger["Exit_Reason"].value_counts()
    fig = base_fig("Exit Reason Mix", 300)
    fig.add_trace(go.Bar(x=reason.index,y=reason.values))
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})


# ============================================================
# TRADE ANALYTICS
# ============================================================

with tabs[2]:
    st.markdown('<div class="section">Return Drivers</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)

    with c1:
        fig = base_fig("Premium Capture %", 350)
        fig.add_trace(go.Bar(
            x=ledger["Exit_Date"],y=ledger["Premium_Capture_%"],
            marker=dict(color=np.where(ledger["Premium_Capture_%"]>=0,C["cyan"],C["red"]))
        ))
        fig.add_hline(y=0,line_dash="dash",line_color=C["muted"])
        st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

    with c2:
        fig = base_fig("Held Days vs P&L", 350)
        fig.add_trace(go.Scatter(
            x=ledger["Days_Held"],y=ledger["Net_PnL"],mode="markers",
            marker=dict(size=9),text=ledger["Exit_Reason"],
            hovertemplate="Days: %{x}<br>P&L: ₹%{y:,.0f}<br>%{text}<extra></extra>"
        ))
        fig.add_hline(y=0,line_dash="dash",line_color=C["muted"])
        fig.update_layout(xaxis_title="Days Held",yaxis_title="Net P&L ₹")
        st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

    st.markdown('<div class="section">Monthly Performance Heatmap</div>', unsafe_allow_html=True)
    tmp = ledger.copy()
    tmp["Exit"] = pd.to_datetime(tmp["Exit_Date"])
    tmp["Year"] = tmp["Exit"].dt.year
    tmp["Month"] = tmp["Exit"].dt.month
    pivot = tmp.pivot_table(index="Year",columns="Month",values="Net_PnL",aggfunc="sum").reindex(columns=range(1,13))
    fig = base_fig("Monthly Net P&L", 350)
    fig.add_trace(go.Heatmap(
        z=pivot.values,x=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        y=pivot.index,hovertemplate="Year %{y}<br>%{x}<br>P&L ₹%{z:,.0f}<extra></extra>",
        colorscale=[[0,C["red"]],[.5,"#151B24"],[1,C["green"]]],
        zmid=0
    ))
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

    st.markdown('<div class="section">Sensitivity — OTM Distance × Stop Multiple</div>', unsafe_allow_html=True)
    s1,s2 = st.columns([1,3])
    with s1:
        metric_choice = st.selectbox("Sensitivity Metric",["Net P&L","Win Rate","Avg ROM"])
        sens_otm = [3,4,5,6,7,8]
        sens_stop = [1.5,2,2.5,3,3.5,4]
    with s2:
        z = []
        for otm in sens_otm:
            row=[]
            for sm in sens_stop:
                l = theoretical_backtest(
                    spot,otm,step,lot,rf,iv_markup,sm,friction,margin_pct,years,vol_window
                )
                if l.empty:
                    row.append(np.nan)
                elif metric_choice=="Net P&L":
                    row.append(l["Net_PnL"].sum())
                elif metric_choice=="Win Rate":
                    row.append((l["Net_PnL"]>0).mean()*100)
                else:
                    row.append(l["ROM_%"].mean())
            z.append(row)
        fig = base_fig(f"{metric_choice} Sensitivity", 360)
        fig.add_trace(go.Heatmap(
            z=z,x=[f"{x:.1f}×" for x in sens_stop],
            y=[f"{x:.1f}%" for x in sens_otm],
            colorscale=[[0,C["red"]],[.5,"#151B24"],[1,C["green"]]],
            hovertemplate="OTM %{y}<br>Stop %{x}<br>Value %{z:.2f}<extra></extra>"
        ))
        st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})


# ============================================================
# MONTE CARLO
# ============================================================

with tabs[3]:
    st.markdown('<div class="section">Bootstrap Monte Carlo Terminal</div>', unsafe_allow_html=True)
    mc = monte_carlo(ledger,int(sims))
    if mc.empty:
        st.warning("Monte Carlo unavailable.")
    else:
        q = mc["Terminal_PnL"].quantile([.05,.25,.5,.75,.95])
        qmin = mc["Minimum_Equity"].quantile([.05,.5,.95])
        a,b,c,d = st.columns(4)
        a.metric("P05 Terminal",money(q.loc[.05]))
        b.metric("Median Terminal",money(q.loc[.5]))
        c.metric("P95 Terminal",money(q.loc[.95]))
        d.metric("P05 Min Equity",money(qmin.loc[.05]))

        c1,c2 = st.columns(2)
        with c1:
            fig = base_fig("Terminal P&L Distribution", 360)
            fig.add_trace(go.Histogram(x=mc["Terminal_PnL"],nbinsx=45))
            fig.add_vline(x=0,line_dash="dash",line_color=C["muted"])
            st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
        with c2:
            fig = base_fig("Minimum Equity Distribution", 360)
            fig.add_trace(go.Histogram(x=mc["Minimum_Equity"],nbinsx=45))
            fig.add_vline(x=0,line_dash="dash",line_color=C["muted"])
            st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

        rng = np.random.default_rng(7)
        returns = ledger["Net_PnL"].to_numpy()
        path_count = min(40,int(sims))
        sampled = rng.choice(returns,size=(path_count,len(returns)),replace=True).cumsum(axis=1)
        fig = base_fig("Sampled Equity Paths", 390)
        for row in sampled:
            fig.add_trace(go.Scatter(
                y=row,mode="lines",line=dict(width=.7),
                opacity=.28,showlegend=False
            ))
        st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
        st.caption("Monte Carlo uses bootstrap resampling of observed cycle P&L. It is not an option-pricing or implied-volatility stochastic model.")


# ============================================================
# TRADE EXPLORER
# ============================================================

with tabs[4]:
    st.markdown('<div class="section">Trade Explorer</div>', unsafe_allow_html=True)
    display = ledger.copy()
    display["Entry"] = pd.to_datetime(display["Entry_Date"]).dt.strftime("%Y-%m-%d")
    display["Exit"] = pd.to_datetime(display["Exit_Date"]).dt.strftime("%Y-%m-%d")
    display["Expiry"] = pd.to_datetime(display["Expiry_Date"]).dt.strftime("%Y-%m-%d")
    display["P&L"] = display["Net_PnL"].round(0)
    display["ROM"] = display["ROM_%"].round(2)
    display["Spot Move"] = display["Spot_Move_%"].round(2)

    cols = ["Entry","Exit","Expiry","Spot_Entry","Put_Strike","Call_Strike",
            "Sold_Prem","Exit_Prem","P&L","ROM","Spot Move","Days_Held","Exit_Reason"]
    st.dataframe(
        display[cols].sort_values("Entry",ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Spot_Entry":st.column_config.NumberColumn("Spot",format="₹%.2f"),
            "Put_Strike":st.column_config.NumberColumn("Put",format="₹%.0f"),
            "Call_Strike":st.column_config.NumberColumn("Call",format="₹%.0f"),
            "Sold_Prem":st.column_config.NumberColumn("Credit",format="₹%.2f"),
            "Exit_Prem":st.column_config.NumberColumn("Exit Prem",format="₹%.2f"),
            "P&L":st.column_config.NumberColumn("Net P&L",format="₹%.0f"),
            "ROM":st.column_config.NumberColumn("ROM %",format="%.2f"),
            "Spot Move":st.column_config.NumberColumn("Spot Move %",format="%.2f"),
        },
        height=520
    )

    csv_bytes = ledger.to_csv(index=False).encode("utf-8")
    st.download_button(
        "↓ Download Full Audit Ledger",
        data=csv_bytes,
        file_name=f"{symbol}_short_strangle_audit.csv",
        mime="text/csv"
    )


# ============================================================
# METHODOLOGY
# ============================================================

with tabs[5]:
    st.markdown('<div class="section">Research Methodology & Disclosures</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="card">
<b>Pricing mode:</b> {mode_label}<br><br>
<b>Theoretical engine:</b> Yahoo Finance underlying prices + rolling {vol_window}-day
realized volatility × IV markup + Black-Scholes option valuation.<br><br>
<b>Entry:</b> first available trading session on/after the 15th of each month.<br>
<b>Exit:</b> first available trading session on/after the 15th of the following month,
unless the combined option mark reaches the configured stop multiple.<br>
<b>Strikes:</b> rounded to the selected NSE strike interval around the target OTM distance.<br>
<b>Costs:</b> turnover friction applied to entry + exit option premium.<br>
<b>Margin:</b> simplified percentage of spot notional, not historical SPAN/ELM.<br>
<b>Expiry:</b> synthetic last-Tuesday rule for this model. Historical research should
use actual contract expiry dates from the relevant NSE dataset.<br><br>
<b>Historical NSE CSV:</b> the application validates and previews uploaded contract data,
but this version does not pretend to execute trades at those historical prices.
An execution-grade version should select the exact expiry/strike contracts and use
historical bid/ask or settlement prices, corporate-action handling, slippage,
lot-size changes, and historical margin requirements.
</div>
""", unsafe_allow_html=True)

    st.markdown("### Model limitations")
    st.write([
        "Black-Scholes is a theoretical mark and does not reproduce the NSE option order book.",
        "Realized volatility is used as a proxy for implied volatility.",
        "Monthly Sharpe/Sortino annualization assumes twelve comparable cycle observations per year.",
        "Bootstrap Monte Carlo assumes the observed P&L distribution is representative and independent.",
        "Theoretical returns should not be interpreted as executable historical performance."
    ])


# ============================================================
# FOOTER
# ============================================================

st.markdown(f"""
<div class="footer">
NSE STRANGLE QUANT TERMINAL V3 · Research interface only ·
Theoretical option pricing is explicitly separated from historical execution data.
Parameters: {symbol} · {years}Y · {y_pct:.1f}% OTM · {stop_mult:.1f}× stop ·
{margin_pct:.1f}% margin · {friction:.1f}% turnover friction.
</div>
""", unsafe_allow_html=True)
