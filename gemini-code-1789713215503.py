"""
NSE SINGLE-STOCK SHORT STRANGLE
Institutional Quant Dashboard
--------------------------------
Streamlit + Plotly

Features
--------
• Responsive institutional dashboard
• Nifty F&O single-stock selection
• Monthly 15th -> 15th short strangle
• Discrete NSE strike snapping
• Rolling realized volatility
• IV markup
• Black-Scholes theoretical pricing
• Stop-loss monitoring
• Margin / ROM analytics
• Equity curve
• Drawdown analysis
• Monthly P&L heatmap
• ROM distribution
• P&L distribution
• Spot move vs P&L
• Premium capture
• Trade audit ledger
• CSV export
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import calendar
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf


# =========================================================
# 1. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="NSE Strangle Quant Dashboard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. INSTITUTIONAL CSS
# =========================================================

st.markdown("""
<style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 0%,
                rgba(25, 118, 210, 0.08),
                transparent 30%
            ),
            #080b10;
        color: #e8edf3;
    }

    .block-container {
        max-width: 1600px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        padding-left: 1.4rem;
        padding-right: 1.4rem;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #0b0f15;
        border-right: 1px solid #202733;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f2f5f8;
    }

    /* ---------- HEADER ---------- */

    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 20px;
        padding: 20px 24px;
        margin-bottom: 18px;
        border: 1px solid #202733;
        border-radius: 14px;
        background: linear-gradient(
            135deg,
            rgba(20, 27, 38, 0.95),
            rgba(11, 15, 21, 0.95)
        );
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }

    .dashboard-title {
        font-size: 1.65rem;
        font-weight: 750;
        letter-spacing: -0.5px;
        color: #f5f7fa;
    }

    .dashboard-subtitle {
        color: #8993a1;
        font-size: 0.82rem;
        margin-top: 4px;
    }

    .status-pill {
        padding: 7px 13px;
        border-radius: 999px;
        background: rgba(0, 204, 150, 0.10);
        border: 1px solid rgba(0, 204, 150, 0.28);
        color: #00cc96;
        font-size: 0.75rem;
        font-weight: 650;
        white-space: nowrap;
    }

    /* ---------- KPI ---------- */

    .kpi-grid {
        display: grid;
        grid-template-columns:
            repeat(5, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 18px;
    }

    .kpi {
        background: linear-gradient(
            145deg,
            #111722,
            #0d121a
        );
        border: 1px solid #202733;
        border-radius: 12px;
        padding: 15px 16px;
        min-height: 105px;
        box-shadow: 0 7px 22px rgba(0,0,0,0.18);
    }

    .kpi-label {
        color: #7f8997;
        text-transform: uppercase;
        font-size: 0.66rem;
        font-weight: 650;
        letter-spacing: 0.7px;
    }

    .kpi-value {
        color: #f3f6f9;
        font-size: 1.42rem;
        font-weight: 750;
        margin-top: 8px;
    }

    .kpi-positive {
        color: #00cc96;
    }

    .kpi-negative {
        color: #ff5b61;
    }

    .kpi-secondary {
        color: #788392;
        font-size: 0.68rem;
        margin-top: 5px;
    }

    /* ---------- SECTION ---------- */

    .section-header {
        font-size: 1rem;
        font-weight: 700;
        color: #e9edf2;
        margin: 15px 0 9px 2px;
    }

    /* ---------- INFO CARDS ---------- */

    .info-card {
        background: #0d121a;
        border: 1px solid #202733;
        border-radius: 12px;
        padding: 15px;
        height: 100%;
    }

    .info-title {
        color: #818b99;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    .info-value {
        color: #eef2f6;
        font-size: 1rem;
        font-weight: 650;
        margin-top: 5px;
    }

    /* ---------- TABLE ---------- */

    div[data-testid="stDataFrame"] {
        border: 1px solid #202733;
        border-radius: 10px;
        overflow: hidden;
    }

    /* ---------- BUTTONS ---------- */

    .stDownloadButton button {
        border-radius: 8px;
        border: 1px solid #293342;
    }

    /* ---------- MOBILE ---------- */

    @media (max-width: 1000px) {

        .kpi-grid {
            grid-template-columns:
                repeat(3, minmax(0, 1fr));
        }

        .dashboard-title {
            font-size: 1.35rem;
        }
    }

    @media (max-width: 650px) {

        .block-container {
            padding-left: 0.55rem;
            padding-right: 0.55rem;
            padding-top: 0.7rem;
        }

        .dashboard-header {
            padding: 15px;
        }

        .dashboard-title {
            font-size: 1.15rem;
        }

        .dashboard-subtitle {
            font-size: 0.72rem;
        }

        .kpi-grid {
            grid-template-columns:
                repeat(2, minmax(0, 1fr));
            gap: 8px;
        }

        .kpi {
            padding: 12px;
            min-height: 90px;
        }

        .kpi-value {
            font-size: 1.05rem;
        }

        .kpi-label {
            font-size: 0.58rem;
        }
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. COLOR PALETTE
# =========================================================

GREEN = "#00CC96"
RED = "#FF5B61"
BLUE = "#4C9AFF"
YELLOW = "#F6C85F"
PURPLE = "#9B7EDE"
TEXT = "#E8EDF3"
MUTED = "#7F8997"
GRID = "#202733"
BG = "#0D121A"


# =========================================================
# 4. NIFTY F&O DIRECTORY
# =========================================================

NIFTY_FNO_DIRECTORY = {

    "RELIANCE.NS": {
        "name": "Reliance Industries",
        "lot": 250,
        "strike_step": 20
    },

    "TCS.NS": {
        "name": "Tata Consultancy Services",
        "lot": 175,
        "strike_step": 50
    },

    "INFY.NS": {
        "name": "Infosys Ltd",
        "lot": 400,
        "strike_step": 20
    },

    "HDFCBANK.NS": {
        "name": "HDFC Bank Ltd",
        "lot": 550,
        "strike_step": 10
    },

    "ICICIBANK.NS": {
        "name": "ICICI Bank Ltd",
        "lot": 700,
        "strike_step": 10
    },

    "SBIN.NS": {
        "name": "State Bank of India",
        "lot": 750,
        "strike_step": 5
    },

    "BHARTIARTL.NS": {
        "name": "Bharti Airtel Ltd",
        "lot": 475,
        "strike_step": 10
    },

    "ITC.NS": {
        "name": "ITC Limited",
        "lot": 1600,
        "strike_step": 5
    },

    "LT.NS": {
        "name": "Larsen & Toubro Ltd",
        "lot": 150,
        "strike_step": 25
    },

    "TATAMOTORS.NS": {
        "name": "Tata Motors",
        "lot": 550,
        "strike_step": 10
    }
}


# =========================================================
# 5. BLACK-SCHOLES
# =========================================================

def black_scholes_price(
    S,
    K,
    T,
    r,
    sigma,
    option_type="call"
):

    if T <= 0.0001:

        if option_type == "call":
            return max(0, S - K)

        return max(0, K - S)

    sigma = max(float(sigma), 0.0001)

    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma ** 2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":

        price = (
            S * norm.cdf(d1)
            - K * np.exp(-r * T) * norm.cdf(d2)
        )

    else:

        price = (
            K * np.exp(-r * T) * norm.cdf(-d2)
            - S * norm.cdf(-d1)
        )

    return max(0.0, float(price))


# =========================================================
# 6. EXPIRY
# =========================================================

def get_last_thursday(year, month):

    last_day = calendar.monthrange(year, month)[1]

    cursor = datetime.date(
        year,
        month,
        last_day
    )

    offset = (cursor.weekday() - 3) % 7

    return cursor - datetime.timedelta(days=offset)


# =========================================================
# 7. DATA DOWNLOAD
# =========================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def fetch_historical_stock_data(
    ticker,
    years=3
):

    end_date = datetime.date.today()

    start_date = (
        end_date
        - datetime.timedelta(
            days=years * 365 + 150
        )
    )

    raw = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False
    )

    if raw.empty:
        return pd.DataFrame()

    if isinstance(
        raw.columns,
        pd.MultiIndex
    ):

        raw.columns = (
            raw.columns
            .get_level_values(0)
        )

    cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    available = [
        c for c in cols
        if c in raw.columns
    ]

    df = raw[available].dropna()

    df.index = pd.to_datetime(
        df.index
    ).date

    return df


# =========================================================
# 8. BACKTEST ENGINE
# =========================================================

def execute_strangle_backtest(
    data,
    y_pct,
    strike_step,
    lot_size,
    risk_free_rate,
    vol_lookback,
    iv_multiplier,
    stop_loss_mult,
    transaction_cost_pct,
    margin_pct,
    lookback_years
):

    df = data.copy()

    # -----------------------------------------
    # Realized volatility
    # -----------------------------------------

    df["Log_Return"] = np.log(
        df["Close"]
        / df["Close"].shift(1)
    )

    df["Realized_Vol"] = (
        df["Log_Return"]
        .rolling(vol_lookback)
        .std()
        * np.sqrt(252)
    )

    df["Realized_Vol"] = (
        df["Realized_Vol"]
        .bfill()
        .clip(lower=0.12)
    )

    trading_dates = sorted(
        list(df.index)
    )

    if not trading_dates:
        return pd.DataFrame(), df

    start_boundary = (
        datetime.date.today()
        - datetime.timedelta(
            days=lookback_years * 365
        )
    )

    first_idx = vol_lookback

    while (
        first_idx < len(trading_dates)
        and trading_dates[first_idx]
        < start_boundary
    ):
        first_idx += 1

    if first_idx >= len(trading_dates):
        return pd.DataFrame(), df

    seed_date = trading_dates[first_idx]
    last_date = trading_dates[-1]

    current_year = seed_date.year
    current_month = seed_date.month

    ledger = []

    # =====================================================
    # MONTHLY CYCLES
    # =====================================================

    while True:

        target_entry = datetime.date(
            current_year,
            current_month,
            15
        )

        if target_entry > last_date:
            break

        valid_entries = [
            d for d in trading_dates
            if d >= target_entry
        ]

        if not valid_entries:
            break

        entry_date = valid_entries[0]

        # -----------------------------------------
        # Next month
        # -----------------------------------------

        if current_month == 12:

            next_year = current_year + 1
            next_month = 1

        else:

            next_year = current_year
            next_month = current_month + 1

        target_exit = datetime.date(
            next_year,
            next_month,
            15
        )

        valid_exits = [
            d for d in trading_dates
            if d >= target_exit
        ]

        if not valid_exits:
            break

        scheduled_exit = valid_exits[0]

        expiry = get_last_thursday(
            next_year,
            next_month
        )

        # -----------------------------------------
        # Entry
        # -----------------------------------------

        S_entry = float(
            df.loc[entry_date, "Close"]
        )

        realized_vol_entry = float(
            df.loc[
                entry_date,
                "Realized_Vol"
            ]
        )

        pricing_vol_entry = (
            realized_vol_entry
            * iv_multiplier
        )

        raw_put = (
            S_entry
            * (1 - y_pct / 100)
        )

        raw_call = (
            S_entry
            * (1 + y_pct / 100)
        )

        K_put = (
            round(raw_put / strike_step)
            * strike_step
        )

        K_call = (
            round(raw_call / strike_step)
            * strike_step
        )

        dte = max(
            1,
            (expiry - entry_date).days
        )

        T = dte / 365

        put_entry = black_scholes_price(
            S_entry,
            K_put,
            T,
            risk_free_rate,
            pricing_vol_entry,
            "put"
        )

        call_entry = black_scholes_price(
            S_entry,
            K_call,
            T,
            risk_free_rate,
            pricing_vol_entry,
            "call"
        )

        initial_credit = (
            put_entry
            + call_entry
        )

        # -----------------------------------------
        # Monitoring
        # -----------------------------------------

        monitoring_dates = [
            d for d in trading_dates
            if entry_date < d <= scheduled_exit
        ]

        actual_exit = scheduled_exit
        exit_reason = "Scheduled Roll"

        put_exit = None
        call_exit = None

        S_exit = float(
            df.loc[
                scheduled_exit,
                "Close"
            ]
        )

        for monitor_date in monitoring_dates:

            S_int = float(
                df.loc[
                    monitor_date,
                    "Close"
                ]
            )

            vol_int = float(
                df.loc[
                    monitor_date,
                    "Realized_Vol"
                ]
            ) * iv_multiplier

            dte_int = max(
                0,
                (expiry - monitor_date).days
            )

            T_int = dte_int / 365

            p_val = black_scholes_price(
                S_int,
                K_put,
                T_int,
                risk_free_rate,
                vol_int,
                "put"
            )

            c_val = black_scholes_price(
                S_int,
                K_call,
                T_int,
                risk_free_rate,
                vol_int,
                "call"
            )

            combined_value = (
                p_val + c_val
            )

            if (
                stop_loss_mult > 0
                and combined_value
                >= initial_credit
                * stop_loss_mult
            ):

                actual_exit = monitor_date
                S_exit = S_int

                put_exit = p_val
                call_exit = c_val

                exit_reason = (
                    f"Stop Loss "
                    f"{stop_loss_mult:.1f}x"
                )

                break

        # -----------------------------------------
        # Scheduled exit
        # -----------------------------------------

        if put_exit is None:

            vol_exit = float(
                df.loc[
                    scheduled_exit,
                    "Realized_Vol"
                ]
            ) * iv_multiplier

            dte_exit = max(
                0,
                (expiry - scheduled_exit).days
            )

            T_exit = dte_exit / 365

            put_exit = black_scholes_price(
                S_exit,
                K_put,
                T_exit,
                risk_free_rate,
                vol_exit,
                "put"
            )

            call_exit = black_scholes_price(
                S_exit,
                K_call,
                T_exit,
                risk_free_rate,
                vol_exit,
                "call"
            )

        # -----------------------------------------
        # P&L
        # -----------------------------------------

        exit_premium = (
            put_exit
            + call_exit
        )

        gross_pnl_share = (
            initial_credit
            - exit_premium
        )

        turnover = (
            initial_credit
            + exit_premium
        )

        friction = (
            turnover
            * transaction_cost_pct
            / 100
        )

        net_pnl_share = (
            gross_pnl_share
            - friction
        )

        total_pnl = (
            net_pnl_share
            * lot_size
        )

        # -----------------------------------------
        # Margin
        # -----------------------------------------

        gross_notional = (
            S_entry * lot_size
        )

        allocated_margin = (
            gross_notional
            * margin_pct
            / 100
        )

        rom = (
            total_pnl
            / allocated_margin
            * 100
        )

        # -----------------------------------------
        # Ledger
        # -----------------------------------------

        ledger.append({

            "Entry_Date": entry_date,

            "Exit_Date": actual_exit,

            "Expiry_Date": expiry,

            "Spot_Entry": S_entry,

            "Spot_Exit": S_exit,

            "Spot_Move_%":
                (S_exit / S_entry - 1)
                * 100,

            "Put_Strike": K_put,

            "Call_Strike": K_call,

            "Put_Entry": put_entry,

            "Call_Entry": call_entry,

            "Sold_Prem":
                initial_credit,

            "Exit_Prem":
                exit_premium,

            "Gross_PnL":
                gross_pnl_share
                * lot_size,

            "Transaction_Cost":
                friction
                * lot_size,

            "Net_PnL":
                total_pnl,

            "Allocated_Margin":
                allocated_margin,

            "ROM_%":
                rom,

            "Days_Held":
                (
                    actual_exit
                    - entry_date
                ).days,

            "Exit_Reason":
                exit_reason
        })

        # -----------------------------------------
        # Increment month
        # -----------------------------------------

        if current_month == 12:

            current_year += 1
            current_month = 1

        else:

            current_month += 1

    return pd.DataFrame(ledger), df


# =========================================================
# 9. PLOTLY THEME
# =========================================================

def base_layout():

    return dict(

        template="plotly_dark",

        paper_bgcolor=BG,

        plot_bgcolor=BG,

        font=dict(
            color=TEXT,
            size=11
        ),

        margin=dict(
            l=45,
            r=20,
            t=50,
            b=40
        ),

        hovermode="x unified",

        xaxis=dict(
            gridcolor=GRID,
            zerolinecolor=GRID
        ),

        yaxis=dict(
            gridcolor=GRID,
            zerolinecolor=GRID
        )
    )


# =========================================================
# 10. SIDEBAR
# =========================================================

st.sidebar.markdown(
    "## ◈ QUANT ENGINE"
)

st.sidebar.caption(
    "NSE Single-Stock Short Strangle"
)

st.sidebar.divider()

selected_symbol = st.sidebar.selectbox(
    "Underlying",
    list(NIFTY_FNO_DIRECTORY.keys()),
    format_func=lambda x:
        f"{x.replace('.NS','')}  ·  "
        f"{NIFTY_FNO_DIRECTORY[x]['name']}"
)

spec = NIFTY_FNO_DIRECTORY[
    selected_symbol
]

configured_lot = spec["lot"]
configured_step = spec["strike_step"]


with st.sidebar.expander(
    "Strategy Parameters",
    expanded=True
):

    y_pct = st.slider(
        "OTM Distance",
        1.0,
        15.0,
        5.0,
        0.5,
        format="%.1f%%"
    )

    strike_step = st.number_input(
        "Strike Interval ₹",
        0.5,
        200.0,
        float(configured_step),
        1.0
    )

    lot_size = st.number_input(
        "Lot Size",
        1,
        20000,
        int(configured_lot),
        25
    )

    lookback_years = st.slider(
        "Backtest Period",
        1,
        4,
        3
    )


with st.sidebar.expander(
    "Risk & Pricing",
    expanded=True
):

    margin_pct = st.slider(
        "Margin / Notional",
        15.0,
        30.0,
        22.0,
        0.5
    )

    stop_loss = st.slider(
        "Stop Loss",
        0.0,
        4.0,
        2.5,
        0.5
    )

    iv_multiplier = st.slider(
        "IV / Realized Vol",
        1.0,
        1.40,
        1.15,
        0.01
    )

    rf_rate = st.slider(
        "Risk-Free Rate",
        4.0,
        9.0,
        6.75,
        0.25
    ) / 100

    friction = st.slider(
        "Total Friction",
        0.0,
        3.0,
        1.2,
        0.1
    )


st.sidebar.divider()

st.sidebar.caption(
    "Model assumptions are theoretical. "
    "Options are priced using Black-Scholes "
    "rather than historical NSE option quotes."
)


# =========================================================
# 11. DATA + BACKTEST
# =========================================================

st.markdown("""
<div class="dashboard-header">

    <div>
        <div class="dashboard-title">
            NSE Single-Stock Short Strangle
        </div>

        <div class="dashboard-subtitle">
            Systematic monthly volatility harvesting
            · 15th entry → 15th exit
            · Black-Scholes theoretical valuation
        </div>
    </div>

    <div class="status-pill">
        ● ENGINE READY
    </div>

</div>
""", unsafe_allow_html=True)


with st.spinner(
    "Loading historical market data..."
):

    stock_history = fetch_historical_stock_data(
        selected_symbol,
        lookback_years
    )


if stock_history.empty:

    st.error(
        "Market data unavailable."
    )

    st.stop()


with st.spinner(
    "Running quantitative simulation..."
):

    ledger_df, enriched_df = (
        execute_strangle_backtest(
            stock_history,
            y_pct,
            strike_step,
            lot_size,
            rf_rate,
            30,
            iv_multiplier,
            stop_loss,
            friction,
            margin_pct,
            lookback_years
        )
    )


if ledger_df.empty:

    st.warning(
        "No complete trading cycles were generated."
    )

    st.stop()


# =========================================================
# 12. METRICS
# =========================================================

ledger_df["Cumulative_PnL"] = (
    ledger_df["Net_PnL"].cumsum()
)

ledger_df["Peak_PnL"] = (
    ledger_df["Cumulative_PnL"].cummax()
)

ledger_df["Drawdown_INR"] = (
    ledger_df["Cumulative_PnL"]
    - ledger_df["Peak_PnL"]
)

ledger_df["Year"] = (
    pd.to_datetime(
        ledger_df["Exit_Date"]
    ).dt.year
)

ledger_df["Month"] = (
    pd.to_datetime(
        ledger_df["Exit_Date"]
    ).dt.month
)

ledger_df["Month_Name"] = (
    pd.to_datetime(
        ledger_df["Exit_Date"]
    ).dt.strftime("%b")
)


total_cycles = len(ledger_df)

winning = ledger_df[
    ledger_df["Net_PnL"] > 0
]

losing = ledger_df[
    ledger_df["Net_PnL"] < 0
]

winning_cycles = len(winning)

win_rate = (
    winning_cycles
    / total_cycles
    * 100
)

net_pnl = ledger_df["Net_PnL"].sum()

gross_wins = (
    winning["Net_PnL"].sum()
)

gross_losses = abs(
    losing["Net_PnL"].sum()
)

profit_factor = (
    gross_wins / gross_losses
    if gross_losses > 0
    else np.nan
)

max_drawdown = abs(
    ledger_df["Drawdown_INR"].min()
)

avg_margin = (
    ledger_df["Allocated_Margin"].mean()
)

total_margin = (
    ledger_df["Allocated_Margin"].sum()
)

avg_rom = ledger_df["ROM_%"].mean()

median_rom = (
    ledger_df["ROM_%"].median()
)

stop_loss_count = (
    ledger_df["Exit_Reason"]
    .str.contains(
        "Stop Loss",
        na=False
    )
    .sum()
)

stop_loss_rate = (
    stop_loss_count
    / total_cycles
    * 100
)

avg_pnl = ledger_df[
    "Net_PnL"
].mean()

best_trade = ledger_df[
    "Net_PnL"
].max()

worst_trade = ledger_df[
    "Net_PnL"
].min()


# =========================================================
# 13. KPI CARDS
# =========================================================

pnl_class = (
    "kpi-positive"
    if net_pnl >= 0
    else "kpi-negative"
)

rom_class = (
    "kpi-positive"
    if avg_rom >= 0
    else "kpi-negative"
)

st.markdown(f"""

<div class="kpi-grid">

<div class="kpi">
    <div class="kpi-label">Net P&L</div>
    <div class="kpi-value {pnl_class}">
        ₹{net_pnl:,.0f}
    </div>
    <div class="kpi-secondary">
        {total_cycles} completed cycles
    </div>
</div>

<div class="kpi">
    <div class="kpi-label">Win Rate</div>
    <div class="kpi-value">
        {win_rate:.1f}%
    </div>
    <div class="kpi-secondary">
        {winning_cycles} wins ·
        {total_cycles-winning_cycles} losses
    </div>
</div>

<div class="kpi">
    <div class="kpi-label">Avg Cycle ROM</div>
    <div class="kpi-value {rom_class}">
        {avg_rom:+.2f}%
    </div>
    <div class="kpi-secondary">
        Median {median_rom:+.2f}%
    </div>
</div>

<div class="kpi">
    <div class="kpi-label">Profit Factor</div>
    <div class="kpi-value">
        {"N/A" if np.isnan(profit_factor)
         else f"{profit_factor:.2f}"}
    </div>
    <div class="kpi-secondary">
        Gross wins / losses
    </div>
</div>

<div class="kpi">
    <div class="kpi-label">Max Drawdown</div>
    <div class="kpi-value kpi-negative">
        ₹{max_drawdown:,.0f}
    </div>
    <div class="kpi-secondary">
        Avg margin ₹{avg_margin:,.0f}
    </div>
</div>

</div>

""", unsafe_allow_html=True)


# =========================================================
# 14. STRATEGY SNAPSHOT
# =========================================================

st.markdown(
    '<div class="section-header">'
    'Strategy Snapshot'
    '</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5, c6 = st.columns(6)

snapshot = [

    ("UNDERLYING",
     selected_symbol.replace(".NS", "")),

    ("LOT SIZE",
     f"{lot_size:,}"),

    ("STRIKE STEP",
     f"₹{strike_step:,.0f}"),

    ("OTM DISTANCE",
     f"{y_pct:.1f}%"),

    ("IV MARKUP",
     f"{iv_multiplier:.2f}x"),

    ("STOP LOSS",
     f"{stop_loss:.1f}x")
]

for col, (label, value) in zip(
    [c1, c2, c3, c4, c5, c6],
    snapshot
):

    with col:

        st.markdown(
            f"""
            <div class="info-card">

                <div class="info-title">
                    {label}
                </div>

                <div class="info-value">
                    {value}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# 15. TABS
# =========================================================

tab_overview, tab_risk, tab_trades, tab_audit = st.tabs(
    [
        "◈ Overview",
        "⚠ Risk Analytics",
        "◎ Trade Analytics",
        "☷ Audit Ledger"
    ]
)


# =========================================================
# 16. OVERVIEW
# =========================================================

with tab_overview:

    st.markdown(
        '<div class="section-header">'
        'Equity & Drawdown'
        '</div>',
        unsafe_allow_html=True
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.72, 0.28]
    )

    fig.add_trace(

        go.Scatter(
            x=ledger_df["Exit_Date"],
            y=ledger_df["Cumulative_PnL"],
            mode="lines+markers",
            name="Cumulative P&L",
            line=dict(
                color=GREEN,
                width=2.5
            ),
            marker=dict(
                size=5
            ),
            hovertemplate=
                "%{x}<br>"
                "Cumulative P&L: ₹%{y:,.0f}"
                "<extra></extra>"
        ),

        row=1,
        col=1
    )

    fig.add_trace(

        go.Scatter(
            x=ledger_df["Exit_Date"],
            y=ledger_df["Drawdown_INR"],
            mode="lines",
            name="Drawdown",
            line=dict(
                color=RED,
                width=1.5
            ),
            fill="tozeroy",
            hovertemplate=
                "%{x}<br>"
                "Drawdown: ₹%{y:,.0f}"
                "<extra></extra>"
        ),

        row=2,
        col=1
    )

    fig.update_layout(
        **base_layout(),
        height=500,
        showlegend=False
    )

    fig.update_yaxes(
        title="₹ P&L",
        row=1,
        col=1
    )

    fig.update_yaxes(
        title="₹ DD",
        row=2,
        col=1
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True
        }
    )


    # ---------------------------------------------
    # Underlying Price
    # ---------------------------------------------

    st.markdown(
        '<div class="section-header">'
        'Underlying Price & Trading Cycles'
        '</div>',
        unsafe_allow_html=True
    )

    price_df = enriched_df.copy()

    fig_price = go.Figure()

    fig_price.add_trace(

        go.Scatter(
            x=list(price_df.index),
            y=price_df["Close"],
            mode="lines",
            name="Spot",
            line=dict(
                color=BLUE,
                width=1.8
            )
        )
    )

    fig_price.add_trace(

        go.Scatter(
            x=ledger_df["Entry_Date"],
            y=ledger_df["Spot_Entry"],
            mode="markers",
            name="Entry",
            marker=dict(
                color=GREEN,
                size=8,
                symbol="triangle-up"
            )
        )
    )

    fig_price.add_trace(

        go.Scatter(
            x=ledger_df["Exit_Date"],
            y=ledger_df["Spot_Exit"],
            mode="markers",
            name="Exit",
            marker=dict(
                color=RED,
                size=8,
                symbol="triangle-down"
            )
        )
    )

    fig_price.update_layout(
        **base_layout(),
        height=390
    )

    fig_price.update_yaxes(
        title="Spot ₹"
    )

    st.plotly_chart(
        fig_price,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


    # ---------------------------------------------
    # Monthly P&L
    # ---------------------------------------------

    st.markdown(
        '<div class="section-header">'
        'Monthly P&L Heatmap'
        '</div>',
        unsafe_allow_html=True
    )

    monthly = (
        ledger_df
        .groupby(["Year", "Month"])["Net_PnL"]
        .sum()
        .reset_index()
    )

    pivot = monthly.pivot(
        index="Year",
        columns="Month",
        values="Net_PnL"
    )

    month_labels = [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec"
    ]

    pivot = pivot.reindex(
        columns=range(1, 13)
    )

    fig_heat = go.Figure(

        go.Heatmap(

            z=pivot.values,

            x=month_labels,

            y=pivot.index,

            colorscale=[
                [0, RED],
                [0.5, "#202733"],
                [1, GREEN]
            ],

            colorbar=dict(
                title="₹ P&L"
            ),

            hovertemplate=
                "Year: %{y}<br>"
                "Month: %{x}<br>"
                "P&L: ₹%{z:,.0f}"
                "<extra></extra>"
        )
    )

    fig_heat.update_layout(
        **base_layout(),
        height=300
    )

    st.plotly_chart(
        fig_heat,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


# =========================================================
# 17. RISK ANALYTICS
# =========================================================

with tab_risk:

    r1, r2, r3, r4 = st.columns(4)

    risk_metrics = [

        (
            "BEST TRADE",
            f"₹{best_trade:,.0f}"
        ),

        (
            "WORST TRADE",
            f"₹{worst_trade:,.0f}"
        ),

        (
            "STOP-LOSS RATE",
            f"{stop_loss_rate:.1f}%"
        ),

        (
            "AVG CYCLE P&L",
            f"₹{avg_pnl:,.0f}"
        )
    ]

    for col, (label, value) in zip(
        [r1, r2, r3, r4],
        risk_metrics
    ):

        with col:

            st.markdown(
                f"""
                <div class="info-card">

                    <div class="info-title">
                        {label}
                    </div>

                    <div class="info-value">
                        {value}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    st.markdown(
        '<div class="section-header">'
        'Cycle ROM Distribution'
        '</div>',
        unsafe_allow_html=True
    )

    fig_rom = go.Figure()

    fig_rom.add_trace(

        go.Histogram(
            x=ledger_df["ROM_%"],
            nbinsx=20,
            name="Cycle ROM",
            marker_line_width=0.5,
            hovertemplate=
                "ROM: %{x:.2f}%"
                "<extra></extra>"
        )
    )

    fig_rom.add_vline(
        x=0,
        line_dash="dash",
        line_width=1.5
    )

    fig_rom.add_vline(
        x=avg_rom,
        line_dash="dot",
        line_width=1.5,
        annotation_text=
            f"Mean {avg_rom:.2f}%"
    )

    fig_rom.update_layout(
        **base_layout(),
        height=370,
        xaxis_title="Cycle ROM %",
        yaxis_title="Number of Cycles"
    )

    st.plotly_chart(
        fig_rom,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


    # ---------------------------------------------
    # Drawdown profile
    # ---------------------------------------------

    st.markdown(
        '<div class="section-header">'
        'Drawdown Profile'
        '</div>',
        unsafe_allow_html=True
    )

    fig_dd = go.Figure()

    fig_dd.add_trace(

        go.Bar(
            x=ledger_df["Exit_Date"],
            y=ledger_df["Drawdown_INR"],
            name="Drawdown",
            marker_color=RED,
            hovertemplate=
                "%{x}<br>"
                "Drawdown: ₹%{y:,.0f}"
                "<extra></extra>"
        )
    )

    fig_dd.update_layout(
        **base_layout(),
        height=340,
        yaxis_title="Drawdown ₹"
    )

    st.plotly_chart(
        fig_dd,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


# =========================================================
# 18. TRADE ANALYTICS
# =========================================================

with tab_trades:

    left, right = st.columns(2)

    # ---------------------------------------------
    # P&L distribution
    # ---------------------------------------------

    with left:

        st.markdown(
            '<div class="section-header">'
            'Cycle P&L Distribution'
            '</div>',
            unsafe_allow_html=True
        )

        fig_pnl = go.Figure()

        fig_pnl.add_trace(

            go.Histogram(
                x=ledger_df["Net_PnL"],
                nbinsx=20,
                name="P&L"
            )
        )

        fig_pnl.add_vline(
            x=0,
            line_dash="dash"
        )

        fig_pnl.update_layout(
            **base_layout(),
            height=360,
            xaxis_title="Net P&L ₹",
            yaxis_title="Cycles"
        )

        st.plotly_chart(
            fig_pnl,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


    # ---------------------------------------------
    # Spot move vs P&L
    # ---------------------------------------------

    with right:

        st.markdown(
            '<div class="section-header">'
            'Spot Move vs Strategy P&L'
            '</div>',
            unsafe_allow_html=True
        )

        fig_scatter = go.Figure()

        fig_scatter.add_trace(

            go.Scatter(
                x=ledger_df["Spot_Move_%"],
                y=ledger_df["Net_PnL"],
                mode="markers",
                marker=dict(
                    size=9,
                    opacity=0.8
                ),
                text=ledger_df["Exit_Reason"],
                hovertemplate=
                    "Spot Move: %{x:.2f}%<br>"
                    "Net P&L: ₹%{y:,.0f}<br>"
                    "Exit: %{text}"
                    "<extra></extra>"
            )
        )

        fig_scatter.add_hline(
            y=0,
            line_dash="dash"
        )

        fig_scatter.add_vline(
            x=0,
            line_dash="dash"
        )

        fig_scatter.update_layout(
            **base_layout(),
            height=360,
            xaxis_title="Spot Move %",
            yaxis_title="Net P&L ₹"
        )

        st.plotly_chart(
            fig_scatter,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


    # ---------------------------------------------
    # Premium capture
    # ---------------------------------------------

    st.markdown(
        '<div class="section-header">'
        'Premium Capture'
        '</div>',
        unsafe_allow_html=True
    )

    ledger_df["Premium_Capture_%"] = np.where(

        ledger_df["Sold_Prem"] > 0,

        (
            ledger_df["Sold_Prem"]
            - ledger_df["Exit_Prem"]
        )
        / ledger_df["Sold_Prem"]
        * 100,

        np.nan
    )

    fig_capture = go.Figure()

    fig_capture.add_trace(

        go.Scatter(
            x=ledger_df["Exit_Date"],
            y=ledger_df[
                "Premium_Capture_%"
            ],
            mode="lines+markers",
            name="Premium Capture",
            line=dict(
                color=PURPLE,
                width=2
            ),
            hovertemplate=
                "%{x}<br>"
                "Capture: %{y:.1f}%"
                "<extra></extra>"
        )
    )

    fig_capture.add_hline(
        y=0,
        line_dash="dash"
    )

    fig_capture.update_layout(
        **base_layout(),
        height=350,
        yaxis_title="Premium Capture %"
    )

    st.plotly_chart(
        fig_capture,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


    # ---------------------------------------------
    # Exit reason
    # ---------------------------------------------

    exit_counts = (
        ledger_df["Exit_Reason"]
        .str.contains(
            "Stop Loss",
            na=False
        )
        .map({
            True: "Stop Loss",
            False: "Scheduled Roll"
        })
        .value_counts()
    )

    fig_exit = go.Figure(

        go.Pie(
            labels=exit_counts.index,
            values=exit_counts.values,
            hole=0.60,
            textinfo="label+percent",
            hovertemplate=
                "%{label}<br>"
                "Cycles: %{value}"
                "<extra></extra>"
        )
    )

    fig_exit.update_layout(
        **base_layout(),
        height=330,
        title="Exit Reason Mix"
    )

    st.plotly_chart(
        fig_exit,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


# =========================================================
# 19. AUDIT LEDGER
# =========================================================

with tab_audit:

    st.markdown(
        '<div class="section-header">'
        'Complete Trade Audit Ledger'
        '</div>',
        unsafe_allow_html=True
    )

    display_cols = [

        "Entry_Date",
        "Exit_Date",
        "Expiry_Date",

        "Spot_Entry",
        "Spot_Exit",
        "Spot_Move_%",

        "Put_Strike",
        "Call_Strike",

        "Sold_Prem",
        "Exit_Prem",

        "Gross_PnL",
        "Transaction_Cost",
        "Net_PnL",

        "Allocated_Margin",
        "ROM_%",

        "Days_Held",
        "Exit_Reason"
    ]

    display_df = ledger_df[
        display_cols
    ].copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        height=580,
        column_config={

            "Spot_Entry":
                st.column_config.NumberColumn(
                    "Entry Spot",
                    format="₹%.2f"
                ),

            "Spot_Exit":
                st.column_config.NumberColumn(
                    "Exit Spot",
                    format="₹%.2f"
                ),

            "Spot_Move_%":
                st.column_config.NumberColumn(
                    "Spot Move",
                    format="%.2f%%"
                ),

            "Put_Strike":
                st.column_config.NumberColumn(
                    "Put Strike",
                    format="₹%.0f"
                ),

            "Call_Strike":
                st.column_config.NumberColumn(
                    "Call Strike",
                    format="₹%.0f"
                ),

            "Sold_Prem":
                st.column_config.NumberColumn(
                    "Sold Premium",
                    format="₹%.2f"
                ),

            "Exit_Prem":
                st.column_config.NumberColumn(
                    "Exit Premium",
                    format="₹%.2f"
                ),

            "Gross_PnL":
                st.column_config.NumberColumn(
                    "Gross P&L",
                    format="₹%.0f"
                ),

            "Transaction_Cost":
                st.column_config.NumberColumn(
                    "Costs",
                    format="₹%.0f"
                ),

            "Net_PnL":
                st.column_config.NumberColumn(
                    "Net P&L",
                    format="₹%.0f"
                ),

            "Allocated_Margin":
                st.column_config.NumberColumn(
                    "Margin",
                    format="₹%.0f"
                ),

            "ROM_%":
                st.column_config.NumberColumn(
                    "ROM",
                    format="%.2f%%"
                )
        }
    )


    # ---------------------------------------------
    # Export
    # ---------------------------------------------

    csv_data = ledger_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="↓ Download Trade Ledger CSV",
        data=csv_data,
        file_name=(
            f"{selected_symbol.replace('.NS','')}"
            "_strangle_backtest.csv"
        ),
        mime="text/csv"
    )


# =========================================================
# 20. MODEL ASSUMPTIONS
# =========================================================

st.divider()

with st.expander(
    "Model Assumptions & Methodology"
):

    st.markdown(
        f"""
### Strategy

- Entry: 15th calendar day or next available trading session
- Exit: 15th of following month or earlier stop-loss
- Short Put: approximately {y_pct:.1f}% OTM
- Short Call: approximately {y_pct:.1f}% OTM
- Expiry: last Thursday of following month
- Lot size: {lot_size:,}
- Strike interval: ₹{strike_step:,.0f}

### Volatility

30-day rolling realized volatility is annualized using:

`Rolling Std(Log Returns) × √252`

The model then applies an IV markup of:

`{iv_multiplier:.2f} × realized volatility`

### Pricing

European Black-Scholes pricing is used to estimate option premiums.

### Risk

Margin is approximated as:

`Spot × Lot Size × Margin %`

Current assumed margin:

`{margin_pct:.1f}%`

### Friction

Transaction friction is applied to:

`Entry Premium + Exit Premium`

Current assumption:

`{friction:.1f}%`

### Important

This is a **theoretical options backtest**.

It does not use historical NSE option-chain bid/ask prices, actual implied volatility,
actual SPAN files, exposure-margin files, slippage, liquidity constraints,
or intraday option prices.

Therefore the results should be interpreted as a model simulation rather than
an execution-grade historical options backtest.
        """
    )
