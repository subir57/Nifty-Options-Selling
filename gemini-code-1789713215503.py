"""
NSE Nifty 50 Single-Stock Short Strangle Backtesting Engine
Mobile & Desktop Optimized Architecture
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

# ---------------------------------------------------------
# Page Configuration & Viewport Optimization
# ---------------------------------------------------------
st.set_page_config(
    page_title="NSE Options Strangle Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto"
)

# ---------------------------------------------------------
# Responsive Custom CSS Injection
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Reduce default container padding on mobile */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        h2 {
            font-size: 1.4rem !important;
        }
        p {
            font-size: 0.9rem !important;
        }
    }

    /* Responsive KPI Metric Container */
    .kpi-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }

    .kpi-card {
        flex: 1 1 calc(20% - 0.75rem);
        min-width: 140px;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        box-sizing: border-box;
    }

    @media (max-width: 768px) {
        .kpi-card {
            flex: 1 1 calc(50% - 0.5rem);
            min-width: 130px;
            padding: 0.6rem 0.75rem;
        }
    }

    .kpi-title {
        font-size: 0.8rem;
        color: #9E9E9E;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .kpi-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #FFFFFF;
    }

    .kpi-sub {
        font-size: 0.75rem;
        color: #00CC96;
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Preset Nifty 50 Single-Stock F&O Specifications
# ---------------------------------------------------------
NIFTY_FNO_DIRECTORY = {
    "RELIANCE.NS": {"name": "Reliance Industries", "lot": 250, "strike_step": 20.0},
    "TCS.NS": {"name": "Tata Consultancy Services", "lot": 175, "strike_step": 50.0},
    "INFY.NS": {"name": "Infosys Ltd", "lot": 400, "strike_step": 20.0},
    "HDFCBANK.NS": {"name": "HDFC Bank Ltd", "lot": 550, "strike_step": 10.0},
    "ICICIBANK.NS": {"name": "ICICI Bank Ltd", "lot": 700, "strike_step": 10.0},
    "SBIN.NS": {"name": "State Bank of India", "lot": 750, "strike_step": 5.0},
    "BHARTIARTL.NS": {"name": "Bharti Airtel Ltd", "lot": 475, "strike_step": 10.0},
    "ITC.NS": {"name": "ITC Limited", "lot": 1600, "strike_step": 5.0},
    "LT.NS": {"name": "Larsen & Toubro Ltd", "lot": 150, "strike_step": 25.0},
    "TATAMOTORS.NS": {"name": "Tata Motors Ltd", "lot": 550, "strike_step": 10.0}
}

# ---------------------------------------------------------
# Mathematical Valuation Engine
# ---------------------------------------------------------
def get_last_thursday(year: int, month: int) -> datetime.date:
    last_day = calendar.monthrange(year, month)[1]
    date_cursor = datetime.date(year, month, last_day)
    day_offset = (date_cursor.weekday() - 3) % 7
    return date_cursor - datetime.timedelta(days=day_offset)

def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    if T <= 0.0001:
        if option_type.lower() == 'call':
            return max(0.0, float(S - K))
        else:
            return max(0.0, float(K - S))
            
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
    return max(0.0, float(price))

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_historical_stock_data(ticker: str, years: int = 3) -> pd.DataFrame:
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=years * 365 + 120)
    
    raw_df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if raw_df.empty:
        return pd.DataFrame()
        
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
        
    cleaned_df = raw_df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    cleaned_df.index = pd.to_datetime(cleaned_df.index).date
    return cleaned_df

def execute_strangle_backtest(
    data: pd.DataFrame,
    y_pct: float,
    strike_step: float,
    lot_size: int,
    risk_free_rate: float,
    vol_lookback: int,
    iv_multiplier: float,
    stop_loss_mult: float,
    transaction_cost_pct: float,
    lookback_years: int
):
    df = data.copy()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Realized_Vol'] = df['Log_Return'].rolling(window=vol_lookback).std() * np.sqrt(252)
    df['Realized_Vol'] = df['Realized_Vol'].bfill().clip(lower=0.12)
    
    trading_dates = sorted(list(df.index))
    if not trading_dates:
        return pd.DataFrame(), pd.DataFrame()
        
    start_boundary = datetime.date.today() - datetime.timedelta(days=lookback_years * 365)
    first_usable_idx = vol_lookback
    while first_usable_idx < len(trading_dates) and trading_dates[first_usable_idx] < start_boundary:
        first_usable_idx += 1
        
    if first_usable_idx >= len(trading_dates):
        return pd.DataFrame(), pd.DataFrame()
        
    seed_date = trading_dates[first_usable_idx]
    last_available_date = trading_dates[-1]
    
    curr_year = seed_date.year
    curr_month = seed_date.month
    trade_ledger = []
    
    while True:
        target_entry_date = datetime.date(curr_year, curr_month, 15)
        if target_entry_date > last_available_date:
            break
            
        candidate_entries = [d for d in trading_dates if d >= target_entry_date]
        if not candidate_entries:
            break
        entry_date = candidate_entries[0]
        
        if curr_month == 12:
            next_year = curr_year + 1
            next_month = 1
        else:
            next_year = curr_year
            next_month = curr_month + 1
            
        target_exit_date = datetime.date(next_year, next_month, 15)
        candidate_exits = [d for d in trading_dates if d >= target_exit_date]
        if not candidate_exits:
            break
        exit_date = candidate_exits[0]
        
        contract_expiry = get_last_thursday(next_year, next_month)
        S_entry = float(df.loc[entry_date, 'Close'])
        pricing_vol_entry = float(df.loc[entry_date, 'Realized_Vol']) * iv_multiplier
        
        raw_put = S_entry * (1.0 - y_pct / 100.0)
        raw_call = S_entry * (1.0 + y_pct / 100.0)
        
        K_put = round(raw_put / strike_step) * strike_step
        K_call = round(raw_call / strike_step) * strike_step
        
        dte_entry = max(1, (contract_expiry - entry_date).days)
        T_entry = dte_entry / 365.0
        
        put_entry_prem = black_scholes_price(S_entry, K_put, T_entry, risk_free_rate, pricing_vol_entry, 'put')
        call_entry_prem = black_scholes_price(S_entry, K_call, T_entry, risk_free_rate, pricing_vol_entry, 'call')
        initial_credit = put_entry_prem + call_entry_prem
        
        intermediate_dates = [d for d in trading_dates if entry_date < d <= exit_date]
        actual_exit_date = exit_date
        exit_reason = "Scheduled Roll"
        put_exit_prem = 0.0
        call_exit_prem = 0.0
        S_exit = float(df.loc[exit_date, 'Close'])
        
        for intermediate_date in intermediate_dates:
            S_int = float(df.loc[intermediate_date, 'Close'])
            vol_int = float(df.loc[intermediate_date, 'Realized_Vol']) * iv_multiplier
            dte_int = max(0, (contract_expiry - intermediate_date).days)
            T_int = dte_int / 365.0
            
            p_val = black_scholes_price(S_int, K_put, T_int, risk_free_rate, vol_int, 'put')
            c_val = black_scholes_price(S_int, K_call, T_int, risk_free_rate, vol_int, 'call')
            combined_cost = p_val + c_val
            
            if stop_loss_mult > 0 and combined_cost >= (initial_credit * stop_loss_mult):
                actual_exit_date = intermediate_date
                S_exit = S_int
                put_exit_prem = p_val
                call_exit_prem = c_val
                exit_reason = f"Stop Loss ({stop_loss_mult:.1f}x)"
                break
        else:
            pricing_vol_exit = float(df.loc[exit_date, 'Realized_Vol']) * iv_multiplier
            dte_exit = max(0, (contract_expiry - exit_date).days)
            T_exit = dte_exit / 365.0
            put_exit_prem = black_scholes_price(S_exit, K_put, T_exit, risk_free_rate, pricing_vol_exit, 'put')
            call_exit_prem = black_scholes_price(S_exit, K_call, T_exit, risk_free_rate, pricing_vol_exit, 'call')
            
        final_debit = put_exit_prem + call_exit_prem
        gross_pnl_share = initial_credit - final_debit
        frictional_deduction = (initial_credit + final_debit) * (transaction_cost_pct / 100.0)
        net_pnl_share = gross_pnl_share - frictional_deduction
        total_cycle_pnl = net_pnl_share * lot_size
        allocated_margin = S_entry * lot_size * 0.22
        return_on_margin = (total_cycle_pnl / allocated_margin) * 100.0
        
        trade_ledger.append({
            "Entry_Date": entry_date,
            "Exit_Date": actual_exit_date,
            "Expiry_Date": contract_expiry,
            "Spot_Entry": S_entry,
            "Spot_Exit": S_exit,
            "Spot_Move_%": ((S_exit / S_entry) - 1.0) * 100.0,
            "Put_Strike": K_put,
            "Call_Strike": K_call,
            "Sold_Prem": initial_credit,
            "Exit_Prem": final_debit,
            "Net_PnL": total_cycle_pnl,
            "ROM_%": return_on_margin,
            "Exit_Reason": exit_reason
        })
        
        if curr_month == 12:
            curr_year += 1
            curr_month = 1
        else:
            curr_month += 1
            
    return pd.DataFrame(trade_ledger), df

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("Asset Selection")
selected_symbol = st.sidebar.selectbox(
    "Nifty 50 Constituent",
    options=list(NIFTY_FNO_DIRECTORY.keys()),
    format_func=lambda x: f"{x.replace('.NS','')} — {NIFTY_FNO_DIRECTORY[x]['name']}"
)

configured_lot = NIFTY_FNO_DIRECTORY[selected_symbol]["lot"]
configured_step = NIFTY_FNO_DIRECTORY[selected_symbol]["strike_step"]

with st.sidebar.expander("Strategy Parameters", expanded=True):
    y_pct_slider = st.slider("OTM Distance (y %)", 1.0, 15.0, 5.0, 0.5)
    strike_step_param = st.number_input("NSE Strike Step (₹)", 0.5, 200.0, float(configured_step), 1.0)
    lot_size_param = st.number_input("Exchange Lot Size", 1, 20000, int(configured_lot), 25)
    lookback_years_param = st.slider("Lookback (Years)", 1, 4, 3)

with st.sidebar.expander("Risk & Tax Rules", expanded=False):
    iv_markup_param = st.slider("IV Markup Multiplier", 1.0, 1.40, 1.15, 0.01)
    rf_rate_param = st.slider("Risk-Free Rate (%)", 4.0, 9.0, 6.75, 0.25) / 100.0
    stop_loss_param = st.slider("Stop Loss Multiple (x)", 0.0, 4.0, 2.5, 0.5)
    friction_param = st.slider("STT & Friction (%)", 0.0, 3.0, 1.2, 0.1)

# ---------------------------------------------------------
# Simulation Processing
# ---------------------------------------------------------
st.markdown("## Nifty 50 Short Strangle Engine")
st.caption("30-day mid-cycle options harvesting with systematic mid-month rolls.")

with st.spinner("Analyzing historical option cycles..."):
    stock_history = fetch_historical_stock_data(selected_symbol, years=lookback_years_param)

if stock_history.empty:
    st.error("Market data unavailable. Please verify connection.")
    st.stop()

ledger_df, enriched_stock_df = execute_strangle_backtest(
    data=stock_history,
    y_pct=y_pct_slider,
    strike_step=strike_step_param,
    lot_size=lot_size_param,
    risk_free_rate=rf_rate_param,
    vol_lookback=30,
    iv_multiplier=iv_markup_param,
    stop_loss_mult=stop_loss_param,
    transaction_cost_pct=friction_param,
    lookback_years=lookback_years_param
)

if ledger_df.empty:
    st.warning("No complete cycles generated for the selected lookback.")
    st.stop()

# Performance Metrics Calculation
ledger_df['Cumulative_PnL'] = ledger_df['Net_PnL'].cumsum()
ledger_df['Peak_PnL'] = ledger_df['Cumulative_PnL'].cummax()
ledger_df['Drawdown_INR'] = ledger_df['Cumulative_PnL'] - ledger_df['Peak_PnL']

total_cycles = len(ledger_df)
winning_cycles = len(ledger_df[ledger_df['Net_PnL'] > 0])
win_rate = (winning_cycles / total_cycles) * 100.0
cumulative_net_pnl = ledger_df['Net_PnL'].sum()

gross_wins = ledger_df[ledger_df['Net_PnL'] > 0]['Net_PnL'].sum()
gross_losses = abs(ledger_df[ledger_df['Net_PnL'] < 0]['Net_PnL'].sum())
profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else np.nan
max_dd = abs(ledger_df['Drawdown_INR'].min())

# ---------------------------------------------------------
# Responsive KPI Cards
# ---------------------------------------------------------
pnl_color = "#00CC96" if cumulative_net_pnl >= 0 else "#EF553B"

st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card">
        <div class="kpi-title">Net Profit</div>
        <div class="kpi-value" style="color: {pnl_color};">₹{cumulative_net_pnl:,.0f}</div>
        <div class="kpi-sub">{total_cycles} Total Cycles</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Win Rate</div>
        <div class="kpi-value">{win_rate:.1f}%</div>
        <div class="kpi-sub">{winning_cycles} Wins / {total_cycles - winning_cycles} Losses</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Profit Factor</div>
        <div class="kpi-value">{"N/A" if np.isnan(profit_factor) else f"{profit_factor:.2f}"}</div>
        <div class="kpi-sub">Win/Loss Ratio</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Max Drawdown</div>
        <div class="kpi-value" style="color: #EF553B;">₹{max_dd:,.0f}</div>
        <div class="kpi-sub">Peak-to-Trough</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Interactive Plotly Views (Touch & Desktop Friendly)
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Equity Curve", "📅 Cycle PnL", "📋 Audit Ledger"])

plotly_layout_defaults = dict(
    template='plotly_dark',
    autosize=True,
    margin=dict(l=10, r=10, t=30, b=20),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=10)
    ),
    hovermode="x unified"
)

with tab1:
    fig_equity = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.7, 0.3]
    )
    fig_equity.add_trace(
        go.Scatter(
            x=ledger_df['Exit_Date'],
            y=ledger_df['Cumulative_PnL'],
            mode='lines+markers',
            name='Cumulative PnL',
            line=dict(color='#00CC96', width=2),
            marker=dict(size=5)
        ),
        row=1, col=1
    )
    fig_equity.add_trace(
        go.Scatter(
            x=ledger_df['Exit_Date'],
            y=ledger_df['Drawdown_INR'],
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='#EF553B', width=1.5),
            fillcolor='rgba(239, 85, 59, 0.25)'
        ),
        row=2, col=1
    )
    fig_equity.update_layout(**plotly_layout_defaults, height=420)
    fig_equity.update_yaxes(title_text="PnL (₹)", row=1, col=1)
    fig_equity.update_yaxes(title_text="DD (₹)", row=2, col=1)
    st.plotly_chart(fig_equity, use_container_width=True, config={'displayModeBar': False, 'responsive': True})

with tab2:
    colors = ['#00CC96' if x > 0 else '#EF553B' for x in ledger_df['Net_PnL']]
    fig_bar = go.Figure(
        data=[
            go.Bar(
                x=ledger_df['Exit_Date'],
                y=ledger_df['Net_PnL'],
                marker_color=colors,
                hovertemplate="Exit: %{x}<br>Net PnL: ₹%{y:,.0f}<extra></extra>"
            )
        ]
    )
    fig_bar.update_layout(**plotly_layout_defaults, height=360)
    fig_bar.update_yaxes(title_text="Net PnL (₹)")
    st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False, 'responsive': True})

with tab3:
    # Modern st.dataframe with responsive column formatting
    display_df = ledger_df[[
        "Entry_Date", "Exit_Date", "Spot_Entry", "Spot_Exit", "Spot_Move_%",
        "Put_Strike", "Call_Strike", "Sold_Prem", "Exit_Prem", "Net_PnL", "ROM_%", "Exit_Reason"
    ]].copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Entry_Date": st.column_config.DateColumn("Entry Date", format="YYYY-MM-DD"),
            "Exit_Date": st.column_config.DateColumn("Exit Date", format="YYYY-MM-DD"),
            "Spot_Entry": st.column_config.NumberColumn("Entry Spot", format="₹%.2f"),
            "Spot_Exit": st.column_config.NumberColumn("Exit Spot", format="₹%.2f"),
            "Spot_Move_%": st.column_config.NumberColumn("Move", format="%+.1f%%"),
            "Put_Strike": st.column_config.NumberColumn("Put (K)", format="%.0f"),
            "Call_Strike": st.column_config.NumberColumn("Call (K)", format="%.0f"),
            "Sold_Prem": st.column_config.NumberColumn("Sold ₹", format="₹%.1f"),
            "Exit_Prem": st.column_config.NumberColumn("Exit ₹", format="₹%.1f"),
            "Net_PnL": st.column_config.NumberColumn("Net PnL", format="₹%.0f"),
            "ROM_%": st.column_config.NumberColumn("ROM", format="%+.1f%%"),
            "Exit_Reason": st.column_config.TextColumn("Trigger")
        }
    )

    st.download_button(
        label="📥 Download Audit Ledger (CSV)",
        data=ledger_df.to_csv(index=False).encode('utf-8'),
        file_name=f"{selected_symbol}_strangle_audit.csv",
        mime="text/csv",
        use_container_width=True
    )
