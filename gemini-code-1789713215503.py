"""
NSE Nifty 50 Single-Stock Short Strangle Backtesting Engine
Framework: Streamlit, Pandas, NumPy, SciPy, Plotly, yfinance

Strategy Mechanics:
  1) Entry: 15th calendar day of Month M (or next valid trading session).
  2) Sell 1 Lot Put at (X - y%) rounded to nearest exchange strike step (Month M+1 expiry).
  3) Sell 1 Lot Call at (X + y%) rounded to nearest exchange strike step (Month M+1 expiry).
  4) Exit: 15th calendar day of Month M+1 (or next valid trading session).
  5) Includes transaction friction, statutory STT (0.1%), stop-loss thresholds, and margin tracking.
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
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="NSE Stock Options Strangle Backtester",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Preset Nifty 50 Single-Stock F&O Specifications
# ---------------------------------------------------------
NIFTY_FNO_DIRECTORY = {
    "RELIANCE.NS": {"name": "Reliance Industries Ltd", "lot": 250, "strike_step": 20.0},
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
# Calendar & Valuation Functions
# ---------------------------------------------------------
def get_last_thursday(year: int, month: int) -> datetime.date:
    """Calculates the standard monthly derivative expiry (last Thursday) on the NSE."""
    last_day = calendar.monthrange(year, month)[1]
    date_cursor = datetime.date(year, month, last_day)
    day_offset = (date_cursor.weekday() - 3) % 7
    return date_cursor - datetime.timedelta(days=day_offset)

def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """
    Computes theoretical European option price via Black-Scholes-Merton formulation.
    Handles T <= 0 boundaries through intrinsic value convergence.
    """
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
    """Fetches daily adjusted OHLCV data from Yahoo Finance with volatility padding."""
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

# ---------------------------------------------------------
# Simulation Algorithm
# ---------------------------------------------------------
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
    """Executes the monthly mid-cycle short strangle strategy over the lookback horizon."""
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
        exit_reason = "Scheduled Roll (15th)"
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
                exit_reason = f"Stop Loss Trigger ({stop_loss_mult:.1f}x)"
                break
        else:
            pricing_vol_exit = float(df.loc[exit_date, 'Realized_Vol']) * iv_multiplier
            dte_exit = max(0, (contract_expiry - exit_date).days)
            T_exit = dte_exit / 365.0
            put_exit_prem = black_scholes_price(S_exit, K_put, T_exit, risk_free_rate, pricing_vol_exit, 'put')
            call_exit_prem = black_scholes_price(S_exit, K_call, T_exit, risk_free_rate, pricing_vol_exit, 'call')
            
        final_debit = put_exit_prem + call_exit_prem
        gross_pnl_share = initial_credit - final_debit
        
        total_turnover = initial_credit + final_debit
        frictional_deduction = total_turnover * (transaction_cost_pct / 100.0)
        net_pnl_share = gross_pnl_share - frictional_deduction
        total_cycle_pnl = net_pnl_share * lot_size
        
        gross_notional = S_entry * lot_size
        allocated_margin = gross_notional * 0.22
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
            "Put_Entry_Price": put_entry_prem,
            "Call_Entry_Price": call_entry_prem,
            "Total_Sold_Premium": initial_credit,
            "Total_Closed_Premium": final_debit,
            "Net_PnL_Per_Share": net_pnl_share,
            "Total_Net_PnL": total_cycle_pnl,
            "Margin_Allocated": allocated_margin,
            "Return_On_Margin_%": return_on_margin,
            "Exit_Mode": exit_reason
        })
        
        if curr_month == 12:
            curr_year += 1
            curr_month = 1
        else:
            curr_month += 1
            
    return pd.DataFrame(trade_ledger), df

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.markdown("<h2 style='text-align: left;'>NSE Nifty 50 Short Strangle Backtesting Engine</h2>", unsafe_allow_html=True)
st.markdown(
    "Quantitative testbed evaluating systematic 30-day rolling short strangles on Indian single-stock derivatives. "
    "Trades enter on the 15th of month M and close on the 15th of month M+1, modeling discrete exchange strikes, "
    "statutory STT, transaction costs, and physical delivery avoidance."
)

st.sidebar.header("Asset & Contract Selection")
selected_symbol = st.sidebar.selectbox(
    "Select Nifty 50 Constituent",
    options=list(NIFTY_FNO_DIRECTORY.keys()),
    format_func=lambda x: f"{x} — {NIFTY_FNO_DIRECTORY[x]['name']}"
)

configured_lot = NIFTY_FNO_DIRECTORY[selected_symbol]["lot"]
configured_step = NIFTY_FNO_DIRECTORY[selected_symbol]["strike_step"]

st.sidebar.subheader("Strategy Configuration")
y_pct_slider = st.sidebar.slider("OTM Distance (y %)", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
strike_step_param = st.sidebar.number_input("NSE Strike Step (INR)", min_value=0.5, max_value=200.0, value=float(configured_step), step=1.0)
lot_size_param = st.sidebar.number_input("Exchange Lot Size (Shares)", min_value=1, max_value=20000, value=int(configured_lot), step=25)
lookback_years_param = st.sidebar.slider("Historical Window (Years)", min_value=1, max_value=4, value=3)

st.sidebar.subheader("Risk & Cost Parameters")
iv_markup_param = st.sidebar.slider("IV / Realized Vol Multiplier", min_value=1.0, max_value=1.40, value=1.15, step=0.01)
rf_rate_param = st.sidebar.slider("Indian Risk-Free Rate (%)", min_value=4.0, max_value=9.0, value=6.75, step=0.25) / 100.0
stop_loss_param = st.sidebar.slider("Stop Loss Multiple (x Credit, 0 = Off)", min_value=0.0, max_value=4.0, value=2.5, step=0.5)
friction_param = st.sidebar.slider("Friction & Taxes (% of Premium)", min_value=0.0, max_value=3.0, value=1.2, step=0.1)

# ---------------------------------------------------------
# Execution & Analytics
# ---------------------------------------------------------
with st.spinner(f"Fetching data and running backtest for {selected_symbol}..."):
    stock_history = fetch_historical_stock_data(selected_symbol, years=lookback_years_param)

if stock_history.empty:
    st.error(f"Failed to fetch market data for {selected_symbol}. Please check connection or ticker validity.")
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
    st.warning("Insufficient data generated across the selected lookback range.")
    st.stop()

ledger_df['Cumulative_PnL'] = ledger_df['Total_Net_PnL'].cumsum()
ledger_df['Peak_PnL'] = ledger_df['Cumulative_PnL'].cummax()
ledger_df['Drawdown_INR'] = ledger_df['Cumulative_PnL'] - ledger_df['Peak_PnL']

total_executions = len(ledger_df)
winning_executions = len(ledger_df[ledger_df['Total_Net_PnL'] > 0])
win_rate_pct = (winning_executions / total_executions) * 100.0 if total_executions > 0 else 0.0
net_cumulative_profit = ledger_df['Total_Net_PnL'].sum()

gross_wins = ledger_df[ledger_df['Total_Net_PnL'] > 0]['Total_Net_PnL'].sum()
gross_losses = abs(ledger_df[ledger_df['Total_Net_PnL'] < 0]['Total_Net_PnL'].sum())
profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else np.nan
peak_drawdown = abs(ledger_df['Drawdown_INR'].min())
mean_margin = ledger_df['Margin_Allocated'].mean()
annualized_rom = (net_cumulative_profit / mean_margin) / lookback_years_param * 100.0 if mean_margin > 0 else 0.0

# ---------------------------------------------------------
# Dashboard Display
# ---------------------------------------------------------
st.markdown("### Strategy Performance Overview")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Cumulative Net PnL", f"₹{net_cumulative_profit:,.0f}")
kpi2.metric("Win Rate", f"{win_rate_pct:.1f}%", f"{winning_executions}/{total_executions} Wins")
kpi3.metric("Profit Factor", f"{profit_factor:.2f}" if not np.isnan(profit_factor) else "N/A")
kpi4.metric("Max Strategy Drawdown", f"₹{peak_drawdown:,.0f}")
kpi5.metric("Annualized Return on Margin", f"{annualized_rom:.1f}%")

st.markdown("---")

tab_equity, tab_cycles, tab_underlying = st.tabs(["Equity Curve & Drawdowns", "Cycle Distribution", "Price Action vs. Strikes"])

with tab_equity:
    fig_curve = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Cumulative Net Strategy Realized Profit (INR)", "Underwater Strategy Drawdown (INR)"),
        row_heights=[0.7, 0.3]
    )
    fig_curve.add_trace(
        go.Scatter(
            x=ledger_df['Exit_Date'],
            y=ledger_df['Cumulative_PnL'],
            mode='lines+markers',
            name='Net Cumulative PnL',
            line=dict(color='#00CC96', width=2.5),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    fig_curve.add_trace(
        go.Scatter(
            x=ledger_df['Exit_Date'],
            y=ledger_df['Drawdown_INR'],
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='#EF553B', width=1.5),
            fillcolor='rgba(239, 85, 59, 0.2)'
        ),
        row=2, col=1
    )
    fig_curve.update_layout(height=520, template='plotly_dark', margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    st.plotly_chart(fig_curve, use_container_width=True)

with tab_cycles:
    cycle_colors = ['#00CC96' if pnl > 0 else '#EF553B' for pnl in ledger_df['Total_Net_PnL']]
    fig_bar = go.Figure(
        data=[
            go.Bar(
                x=ledger_df['Exit_Date'],
                y=ledger_df['Total_Net_PnL'],
                marker_color=cycle_colors,
                text=[f"₹{pnl:,.0f}" for pnl in ledger_df['Total_Net_PnL']],
                textposition='auto'
            )
        ]
    )
    fig_bar.update_layout(
        title="Cycle-by-Cycle Net Realized Profit/Loss (INR)",
        xaxis_title="Roll Exit Date",
        yaxis_title="Net PnL (INR)",
        height=450,
        template='plotly_dark',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with tab_underlying:
    fig_scatter = go.Figure()
    fig_scatter.add_trace(
        go.Scatter(
            x=enriched_stock_df.index,
            y=enriched_stock_df['Close'],
            mode='lines',
            name='Stock Close',
            line=dict(color='#636EFA', width=1.5)
        )
    )
    fig_scatter.add_trace(
        go.Scatter(
            x=ledger_df['Entry_Date'],
            y=ledger_df['Call_Strike'],
            mode='markers',
            name='Sold Call Strike',
            marker=dict(symbol='triangle-up', size=8, color='#FFA15A')
        )
    )
    fig_scatter.add_trace(
        go.Scatter(
            x=ledger_df['Entry_Date'],
            y=ledger_df['Put_Strike'],
            mode='markers',
            name='Sold Put Strike',
            marker=dict(symbol='triangle-down', size=8, color='#AB63FA')
        )
    )
    fig_scatter.update_layout(
        title=f"{selected_symbol} Spot Trajectory vs. Strangle Strike Grid",
        xaxis_title="Date",
        yaxis_title="Stock Price (INR)",
        height=480,
        template='plotly_dark',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------
# Audit Ledger
# ---------------------------------------------------------
st.markdown("### Execution Audit Ledger")
view_df = ledger_df[[
    "Entry_Date", "Exit_Date", "Spot_Entry", "Spot_Exit", "Spot_Move_%",
    "Put_Strike", "Call_Strike", "Total_Sold_Premium", "Total_Closed_Premium",
    "Net_PnL_Per_Share", "Total_Net_PnL", "Return_On_Margin_%", "Exit_Mode"
]].copy()

view_df.rename(columns={
    "Spot_Entry": "Entry Spot",
    "Spot_Exit": "Exit Spot",
    "Spot_Move_%": "Move %",
    "Put_Strike": "Put Strike",
    "Call_Strike": "Call Strike",
    "Total_Sold_Premium": "Sold Prem",
    "Total_Closed_Premium": "Exit Prem",
    "Net_PnL_Per_Share": "Net PnL/Sh",
    "Total_Net_PnL": "Net PnL (₹)",
    "Return_On_Margin_%": "ROM %",
    "Exit_Mode": "Trigger"
}, inplace=True)

st.dataframe(
    view_df.style.format({
        "Entry Spot": "₹{:.2f}",
        "Exit Spot": "₹{:.2f}",
        "Move %": "{:+.2f}%",
        "Put Strike": "{:.1f}",
        "Call Strike": "{:.1f}",
        "Sold Prem": "₹{:.2f}",
        "Exit Prem": "₹{:.2f}",
        "Net PnL/Sh": "₹{:.2f}",
        "Net PnL (₹)": "₹{:,.2f}",
        "ROM %": "{:+.2f}%"
    }),
    use_container_width=True
)

st.download_button(
    label="Download Audit Ledger CSV",
    data=ledger_df.to_csv(index=False).encode('utf-8'),
    file_name=f"{selected_symbol}_strangle_audit.csv",
    mime="text/csv"
)