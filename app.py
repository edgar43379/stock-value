"""
The "Smart" Intrinsic Value Calculator
Features:
- yfinance (Price/Shares) + Manual Overrides (FCF/Debt)
- Automatic Sector Detection + Dynamic Default Assumptions by Sector
- Historical Reality Check: 5-Year Revenue CAGR from stock.financials
- Sensitivity Analysis: 3x3 matrix (Growth ±1%, Discount ±0.5%)
- Export to CSV: Download Model from sidebar ({ticker}_valuation.csv)
"""

import streamlit as st
import yfinance as yf
import pandas as pd

# Page config
st.set_page_config(
    page_title="Smart Valuation Tool",
    page_icon="🧠",
    layout="wide",
)

# --- 1. SESSION STATE SETUP ---
# We store defaults here so sliders can update automatically
if "fcf" not in st.session_state:
    st.session_state.fcf = 0.0
if "shares" not in st.session_state:
    st.session_state.shares = 0.0
if "price" not in st.session_state:
    st.session_state.price = 0.0
if "net_debt" not in st.session_state:
    st.session_state.net_debt = 0.0
if "sector" not in st.session_state:
    st.session_state.sector = "Unknown"
    
# DEFAULT RATES (These will change based on sector)
if "def_growth" not in st.session_state:
    st.session_state.def_growth = 5.0
if "def_discount" not in st.session_state:
    st.session_state.def_discount = 9.0
if "def_terminal" not in st.session_state:
    st.session_state.def_terminal = 2.5
if "ticker" not in st.session_state:
    st.session_state.ticker = "KO"
if "export_intrinsic_value" not in st.session_state:
    st.session_state.export_intrinsic_value = None
if "export_upside_pct" not in st.session_state:
    st.session_state.export_upside_pct = None
if "fcf_source" not in st.session_state:
    st.session_state.fcf_source = None  # "manual" | "fallback" | None
if "last_op_cash" not in st.session_state:
    st.session_state.last_op_cash = None
if "last_capex" not in st.session_state:
    st.session_state.last_capex = None

# --- 2. HELPERS & SECTOR LOGIC ---
def safe_float(value, default=0.0):
    """Handle None or invalid values; return default so the app never crashes on math."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_sector_defaults(sector):
    """Returns (Growth %, Discount %, Terminal %) based on yfinance sector.
    Fallback: Growth=5%, Discount=9%, Terminal=2.5%."""
    defaults = {
        "Technology": (12.0, 9.0, 3.0),           # High growth, med risk
        "Consumer Defensive": (4.5, 7.0, 2.5),    # Low growth, low risk (Coke, P&G)
        "Consumer Cyclical": (5.0, 10.0, 2.5),    # Econ sensitive (Ford, Amazon)
        "Financial Services": (5.0, 10.0, 2.5),   # Banks (JPM)
        "Healthcare": (5.0, 8.0, 2.5),            # Stable (JNJ)
        "Energy": (3.0, 9.0, 2.0),                # Slow, cyclical (Exxon)
        "Utilities": (3.0, 6.5, 2.5),             # Very safe (Duke Energy)
        "Industrials": (4.0, 9.0, 2.5),           # General (Boeing, CAT)
        "Real Estate": (3.5, 8.0, 2.5),           # REITs
        "Communication Services": (8.0, 9.0, 2.5),  # Google, Meta
    }
    return defaults.get(sector, (5.0, 9.0, 2.5))


def run_dcf(fcf, net_debt, shares, growth_pct, discount_pct, terminal_pct):
    """Run DCF and return intrinsic value per share. Rates in decimal (e.g. 0.09)."""
    if shares <= 0:
        return 0.0
    g, r, t = growth_pct, discount_pct, terminal_pct
    future_cfs = [fcf * ((1 + g) ** y) for y in range(1, 6)]
    discount_factors = [(1 + r) ** y for y in range(1, 6)]
    pv_cfs = sum(cf / df for cf, df in zip(future_cfs, discount_factors))
    final_fcf = future_cfs[-1]
    tv = (final_fcf * (1 + t)) / (r - t) if r > t else 0.0
    pv_tv = tv / ((1 + r) ** 5)
    ev = pv_cfs + pv_tv
    equity = ev - net_debt
    return equity / shares


def revenue_cagr_5y(ticker_symbol):
    """Fetch last 5 years revenue from yfinance financials; return CAGR as percentage or None."""
    try:
        stock = yf.Ticker(ticker_symbol)
        fin = stock.financials
        if fin is None or fin.empty:
            return None
        # Find revenue row (yfinance uses 'Total Revenue' or 'Revenue')
        rev_row = None
        for label in ("Total Revenue", "Revenue", "Total revenues"):
            if label in fin.index:
                rev_row = fin.loc[label]
                break
        if rev_row is None:
            return None
        # Columns are typically dates (newest first); we need oldest and newest for 5Y CAGR
        rev_row = rev_row.dropna()
        if len(rev_row) < 2:
            return None
        rev_row = rev_row.sort_index()
        oldest = safe_float(rev_row.iloc[0])
        newest = safe_float(rev_row.iloc[-1])
        if oldest <= 0:
            return None
        cagr = (newest / oldest) ** (1 / 5) - 1
        return cagr * 100
    except Exception:
        return None

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🧠 Smart Settings")
    st.info(f"Detected Sector: **{st.session_state.sector}**")

    st.markdown("---")
    st.caption("**Auto-Recommended Rates:**")
    st.write(f"Growth: {st.session_state.def_growth}%")
    st.write(f"Discount: {st.session_state.def_discount}%")
    st.write(f"Terminal: {st.session_state.def_terminal}%")

    st.markdown("---")
    st.subheader("Download Model")
    iv = st.session_state.export_intrinsic_value
    up = st.session_state.export_upside_pct
    ticker_safe = (st.session_state.ticker or "KO").strip() or "KO"
    csv_lines = [
        "Metric,Value",
        f"Ticker,{ticker_safe}",
        f"Price,{st.session_state.price}",
        f"FCF,{st.session_state.fcf}",
        f"Net Debt,{st.session_state.net_debt}",
        f"Shares,{st.session_state.shares}",
        f"Intrinsic Value,{iv if iv is not None else ''}",
        f"Upside/Downside %,{up if up is not None else ''}",
    ]
    csv_string = "\n".join(csv_lines)
    st.download_button(
        label="Download Model (CSV)",
        data=csv_string,
        file_name=f"{ticker_safe}_valuation.csv",
        mime="text/csv",
        key="download_valuation",
    )

# --- 4. MAIN APP ---
st.title("🧠 Smart Intrinsic Value Calculator")

# INPUT SECTION
st.header("1. Financial Inputs")
col1, col2 = st.columns([1, 2])

with col1:
    ticker = st.text_input("Stock Ticker", value=st.session_state.ticker, key="ticker_input").upper()
    st.session_state.ticker = ticker

    if st.button("Fetch Data (yfinance)"):
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info or {}

                # 1. Basics (safe fallbacks)
                st.session_state.price = safe_float(
                    info.get('currentPrice') or info.get('regularMarketPrice')
                )
                st.session_state.shares = safe_float(info.get('sharesOutstanding'))
                st.session_state.sector = info.get('sector') or "Unknown"

                # 2. Sector auto-detection → update slider defaults
                g, d, t = get_sector_defaults(st.session_state.sector)
                st.session_state.def_growth = g
                st.session_state.def_discount = d
                st.session_state.def_terminal = t

                # 3. Robust Unlevered FCF: Primary = operatingCashflow + capitalExpenditures (CapEx usually negative)
                operating_cash = info.get('operatingCashflow')
                capex = info.get('capitalExpenditures')
                st.session_state.last_op_cash = operating_cash
                st.session_state.last_capex = capex
                if operating_cash is not None and capex is not None:
                    st.session_state.fcf = safe_float(operating_cash + capex)
                    st.session_state.fcf_source = "manual"
                else:
                    raw_fcf = info.get('freeCashflow')
                    if raw_fcf is not None:
                        st.session_state.fcf = safe_float(raw_fcf)
                        st.session_state.fcf_source = "fallback"
                    else:
                        st.session_state.fcf = 0.0
                        st.session_state.fcf_source = "fallback"

                # 4. Net Debt: totalDebt - totalCash (missing keys → 0)
                total_debt = info.get('totalDebt')
                total_cash = info.get('totalCash')
                st.session_state.net_debt = safe_float(total_debt) - safe_float(total_cash)

                st.success(f"Loaded {ticker}! Sector: {st.session_state.sector}")

            except Exception as e:
                st.error(f"Could not load data for **{ticker}**. Check the ticker or try again. Details: {e}")

with col2:
    st.warning("⚠️ **Manual override:** Verify and adjust these numbers against the company's 10-K before relying on the valuation.")
    if st.session_state.fcf_source == "manual":
        st.info("✅ FCF was calculated as **Unlevered**: Operating Cash Flow + CapEx. You can overwrite below.")
    elif st.session_state.fcf_source == "fallback":
        st.warning("⚠️ FCF could not be calculated from Operating Cash Flow + CapEx; fallback or default was used. Check and overwrite if needed.")
    st.write("**Debug (raw from yfinance):** Operating Cash Flow =", st.session_state.last_op_cash, " | Capital Expenditures =", st.session_state.last_capex)
    st.caption("👇 Edit any field; values persist when you move sliders.")

    fcf_input = st.number_input(
        "Free Cash Flow ($)", value=float(st.session_state.fcf), format="%f", key="fcf"
    )
    net_debt_input = st.number_input(
        "Net Debt ($)", value=float(st.session_state.net_debt), format="%f", key="net_debt"
    )
    shares_input = st.number_input(
        "Shares Outstanding", value=float(st.session_state.shares), format="%f", key="shares"
    )
    price_input = st.number_input(
        "Current Price ($)", value=float(st.session_state.price), format="%f", key="price"
    )

st.markdown("---")

# VALUATION SECTION
st.header("2. Assumptions (Auto-filled by Sector)")

# Historical Reality Check (5-Year Revenue CAGR) — wrapped so app never crashes
try:
    cagr_pct = revenue_cagr_5y(st.session_state.ticker)
    if cagr_pct is not None:
        st.caption(f"📉 **5-Year Revenue CAGR:** **{cagr_pct:.1f}%**")
    else:
        st.caption("📉 **5-Year Revenue CAGR:** Historical data unavailable")
except Exception:
    st.caption("📉 **5-Year Revenue CAGR:** Historical data unavailable")

# SLIDERS (Now connected to session_state defaults)
col_a, col_b, col_c = st.columns(3)

with col_a:
    growth_rate = st.slider("Growth Rate (Next 5 Years)", 0.0, 30.0, st.session_state.def_growth, 0.5) / 100
with col_b:
    discount_rate = st.slider("Discount Rate (Risk)", 4.0, 15.0, st.session_state.def_discount, 0.5) / 100
with col_c:
    terminal_growth = st.slider("Terminal Growth", 0.0, 5.0, st.session_state.def_terminal, 0.1) / 100

# CALCULATIONS
if shares_input > 0:
    # 1. Project Cash Flows
    future_cash_flows = []
    discount_factors = []
    
    for year in range(1, 6):
        cf = fcf_input * ((1 + growth_rate) ** year)
        future_cash_flows.append(cf)
        discount_factors.append((1 + discount_rate) ** year)
        
    # 2. Terminal Value
    final_year_fcf = future_cash_flows[-1]
    tv = (final_year_fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    pv_tv = tv / ((1 + discount_rate) ** 5)
    
    # 3. Sum It Up
    enterprise_value = sum([cf / df for cf, df in zip(future_cash_flows, discount_factors)]) + pv_tv
    equity_value = enterprise_value - net_debt_input
    intrinsic_value = equity_value / shares_input
    
    # DISPLAY RESULTS
    st.header("3. Results")
    
    # The Bridge
    st.markdown(f"""
    #### Valuation Bridge
    | Metric | Value per Share |
    | :--- | :--- |
    | **Enterprise Value** | **${(enterprise_value/shares_input):.2f}** |
    | *(-) Net Debt* | *${(net_debt_input/shares_input):.2f}* |
    | **(=) Equity Value** | **${intrinsic_value:.2f}** |
    """)
    
    # Big Metric
    delta = intrinsic_value - price_input
    upside_pct = (delta / price_input) * 100 if price_input else 0.0
    st.session_state.export_intrinsic_value = intrinsic_value
    st.session_state.export_upside_pct = upside_pct

    st.metric(label="Intrinsic Value per Share", value=f"${intrinsic_value:.2f}", delta=f"{delta:.2f}")
    
    if delta > 0:
        st.success(f"✅ UNDERVALUED by {upside_pct:.1f}%")
    else:
        st.error(f"❌ OVERVALUED by {abs(upside_pct):.1f}%")

    # --- 4. Sensitivity Analysis ---
    st.header("4. Sensitivity Analysis")
    with st.expander("What-if: Intrinsic Value by Growth vs Discount Rate", expanded=True):
        growth_pct_user = growth_rate * 100
        discount_pct_user = discount_rate * 100
        growth_vals = [growth_pct_user - 1.0, growth_pct_user, growth_pct_user + 1.0]
        discount_vals = [discount_pct_user - 0.5, discount_pct_user, discount_pct_user + 0.5]
        rows = []
        for g in growth_vals:
            row = {}
            for d in discount_vals:
                iv = run_dcf(
                    fcf_input, net_debt_input, shares_input,
                    g / 100, d / 100, terminal_growth
                )
                row[f"Disc {d:.1f}%"] = round(iv, 2)
            rows.append(row)
        sens_df = pd.DataFrame(rows, index=[f"Growth {g:.1f}%" for g in growth_vals])
        st.caption("Rows: Growth ±1%; Columns: Discount ±0.5%. Best-case (max intrinsic value) highlighted.")
        st.dataframe(sens_df.style.highlight_max(axis=None).format("{:.2f}"), use_container_width=True)
else:
    st.session_state.export_intrinsic_value = None
    st.session_state.export_upside_pct = None
    st.warning("Enter valid data to see the valuation.")
