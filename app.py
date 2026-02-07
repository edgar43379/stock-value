"""
The Free Intrinsic Value Calculator
A 100% free, local stock valuation dashboard using Streamlit and yfinance.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Free Intrinsic Value Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for manual overrides
if "fcf" not in st.session_state:
    st.session_state.fcf = None
if "shares" not in st.session_state:
    st.session_state.shares = None
if "price" not in st.session_state:
    st.session_state.price = None
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = ""


def fetch_stock_data(ticker: str) -> tuple:
    """Fetch FCF, shares outstanding, and market price from yfinance."""
    try:
        stock = yf.Ticker(ticker.upper().strip())
        info = stock.info

        fcf = info.get("freeCashflow") or info.get("operatingCashflow")
        shares = info.get("sharesOutstanding")
        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")

        return (fcf, shares, price)
    except Exception:
        return (None, None, None)


def calculate_intrinsic_value(
    fcf: float,
    shares: float,
    net_debt: float,
    growth_rate: float,
    discount_rate: float,
    terminal_growth: float,
) -> tuple[float, float, float]:
    """
    DCF model: 5-year projection + terminal value.
    Returns (ev_per_share, net_debt_per_share, intrinsic_value_per_share).
    """
    if not fcf or not shares or fcf <= 0 or shares <= 0:
        return (0.0, 0.0, 0.0)

    growth = growth_rate / 100
    discount = discount_rate / 100
    term_growth = terminal_growth / 100
    net_debt_val = float(net_debt or 0)

    pv_cf = 0.0
    cf = float(fcf)
    for year in range(1, 6):
        cf *= 1 + growth
        pv_cf += cf / ((1 + discount) ** year)

    terminal_value = cf * (1 + term_growth) / (discount - term_growth)
    pv_terminal = terminal_value / ((1 + discount) ** 5)
    enterprise_value = pv_cf + pv_terminal
    equity_value = enterprise_value - net_debt_val

    ev_per_share = enterprise_value / shares
    net_debt_per_share = net_debt_val / shares
    intrinsic_value_per_share = equity_value / shares

    return (ev_per_share, net_debt_per_share, intrinsic_value_per_share)


# ============ SIDEBAR (Educational) ============
with st.sidebar:
    st.title("📚 Learn the Basics")
    st.markdown("---")
    st.subheader("Free Cash Flow (FCF)")
    st.markdown("""
    **Free Cash Flow** is the cash a company generates after paying for operations 
    and capital expenditures (like equipment or buildings).
    
    - **Why it matters:** It shows how much money is left for shareholders, 
    debt repayment, or reinvestment.
    - **Higher is better:** More FCF = more flexibility and potential dividends.
    """)

    st.markdown("---")
    st.subheader("Discount Rate (Risk)")
    st.markdown("""
    The **Discount Rate** converts future cash flows into today's value.
    
    - **Higher rate = more conservative:** You're saying future money is riskier.
    - **Typical range:** 8–12% for stable companies, 10–15% for riskier ones.
    - **Think of it as:** Your required return for taking on the investment risk.
    """)

    st.markdown("---")
    st.subheader("Growth Rate")
    st.markdown("""
    How fast you expect cash flow to grow in the next 5 years. 
    Be realistic—most companies grow 2–8% annually.
    """)

    st.markdown("---")
    st.subheader("Terminal Growth Rate")
    st.markdown("""
    Long-term perpetual growth after year 5. Usually 1–3%, 
    roughly in line with GDP growth.
    """)


# ============ MAIN CONTENT ============
st.title("📊 The Free Intrinsic Value Calculator")
st.caption("100% free • No API keys • All data via yfinance")

st.markdown("---")

# ----- 1. INPUT SECTION -----
st.header("1️⃣ Input")
col1, col2, col3 = st.columns(3)

with col1:
    ticker = st.text_input("Stock Ticker", value="KO", placeholder="e.g., AAPL, MSFT, KO").strip().upper()
    if ticker:
        if st.session_state.last_ticker != ticker:
            st.session_state.fcf = None
            st.session_state.shares = None
            st.session_state.price = None
            st.session_state.last_ticker = ticker
        if st.button("Fetch Data"):
            fcf_raw, shares_raw, price_raw = fetch_stock_data(ticker)
            st.session_state.fcf = fcf_raw
            st.session_state.shares = shares_raw
            st.session_state.price = price_raw
            st.session_state.last_ticker = ticker
            st.rerun()

with col2:
    fcf_val = st.session_state.fcf
    shares_val = st.session_state.shares
    price_val = st.session_state.price

    # Handle None/missing - default to 0 and allow manual input
    fcf_display = fcf_val if fcf_val is not None else 0
    shares_display = shares_val if shares_val is not None else 0
    price_display = price_val if price_val is not None else 0

    def safe_int(val):
        if val is None or val == 0:
            return 0
        return int(val) if val == int(val) else int(val)

    fcf_input = st.number_input(
        "Free Cash Flow ($)",
        value=safe_int(fcf_display),
        step=1_000_000,
        format="%d",
        help="Manually enter if yfinance returns missing data.",
    )
    shares_input = st.number_input(
        "Shares Outstanding",
        value=safe_int(shares_display),
        step=1_000_000,
        format="%d",
        help="Manually enter if yfinance returns missing data.",
    )
    net_debt_input = st.number_input(
        "Net Debt ($)",
        value=0,
        step=1_000_000,
        format="%d",
        help="Total Debt minus Cash. If Cash > Debt, enter a negative number.",
    )

with col3:
    market_price = st.number_input(
        "Market Price ($)",
        value=float(price_display) if price_display else 0.0,
        step=0.01,
        format="%.2f",
        help="Current stock price. Edit if needed.",
    )
    st.metric("Current Market Price", f"${market_price:,.2f}")

st.markdown("---")

# ----- 2. ASSUMPTIONS SECTION -----
st.header("2️⃣ Assumptions (The Learning Part)")
col_a, col_b, col_c = st.columns(3)

with col_a:
    growth_rate = st.slider(
        "Growth Rate (Next 5 Years) (%)",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help="Expected annual FCF growth.",
    )

with col_b:
    discount_rate = st.slider(
        "Discount Rate (Risk) (%)",
        min_value=5.0,
        max_value=15.0,
        value=10.0,
        step=0.5,
        help="Your required return / risk adjustment.",
    )

with col_c:
    terminal_growth = st.slider(
        "Terminal Growth Rate (%)",
        min_value=1.0,
        max_value=4.0,
        value=2.5,
        step=0.25,
        help="Perpetual growth after year 5.",
    )

st.markdown("---")

# ----- 3. OUTPUT -----
st.header("3️⃣ Result")

ev_per_share, net_debt_per_share, intrinsic_value = calculate_intrinsic_value(
    fcf_input, shares_input, net_debt_input, growth_rate, discount_rate, terminal_growth
)

if intrinsic_value > 0:
    st.caption("**Valuation breakdown**")
    st.markdown(
        f"Enterprise Value per Share: **${ev_per_share:,.2f}**  \n"
        f"(-) Net Debt per Share: **${net_debt_per_share:,.2f}**  \n"
        f"(=) Equity Value per Share: **${intrinsic_value:,.2f}**"
    )
    st.markdown("")
    st.metric("Intrinsic Value per Share", f"${intrinsic_value:,.2f}")

    # Status message
    if market_price and market_price > 0:
        pct_diff = ((intrinsic_value - market_price) / market_price) * 100
        if pct_diff > 0:
            st.success(f"✅ **Undervalued by {pct_diff:.1f}%** — Intrinsic value is higher than market price.")
        else:
            st.error(f"⚠️ **Overvalued by {abs(pct_diff):.1f}%** — Market price exceeds intrinsic value.")
    else:
        st.info("Enter a market price above to see undervalued/overvalued status.")

    # Bar chart
    if market_price and market_price > 0:
        df_chart = pd.DataFrame(
            {
                "Metric": ["Intrinsic Value", "Current Price"],
                "Value ($)": [intrinsic_value, market_price],
            }
        )
        fig = px.bar(
            df_chart,
            x="Metric",
            y="Value ($)",
            color="Metric",
            color_discrete_map={"Intrinsic Value": "#28a745", "Current Price": "#6c757d"},
            title="Intrinsic Value vs Current Price",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Value ($)")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("⚠️ Enter valid Free Cash Flow and Shares Outstanding to see the intrinsic value.")
    st.info("If yfinance returns missing data, use the manual input fields above.")
