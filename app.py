"""
Analyst Terminal: Financial Terminal UI + 4 Tabs
- Custom dark theme, metric cards, format_currency for all big numbers
- Sidebar: Ticker + Fetch at top, Status, Download at bottom
- Header: Ticker-tape (Company Name | Price / Change)
- Tabs: 💎 Valuation | 🧠 Guru Checklists | 📈 Financials | 📊 Deep Dive
"""

import gc
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

# Set True to show RAM usage in sidebar (requires psutil)
DEBUG_MODE = False

def force_garbage_collection():
    """Run gc.collect() to free memory after expensive operations."""
    gc.collect()

# Page config
st.set_page_config(
    page_title="Analyst Terminal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- FINANCIAL TERMINAL THEME (Custom CSS) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Roboto+Mono:wght@400;500&display=swap');
/* Main app background */
.stApp { background-color: #0E1117; }
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 100%; }
/* Typography */
p, span, label, .stMarkdown { font-family: 'Inter', sans-serif !important; color: #FAFAFA !important; }
h1, h2, h3 { font-family: 'Inter', sans-serif !important; color: #FAFAFA !important; font-weight: 600 !important; }
/* Metric cards: financial widget look */
[data-testid="stMetric"] {
    background-color: #262730 !important;
    border: 1px solid #3a3a4a !important;
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
}
[data-testid="stMetric"] label { font-size: 0.8rem !important; opacity: 0.9 !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { font-family: 'Roboto Mono', Consolas, monospace !important; font-size: 1.25rem !important; }
/* Inputs: sleek minimal */
.stTextInput input, .stNumberInput input {
    background-color: #262730 !important;
    border: 1px solid #3a3a4a !important;
    color: #FAFAFA !important;
    font-family: 'Roboto Mono', monospace !important;
    border-radius: 6px !important;
}
.stTextInput input:focus, .stNumberInput input:focus { border-color: #6366f1 !important; box-shadow: 0 0 0 1px #6366f1 !important; }
/* Buttons */
.stButton button {
    background-color: #262730 !important;
    border: 1px solid #3a3a4a !important;
    color: #FAFAFA !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton button:hover { border-color: #6366f1 !important; background-color: #1e1e2e !important; }
/* Tabs */
.stTabs [data-baseweb="tab-list"] { background-color: #262730 !important; border-radius: 8px !important; gap: 4px !important; }
.stTabs [data-baseweb="tab"] { color: #94a3b8 !important; font-family: 'Inter', sans-serif !important; }
.stTabs [aria-selected="true"] { background-color: #0E1117 !important; color: #FAFAFA !important; }
/* Sidebar */
[data-testid="stSidebar"] { background-color: #0E1117 !important; }
[data-testid="stSidebar"] .stMarkdown { color: #e2e8f0 !important; }
/* Expander / containers */
.streamlit-expanderHeader { background-color: #262730 !important; border-radius: 6px !important; }
/* DataFrames */
[data-testid="stDataFrame"] { border: 1px solid #3a3a4a !important; border-radius: 6px !important; }
</style>
""", unsafe_allow_html=True)

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
if "operating_cash" not in st.session_state:
    st.session_state.operating_cash = 0.0
if "capex" not in st.session_state:
    st.session_state.capex = 0.0
if "show_snapshot" not in st.session_state:
    st.session_state.show_snapshot = False
if "cached_info" not in st.session_state:
    st.session_state.cached_info = None
if "cached_financials" not in st.session_state:
    st.session_state.cached_financials = None
if "cached_balance_sheet" not in st.session_state:
    st.session_state.cached_balance_sheet = None
if "cached_cashflow" not in st.session_state:
    st.session_state.cached_cashflow = None
if "company_name" not in st.session_state:
    st.session_state.company_name = ""

# --- 2. HELPERS (safe_float, format_currency, sector, DCF) ---
def safe_float(value, default=0.0):
    """Handle None or invalid values; return default so the app never crashes on math."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_currency(value):
    """Format large numbers as $1.2T, $1.2B, $1.2M, $1.2K. Raw numbers never shown."""
    v = safe_float(value, None)
    if v is None or (isinstance(value, float) and value != value):
        return "N/A"
    v = float(v)
    if abs(v) >= 1e12:
        return f"${v / 1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:.2f}K"
    return f"${v:,.2f}"


def format_market_cap(value):
    """Format market cap as $T/$B/$M (alias for format_currency for size)."""
    return format_currency(value)


def _format_big(v):
    """Format big numbers for Deep Dive; use $ for currency context."""
    if v is None or (isinstance(v, float) and v != v):
        return "N/A"
    v = safe_float(v, None)
    if v is None:
        return "N/A"
    if abs(v) >= 1e12:
        return f"${v / 1e12:.2f}B"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:.2f}K"
    return f"${v:,.2f}"


def format_deep_value(value, kind="number", neutral_color=False):
    """Format for Deep Dive grid. Returns (display_str, color) with color one of 'green','red','neutral'."""
    if value is None or (isinstance(value, float) and value != value):
        return ("N/A", "neutral")
    v = safe_float(value, None)
    if v is None:
        return ("N/A", "neutral")
    if kind == "big":
        s = _format_big(v)
    elif kind == "percent":
        s = f"{v * 100:.2f}%" if abs(v) <= 1.5 else f"{v:.2f}%"
    elif kind == "currency":
        s = f"${v:,.2f}"
    elif kind == "ratio":
        s = f"{v:.2f}"
    else:
        s = f"{v:,.2f}" if abs(v) >= 1000 else f"{v:.2f}"
    if neutral_color:
        return (s, "neutral")
    if v > 0:
        return (s, "green")
    if v < 0:
        return (s, "red")
    return (s, "neutral")


def display_grid(title, data_dict, neutral_keys=None):
    """Display a category as subheader + 4-column grid. data_dict: { 'Label': (display_str, color) } or { 'Label': raw_value }.
    neutral_keys: set of keys that should not be green/red (e.g. P/E, PEG)."""
    st.subheader(title)
    neutral_keys = neutral_keys or set()
    items = list(data_dict.items())
    if not items:
        st.caption("No data")
        return
    cols = st.columns(4)
    for i, (k, v) in enumerate(items):
        with cols[i % 4]:
            if isinstance(v, tuple):
                display_str, color = v
            else:
                display_str, color = format_deep_value(v, "number", neutral_color=(k in neutral_keys))
            st.caption(f"**{k}**")
            if color == "green":
                st.markdown(f":green[{display_str}]")
            elif color == "red":
                st.markdown(f":red[{display_str}]")
            else:
                st.markdown(display_str)
    st.markdown("")


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


def revenue_cagr_from_financials(fin):
    """Compute 5Y revenue CAGR from a financials DataFrame (no yfinance call). Return percentage or None."""
    if fin is None or not isinstance(fin, pd.DataFrame) or fin.empty:
        return None
    rev_row = None
    for label in ("Total Revenue", "Revenue", "Total revenues"):
        if label in fin.index:
            rev_row = fin.loc[label].dropna()
            break
    if rev_row is None or len(rev_row) < 2:
        return None
    rev_row = rev_row.sort_index()
    oldest = safe_float(rev_row.iloc[0])
    newest = safe_float(rev_row.iloc[-1])
    if oldest <= 0:
        return None
    cagr = (newest / oldest) ** (1 / 5) - 1
    return cagr * 100


@st.cache_data(ttl=3600, max_entries=50)
def fetch_peer_metrics(tickers_list):
    """Fetch P/E, EV/EBITDA, Profit Margin for a list of tickers. Returns list of dicts; safe against missing data."""
    results = []
    for t in tickers_list:
        t = (t or "").strip().upper()
        if not t:
            continue
        try:
            stock = yf.Ticker(t)
            info = stock.info or {}
            mcap = safe_float(info.get("marketCap"))
            ev = safe_float(info.get("enterpriseValue"))
            pe = safe_float(info.get("trailingPE")) or safe_float(info.get("forwardPE"))
            ebitda = safe_float(info.get("ebitda"))
            ev_ebitda = (ev / ebitda) if ebitda and ebitda != 0 else None
            profit_margin = None
            if "profitMargins" in info and info["profitMargins"] is not None:
                profit_margin = safe_float(info["profitMargins"]) * 100
            results.append({
                "Ticker": t,
                "P/E": pe if pe else None,
                "EV/EBITDA": round(ev_ebitda, 2) if ev_ebitda is not None else None,
                "Profit Margin %": round(profit_margin, 2) if profit_margin is not None else None,
            })
        except Exception:
            results.append({"Ticker": t, "P/E": None, "EV/EBITDA": None, "Profit Margin %": None})
    return results


def safe_df(df, copy=False):
    """Return DataFrame if valid and non-empty; else None. Use copy=False to avoid duplicating in memory."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    return df.copy() if copy else df


def fetch_ticker_data(ticker):
    """Single yfinance fetch: update all session_state. Returns (success, error_message)."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return False, "Enter a ticker."
    # Free memory: drop old heavy cached data before loading new ticker
    for k in ("cached_info", "cached_financials", "cached_balance_sheet", "cached_cashflow"):
        st.session_state.pop(k, None)
    force_garbage_collection()
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        st.session_state.price = safe_float(info.get('currentPrice') or info.get('regularMarketPrice'))
        st.session_state.shares = safe_float(info.get('sharesOutstanding'))
        st.session_state.sector = info.get('sector') or "Unknown"
        st.session_state.company_name = info.get('shortName') or info.get('longName') or ticker
        g, d, t = get_sector_defaults(st.session_state.sector)
        st.session_state.def_growth, st.session_state.def_discount, st.session_state.def_terminal = g, d, t
        operating_cash, capex = info.get('operatingCashflow'), info.get('capitalExpenditures')
        st.session_state.last_op_cash, st.session_state.last_capex = operating_cash, capex
        st.session_state.operating_cash = safe_float(operating_cash)
        st.session_state.capex = safe_float(capex)  # Yahoo often sends negative, e.g. -2B
        st.session_state.fcf = st.session_state.operating_cash + st.session_state.capex
        st.session_state.fcf_source = "manual" if (operating_cash is not None and capex is not None) else "fallback"
        for k in ("ocf_input", "capex_input"):
            st.session_state.pop(k, None)  # reset widget state so inputs show new fetched values
        total_debt, total_cash = info.get('totalDebt'), info.get('totalCash')
        st.session_state.net_debt = safe_float(total_debt) - safe_float(total_cash)
        st.session_state.cached_info = info
        try:
            st.session_state.cached_financials = safe_df(stock.financials)
            st.session_state.cached_balance_sheet = safe_df(stock.balance_sheet)
            st.session_state.cached_cashflow = safe_df(stock.cashflow)
        except Exception:
            st.session_state.cached_financials = st.session_state.cached_balance_sheet = st.session_state.cached_cashflow = None
        st.session_state.show_snapshot = True
        st.session_state.snap_market_cap = info.get("marketCap")
        st.session_state.snap_prev_close = info.get("regularMarketPreviousClose")
        st.session_state.snap_trailing_pe = info.get("trailingPE")
        st.session_state.snap_forward_pe = info.get("forwardPE")
        st.session_state.snap_peg = info.get("pegRatio")
        st.session_state.snap_beta = info.get("beta")
        force_garbage_collection()
        return True, None
    except Exception as e:
        return False, str(e)


# --- 3. PRO SIDEBAR (Ticker + Fetch top, Status, Download bottom) ---
with st.sidebar:
    st.title("Terminal")
    ticker_side = st.text_input("Ticker", value=st.session_state.ticker, key="ticker_sidebar").strip().upper()
    st.session_state.ticker = ticker_side or st.session_state.ticker
    if st.button("Fetch Data", key="fetch_sidebar", type="primary"):
        with st.spinner("Loading..."):
            ok, err = fetch_ticker_data(st.session_state.ticker)
            if ok:
                st.success(f"Loaded {st.session_state.ticker}")
            else:
                st.error(err or "Fetch failed")
    st.markdown("---")
    st.caption("Sector & rates")
    st.info(f"**{st.session_state.sector}**")
    st.write(f"Growth: {st.session_state.def_growth}% · Discount: {st.session_state.def_discount}% · Terminal: {st.session_state.def_terminal}%")
    st.markdown("---")
    # Status (US market hours approx 9:30–16:00 ET → 14:30–21:00 UTC)
    try:
        from datetime import timezone
        utc_now = datetime.now(timezone.utc)
        is_weekday = utc_now.weekday() < 5
        hour_utc = utc_now.hour + utc_now.minute / 60
        market_open = is_weekday and (14.5 <= hour_utc <= 21.0)
        status = "Market Open" if market_open else "Market Closed"
        status_icon = "🟢" if market_open else "🔴"
    except Exception:
        status, status_icon = "—", "⚪"
    st.caption(f"{status_icon} **{status}**")
    st.markdown("---")
    st.caption("Export")
    ticker_safe = (st.session_state.ticker or "KO").strip() or "KO"
    csv_lines = [
        "Metric,Value",
        f"Ticker,{ticker_safe}",
        f"Price,{st.session_state.price}",
        f"FCF,{st.session_state.fcf}",
        f"Net Debt,{st.session_state.net_debt}",
        f"Shares,{st.session_state.shares}",
        f"Intrinsic Value,{st.session_state.export_intrinsic_value or ''}",
        f"Upside %,{st.session_state.export_upside_pct or ''}",
    ]
    st.download_button(
        label="📥 Download Model (CSV)",
        data="\n".join(csv_lines),
        file_name=f"{ticker_safe}_valuation.csv",
        mime="text/csv",
        key="download_valuation",
    )
    if DEBUG_MODE:
        try:
            import psutil
            proc = psutil.Process()
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            st.caption("RAM (debug)")
            st.metric("Memory", f"{rss_mb:.1f} MB")
        except ImportError:
            st.caption("RAM (debug): install psutil")

# --- 4. MAIN APP ---
# Ticker-tape header: Company Name (H1) | Price / Change (H2) — same line, columns
header_left, header_right = st.columns([2, 1])
with header_left:
    name = st.session_state.get("company_name") or st.session_state.ticker or "Analyst Terminal"
    st.title(f"📈 {name}")
with header_right:
    if st.session_state.get("show_snapshot"):
        price_s = safe_float(st.session_state.price)
        prev_close = safe_float(st.session_state.get("snap_prev_close"), None)
        delta = (price_s - prev_close) if prev_close and prev_close != 0 else None
        st.title(f"${price_s:.2f}" if price_s else "—")
        if delta is not None:
            st.caption(f"{delta:+.2f} ({'+' if delta >= 0 else ''}{(delta/prev_close)*100:.2f}%)")
    else:
        st.caption("Fetch data in sidebar →")
st.markdown("---")

# ========== SECTION 1: COMPANY SNAPSHOT (above tabs, only after Fetch) ==========
if st.session_state.get("show_snapshot", False):
    price_s = safe_float(st.session_state.price)
    prev_close = safe_float(st.session_state.get("snap_prev_close"), None)
    delta_price = (price_s - prev_close) if prev_close is not None and prev_close else None
    mcap_str = format_currency(st.session_state.get("snap_market_cap"))
    pe_t = safe_float(st.session_state.get("snap_trailing_pe"), None)
    pe_f = safe_float(st.session_state.get("snap_forward_pe"), None)
    peg_s = safe_float(st.session_state.get("snap_peg"), None)
    beta_s = safe_float(st.session_state.get("snap_beta"), None)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Price", f"${price_s:.2f}" if price_s else "N/A", f"{delta_price:+.2f}" if delta_price is not None else None)
    with c2:
        val_txt = f"P/E: {pe_t:.1f}" if pe_t and pe_t > 0 else "P/E: N/A"
        val_txt += f" | Fwd: {pe_f:.1f}" if pe_f and pe_f > 0 else " | Fwd: N/A"
        st.metric("Valuation", val_txt, None)
    with c3:
        st.metric("Growth", f"PEG: {peg_s:.2f}" if peg_s and peg_s > 0 else "N/A", None)
    with c4:
        st.metric("Volatility", f"Beta: {beta_s:.2f}" if beta_s is not None else "N/A", None)
    with c5:
        st.metric("Size", mcap_str, None)
    st.markdown("---")

tab_valuation, tab_guru, tab_datalab, tab_deep = st.tabs(["💎 Valuation", "🧠 Guru Checklists", "📈 Financials", "📊 Deep Dive"])

# ========== TAB 1: INTRINSIC VALUE (existing calculator) ==========
with tab_valuation:
    st.header("Intrinsic Value Calculator")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.caption(f"**Ticker:** {st.session_state.ticker or '—'} · Set in sidebar")
        if st.session_state.fcf_source == "manual":
            st.caption("✅ FCF: Unlevered (OpCash + CapEx)")
        elif st.session_state.fcf_source == "fallback":
            st.caption("⚠️ FCF: Fallback — verify in 10-K")
    with col2:
        st.caption("**Manual override:** Check vs 10-K. Edit OCF and CapEx; values persist.")
        # Build Your Own FCF: OCF + CapEx (CapEx usually negative from Yahoo)
        ocf_col, capex_col = st.columns(2)
        with ocf_col:
            operating_input = st.number_input(
                "Operating Cash Flow (OCF)",
                value=float(st.session_state.operating_cash),
                format="%f",
                key="ocf_input",
            )
        with capex_col:
            capex_input = st.number_input(
                "Less: CapEx",
                value=float(st.session_state.capex),
                format="%f",
                key="capex_input",
                help="Yahoo often sends this as negative (e.g. -2B).",
            )
        ignore_capex = st.checkbox("Ignore CapEx? (use 0)", key="ignore_capex", help="Use for service companies with low capex.")
        capex_for_fcf = 0.0 if ignore_capex else capex_input
        calculated_fcf = operating_input + capex_for_fcf
        st.session_state.fcf = calculated_fcf
        st.success(f"✅ **Free Cash Flow to Use:** {format_currency(calculated_fcf)}")
        net_debt_input = st.number_input("Net Debt ($)", value=float(st.session_state.net_debt), format="%f", key="net_debt")
        shares_input = st.number_input("Shares Outstanding", value=float(st.session_state.shares), format="%f", key="shares")
        price_input = st.number_input("Current Price ($)", value=float(st.session_state.price), format="%f", key="price")

    st.markdown("---")
    st.header("2. Assumptions (Auto-filled by Sector)")

    cagr_pct = revenue_cagr_from_financials(st.session_state.get("cached_financials"))
    if cagr_pct is not None:
        st.caption(f"📉 **5-Year Revenue CAGR:** **{cagr_pct:.1f}%**")
    else:
        st.caption("📉 **5-Year Revenue CAGR:** Fetch data first or unavailable")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        growth_rate = st.slider("Growth Rate (Next 5 Years)", 0.0, 30.0, st.session_state.def_growth, 0.5) / 100
    with col_b:
        discount_rate = st.slider("Discount Rate (Risk)", 4.0, 15.0, st.session_state.def_discount, 0.5) / 100
    with col_c:
        terminal_growth = st.slider("Terminal Growth", 0.0, 5.0, st.session_state.def_terminal, 0.1) / 100

    if shares_input > 0:
        future_cash_flows = []
        discount_factors = []
        for year in range(1, 6):
            cf = calculated_fcf * ((1 + growth_rate) ** year)
            future_cash_flows.append(cf)
            discount_factors.append((1 + discount_rate) ** year)
        final_year_fcf = future_cash_flows[-1]
        tv = (final_year_fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        pv_tv = tv / ((1 + discount_rate) ** 5)
        enterprise_value = sum([cf / df for cf, df in zip(future_cash_flows, discount_factors)]) + pv_tv
        equity_value = enterprise_value - net_debt_input
        intrinsic_value = equity_value / shares_input

        st.header("3. Results")
        st.markdown(f"""
        #### Valuation Bridge
        | Metric | Value per Share |
        | :--- | :--- |
        | **Enterprise Value** | **${(enterprise_value/shares_input):.2f}** |
        | *(-) Net Debt* | *${(net_debt_input/shares_input):.2f}* |
        | **(=) Equity Value** | **${intrinsic_value:.2f}** |
        """)
        delta = intrinsic_value - price_input
        upside_pct = (delta / price_input) * 100 if price_input else 0.0
        st.session_state.export_intrinsic_value = intrinsic_value
        st.session_state.export_upside_pct = upside_pct
        st.metric(label="Intrinsic Value per Share", value=f"${intrinsic_value:.2f}", delta=f"{delta:.2f}")
        if delta > 0:
            st.success(f"✅ UNDERVALUED by {upside_pct:.1f}%")
        else:
            st.error(f"❌ OVERVALUED by {abs(upside_pct):.1f}%")

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
                        calculated_fcf, net_debt_input, shares_input,
                        g / 100, d / 100, terminal_growth
                    )
                    row[f"Disc {d:.1f}%"] = round(iv, 2)
                rows.append(row)
            sens_df = pd.DataFrame(rows, index=[f"Growth {g:.1f}%" for g in growth_vals])
            st.caption("Rows: Growth ±1%; Columns: Discount ±0.5%. Best-case (max intrinsic value) highlighted.")
            st.dataframe(sens_df.style.highlight_max(axis=None).format("{:.2f}"), use_container_width=True)
        force_garbage_collection()
    else:
        st.session_state.export_intrinsic_value = None
        st.session_state.export_upside_pct = None
        st.warning("Enter valid data to see the valuation.")

# ========== TAB 2: GURU ANALYSIS (cached_info) ==========
with tab_guru:
    st.header("Guru Checklists")
    info_g = st.session_state.get("cached_info") or {}
    if not info_g:
        st.info("Fetch data in the **💎 Valuation** tab first.")
    else:
        def _icon(passed):
            return "✅" if passed is True else ("❌" if passed is False else "➖")

        col_buffett, col_lynch, col_greenblatt = st.columns(3)

        with col_buffett:
            st.subheader("Warren Buffett (Quality)")
            roe = safe_float(info_g.get("returnOnEquity"), None)
            roe_pass = roe is not None and roe > 0.15
            st.write(_icon(roe_pass), " **ROE > 15%:**", f"{roe*100:.1f}%" if roe is not None else "N/A")
            profit_m = safe_float(info_g.get("profitMargins"), None)
            margin_pass = profit_m is not None and profit_m > 0.20
            st.write(_icon(margin_pass), " **Profit Margin > 20%:**", f"{profit_m*100:.1f}%" if profit_m is not None else "N/A")
            dte = safe_float(info_g.get("debtToEquity"), None)
            debt_pass = dte is not None and dte < 1.0
            st.write(_icon(debt_pass), " **Debt/Equity < 1.0:**", f"{dte:.2f}" if dte is not None else "N/A")

        with col_lynch:
            st.subheader("Peter Lynch (Growth)")
            peg = safe_float(info_g.get("pegRatio"), None)
            peg_pass = peg is not None and peg < 1.2
            st.write(_icon(peg_pass), " **PEG < 1.2:**", f"{peg:.2f}" if peg is not None else "N/A")
            growth_raw = info_g.get("earningsGrowth") or info_g.get("revenueGrowth")
            growth_pct = safe_float(growth_raw, None)
            if growth_pct is not None and growth_pct > 1:
                growth_pct = growth_pct / 100
            growth_pass = growth_pct is not None and 0.10 <= growth_pct <= 0.25
            st.write(_icon(growth_pass), " **Earnings Growth 10–25%:**", f"{growth_pct*100:.1f}%" if growth_pct is not None else "N/A")

        with col_greenblatt:
            st.subheader("Joel Greenblatt (Value)")
            roic = safe_float(info_g.get("returnOnInvestedCapital") or info_g.get("returnOnCapital"), None)
            roa = safe_float(info_g.get("returnOnAssets"), None)
            roc = roic if (roic is not None and roic != 0) else roa
            roc_pass = roc is not None and roc > 0.25
            st.write(_icon(roc_pass), " **ROC > 25%:**", f"{roc*100:.1f}%" if roc is not None else "N/A")

# ========== TAB 3: FINANCIALS (cached_financials / balance_sheet / cashflow) ==========
with tab_datalab:
    st.header("Financials")
    fin = st.session_state.get("cached_financials")
    bs = st.session_state.get("cached_balance_sheet")
    cf = st.session_state.get("cached_cashflow")
    info_lab = st.session_state.get("cached_info") or {}
    if fin is None or fin.empty:
        st.info("Fetch data in the **💎 Valuation** tab first.")
    else:
        st.subheader("Revenue vs Net Income (last 5 years)")
        rev_row = None
        for r in ("Total Revenue", "Revenue", "Total revenues"):
            if r in fin.index:
                rev_row = fin.loc[r]
                break
        ni_row = None
        for r in ("Net Income", "Net Income Common Stockholders"):
            if r in fin.index:
                ni_row = fin.loc[r]
                break
        if rev_row is not None and ni_row is not None:
            chart_df = pd.DataFrame({
                "Revenue": rev_row.reindex(fin.columns).fillna(0).astype(float),
                "Net Income": ni_row.reindex(fin.columns).fillna(0).astype(float),
            })
            chart_df.index = pd.to_datetime(chart_df.index).strftime("%Y")
            st.bar_chart(chart_df)
        else:
            st.info("Revenue or Net Income series not found.")

        st.subheader("Historical ratios (Price/Book, ROIC, FCF Yield)")
        mcap = safe_float(info_lab.get("marketCap"))
        rows_hist = []
        if bs is not None and not bs.empty and fin is not None:
            eq_row = None
            for r in ("Total Stockholder Equity", "Stockholders Equity", "Total Equity"):
                if r in bs.index:
                    eq_row = bs.loc[r]
                    break
            for col in fin.columns:
                try:
                    yr = pd.to_datetime(col).strftime("%Y") if hasattr(col, 'strftime') else str(col)[:4]
                    eq_val = safe_float(eq_row[col], None) if (eq_row is not None and col in eq_row.index) else None
                    pb = (mcap / eq_val) if eq_val and eq_val > 0 else None
                    ni_val = safe_float(ni_row[col], None) if ni_row is not None else None
                    roic_yr = (ni_val / eq_val * 100) if eq_val and eq_val > 0 and ni_val is not None else None
                    fcf_val = None
                    if cf is not None and not cf.empty and col in cf.columns:
                        for f in ("Free Cash Flow", "Operating Cash Flow"):
                            if f in cf.index:
                                fcf_val = safe_float(cf.loc[f][col], None)
                                break
                    fcf_yield = (fcf_val / mcap * 100) if mcap and mcap > 0 and fcf_val is not None else None
                    rows_hist.append({
                        "Year": yr,
                        "P/B": f"{pb:.2f}" if pb is not None else "—",
                        "ROIC %": f"{roic_yr:.1f}%" if roic_yr is not None else "—",
                        "FCF Yield %": f"{fcf_yield:.1f}%" if fcf_yield is not None else "—",
                    })
                except Exception:
                    continue
            if rows_hist:
                st.dataframe(pd.DataFrame(rows_hist), use_container_width=True, hide_index=True)
            else:
                st.caption("Could not compute historical ratios.")
        else:
            st.caption("Balance sheet needed for P/B and ROIC by year.")

# ========== TAB 4: DEEP DIVE (cached_info) ==========
with tab_deep:
    st.header("📊 Deep Dive")
    info_d = st.session_state.get("cached_info") or {}
    if not info_d:
        st.info("Fetch data in the **💎 Valuation** tab first.")
    else:
        # ROIC: use returnOnInvestedCapital or approximate
        roic_val = info_d.get("returnOnInvestedCapital") or info_d.get("returnOnCapital")
        ev_ebitda = None
        if safe_float(info_d.get("ebitda")):
            ev_ebitda = safe_float(info_d.get("enterpriseValue")) / safe_float(info_d.get("ebitda"))

        valuation = {
            "Market Cap": format_deep_value(info_d.get("marketCap"), "big", neutral_color=True),
            "Enterprise Value": format_deep_value(info_d.get("enterpriseValue"), "big", neutral_color=True),
            "Trailing P/E": format_deep_value(info_d.get("trailingPE"), "ratio", neutral_color=True),
            "Forward P/E": format_deep_value(info_d.get("forwardPE"), "ratio", neutral_color=True),
            "PEG Ratio": format_deep_value(info_d.get("pegRatio"), "ratio", neutral_color=True),
            "Price/Sales": format_deep_value(info_d.get("priceToSalesTrailing12Months"), "ratio", neutral_color=True),
            "Price/Book": format_deep_value(info_d.get("priceToBook"), "ratio", neutral_color=True),
            "EV/EBITDA": format_deep_value(ev_ebitda, "ratio", neutral_color=True),
        }
        financials = {
            "Revenue (ttm)": format_deep_value(info_d.get("totalRevenue"), "big"),
            "Net Income (ttm)": format_deep_value(info_d.get("netIncomeToCommon"), "big"),
            "EPS (ttm)": format_deep_value(info_d.get("trailingEps"), "currency"),
            "Diluted EPS": format_deep_value(info_d.get("dilutedEps"), "currency"),
            "EBITDA": format_deep_value(info_d.get("ebitda"), "big"),
            "Total Cash": format_deep_value(info_d.get("totalCash"), "big"),
            "Total Debt": format_deep_value(info_d.get("totalDebt"), "big"),
            "Book Value/Share": format_deep_value(info_d.get("bookValue"), "currency"),
        }
        profitability = {
            "ROA": format_deep_value(info_d.get("returnOnAssets"), "percent"),
            "ROE": format_deep_value(info_d.get("returnOnEquity"), "percent"),
            "ROIC": format_deep_value(roic_val, "percent"),
            "Gross Margin": format_deep_value(info_d.get("grossMargins"), "percent"),
            "Operating Margin": format_deep_value(info_d.get("operatingMargins"), "percent"),
            "Profit Margin": format_deep_value(info_d.get("profitMargins"), "percent"),
            "Payout Ratio": format_deep_value(info_d.get("payoutRatio"), "percent"),
        }
        technicals = {
            "Beta": format_deep_value(info_d.get("beta"), "ratio", neutral_color=True),
            "52-Week High": format_deep_value(info_d.get("fiftyTwoWeekHigh"), "currency", neutral_color=True),
            "52-Week Low": format_deep_value(info_d.get("fiftyTwoWeekLow"), "currency", neutral_color=True),
            "50-Day MA": format_deep_value(info_d.get("fiftyDayAverage"), "currency", neutral_color=True),
            "200-Day MA": format_deep_value(info_d.get("twoHundredDayAverage"), "currency", neutral_color=True),
            "Avg Volume": format_deep_value(info_d.get("averageVolume"), "big", neutral_color=True),
            "Short Ratio": format_deep_value(info_d.get("shortRatio"), "ratio", neutral_color=True),
            "Shares Out.": format_deep_value(info_d.get("sharesOutstanding"), "big", neutral_color=True),
            "Float": format_deep_value(info_d.get("floatShares"), "big", neutral_color=True),
        }
        display_grid("Valuation", valuation)
        display_grid("Financials", financials)
        display_grid("Profitability & Effectiveness", profitability)
        display_grid("Technicals & Market", technicals)
