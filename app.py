"""
Valuation Dashboard: Smart Intrinsic Value Calculator
- Main: DCF with Manual Overrides, Sector Defaults, Sensitivity, Export
- Data Explorer: Financials, Key Ratios (TTM/LFY), Revenue vs Net Income chart
- Valuation Models: DCF (+ Terminal Multiple), DDM, Comparable Comps
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
if "show_snapshot" not in st.session_state:
    st.session_state.show_snapshot = False

# --- 2. HELPERS & SECTOR LOGIC ---
def safe_float(value, default=0.0):
    """Handle None or invalid values; return default so the app never crashes on math."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_market_cap(value):
    """Format market cap as T/B/M. Returns 'N/A' if missing or invalid."""
    v = safe_float(value, None)
    if v is None or v <= 0:
        return "N/A"
    if v >= 1e12:
        return f"{v / 1e12:.2f}T"
    if v >= 1e9:
        return f"{v / 1e9:.2f}B"
    if v >= 1e6:
        return f"{v / 1e6:.2f}M"
    return f"{v:,.0f}"


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


@st.cache_data(ttl=300)
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


def safe_df(df):
    """Return DataFrame if valid and non-empty; else None."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    return df

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

# --- 4. MAIN APP (TABS) ---
st.title("🧠 Valuation Dashboard")
tab_main, tab_explorer, tab_models, tab_guru = st.tabs(["Main", "Data Explorer", "Valuation Models", "Guru Checklists"])

# ========== TAB 1: MAIN ==========
with tab_main:
    st.header("Smart Intrinsic Value Calculator")

    # --- Company Snapshot (only after Fetch Data) ---
    if st.session_state.get("show_snapshot", False):
        price_s = safe_float(st.session_state.price)
        prev_close = safe_float(st.session_state.get("snap_prev_close"), None)
        delta_price = (price_s - prev_close) if prev_close is not None and prev_close else None
        mcap_str = format_market_cap(st.session_state.get("snap_market_cap"))
        pe_t = safe_float(st.session_state.get("snap_trailing_pe"), None)
        pe_f = safe_float(st.session_state.get("snap_forward_pe"), None)
        peg_s = safe_float(st.session_state.get("snap_peg"), None)
        div_y = safe_float(st.session_state.get("snap_div_yield"), None)
        if div_y is not None and div_y <= 1 and div_y > 0:
            div_y_pct = div_y * 100
        elif div_y is not None:
            div_y_pct = div_y
        else:
            div_y_pct = None
        beta_s = safe_float(st.session_state.get("snap_beta"), None)
        high52 = safe_float(st.session_state.get("snap_52w_high"), None)

        with st.container():
            st.caption(f"**Company Snapshot** · Market Cap: {mcap_str}")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric(
                    "Price",
                    f"${price_s:.2f}" if price_s else "N/A",
                    f"{delta_price:+.2f}" if delta_price is not None else None,
                )
            with c2:
                val_txt = f"P/E: {pe_t:.1f}" if pe_t is not None and pe_t > 0 else "P/E: N/A"
                val_txt += f" | Fwd: {pe_f:.1f}" if pe_f is not None and pe_f > 0 else " | Fwd: N/A"
                st.metric("Valuation", val_txt, None)
            with c3:
                gv_txt = f"PEG: {peg_s:.2f}" if peg_s is not None and peg_s > 0 else "PEG: N/A"
                gv_txt += f" | Div: {div_y_pct:.2f}%" if div_y_pct is not None else " | Div: N/A"
                st.metric("Growth / Value", gv_txt, None)
            with c4:
                vol_txt = f"Beta: {beta_s:.2f}" if beta_s is not None else "Beta: N/A"
                vol_txt += f" | 52W High: ${high52:.2f}" if high52 is not None and high52 > 0 else " | 52W High: N/A"
                st.metric("Volatility / Range", vol_txt, None)
            st.markdown("---")

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

                    # 3. Robust Unlevered FCF
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

                    # 4. Net Debt
                    total_debt = info.get('totalDebt')
                    total_cash = info.get('totalCash')
                    st.session_state.net_debt = safe_float(total_debt) - safe_float(total_cash)

                    # 5. Snapshot data (for Company Snapshot dashboard)
                    st.session_state.show_snapshot = True
                    st.session_state.snap_market_cap = info.get("marketCap")
                    st.session_state.snap_prev_close = info.get("regularMarketPreviousClose")
                    st.session_state.snap_trailing_pe = info.get("trailingPE")
                    st.session_state.snap_forward_pe = info.get("forwardPE")
                    st.session_state.snap_peg = info.get("pegRatio")
                    st.session_state.snap_div_yield = info.get("dividendYield") or info.get("yield")
                    st.session_state.snap_beta = info.get("beta")
                    st.session_state.snap_52w_high = info.get("fiftyTwoWeekHigh")
                    st.session_state.snap_52w_low = info.get("fiftyTwoWeekLow")

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

    st.header("2. Assumptions (Auto-filled by Sector)")

    try:
        cagr_pct = revenue_cagr_5y(st.session_state.ticker)
        if cagr_pct is not None:
            st.caption(f"📉 **5-Year Revenue CAGR:** **{cagr_pct:.1f}%**")
        else:
            st.caption("📉 **5-Year Revenue CAGR:** Historical data unavailable")
    except Exception:
        st.caption("📉 **5-Year Revenue CAGR:** Historical data unavailable")

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
            cf = fcf_input * ((1 + growth_rate) ** year)
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

# ========== TAB 2: DATA EXPLORER ==========
with tab_explorer:
    st.header("Data Explorer")
    ticker_ex = (st.session_state.ticker or "KO").strip().upper()
    try:
        stock_ex = yf.Ticker(ticker_ex)
        info_ex = stock_ex.info or {}
    except Exception:
        stock_ex = None
        info_ex = {}

    # --- Financials (transposed: dates as columns) ---
    st.subheader("Financial Statements (dates as columns)")
    if stock_ex is None:
        st.info("Data unavailable. Fetch data for this ticker in the Main tab first.")
    else:
        for label, get_df in [
            ("Income Statement", lambda s=stock_ex: safe_df(s.financials)),
            ("Balance Sheet", lambda s=stock_ex: safe_df(s.balance_sheet)),
            ("Cash Flow", lambda s=stock_ex: safe_df(s.cashflow)),
        ]:
            with st.expander(label):
                df = get_df()
                if df is not None and not df.empty:
                    try:
                        transposed = df.T
                        transposed.index = pd.to_datetime(transposed.index).strftime("%Y-%m-%d")
                        st.dataframe(transposed.fillna("").astype(str), use_container_width=True)
                    except Exception:
                        st.dataframe(df.fillna("").astype(str), use_container_width=True)
                else:
                    st.info("Data unavailable")

    # --- Key Ratios (TTM & LFY) ---
    st.subheader("Key Ratios (Instant Check)")
    mcap = safe_float(info_ex.get("marketCap"))
    ev = safe_float(info_ex.get("enterpriseValue"))
    equity = safe_float(info_ex.get("totalStockholderEquity"))
    ebitda = safe_float(info_ex.get("ebitda"))
    fcf = safe_float(info_ex.get("freeCashflow"))
    op_cash = info_ex.get("operatingCashflow")
    capex = info_ex.get("capitalExpenditures")
    if op_cash is not None and capex is not None:
        fcf_calc = safe_float(op_cash + capex)
    else:
        fcf_calc = fcf
    ebit = safe_float(info_ex.get("ebit"))
    total_debt_ex = safe_float(info_ex.get("totalDebt"))
    tax_rate = safe_float(info_ex.get("taxRate"), 0.25)
    if tax_rate <= 0 or tax_rate > 1:
        tax_rate = 0.25
    invested_cap = total_debt_ex + equity if (total_debt_ex + equity) else None
    nopat = ebit * (1 - tax_rate) if ebit else None
    roic = (nopat / invested_cap * 100) if invested_cap and invested_cap != 0 and nopat is not None else None

    def ratio_val(num, denom, fmt=".2f"):
        if denom is None or denom == 0 or num is None:
            return "—"
        return format(num / denom, fmt)

    def _lfy_pb(s):
        try:
            bs = safe_df(s.balance_sheet)
            if bs is None or bs.empty:
                return "—"
            eq_row = None
            for r in ("Total Stockholder Equity", "Stockholders Equity", "Total Equity"):
                if r in bs.index:
                    eq_row = bs.loc[r]
                    break
            if eq_row is None:
                return "—"
            eq = safe_float(eq_row.iloc[0]) if len(eq_row) else 0
            return ratio_val(mcap, eq) if eq else "—"
        except Exception:
            return "—"

    rows_ratios = [
        ("Price / Book", ratio_val(mcap, equity), _lfy_pb(stock_ex) if stock_ex else "—"),
        ("EV / EBITDA", ratio_val(ev, ebitda), "—"),
        ("FCF Yield %", (format(fcf_calc / mcap * 100, ".2f") + "%") if mcap and mcap != 0 else "—", "—"),
        ("ROIC %", (format(roic, ".2f") + "%") if roic is not None else "—", "—"),
    ]
    ratios_df = pd.DataFrame(rows_ratios, columns=["Ratio", "TTM", "Last Fiscal Year"])
    st.dataframe(ratios_df, use_container_width=True, hide_index=True)

    # --- Bar chart: Revenue vs Net Income ---
    st.subheader("Revenue vs Net Income")
    fin = safe_df(stock_ex.financials) if stock_ex else None
    if fin is not None and not fin.empty:
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
            st.info("Revenue or Net Income series not found for chart.")
    else:
        st.info("Data unavailable for chart.")

# ========== TAB 3: VALUATION MODELS ==========
with tab_models:
    st.header("Valuation Models")
    ticker_m = (st.session_state.ticker or "KO").strip().upper()
    fcf_m = safe_float(st.session_state.fcf)
    net_debt_m = safe_float(st.session_state.net_debt)
    shares_m = safe_float(st.session_state.shares)
    price_m = safe_float(st.session_state.price)

    # --- DCF (with Terminal Multiple option) ---
    with st.expander("Discounted Cash Flow (DCF)", expanded=True):
        g_m = st.slider("Growth % (DCF)", 0.0, 30.0, st.session_state.def_growth, 0.5, key="dcf_g") / 100
        r_m = st.slider("Discount % (DCF)", 4.0, 15.0, st.session_state.def_discount, 0.5, key="dcf_r") / 100
        terminal_method = st.radio("Terminal Value Method", ["Perpetuity Growth", "Exit Multiple (e.g. 10x FCF)"], key="tv_method")
        if terminal_method == "Perpetuity Growth":
            t_m = st.slider("Terminal Growth %", 0.0, 5.0, st.session_state.def_terminal, 0.1, key="dcf_t") / 100
            exit_mult = None
        else:
            exit_mult = st.number_input("Exit Multiple (× Final Year FCF)", min_value=1.0, max_value=30.0, value=10.0, step=0.5, key="exit_mult")
            t_m = None
        if shares_m and shares_m > 0:
            future_cfs = [fcf_m * ((1 + g_m) ** y) for y in range(1, 6)]
            discount_factors = [(1 + r_m) ** y for y in range(1, 6)]
            pv_cfs = sum(cf / df for cf, df in zip(future_cfs, discount_factors))
            final_fcf = future_cfs[-1]
            if terminal_method == "Perpetuity Growth" and t_m is not None and r_m > t_m:
                tv = (final_fcf * (1 + t_m)) / (r_m - t_m)
            elif exit_mult is not None:
                tv = exit_mult * final_fcf
            else:
                tv = 0.0
            pv_tv = tv / ((1 + r_m) ** 5)
            ev = pv_cfs + pv_tv
            equity = ev - net_debt_m
            iv_dcf = equity / shares_m
            st.metric("DCF Intrinsic Value per Share", f"${iv_dcf:.2f}", f"{(iv_dcf - price_m):.2f}" if price_m else None)
        else:
            st.warning("Enter Shares (and FCF) in the Main tab first.")

    # --- Dividend Discount Model (DDM) ---
    with st.expander("Dividend Discount Model (DDM)"):
        try:
            stock_ddm = yf.Ticker(ticker_m)
            info_ddm = stock_ddm.info or {}
            div_rate = safe_float(info_ddm.get("dividendRate"))
        except Exception:
            div_rate = 0.0
        div_annual = st.number_input("Annual Dividend ($)", value=div_rate if div_rate else 0.0, min_value=0.0, step=0.01, key="ddm_div")
        g_ddm = st.slider("Dividend Growth %", 0.0, 15.0, 5.0, 0.5, key="ddm_g") / 100
        r_ddm = st.slider("Cost of Equity %", 4.0, 15.0, 8.0, 0.5, key="ddm_r") / 100
        if not div_annual or div_annual <= 0:
            st.warning("⚠️ This company does not pay dividends (or dividend = 0). Use DCF instead.")
        elif r_ddm <= g_ddm:
            st.warning("Cost of Equity must be greater than Growth Rate for the formula to work.")
        else:
            value_ddm = div_annual / (r_ddm - g_ddm)
            st.metric("DDM Value per Share", f"${value_ddm:.2f}", f"vs Price ${price_m:.2f}" if price_m else None)

    # --- Comparable Company Analysis (Comps) ---
    with st.expander("Comparable Company Analysis (Comps)"):
        peers_str = st.text_input("Peer tickers (comma-separated)", value="PEP, MNST, KDP", key="peers_input")
        peer_list = [t.strip().upper() for t in (peers_str or "").split(",") if t.strip()]
        if ticker_m and ticker_m not in peer_list:
            peer_list = [ticker_m] + peer_list
        if peer_list:
            with st.spinner("Fetching peer metrics..."):
                comps_data = fetch_peer_metrics(tuple(peer_list))
            if comps_data:
                comps_df = pd.DataFrame(comps_data)
                # Highlight main ticker row
                def _highlight_main(row):
                    if row["Ticker"] == ticker_m:
                        return ["background-color: #e7f3ff; font-weight: bold"] * len(row)
                    return [""] * len(row)
                st.caption(f"**Bold row** = main ticker ({ticker_m}) vs peer average.")
                st.dataframe(
                    comps_df.style.apply(_highlight_main, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No peer data returned.")
        else:
            st.info("Enter at least one peer ticker.")

# ========== TAB 4: GURU CHECKLISTS ==========
with tab_guru:
    st.header("Guru Checklists")
    ticker_g = (st.session_state.ticker or "KO").strip().upper()
    try:
        stock_g = yf.Ticker(ticker_g)
        info_g = stock_g.info or {}
    except Exception:
        stock_g = None
        info_g = {}

    def _icon(passed):
        if passed is True:
            return "✅"
        if passed is False:
            return "❌"
        return "➖"

    col_buffett, col_lynch, col_greenblatt = st.columns(3)

    # --- Warren Buffett (Quality & Moat) ---
    with col_buffett:
        st.subheader("Warren Buffett")
        st.caption("Quality & Moat")
        roe = safe_float(info_g.get("returnOnEquity"), None)
        if roe is not None and roe == 0.0:
            roe = None
        roe_pass = roe is not None and roe > 0.15
        profit_m = safe_float(info_g.get("profitMargins"), None)
        if profit_m is not None and profit_m == 0.0 and info_g.get("profitMargins") is None:
            profit_m = None
        oper_m = safe_float(info_g.get("operatingMargins"), None)
        margin_pass = (profit_m is not None and profit_m > 0.20) or (oper_m is not None and oper_m > 0.15)
        if profit_m is None and oper_m is None:
            margin_pass = None
        dte = safe_float(info_g.get("debtToEquity"), None)
        if dte is not None and dte == 0.0 and info_g.get("debtToEquity") is None:
            dte = None
        debt_pass = (dte is not None and dte < 1.0) if dte is not None else None
        buffett_total = sum(1 for x in [roe_pass, margin_pass, debt_pass] if x is True)
        buffett_max = sum(1 for x in [roe_pass, margin_pass, debt_pass] if x is not None)
        st.metric("Buffett Score", f"{buffett_total}/{buffett_max} Pass", None)
        st.write(_icon(roe_pass), " **ROE:**", f"{roe*100:.1f}%" if roe is not None else "—", "(>15% pass)")
        margin_txt = (f"Profit {profit_m*100:.1f}%" if profit_m is not None else "—") + (f", Oper {oper_m*100:.1f}%" if oper_m is not None else "")
        st.write(_icon(margin_pass), " **Margin:**", margin_txt, "(Profit>20% or Oper>15%)")
        st.write(_icon(debt_pass), " **Debt/Equity:**", f"{dte:.2f}" if dte is not None else "—", "(<1.0 pass)")

    # --- Peter Lynch (GARP) ---
    with col_lynch:
        st.subheader("Peter Lynch")
        st.caption("Growth at a Reasonable Price")
        peg = safe_float(info_g.get("pegRatio"), None)
        if peg is not None and peg == 0.0:
            peg = None
        peg_pass = (peg is not None and peg < 1.0) if peg is not None else None
        peg_warn = peg is not None and peg > 1.5
        earn_g = info_g.get("earningsGrowth")
        rev_g = info_g.get("revenueGrowth")
        if earn_g is not None:
            earn_g = safe_float(earn_g, None)
        if rev_g is not None:
            rev_g = safe_float(rev_g, None)
        growth_pct = earn_g if earn_g is not None else rev_g
        if growth_pct is not None and growth_pct > 1:
            growth_pct = growth_pct / 100
        if growth_pct is not None and 0.10 <= growth_pct <= 0.25:
            growth_pass = True
        elif growth_pct is not None:
            growth_pass = False
        else:
            growth_pass = None
        inv_pass = None
        try:
            if stock_g is not None:
                bs = safe_df(stock_g.balance_sheet)
                fin = safe_df(stock_g.financials)
                if bs is not None and not bs.empty and fin is not None and not fin.empty:
                    inv_row = None
                    for r in ("Inventory", "Total Inventory", "Inventories"):
                        if r in bs.index:
                            inv_row = bs.loc[r].dropna()
                            break
                    rev_row = None
                    for r in ("Total Revenue", "Revenue", "Total revenues"):
                        if r in fin.index:
                            rev_row = fin.loc[r].dropna()
                            break
                    if inv_row is not None and len(inv_row) >= 2 and rev_row is not None and len(rev_row) >= 2:
                        inv_row = inv_row.sort_index()
                        rev_row = rev_row.sort_index()
                        inv_curr = safe_float(inv_row.iloc[-1])
                        inv_prev = safe_float(inv_row.iloc[-2])
                        rev_curr = safe_float(rev_row.iloc[-1])
                        rev_prev = safe_float(rev_row.iloc[-2])
                        if inv_prev and rev_prev:
                            inv_growth = (inv_curr - inv_prev) / inv_prev
                            rev_growth = (rev_curr - rev_prev) / rev_prev
                            inv_pass = inv_growth < rev_growth
        except Exception:
            pass
        lynch_total = sum(1 for x in [peg_pass, growth_pass, inv_pass] if x is True)
        lynch_max = sum(1 for x in [peg_pass, growth_pass, inv_pass] if x is not None)
        st.metric("Lynch Score", f"{lynch_total}/{lynch_max} Pass", None)
        st.write(_icon(peg_pass), " **PEG Ratio:**", f"{peg:.2f}" if peg is not None else "—", "(<1.0 pass)")
        if peg_warn:
            st.warning("PEG > 1.5 — may be overvalued.")
        st.write(_icon(growth_pass), " **Growth (Earn/Rev):**", f"{growth_pct*100:.1f}%" if growth_pct is not None else "—", "(10–25% pass)")
        st.write(_icon(inv_pass), " **Inv Growth < Rev Growth:**", "Yes" if inv_pass is True else "No" if inv_pass is False else "—")

    # --- Joel Greenblatt (Magic Formula) ---
    with col_greenblatt:
        st.subheader("Joel Greenblatt")
        st.caption("The Magic Formula")
        roa = safe_float(info_g.get("returnOnAssets"), None)
        roic = safe_float(info_g.get("returnOnInvestedCapital"), None)
        roc = roic if (roic is not None and roic != 0) else roa
        if roc is not None and roc == 0.0 and roa is None and roic is None:
            roc = None
        roc_pass = (roc is not None and roc > 0.25) if roc is not None else None
        ebit = safe_float(info_g.get("ebit"))
        ev = safe_float(info_g.get("enterpriseValue"))
        earn_yield = (ebit / ev) if ev and ev != 0 else None
        yield_pass = (earn_yield is not None and earn_yield > 0.10) if earn_yield is not None else None
        greenblatt_total = sum(1 for x in [roc_pass, yield_pass] if x is True)
        greenblatt_max = sum(1 for x in [roc_pass, yield_pass] if x is not None)
        st.metric("Greenblatt Score", f"{greenblatt_total}/{greenblatt_max} Pass", None)
        st.write(_icon(roc_pass), " **ROC (ROA/ROIC):**", f"{roc*100:.1f}%" if roc is not None else "—", "(>25% pass)")
        st.write(_icon(yield_pass), " **Earnings Yield (EBIT/EV):**", f"{earn_yield*100:.1f}%" if earn_yield is not None else "—", "(>10% pass)")
