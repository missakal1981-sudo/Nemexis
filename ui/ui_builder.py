from __future__ import annotations

import streamlit as st

from models.asset import Asset
from models.platform import Platform


def build_platform_from_ui() -> Platform:
    st.sidebar.header("Platform Inputs")

    platform_name = st.sidebar.text_input("Platform name", value="Nemexis Platform")
    debt_ratio = st.sidebar.number_input("Debt ratio", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
    equity_ratio = st.sidebar.number_input("Equity ratio", min_value=0.0, max_value=1.0, value=0.40, step=0.05)
    debt_rate = st.sidebar.number_input("Debt rate", min_value=0.0, max_value=1.0, value=0.065, step=0.005, format="%.4f")
    debt_tenor = st.sidebar.number_input("Debt tenor (years)", min_value=1, max_value=50, value=20)
    target_irr = st.sidebar.number_input("Target IRR", min_value=-0.5, max_value=1.0, value=0.08, step=0.01, format="%.4f")
    covenant_dscr = st.sidebar.number_input("Covenant DSCR", min_value=0.5, max_value=5.0, value=1.35, step=0.05)
    tax_rate = st.sidebar.number_input("Tax rate", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.4f")
    reserve_pct = st.sidebar.number_input("Reserve % of EBITDA", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.4f")

    st.header("Assets")

    asset_count = st.number_input("Number of assets", min_value=1, max_value=10, value=1, step=1)

    assets = []
    for i in range(asset_count):
        st.subheader(f"Asset {i+1}")
        c1, c2, c3 = st.columns(3)

        with c1:
            name = st.text_input(f"Asset name {i+1}", value=f"Asset {i+1}")
            asset_type = st.selectbox(f"Asset type {i+1}", ["generation", "transmission", "battery", "hybrid"], key=f"type_{i}")
            revenue_mode = st.selectbox(f"Revenue mode {i+1}", ["fixed", "tolling", "transmission"], key=f"rev_{i}")

        with c2:
            capacity_mw = st.number_input(f"Capacity MW {i+1}", min_value=0.0, value=100.0, step=10.0)
            capex = st.number_input(f"CAPEX {i+1}", min_value=0.0, value=100_000_000.0, step=10_000_000.0, format="%.2f")
            annual_revenue = st.number_input(f"Annual revenue {i+1}", min_value=0.0, value=20_000_000.0, step=1_000_000.0, format="%.2f")

        with c3:
            toll = st.number_input(f"Toll $/kW-month {i+1}", min_value=0.0, value=0.0, step=0.5, format="%.4f")
            fixed_om = st.number_input(f"Fixed O&M annual {i+1}", min_value=0.0, value=2_000_000.0, step=500_000.0, format="%.2f")
            operating_years = st.number_input(f"Operating years {i+1}", min_value=1, max_value=50, value=25)

        asset = Asset(
            name=name,
            asset_type=asset_type,
            capacity_mw=capacity_mw,
            capex=capex,
            revenue_mode=revenue_mode,
            toll_usd_per_kw_month=(toll if revenue_mode in ("tolling", "transmission") and toll > 0 else None),
            annual_revenue=(annual_revenue if revenue_mode == "fixed" else None),
            fixed_om_annual=fixed_om,
            operating_years=operating_years,
        )
        assets.append(asset)

    return Platform(
        name=platform_name,
        assets=assets,
        debt_ratio=debt_ratio,
        equity_ratio=equity_ratio,
        debt_rate=debt_rate,
        debt_tenor_years=int(debt_tenor),
        target_irr=target_irr,
        covenant_dscr=covenant_dscr,
        tax_rate=tax_rate,
        reserve_pct_of_ebitda=reserve_pct,
    )
