from __future__ import annotations

import json
import streamlit as st

from ui.ui_builder import build_platform_from_ui
from engine.deterministic import DeterministicEngine
from engine.montecarlo import MonteCarloEngine
from memo.memo_generator import generate_memo


st.set_page_config(page_title="Nemexis", layout="wide")


def main():
    st.title("Nemexis")
    st.caption("Deterministic infrastructure engine + Monte Carlo + memo generator")

    try:
        platform = build_platform_from_ui()

        col1, col2 = st.columns([1, 1])
        with col1:
            run_det = st.button("Run deterministic", use_container_width=True)
        with col2:
            run_mc = st.button("Run Monte Carlo", use_container_width=True)

        if run_det or run_mc:
            det = DeterministicEngine(platform)
            result = det.run()

            st.header("Deterministic Ledger")
            st.write(
                {
                    "total_capex": result.total_capex,
                    "total_revenue": result.total_revenue,
                    "total_om": result.total_om,
                    "total_ebitda": result.total_ebitda,
                    "debt": result.debt,
                    "equity": result.equity,
                    "annual_debt_service": result.annual_debt_service,
                    "cfads": result.cfads,
                    "dscr": result.dscr,
                    "equity_irr": result.equity_irr,
                }
            )

            mc_summary = {
                "IRR": {"P10": None, "P50": None, "P90": None, "Prob_lt_target": None},
                "DSCR": {"P10": None, "P50": None, "P90": None, "Prob_lt_covenant": None},
                "counts": {"iterations": 0, "irr_samples": 0, "dscr_samples": 0},
            }

            if run_mc:
                mc = MonteCarloEngine(platform, iterations=10000, seed=42)
                mc_summary = mc.run()

                st.header("Monte Carlo Summary")
                st.code(json.dumps(mc_summary, indent=2), language="json")

            st.header("Final Memo")
            memo = generate_memo(
                result=result,
                mc_summary=mc_summary,
                platform_name=platform.name,
                target_irr=platform.target_irr,
                covenant_dscr=platform.covenant_dscr,
            )
            st.markdown(memo)

    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
if st.button("Run Downside"):
    res_down = run_deterministic(apply_scenario(platform, "Downside"))

if st.button("Run Sensitivity"):
    sens = run_sensitivity(platform)
    st.json(sens)
