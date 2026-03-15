from __future__ import annotations

from engine.deterministic import PlatformResult


def pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x*100:.2f}%"


def xnum(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"${x:,.0f}"


def generate_memo(result: PlatformResult, mc_summary: dict, platform_name: str, target_irr: float, covenant_dscr: float) -> str:
    irr = result.equity_irr
    dscr = result.dscr

    if irr is None:
        rec = "Defer"
    elif irr < target_irr or dscr < covenant_dscr:
        rec = "Proceed with Mitigation"
    else:
        rec = "Proceed"

    return f"""
## Investment Committee Memo

**Project:** {platform_name}  
**Recommendation:** {rec}

### Base case
- Total CAPEX: {xnum(result.total_capex)}
- Revenue: {xnum(result.total_revenue)}
- EBITDA: {xnum(result.total_ebitda)}
- Debt: {xnum(result.debt)}
- Equity: {xnum(result.equity)}
- Annual debt service: {xnum(result.annual_debt_service)}
- CFADS: {xnum(result.cfads)}
- DSCR: {result.dscr:.2f}x
- Equity IRR: {pct(result.equity_irr)}

### Risk view
- Monte Carlo IRR P10 / P50 / P90: {pct(mc_summary["IRR"]["P10"])} / {pct(mc_summary["IRR"]["P50"])} / {pct(mc_summary["IRR"]["P90"])}
- Probability IRR below target: {pct(mc_summary["IRR"]["Prob_lt_target"])}
- Monte Carlo DSCR P10 / P50 / P90: {mc_summary["DSCR"]["P10"]:.2f}x / {mc_summary["DSCR"]["P50"]:.2f}x / {mc_summary["DSCR"]["P90"]:.2f}x
- Probability DSCR below covenant: {pct(mc_summary["DSCR"]["Prob_lt_covenant"])}

### Asset detail
""" + "\n".join(
        f"- {a.name}: revenue {xnum(a.annual_revenue)}, O&M {xnum(a.annual_om)}, EBITDA {xnum(a.annual_ebitda)}"
        for a in result.asset_results
    ) + "\n"
