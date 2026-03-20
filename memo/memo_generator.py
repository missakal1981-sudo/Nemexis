from __future__ import annotations

from engine.deterministic import PlatformResult


def fmt_pct(x):
    if x is None:
        return "n/a"
    return f"{x:.2%}"


def fmt_num(x):
    if x is None:
        return "n/a"
    return f"${x:,.0f}"


def fmt_x(x):
    if x is None:
        return "n/a"
    return f"{x:.2f}x"


def generate_memo(
    result: PlatformResult,
    mc_summary: dict,
    platform_name: str,
    target_irr: float,
    covenant_dscr: float,
) -> str:
    irr = result.equity_irr
    dscr = result.dscr

    if irr is None:
        recommendation = "Defer"
    elif irr < target_irr or dscr < covenant_dscr:
        recommendation = "Proceed with Mitigation"
    else:
        recommendation = "Proceed"

    mc_irr = mc_summary.get("IRR", {}) if mc_summary else {}
    mc_dscr = mc_summary.get("DSCR", {}) if mc_summary else {}

    mc_irr_p10 = mc_irr.get("P10")
    mc_irr_p50 = mc_irr.get("P50")
    mc_irr_p90 = mc_irr.get("P90")
    mc_irr_prob = mc_irr.get("Prob_lt_target")

    mc_dscr_p10 = mc_dscr.get("P10")
    mc_dscr_p50 = mc_dscr.get("P50")
    mc_dscr_p90 = mc_dscr.get("P90")
    mc_dscr_prob = mc_dscr.get("Prob_lt_covenant")

    asset_lines = []
    for a in result.asset_results:
        asset_lines.append(
            f"- {a.name}: revenue {fmt_num(a.annual_revenue)}, "
            f"O&M {fmt_num(a.annual_om)}, EBITDA {fmt_num(a.annual_ebitda)}"
        )

    assets_block = "\n".join(asset_lines) if asset_lines else "- No asset detail available"

    return f"""
## Investment Committee Memo

**Project:** {platform_name}  
**Recommendation:** {recommendation}

### Base case
- Total CAPEX: {fmt_num(result.total_capex)}
- Revenue: {fmt_num(result.total_revenue)}
- EBITDA: {fmt_num(result.total_ebitda)}
- Debt: {fmt_num(result.debt)}
- Equity: {fmt_num(result.equity)}
- Annual debt service: {fmt_num(result.annual_debt_service)}
- CFADS: {fmt_num(result.cfads)}
- DSCR: {fmt_x(result.dscr)}
- Equity IRR: {fmt_pct(result.equity_irr)}

### Threshold checks
- Target equity IRR: {fmt_pct(target_irr)}
- Covenant DSCR: {fmt_x(covenant_dscr)}

### Monte Carlo summary
- IRR P10 / P50 / P90: {fmt_pct(mc_irr_p10)} / {fmt_pct(mc_irr_p50)} / {fmt_pct(mc_irr_p90)}
- Probability IRR below target: {fmt_pct(mc_irr_prob)}
- DSCR P10 / P50 / P90: {fmt_x(mc_dscr_p10)} / {fmt_x(mc_dscr_p50)} / {fmt_x(mc_dscr_p90)}
- Probability DSCR below covenant: {fmt_pct(mc_dscr_prob)}

### Asset detail
{assets_block}
""".strip()
