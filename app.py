from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, asdict
from enum import Enum
from statistics import median
from typing import Dict, List, Optional, Tuple


# ============================================================
# Nemexis v11
# Multi-asset deterministic engine + Monte Carlo + memo builder
# Math is computed by code, not by the LLM.
# ============================================================


class RevenueModel(str, Enum):
    PPA = "ppa"
    TOLLING = "tolling"
    AVAILABILITY = "availability"
    TRANSMISSION = "transmission"
    HYBRID = "hybrid"


class Recommendation(str, Enum):
    PROCEED = "Proceed"
    PROCEED_WITH_MITIGATION = "Proceed with Mitigation"
    DEFER = "Defer"
    REJECT = "Reject"


@dataclass
class TriangularInput:
    low: float
    mode: float
    high: float

    def sample(self) -> float:
        return random.triangular(self.low, self.high, self.mode)


@dataclass
class ScenarioSettings:
    name: str
    debt_ratio: float
    equity_ratio: float
    debt_capped: bool = False
    debt_cap_amount: Optional[float] = None


@dataclass
class AssetRevenueSpec:
    model: RevenueModel
    annual_revenue: Optional[float] = None
    monthly_capacity_rate_per_kw: Optional[float] = None
    monthly_fixed_om_per_kw: Optional[float] = None
    variable_om_per_mwh: Optional[float] = None
    annual_escalation: float = 0.0
    annual_volume_mwh: Optional[float] = None
    transmission_rate_per_kw_month: Optional[float] = None
    contracted_kw: Optional[float] = None
    availability_payment_annual: Optional[float] = None


@dataclass
class AssetTechSpec:
    nameplate_mw: Optional[float] = None
    availability: float = 1.0
    capacity_factor: Optional[float] = None
    uptime_requirement: Optional[float] = None


@dataclass
class AssetInputs:
    name: str
    asset_type: str
    capex: float
    construction_start_year: int
    cod_year: int
    operating_life_years: int
    revenue: AssetRevenueSpec
    tech: AssetTechSpec = field(default_factory=AssetTechSpec)
    base_fixed_om_annual: float = 0.0
    reserves_annual: float = 0.0
    tax_rate: float = 0.0
    capex_phasing: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.3])
    load_ramp: List[float] = field(default_factory=lambda: [1.0])
    notes: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.capex <= 0:
            raise ValueError(f"{self.name}: CAPEX must be positive")
        if not self.capex_phasing or not math.isclose(sum(self.capex_phasing), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{self.name}: CAPEX phasing must sum to 1.0")
        if self.operating_life_years <= 0:
            raise ValueError(f"{self.name}: operating_life_years must be positive")
        if self.cod_year < self.construction_start_year:
            raise ValueError(f"{self.name}: COD cannot be before construction start")
        if self.revenue.model == RevenueModel.TRANSMISSION:
            if self.revenue.transmission_rate_per_kw_month is None or self.revenue.contracted_kw is None:
                raise ValueError(f"{self.name}: transmission asset needs rate_per_kw_month and contracted_kw")
        if self.revenue.model == RevenueModel.TOLLING:
            if self.revenue.monthly_capacity_rate_per_kw is None or self.tech.nameplate_mw is None:
                raise ValueError(f"{self.name}: tolling asset needs nameplate_mw and monthly_capacity_rate_per_kw")


@dataclass
class PlatformInputs:
    project_name: str
    start_year: int
    debt_interest_rate: float
    debt_tenor_years: int
    target_equity_irr: float
    minimum_dscr: Optional[float]
    assets: List[AssetInputs]
    scenario_a: ScenarioSettings
    scenario_b: ScenarioSettings
    capex_overrun_pct: float = 0.0
    delay_months: int = 0
    monte_carlo_iterations: int = 10000

    def validate(self) -> None:
        if self.debt_interest_rate <= 0:
            raise ValueError("debt_interest_rate must be positive")
        if self.debt_tenor_years <= 0:
            raise ValueError("debt_tenor_years must be positive")
        if not self.assets:
            raise ValueError("At least one asset is required")
        for asset in self.assets:
            asset.validate()
        if not math.isclose(self.scenario_a.debt_ratio + self.scenario_a.equity_ratio, 1.0, abs_tol=1e-9):
            raise ValueError("Scenario A debt_ratio + equity_ratio must equal 1.0")
        if not math.isclose(self.scenario_b.debt_ratio + self.scenario_b.equity_ratio, 1.0, abs_tol=1e-9):
            raise ValueError("Scenario B debt_ratio + equity_ratio must equal 1.0")


@dataclass
class AssetLedger:
    name: str
    capex: float
    debt: float
    equity: float
    annual_revenue_base: float
    annual_cfads_base: float
    annual_debt_service: float
    dscr_base: Optional[float]


@dataclass
class ScenarioLedger:
    name: str
    total_capex: float
    total_debt: float
    total_equity: float
    annual_revenue: float
    annual_cfads: float
    annual_debt_service: float
    dscr: Optional[float]
    equity_irr: Optional[float]
    asset_ledgers: List[AssetLedger] = field(default_factory=list)


@dataclass
class MonteCarloSummary:
    a_irr: Dict[str, Optional[float]]
    b_irr: Dict[str, Optional[float]]
    a_dscr: Dict[str, Optional[float]]
    b_dscr: Dict[str, Optional[float]]
    prob: Dict[str, Optional[float]]
    counts: Dict[str, int]


def pmt(rate: float, nper: int, pv: float) -> float:
    if nper <= 0:
        raise ValueError("nper must be positive")
    if rate == 0:
        return pv / nper
    return rate * pv / (1.0 - (1.0 + rate) ** (-nper))


def npv(rate: float, cashflows: List[float]) -> float:
    return sum(cf / ((1.0 + rate) ** i) for i, cf in enumerate(cashflows))


def irr_bisection(cashflows: List[float], low: float = -0.95, high: float = 1.5, max_iter: int = 200) -> Optional[float]:
    if not cashflows or all(cf >= 0 for cf in cashflows) or all(cf <= 0 for cf in cashflows):
        return None

    f_low = npv(low, cashflows)
    f_high = npv(high, cashflows)

    expand_count = 0
    while f_low * f_high > 0 and expand_count < 30:
        high += 1.0
        f_high = npv(high, cashflows)
        expand_count += 1

    if f_low * f_high > 0:
        return None

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        f_mid = npv(mid, cashflows)
        if abs(f_mid) < 1e-10:
            return mid
        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return (low + high) / 2.0


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    values_sorted = sorted(values)
    idx = (len(values_sorted) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values_sorted[lo]
    frac = idx - lo
    return values_sorted[lo] * (1 - frac) + values_sorted[hi] * frac


class DeterministicEngine:
    def __init__(self, platform: PlatformInputs):
        self.platform = platform
        self.platform.validate()

    def annual_revenue(self, asset: AssetInputs) -> float:
        rev = asset.revenue
        if rev.annual_revenue is not None:
            return rev.annual_revenue
        if rev.model == RevenueModel.TOLLING:
            kw = (asset.tech.nameplate_mw or 0.0) * 1000.0
            cap_rev = kw * (rev.monthly_capacity_rate_per_kw or 0.0) * 12.0
            fixed_om_pass = kw * (rev.monthly_fixed_om_per_kw or 0.0) * 12.0
            if rev.annual_volume_mwh is not None:
                var_om_pass = rev.annual_volume_mwh * (rev.variable_om_per_mwh or 0.0)
            else:
                var_om_pass = 0.0
            return cap_rev + fixed_om_pass + var_om_pass
        if rev.model == RevenueModel.TRANSMISSION:
            return (rev.transmission_rate_per_kw_month or 0.0) * (rev.contracted_kw or 0.0) * 12.0
        if rev.model == RevenueModel.AVAILABILITY:
            return rev.availability_payment_annual or 0.0
        if rev.model == RevenueModel.PPA:
            return rev.annual_revenue or 0.0
        if rev.model == RevenueModel.HYBRID:
            return rev.annual_revenue or 0.0
        raise ValueError(f"Unsupported revenue model: {rev.model}")

    def annual_cfads(self, asset: AssetInputs, revenue_override: Optional[float] = None) -> float:
        revenue = revenue_override if revenue_override is not None else self.annual_revenue(asset)
        taxable_base = max(0.0, revenue - asset.base_fixed_om_annual)
        taxes = taxable_base * asset.tax_rate
        cfads = revenue - asset.base_fixed_om_annual - taxes - asset.reserves_annual
        return cfads

    def apply_overrun(self, asset: AssetInputs) -> float:
        return asset.capex * (1.0 + self.platform.capex_overrun_pct)

    def scenario_asset_financing(self, capex: float, scenario: ScenarioSettings) -> Tuple[float, float]:
        if scenario.debt_capped:
            debt = scenario.debt_cap_amount or 0.0
            equity = max(0.0, capex - debt)
            return debt, equity
        debt = capex * scenario.debt_ratio
        equity = capex * scenario.equity_ratio
        return debt, equity

    def build_scenario(self, scenario: ScenarioSettings) -> ScenarioLedger:
        asset_ledgers: List[AssetLedger] = []
        total_capex = 0.0
        total_debt = 0.0
        total_equity = 0.0
        total_revenue = 0.0
        total_cfads = 0.0
        total_debt_service = 0.0

        for asset in self.platform.assets:
            stressed_capex = self.apply_overrun(asset)
            debt, equity = self.scenario_asset_financing(stressed_capex, scenario)
            revenue = self.annual_revenue(asset)
            cfads = self.annual_cfads(asset)
            debt_service = pmt(self.platform.debt_interest_rate, self.platform.debt_tenor_years, debt) if debt > 0 else 0.0
            dscr = (cfads / debt_service) if debt_service > 0 else None

            asset_ledgers.append(
                AssetLedger(
                    name=asset.name,
                    capex=stressed_capex,
                    debt=debt,
                    equity=equity,
                    annual_revenue_base=revenue,
                    annual_cfads_base=cfads,
                    annual_debt_service=debt_service,
                    dscr_base=dscr,
                )
            )
            total_capex += stressed_capex
            total_debt += debt
            total_equity += equity
            total_revenue += revenue
            total_cfads += cfads
            total_debt_service += debt_service

        dscr_total = (total_cfads / total_debt_service) if total_debt_service > 0 else None
        irr = self._platform_equity_irr(scenario, total_equity, total_cfads, total_debt_service)

        return ScenarioLedger(
            name=scenario.name,
            total_capex=total_capex,
            total_debt=total_debt,
            total_equity=total_equity,
            annual_revenue=total_revenue,
            annual_cfads=total_cfads,
            annual_debt_service=total_debt_service,
            dscr=dscr_total,
            equity_irr=irr,
            asset_ledgers=asset_ledgers,
        )

    def _platform_equity_irr(
        self,
        scenario: ScenarioSettings,
        total_equity: float,
        total_cfads: float,
        total_debt_service: float,
    ) -> Optional[float]:
        construction_years = max(asset.cod_year - asset.construction_start_year for asset in self.platform.assets)
        construction_years = max(1, construction_years)
        operating_years = max(asset.operating_life_years for asset in self.platform.assets)

        equity_draws = [-(total_equity / construction_years) for _ in range(construction_years)]
        annual_equity_cash = total_cfads - total_debt_service
        ops = [annual_equity_cash for _ in range(operating_years)]
        return irr_bisection(equity_draws + ops)


class MonteCarloEngine:
    def __init__(
        self,
        platform: PlatformInputs,
        om_pct: TriangularInput,
        tax_rate: TriangularInput,
        reserves_pct_of_ebitda: TriangularInput,
        revenue_factor: TriangularInput,
        capex_factor: TriangularInput,
    ):
        self.platform = platform
        self.om_pct = om_pct
        self.tax_rate = tax_rate
        self.reserves_pct_of_ebitda = reserves_pct_of_ebitda
        self.revenue_factor = revenue_factor
        self.capex_factor = capex_factor

    def run(self) -> MonteCarloSummary:
        irr_a_vals: List[float] = []
        irr_b_vals: List[float] = []
        dscr_a_vals: List[float] = []
        dscr_b_vals: List[float] = []

        for _ in range(self.platform.monte_carlo_iterations):
            sampled_assets = []
            for asset in self.platform.assets:
                revenue_mult = self.revenue_factor.sample()
                capex_mult = self.capex_factor.sample()
                om_pct = self.om_pct.sample()
                tax_rate = self.tax_rate.sample()
                reserves_pct = self.reserves_pct_of_ebitda.sample()

                revenue_base = DeterministicEngine(self.platform).annual_revenue(asset)
                sampled_asset = AssetInputs(
                    name=asset.name,
                    asset_type=asset.asset_type,
                    capex=asset.capex * capex_mult,
                    construction_start_year=asset.construction_start_year,
                    cod_year=asset.cod_year,
                    operating_life_years=asset.operating_life_years,
                    revenue=asset.revenue,
                    tech=asset.tech,
                    base_fixed_om_annual=asset.capex * om_pct,
                    reserves_annual=revenue_base * reserves_pct,
                    tax_rate=tax_rate,
                    capex_phasing=asset.capex_phasing,
                    load_ramp=asset.load_ramp,
                    notes=asset.notes,
                )
                sampled_assets.append((sampled_asset, revenue_base * revenue_mult))

            a = self._run_sample(self.platform.scenario_a, sampled_assets)
            b = self._run_sample(self.platform.scenario_b, sampled_assets)

            if a.equity_irr is not None:
                irr_a_vals.append(a.equity_irr)
            if b.equity_irr is not None:
                irr_b_vals.append(b.equity_irr)
            if a.dscr is not None:
                dscr_a_vals.append(a.dscr)
            if b.dscr is not None:
                dscr_b_vals.append(b.dscr)

        target = self.platform.target_equity_irr
        covenant = self.platform.minimum_dscr
        prob = {
            "A_IRR_lt_target": (sum(v < target for v in irr_a_vals) / len(irr_a_vals)) if irr_a_vals else None,
            "B_IRR_lt_target": (sum(v < target for v in irr_b_vals) / len(irr_b_vals)) if irr_b_vals else None,
            "A_DSCR_lt_covenant": (sum(v < covenant for v in dscr_a_vals) / len(dscr_a_vals)) if covenant and dscr_a_vals else None,
            "B_DSCR_lt_covenant": (sum(v < covenant for v in dscr_b_vals) / len(dscr_b_vals)) if covenant and dscr_b_vals else None,
        }

        return MonteCarloSummary(
            a_irr=self._summary_block(irr_a_vals),
            b_irr=self._summary_block(irr_b_vals),
            a_dscr=self._summary_block(dscr_a_vals),
            b_dscr=self._summary_block(dscr_b_vals),
            prob=prob,
            counts={
                "iterations": self.platform.monte_carlo_iterations,
                "irr_a_samples": len(irr_a_vals),
                "irr_b_samples": len(irr_b_vals),
                "dscr_a_samples": len(dscr_a_vals),
                "dscr_b_samples": len(dscr_b_vals),
            },
        )

    def _run_sample(self, scenario: ScenarioSettings, sampled_assets: List[Tuple[AssetInputs, float]]) -> ScenarioLedger:
        total_capex = 0.0
        total_debt = 0.0
        total_equity = 0.0
        total_revenue = 0.0
        total_cfads = 0.0
        total_debt_service = 0.0

        for asset, sampled_revenue in sampled_assets:
            stressed_capex = asset.capex * (1.0 + self.platform.capex_overrun_pct)
            if scenario.debt_capped:
                debt = scenario.debt_cap_amount or 0.0
                equity = max(0.0, stressed_capex - debt)
            else:
                debt = stressed_capex * scenario.debt_ratio
                equity = stressed_capex * scenario.equity_ratio

            taxable_base = max(0.0, sampled_revenue - asset.base_fixed_om_annual)
            taxes = taxable_base * asset.tax_rate
            cfads = sampled_revenue - asset.base_fixed_om_annual - taxes - asset.reserves_annual
            debt_service = pmt(self.platform.debt_interest_rate, self.platform.debt_tenor_years, debt) if debt > 0 else 0.0

            total_capex += stressed_capex
            total_debt += debt
            total_equity += equity
            total_revenue += sampled_revenue
            total_cfads += cfads
            total_debt_service += debt_service

        dscr = (total_cfads / total_debt_service) if total_debt_service > 0 else None
        irr = self._sample_irr(total_equity, total_cfads, total_debt_service)

        return ScenarioLedger(
            name=scenario.name,
            total_capex=total_capex,
            total_debt=total_debt,
            total_equity=total_equity,
            annual_revenue=total_revenue,
            annual_cfads=total_cfads,
            annual_debt_service=total_debt_service,
            dscr=dscr,
            equity_irr=irr,
        )

    def _sample_irr(self, total_equity: float, total_cfads: float, total_debt_service: float) -> Optional[float]:
        construction_years = max(asset.cod_year - asset.construction_start_year for asset in self.platform.assets)
        construction_years = max(1, construction_years)
        operating_years = max(asset.operating_life_years for asset in self.platform.assets)
        equity_draws = [-(total_equity / construction_years) for _ in range(construction_years)]
        annual_equity_cash = total_cfads - total_debt_service
        ops = [annual_equity_cash for _ in range(operating_years)]
        return irr_bisection(equity_draws + ops)

    @staticmethod
    def _summary_block(values: List[float]) -> Dict[str, Optional[float]]:
        return {
            "P10": percentile(values, 0.10),
            "P50": percentile(values, 0.50),
            "P90": percentile(values, 0.90),
            "P95": percentile(values, 0.95),
        }


class MemoBuilder:
    @staticmethod
    def recommendation(
        a: ScenarioLedger,
        b: ScenarioLedger,
        mc: Optional[MonteCarloSummary],
        target_irr: float,
        min_dscr: Optional[float],
    ) -> Recommendation:
        a_bad = a.equity_irr is not None and a.equity_irr < target_irr
        b_bad = b.equity_irr is not None and b.equity_irr < target_irr
        a_dscr_bad = min_dscr is not None and a.dscr is not None and a.dscr < min_dscr
        b_dscr_bad = min_dscr is not None and b.dscr is not None and b.dscr < min_dscr

        mc_bad = False
        if mc is not None:
            pa = mc.prob.get("A_IRR_lt_target")
            pb = mc.prob.get("B_IRR_lt_target")
            if pa is not None and pb is not None and max(pa, pb) > 0.65:
                mc_bad = True

        if (a_bad and b_bad) or (a_dscr_bad or b_dscr_bad) or mc_bad:
            return Recommendation.DEFER
        if a_bad or b_bad:
            return Recommendation.PROCEED_WITH_MITIGATION
        return Recommendation.PROCEED

    @staticmethod
    def build(platform: PlatformInputs, a: ScenarioLedger, b: ScenarioLedger, mc: Optional[MonteCarloSummary]) -> str:
        rec = MemoBuilder.recommendation(a, b, mc, platform.target_equity_irr, platform.minimum_dscr)
        lines = []
        lines.append("**MEMO**")
        lines.append("")
        lines.append("**To:** Investment Committee")
        lines.append("**From:** Nemexis Master")
        lines.append("**Date:** [Insert Date]")
        lines.append(f"**Subject:** {platform.project_name}")
        lines.append("")
        lines.append(f"**Recommendation:** {rec.value}")
        lines.append("")
        lines.append("**Deterministic assessment**")
        lines.append(f"- Case A DSCR: {fmt_ratio(a.dscr)}; IRR: {fmt_pct(a.equity_irr)}")
        lines.append(f"- Case B DSCR: {fmt_ratio(b.dscr)}; IRR: {fmt_pct(b.equity_irr)}")
        if mc is not None:
            lines.append("")
            lines.append("**Monte Carlo risk view**")
            lines.append(f"- Prob(Case A IRR < target): {fmt_pct(mc.prob.get('A_IRR_lt_target'))}")
            lines.append(f"- Prob(Case B IRR < target): {fmt_pct(mc.prob.get('B_IRR_lt_target'))}")
            if platform.minimum_dscr is not None:
                lines.append(f"- Prob(Case A DSCR < covenant): {fmt_pct(mc.prob.get('A_DSCR_lt_covenant'))}")
                lines.append(f"- Prob(Case B DSCR < covenant): {fmt_pct(mc.prob.get('B_DSCR_lt_covenant'))}")
        lines.append("")
        lines.append("**Key risk drivers**")
        lines.append("1. Multi-asset execution risk across generation and transmission.")
        lines.append("2. Load ramp and utilization timing risk versus fixed financing obligations.")
        lines.append("3. CAPEX overrun and contract recovery risk through EPC, LDs, change orders, and insurance structure.")
        lines.append("")
        lines.append("**Next actions**")
        lines.append("1. Confirm revenue contracts, guarantees, and ramp schedule.")
        lines.append("2. Reconcile debt structure to actual covenant package and reserve mechanics.")
        lines.append("3. Validate top technical failure modes through independent technical diligence.")
        return "\n".join(lines)


def fmt_money(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"${x:,.0f}"


def fmt_ratio(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x:.2f}x"


def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:.2f}%"


def to_math_claims(a: ScenarioLedger, b: ScenarioLedger) -> Dict[str, object]:
    return {
        "claims": [
            {
                "id": "total_capex_case_a",
                "type": "arithmetic",
                "expr": str(a.total_capex),
                "inputs": {"scenario": a.name},
                "expected": a.total_capex,
                "units": "USD",
            },
            {
                "id": "total_debt_case_a",
                "type": "arithmetic",
                "expr": str(a.total_debt),
                "inputs": {"scenario": a.name},
                "expected": a.total_debt,
                "units": "USD",
            },
            {
                "id": "total_equity_case_a",
                "type": "arithmetic",
                "expr": str(a.total_equity),
                "inputs": {"scenario": a.name},
                "expected": a.total_equity,
                "units": "USD",
            },
            {
                "id": "debt_service_case_a",
                "type": "pmt",
                "inputs": {
                    "rate": a.asset_ledgers[0].annual_debt_service / a.asset_ledgers[0].debt if a.asset_ledgers and a.asset_ledgers[0].debt else 0.0,
                    "nper": 1,
                    "pv": a.total_debt,
                },
                "expected": a.annual_debt_service,
                "units": "USD_per_year",
            },
            {
                "id": "total_capex_case_b",
                "type": "arithmetic",
                "expr": str(b.total_capex),
                "inputs": {"scenario": b.name},
                "expected": b.total_capex,
                "units": "USD",
            },
            {
                "id": "total_debt_case_b",
                "type": "arithmetic",
                "expr": str(b.total_debt),
                "inputs": {"scenario": b.name},
                "expected": b.total_debt,
                "units": "USD",
            },
            {
                "id": "total_equity_case_b",
                "type": "arithmetic",
                "expr": str(b.total_equity),
                "inputs": {"scenario": b.name},
                "expected": b.total_equity,
                "units": "USD",
            },
            {
                "id": "debt_service_case_b",
                "type": "pmt",
                "inputs": {
                    "rate": b.asset_ledgers[0].annual_debt_service / b.asset_ledgers[0].debt if b.asset_ledgers and b.asset_ledgers[0].debt else 0.0,
                    "nper": 1,
                    "pv": b.total_debt,
                },
                "expected": b.annual_debt_service,
                "units": "USD_per_year",
            },
        ]
    }


def render_output(platform: PlatformInputs, a: ScenarioLedger, b: ScenarioLedger, mc: MonteCarloSummary) -> str:
    memo = MemoBuilder.build(platform, a, b, mc)
    lines = []
    lines.append("# Nemexis Output (v11)")
    lines.append("")
    lines.append("## Deterministic Ledger")
    lines.append(f"- Total CAPEX (incl. overrun): {fmt_money(a.total_capex)}")
    lines.append(f"- Case A: Debt {fmt_money(a.total_debt)}, Equity {fmt_money(a.total_equity)}, DSCR {fmt_ratio(a.dscr)}, IRR {fmt_pct(a.equity_irr)}")
    lines.append(f"- Case B: Debt {fmt_money(b.total_debt)}, Equity {fmt_money(b.total_equity)}, DSCR {fmt_ratio(b.dscr)}, IRR {fmt_pct(b.equity_irr)}")
    lines.append("")
    lines.append("## Monte Carlo Summary")
    lines.append("```json")
    lines.append(json.dumps(asdict(mc), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Final Memo")
    lines.append(memo)
    lines.append("")
    lines.append("## Math Claims (JSON)")
    lines.append("```json")
    lines.append(json.dumps(to_math_claims(a, b), indent=2))
    lines.append("```")
    return "\n".join(lines)


def example_clamster() -> PlatformInputs:
    generation = AssetInputs(
        name="Generation Phase 1",
        asset_type="generation",
        capex=677_000_000,
        construction_start_year=2026,
        cod_year=2029,
        operating_life_years=25,
        revenue=AssetRevenueSpec(
            model=RevenueModel.TOLLING,
            annual_revenue=124_000_000,
        ),
        tech=AssetTechSpec(nameplate_mw=360, availability=0.95, capacity_factor=0.50),
        base_fixed_om_annual=18_000_000,
        reserves_annual=3_000_000,
        tax_rate=0.20,
    )

    transmission = AssetInputs(
        name="Transmission Phase 2",
        asset_type="transmission",
        capex=504_000_000,
        construction_start_year=2026,
        cod_year=2030,
        operating_life_years=25,
        revenue=AssetRevenueSpec(
            model=RevenueModel.TRANSMISSION,
            annual_revenue=63_000_000,
            transmission_rate_per_kw_month=3.50,
            contracted_kw=1_500_000,
        ),
        tech=AssetTechSpec(nameplate_mw=2000, availability=0.98),
        base_fixed_om_annual=10_000_000,
        reserves_annual=2_000_000,
        tax_rate=0.20,
    )

    return PlatformInputs(
        project_name="Infrastructure Investment Evaluation — Clamster / Clarksville Hyperscale Power Platform",
        start_year=2026,
        debt_interest_rate=0.065,
        debt_tenor_years=20,
        target_equity_irr=0.08,
        minimum_dscr=1.35,
        assets=[generation, transmission],
        scenario_a=ScenarioSettings(name="Case A (Debt capped)", debt_ratio=0.0, equity_ratio=1.0, debt_capped=True, debt_cap_amount=0.6 * (677_000_000 + 504_000_000)),
        scenario_b=ScenarioSettings(name="Case B (Pro-rata)", debt_ratio=0.6, equity_ratio=0.4, debt_capped=False),
        capex_overrun_pct=0.15,
        monte_carlo_iterations=10000,
    )


def main() -> None:
    platform = example_clamster()
    det = DeterministicEngine(platform)
    a = det.build_scenario(platform.scenario_a)
    b = det.build_scenario(platform.scenario_b)

    mc_engine = MonteCarloEngine(
        platform=platform,
        om_pct=TriangularInput(0.01, 0.025, 0.04),
        tax_rate=TriangularInput(0.0, 0.20, 0.30),
        reserves_pct_of_ebitda=TriangularInput(0.0, 0.02, 0.05),
        revenue_factor=TriangularInput(0.80, 1.00, 1.10),
        capex_factor=TriangularInput(0.95, 1.00, 1.20),
    )
    mc = mc_engine.run()

    print(render_output(platform, a, b, mc))


if __name__ == "__main__":
    main()
