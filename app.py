from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ============================================================
# Nemexis v11.1
# Deterministic engine first. Memo never invents numbers.
# Multi-asset / multi-scenario / Monte Carlo structure.
# ============================================================


class RevenueModel(str, Enum):
    PPA = "ppa"
    TOLLING = "tolling"
    AVAILABILITY = "availability"
    TRANSMISSION = "transmission"
    HYBRID = "hybrid"
    FIXED = "fixed"


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
    monthly_fixed_om_pass_through_per_kw: Optional[float] = None
    variable_om_pass_through_per_mwh: Optional[float] = None
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

        # Universal rule: an explicit annual_revenue is sufficient input for any
        # revenue model. Only require model-specific fields when revenue must be
        # derived by the engine.
        if self.revenue.annual_revenue is not None:
            return

        if self.revenue.model == RevenueModel.TRANSMISSION:
            if self.revenue.transmission_rate_per_kw_month is None or self.revenue.contracted_kw is None:
                raise ValueError(f"{self.name}: transmission asset needs transmission_rate_per_kw_month and contracted_kw unless annual_revenue is provided")
        if self.revenue.model == RevenueModel.TOLLING:
            if self.tech.nameplate_mw is None:
                raise ValueError(f"{self.name}: tolling asset needs nameplate_mw unless annual_revenue is provided")
            if self.revenue.monthly_capacity_rate_per_kw is None:
                raise ValueError(f"{self.name}: tolling asset needs monthly_capacity_rate_per_kw unless annual_revenue is provided")


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
        if self.debt_interest_rate < 0:
            raise ValueError("debt_interest_rate must be >= 0")
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
        if self.scenario_a.debt_capped and self.scenario_a.debt_cap_amount is None:
            raise ValueError("Scenario A debt-capped mode requires debt_cap_amount")
        if self.scenario_b.debt_capped and self.scenario_b.debt_cap_amount is None:
            raise ValueError("Scenario B debt-capped mode requires debt_cap_amount")


@dataclass
class AssetLedger:
    name: str
    capex: float
    debt: float
    equity: float
    annual_revenue: float
    annual_cfads: float
    annual_debt_service: float
    dscr: Optional[float]


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
    if pv <= 0:
        return 0.0
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
    while f_low * f_high > 0 and expand_count < 50:
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
    vals = sorted(values)
    idx = (len(vals) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return vals[lo]
    frac = idx - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


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
            fixed_om_pass = kw * (rev.monthly_fixed_om_pass_through_per_kw or 0.0) * 12.0
            variable_om_pass = (rev.annual_volume_mwh or 0.0) * (rev.variable_om_pass_through_per_mwh or 0.0)
            return cap_rev + fixed_om_pass + variable_om_pass
        if rev.model == RevenueModel.TRANSMISSION:
            return (rev.transmission_rate_per_kw_month or 0.0) * (rev.contracted_kw or 0.0) * 12.0
        if rev.model == RevenueModel.AVAILABILITY:
            return rev.availability_payment_annual or 0.0
        if rev.model in {RevenueModel.PPA, RevenueModel.HYBRID, RevenueModel.FIXED}:
            return rev.annual_revenue or 0.0
        raise ValueError(f"Unsupported revenue model: {rev.model}")

    def annual_cfads(self, asset: AssetInputs, revenue_override: Optional[float] = None) -> float:
        revenue = self.annual_revenue(asset) if revenue_override is None else revenue_override
        taxable_base = max(0.0, revenue - asset.base_fixed_om_annual)
        taxes = taxable_base * asset.tax_rate
        return revenue - asset.base_fixed_om_annual - taxes - asset.reserves_annual

    def stressed_capex(self, asset: AssetInputs) -> float:
        return asset.capex * (1.0 + self.platform.capex_overrun_pct)

    def scenario_financing(self, stressed_capex: float, scenario: ScenarioSettings) -> Tuple[float, float]:
        if scenario.debt_capped:
            debt = min(scenario.debt_cap_amount or 0.0, stressed_capex)
            equity = stressed_capex - debt
            return debt, equity
        debt = stressed_capex * scenario.debt_ratio
        equity = stressed_capex * scenario.equity_ratio
        return debt, equity

    def build_scenario(self, scenario: ScenarioSettings) -> ScenarioLedger:
        asset_ledgers: List[AssetLedger] = []
        total_capex = total_debt = total_equity = 0.0
        total_revenue = total_cfads = total_debt_service = 0.0

        for asset in self.platform.assets:
            capex = self.stressed_capex(asset)
            debt, equity = self.scenario_financing(capex, scenario)
            revenue = self.annual_revenue(asset)
            cfads = self.annual_cfads(asset)
            debt_service = pmt(self.platform.debt_interest_rate, self.platform.debt_tenor_years, debt) if debt > 0 else 0.0
            dscr = cfads / debt_service if debt_service > 0 else None
            asset_ledgers.append(AssetLedger(asset.name, capex, debt, equity, revenue, cfads, debt_service, dscr))
            total_capex += capex
            total_debt += debt
            total_equity += equity
            total_revenue += revenue
            total_cfads += cfads
            total_debt_service += debt_service

        total_dscr = total_cfads / total_debt_service if total_debt_service > 0 else None
        irr = self.platform_equity_irr(total_equity, total_cfads, total_debt_service)
        self._check_identity(total_capex, total_debt, total_equity, scenario.name)

        return ScenarioLedger(
            name=scenario.name,
            total_capex=total_capex,
            total_debt=total_debt,
            total_equity=total_equity,
            annual_revenue=total_revenue,
            annual_cfads=total_cfads,
            annual_debt_service=total_debt_service,
            dscr=total_dscr,
            equity_irr=irr,
            asset_ledgers=asset_ledgers,
        )

    @staticmethod
    def _check_identity(total_capex: float, total_debt: float, total_equity: float, scenario_name: str) -> None:
        if not math.isclose(total_capex, total_debt + total_equity, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError(f"Scenario identity fails {scenario_name}: debt+equity={total_debt + total_equity} vs total_capex={total_capex}")

    def platform_equity_irr(self, total_equity: float, total_cfads: float, total_debt_service: float) -> Optional[float]:
        construction_years = max(1, max(a.cod_year - a.construction_start_year for a in self.platform.assets))
        operating_years = max(a.operating_life_years for a in self.platform.assets)
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
        reserves_pct_of_revenue: TriangularInput,
        revenue_factor: TriangularInput,
        capex_factor: TriangularInput,
    ):
        self.platform = platform
        self.om_pct = om_pct
        self.tax_rate = tax_rate
        self.reserves_pct_of_revenue = reserves_pct_of_revenue
        self.revenue_factor = revenue_factor
        self.capex_factor = capex_factor

    def run(self) -> MonteCarloSummary:
        irr_a_vals: List[float] = []
        irr_b_vals: List[float] = []
        dscr_a_vals: List[float] = []
        dscr_b_vals: List[float] = []

        for _ in range(self.platform.monte_carlo_iterations):
            assets: List[Tuple[AssetInputs, float]] = []
            det = DeterministicEngine(self.platform)
            for asset in self.platform.assets:
                base_rev = det.annual_revenue(asset)
                rev = base_rev * self.revenue_factor.sample()
                sampled_capex = asset.capex * self.capex_factor.sample()
                sampled_om = sampled_capex * self.om_pct.sample()
                sampled_tax = self.tax_rate.sample()
                sampled_res = rev * self.reserves_pct_of_revenue.sample()
                assets.append((
                    AssetInputs(
                        name=asset.name,
                        asset_type=asset.asset_type,
                        capex=sampled_capex,
                        construction_start_year=asset.construction_start_year,
                        cod_year=asset.cod_year,
                        operating_life_years=asset.operating_life_years,
                        revenue=asset.revenue,
                        tech=asset.tech,
                        base_fixed_om_annual=sampled_om,
                        reserves_annual=sampled_res,
                        tax_rate=sampled_tax,
                        capex_phasing=asset.capex_phasing,
                        load_ramp=asset.load_ramp,
                        notes=asset.notes,
                    ),
                    rev,
                ))

            a = self._run_sample(self.platform.scenario_a, assets)
            b = self._run_sample(self.platform.scenario_b, assets)
            if a.equity_irr is not None:
                irr_a_vals.append(a.equity_irr)
            if b.equity_irr is not None:
                irr_b_vals.append(b.equity_irr)
            if a.dscr is not None:
                dscr_a_vals.append(a.dscr)
            if b.dscr is not None:
                dscr_b_vals.append(b.dscr)

        covenant = self.platform.minimum_dscr
        target = self.platform.target_equity_irr
        prob = {
            "A_IRR_lt_target": (sum(v < target for v in irr_a_vals) / len(irr_a_vals)) if irr_a_vals else None,
            "B_IRR_lt_target": (sum(v < target for v in irr_b_vals) / len(irr_b_vals)) if irr_b_vals else None,
            "A_DSCR_lt_covenant": (sum(v < covenant for v in dscr_a_vals) / len(dscr_a_vals)) if covenant is not None and dscr_a_vals else None,
            "B_DSCR_lt_covenant": (sum(v < covenant for v in dscr_b_vals) / len(dscr_b_vals)) if covenant is not None and dscr_b_vals else None,
        }
        return MonteCarloSummary(
            a_irr=self._summary(irr_a_vals),
            b_irr=self._summary(irr_b_vals),
            a_dscr=self._summary(dscr_a_vals),
            b_dscr=self._summary(dscr_b_vals),
            prob=prob,
            counts={
                "iterations": self.platform.monte_carlo_iterations,
                "irr_a_samples": len(irr_a_vals),
                "irr_b_samples": len(irr_b_vals),
                "dscr_a_samples": len(dscr_a_vals),
                "dscr_b_samples": len(dscr_b_vals),
            },
        )

    def _run_sample(self, scenario: ScenarioSettings, assets: List[Tuple[AssetInputs, float]]) -> ScenarioLedger:
        total_capex = total_debt = total_equity = 0.0
        total_rev = total_cfads = total_debt_service = 0.0

        for asset, sampled_revenue in assets:
            stressed_capex = asset.capex * (1.0 + self.platform.capex_overrun_pct)
            if scenario.debt_capped:
                debt = min(scenario.debt_cap_amount or 0.0, stressed_capex)
                equity = stressed_capex - debt
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
            total_rev += sampled_revenue
            total_cfads += cfads
            total_debt_service += debt_service

        dscr = total_cfads / total_debt_service if total_debt_service > 0 else None
        irr = self._sample_irr(total_equity, total_cfads, total_debt_service)
        return ScenarioLedger(scenario.name, total_capex, total_debt, total_equity, total_rev, total_cfads, total_debt_service, dscr, irr)

    def _sample_irr(self, total_equity: float, total_cfads: float, total_debt_service: float) -> Optional[float]:
        construction_years = max(1, max(a.cod_year - a.construction_start_year for a in self.platform.assets))
        operating_years = max(a.operating_life_years for a in self.platform.assets)
        draws = [-(total_equity / construction_years) for _ in range(construction_years)]
        ops = [total_cfads - total_debt_service for _ in range(operating_years)]
        return irr_bisection(draws + ops)

    @staticmethod
    def _summary(values: List[float]) -> Dict[str, Optional[float]]:
        return {
            "P10": percentile(values, 0.10),
            "P50": percentile(values, 0.50),
            "P90": percentile(values, 0.90),
            "P95": percentile(values, 0.95),
        }


class MemoBuilder:
    @staticmethod
    def recommendation(platform: PlatformInputs, a: ScenarioLedger, b: ScenarioLedger, mc: MonteCarloSummary) -> Recommendation:
        target = platform.target_equity_irr
        covenant = platform.minimum_dscr
        fail_irr = (a.equity_irr is not None and a.equity_irr < target) and (b.equity_irr is not None and b.equity_irr < target)
        fail_dscr = False
        if covenant is not None:
            fail_dscr = ((a.dscr is not None and a.dscr < covenant) or (b.dscr is not None and b.dscr < covenant))
        high_prob = False
        pa = mc.prob.get("A_IRR_lt_target")
        pb = mc.prob.get("B_IRR_lt_target")
        if pa is not None and pb is not None and max(pa, pb) > 0.65:
            high_prob = True
        if fail_irr or fail_dscr or high_prob:
            return Recommendation.DEFER
        return Recommendation.PROCEED_WITH_MITIGATION

    @staticmethod
    def build(platform: PlatformInputs, a: ScenarioLedger, b: ScenarioLedger, mc: MonteCarloSummary) -> str:
        rec = MemoBuilder.recommendation(platform, a, b, mc)
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
        lines.append(f"- Case A: Debt {fmt_money(a.total_debt)}, Equity {fmt_money(a.total_equity)}, DSCR {fmt_ratio(a.dscr)}, IRR {fmt_pct(a.equity_irr)}")
        lines.append(f"- Case B: Debt {fmt_money(b.total_debt)}, Equity {fmt_money(b.total_equity)}, DSCR {fmt_ratio(b.dscr)}, IRR {fmt_pct(b.equity_irr)}")
        lines.append("")
        lines.append("**Monte Carlo risk view**")
        lines.append(f"- Prob(Case A IRR < target): {fmt_pct(mc.prob.get('A_IRR_lt_target'))}")
        lines.append(f"- Prob(Case B IRR < target): {fmt_pct(mc.prob.get('B_IRR_lt_target'))}")
        if platform.minimum_dscr is not None:
            lines.append(f"- Prob(Case A DSCR < covenant): {fmt_pct(mc.prob.get('A_DSCR_lt_covenant'))}")
            lines.append(f"- Prob(Case B DSCR < covenant): {fmt_pct(mc.prob.get('B_DSCR_lt_covenant'))}")
        lines.append("")
        lines.append("**Key risk drivers**")
        lines.append("1. Asset-level execution and commissioning risk.")
        lines.append("2. Contracted revenue durability and ramp timing risk.")
        lines.append("3. CAPEX overrun recovery through contracts, insurance, and sponsor support.")
        return "\n".join(lines)


class OutputRenderer:
    @staticmethod
    def math_claims(a: ScenarioLedger, b: ScenarioLedger, platform: PlatformInputs) -> Dict[str, object]:
        return {
            "claims": [
                {
                    "id": "case_a_total_capex",
                    "type": "arithmetic",
                    "expr": str(a.total_capex),
                    "inputs": {"scenario": a.name},
                    "expected": a.total_capex,
                    "units": "USD",
                },
                {
                    "id": "case_a_total_debt",
                    "type": "arithmetic",
                    "expr": str(a.total_debt),
                    "inputs": {"scenario": a.name},
                    "expected": a.total_debt,
                    "units": "USD",
                },
                {
                    "id": "case_a_total_equity",
                    "type": "arithmetic",
                    "expr": str(a.total_equity),
                    "inputs": {"scenario": a.name},
                    "expected": a.total_equity,
                    "units": "USD",
                },
                {
                    "id": "case_a_debt_service",
                    "type": "pmt",
                    "inputs": {
                        "rate": platform.debt_interest_rate,
                        "nper": platform.debt_tenor_years,
                        "pv": a.total_debt,
                    },
                    "expected": a.annual_debt_service,
                    "units": "USD_per_year",
                },
                {
                    "id": "case_b_total_capex",
                    "type": "arithmetic",
                    "expr": str(b.total_capex),
                    "inputs": {"scenario": b.name},
                    "expected": b.total_capex,
                    "units": "USD",
                },
                {
                    "id": "case_b_total_debt",
                    "type": "arithmetic",
                    "expr": str(b.total_debt),
                    "inputs": {"scenario": b.name},
                    "expected": b.total_debt,
                    "units": "USD",
                },
                {
                    "id": "case_b_total_equity",
                    "type": "arithmetic",
                    "expr": str(b.total_equity),
                    "inputs": {"scenario": b.name},
                    "expected": b.total_equity,
                    "units": "USD",
                },
                {
                    "id": "case_b_debt_service",
                    "type": "pmt",
                    "inputs": {
                        "rate": platform.debt_interest_rate,
                        "nper": platform.debt_tenor_years,
                        "pv": b.total_debt,
                    },
                    "expected": b.annual_debt_service,
                    "units": "USD_per_year",
                },
            ]
        }

    @staticmethod
    def render(platform: PlatformInputs, a: ScenarioLedger, b: ScenarioLedger, mc: MonteCarloSummary) -> str:
        lines: List[str] = []
        lines.append("# Nemexis Output (v11.1)")
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
        lines.append(MemoBuilder.build(platform, a, b, mc))
        lines.append("")
        lines.append("## Math Claims (JSON)")
        lines.append("```json")
        lines.append(json.dumps(OutputRenderer.math_claims(a, b, platform), indent=2))
        lines.append("```")
        return "\n".join(lines)


# ------------------------------------------------------------
# Example prompt pack: Clamster / Clarksville platform
# ------------------------------------------------------------


def example_clamster() -> PlatformInputs:
    generation = AssetInputs(
        name="BTM Gas Generation",
        asset_type="generation",
        capex=677_000_000,
        construction_start_year=2026,
        cod_year=2029,
        operating_life_years=25,
        revenue=AssetRevenueSpec(model=RevenueModel.FIXED, annual_revenue=124_000_000),
        tech=AssetTechSpec(nameplate_mw=360, availability=0.95, capacity_factor=0.50),
        base_fixed_om_annual=18_000_000,
        reserves_annual=3_000_000,
        tax_rate=0.20,
    )

    transmission = AssetInputs(
        name="Private 500kV Transmission",
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

    base_total_capex = generation.capex + transmission.capex
    base_debt = base_total_capex * 0.60

    return PlatformInputs(
        project_name="Infrastructure Investment Evaluation — Clamster / Clarksville Hyperscale Power Platform",
        start_year=2026,
        debt_interest_rate=0.065,
        debt_tenor_years=20,
        target_equity_irr=0.08,
        minimum_dscr=1.35,
        assets=[generation, transmission],
        scenario_a=ScenarioSettings(
            name="Case A (Debt capped)",
            debt_ratio=0.0,
            equity_ratio=1.0,
            debt_capped=True,
            debt_cap_amount=base_debt,
        ),
        scenario_b=ScenarioSettings(
            name="Case B (Pro-rata)",
            debt_ratio=0.60,
            equity_ratio=0.40,
            debt_capped=False,
        ),
        capex_overrun_pct=0.15,
        monte_carlo_iterations=10000,
    )


# ------------------------------------------------------------
# Runner
# ------------------------------------------------------------


def main() -> None:
    platform = example_clamster()
    det = DeterministicEngine(platform)
    scenario_a = det.build_scenario(platform.scenario_a)
    scenario_b = det.build_scenario(platform.scenario_b)

    mc = MonteCarloEngine(
        platform=platform,
        om_pct=TriangularInput(0.01, 0.025, 0.04),
        tax_rate=TriangularInput(0.00, 0.20, 0.30),
        reserves_pct_of_revenue=TriangularInput(0.00, 0.02, 0.05),
        revenue_factor=TriangularInput(0.80, 1.00, 1.10),
        capex_factor=TriangularInput(0.95, 1.00, 1.20),
    ).run()

    print(OutputRenderer.render(platform, scenario_a, scenario_b, mc))


if __name__ == "__main__":
    main()
