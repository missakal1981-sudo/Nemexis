from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from models.asset import Asset
from models.platform import Platform
from engine.finance import pmt, irr_bisection


@dataclass
class AssetResult:
    name: str
    annual_revenue: float
    annual_om: float
    annual_ebitda: float


@dataclass
class PlatformResult:
    total_capex: float
    total_revenue: float
    total_om: float
    total_ebitda: float
    debt: float
    equity: float
    annual_debt_service: float
    cfads: float
    dscr: float
    equity_irr: float | None
    cashflows_to_equity: List[float]
    asset_results: List[AssetResult]


class DeterministicEngine:
    def __init__(self, platform: Platform):
        self.platform = platform
        self.platform.validate()

    def _annual_revenue_for_asset(self, asset: Asset) -> float:
        if asset.revenue_mode in ("tolling", "transmission"):
            return asset.capacity_mw * 1000.0 * asset.toll_usd_per_kw_month * 12.0 * asset.utilization
        if asset.revenue_mode == "fixed":
            return float(asset.annual_revenue or 0.0)
        raise ValueError(f"Unsupported revenue_mode for {asset.name}: {asset.revenue_mode}")

    def _annual_om_for_asset(self, asset: Asset) -> float:
        variable = 0.0
        if asset.annual_mwh is not None:
            variable = asset.annual_mwh * asset.variable_om_per_mwh
        return asset.fixed_om_annual + variable

    def run(self) -> PlatformResult:
        asset_results: List[AssetResult] = []
        total_capex = 0.0
        total_revenue = 0.0
        total_om = 0.0

        max_construction_years = 0
        max_operating_years = 0

        for asset in self.platform.assets:
            total_capex += asset.capex
            revenue = self._annual_revenue_for_asset(asset)
            om = self._annual_om_for_asset(asset)
            ebitda = revenue - om

            asset_results.append(
                AssetResult(
                    name=asset.name,
                    annual_revenue=revenue,
                    annual_om=om,
                    annual_ebitda=ebitda,
                )
            )

            total_revenue += revenue
            total_om += om
            max_construction_years = max(max_construction_years, asset.construction_years)
            max_operating_years = max(max_operating_years, asset.operating_years)

        total_ebitda = total_revenue - total_om

        debt = total_capex * self.platform.debt_ratio
        equity = total_capex * self.platform.equity_ratio

        annual_debt_service = pmt(self.platform.debt_rate, self.platform.debt_tenor_years, debt)

        taxes = max(0.0, total_ebitda) * self.platform.tax_rate
        reserves = max(0.0, total_ebitda) * self.platform.reserve_pct_of_ebitda
        cfads = total_ebitda - taxes - reserves
        dscr = cfads / annual_debt_service if annual_debt_service > 0 else float("inf")

        cashflows_to_equity: List[float] = []

        # construction years: straight-line equity funding
        for _ in range(max_construction_years):
            cashflows_to_equity.append(-(equity / max_construction_years))

        # operating years
        for year in range(1, max_operating_years + 1):
            debt_service = annual_debt_service if year <= self.platform.debt_tenor_years else 0.0
            cf_equity = total_ebitda - taxes - reserves - debt_service
            cashflows_to_equity.append(cf_equity)

        equity_irr = irr_bisection(cashflows_to_equity)

        return PlatformResult(
            total_capex=total_capex,
            total_revenue=total_revenue,
            total_om=total_om,
            total_ebitda=total_ebitda,
            debt=debt,
            equity=equity,
            annual_debt_service=annual_debt_service,
            cfads=cfads,
            dscr=dscr,
            equity_irr=equity_irr,
            cashflows_to_equity=cashflows_to_equity,
            asset_results=asset_results,
        )
