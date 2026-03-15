from __future__ import annotations

import json
import random
from typing import Dict, List

from models.asset import Asset
from models.platform import Platform
from engine.deterministic import DeterministicEngine
from engine.finance import triangular_sample


def percentile(values: List[float], p: float):
    if not values:
        return None
    values = sorted(values)
    idx = int(round((len(values) - 1) * p))
    return values[idx]


class MonteCarloEngine:
    def __init__(self, platform: Platform, iterations: int = 10000, seed: int = 42):
        self.platform = platform
        self.iterations = iterations
        self.seed = seed

    def _sample_asset(self, asset: Asset, rng: random.Random) -> Asset:
        sampled = Asset(**asset.__dict__)

        if asset.capex_low is not None and asset.capex_base is not None and asset.capex_high is not None:
            sampled.capex = triangular_sample(asset.capex_low, asset.capex_base, asset.capex_high, rng)

        if asset.revenue_low is not None and asset.revenue_base is not None and asset.revenue_high is not None:
            sampled.annual_revenue = triangular_sample(asset.revenue_low, asset.revenue_base, asset.revenue_high, rng)

        if asset.om_low is not None and asset.om_base is not None and asset.om_high is not None:
            sampled.fixed_om_annual = triangular_sample(asset.om_low, asset.om_base, asset.om_high, rng)

        return sampled

    def run(self) -> Dict:
        rng = random.Random(self.seed)

        irrs: List[float] = []
        dscrs: List[float] = []

        for _ in range(self.iterations):
            sampled_assets = [self._sample_asset(a, rng) for a in self.platform.assets]
            sampled_platform = Platform(
                name=self.platform.name,
                assets=sampled_assets,
                debt_ratio=self.platform.debt_ratio,
                equity_ratio=self.platform.equity_ratio,
                debt_rate=self.platform.debt_rate,
                debt_tenor_years=self.platform.debt_tenor_years,
                target_irr=self.platform.target_irr,
                covenant_dscr=self.platform.covenant_dscr,
                tax_rate=self.platform.tax_rate,
                reserve_pct_of_ebitda=self.platform.reserve_pct_of_ebitda,
            )

            result = DeterministicEngine(sampled_platform).run()

            if result.equity_irr is not None:
                irrs.append(result.equity_irr)
            dscrs.append(result.dscr)

        return {
            "IRR": {
                "P10": percentile(irrs, 0.10),
                "P50": percentile(irrs, 0.50),
                "P90": percentile(irrs, 0.90),
                "Prob_lt_target": (sum(1 for x in irrs if x < self.platform.target_irr) / len(irrs)) if irrs else None,
            },
            "DSCR": {
                "P10": percentile(dscrs, 0.10),
                "P50": percentile(dscrs, 0.50),
                "P90": percentile(dscrs, 0.90),
                "Prob_lt_covenant": (sum(1 for x in dscrs if x < self.platform.covenant_dscr) / len(dscrs)) if dscrs else None,
            },
            "counts": {
                "iterations": self.iterations,
                "irr_samples": len(irrs),
                "dscr_samples": len(dscrs),
            },
        }
