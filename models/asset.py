from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal


RevenueMode = Literal[
    "tolling",      # $/kW-month
    "fixed",        # direct annual revenue
    "transmission", # $/kW-month
    "ppa",          # reserved for future
]


@dataclass
class Asset:
    name: str
    asset_type: str  # e.g. generation, transmission, battery, hybrid
    capacity_mw: float
    capex: float

    revenue_mode: RevenueMode = "fixed"

    # Commercial inputs
    toll_usd_per_kw_month: Optional[float] = None
    annual_revenue: Optional[float] = None
    utilization: float = 1.0

    # Cost inputs
    fixed_om_annual: float = 0.0
    variable_om_per_mwh: float = 0.0
    annual_mwh: Optional[float] = None

    # Timing
    construction_years: int = 3
    operating_years: int = 25

    # Risk / MC inputs
    capex_low: Optional[float] = None
    capex_base: Optional[float] = None
    capex_high: Optional[float] = None

    revenue_low: Optional[float] = None
    revenue_base: Optional[float] = None
    revenue_high: Optional[float] = None

    om_low: Optional[float] = None
    om_base: Optional[float] = None
    om_high: Optional[float] = None

    metadata: dict = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Asset name is required.")
        if self.capacity_mw <= 0:
            raise ValueError(f"{self.name}: capacity_mw must be > 0.")
        if self.capex < 0:
            raise ValueError(f"{self.name}: capex must be >= 0.")
        if self.construction_years <= 0:
            raise ValueError(f"{self.name}: construction_years must be > 0.")
        if self.operating_years <= 0:
            raise ValueError(f"{self.name}: operating_years must be > 0.")
        if not (0 < self.utilization <= 1.5):
            raise ValueError(f"{self.name}: utilization must be > 0 and <= 1.5.")

        if self.revenue_mode in ("tolling", "transmission"):
            if self.toll_usd_per_kw_month is None:
                raise ValueError(
                    f"{self.name}: tolling/transmission asset needs toll_usd_per_kw_month."
                )

        if self.revenue_mode == "fixed":
            if self.annual_revenue is None:
                raise ValueError(f"{self.name}: fixed revenue asset needs annual_revenue.")
