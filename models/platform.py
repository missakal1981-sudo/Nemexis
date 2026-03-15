from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from models.asset import Asset


@dataclass
class Platform:
    name: str
    assets: List[Asset] = field(default_factory=list)

    debt_ratio: float = 0.60
    equity_ratio: float = 0.40
    debt_rate: float = 0.065
    debt_tenor_years: int = 20
    target_irr: float = 0.08
    covenant_dscr: float = 1.35

    tax_rate: float = 0.0
    reserve_pct_of_ebitda: float = 0.0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Platform name is required.")
        if not self.assets:
            raise ValueError("At least one asset is required.")
        if abs((self.debt_ratio + self.equity_ratio) - 1.0) > 1e-9:
            raise ValueError("Debt ratio + equity ratio must equal 1.0.")
        if self.debt_rate < 0:
            raise ValueError("Debt rate must be >= 0.")
        if self.debt_tenor_years <= 0:
            raise ValueError("Debt tenor must be > 0.")
        if self.target_irr < -1:
            raise ValueError("Target IRR is invalid.")
        if self.covenant_dscr <= 0:
            raise ValueError("Covenant DSCR must be > 0.")

        for asset in self.assets:
            asset.validate()
