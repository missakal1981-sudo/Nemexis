class Scenario:
    BASE = "Base Case"
    DOWNSIDE = "Downside"
    UPSIDE = "Upside"
  def apply_scenario(platform, scenario):
    p = platform.copy()

    if scenario == Scenario.DOWNSIDE:
        for a in p.assets:
            a.revenue *= 0.9
            a.capex *= 1.1
        p.debt_rate += 0.01

    elif scenario == Scenario.UPSIDE:
        for a in p.assets:
            a.revenue *= 1.1
            a.capex *= 0.95

    return p
