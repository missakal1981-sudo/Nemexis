def run_sensitivity(platform):
    results = {}

    for capex_shift in [-0.2, -0.1, 0, 0.1, 0.2]:
        p = platform.copy()
        for a in p.assets:
            a.capex *= (1 + capex_shift)

        res = run_deterministic(p)
        results[f"capex_{capex_shift}"] = res["equity_irr"]

    return results
