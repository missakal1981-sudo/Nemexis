import os
import json
import time
import math
import random
import datetime
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components

from openai import OpenAI
import anthropic
import requests

# ============================
# Nemexis v10.3.6
# Universal architecture:
# - LLMs produce narrative only (no math JSON requirement).
# - Engine computes ledger deterministically.
# - Monte Carlo (10,000) quantifies uncertainty with triangular defaults.
# - Iterative loop always ends with a final memo (LLM can't block it).
# ============================

# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="Nemexis v10.3.6", layout="wide")
st.title("Nemexis v10.3.6 — Universal Reliability (Deterministic Ledger + Monte Carlo 10,000)")

# ----------------------------
# Secrets
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
XAI_API_KEY = os.getenv("XAI_API_KEY", "").strip()
APP_PASSWORD = os.getenv("NEMEXIS_PASSWORD", "").strip()

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# ----------------------------
# Password UX
# ----------------------------
if "unlocked" not in st.session_state:
    st.session_state["unlocked"] = False

if APP_PASSWORD:
    st.markdown("### Access")
    pw = st.text_input("Password", type="password", value="", placeholder="Enter password")
    unlock = st.button("Unlock")
    if unlock:
        if pw == APP_PASSWORD:
            st.session_state["unlocked"] = True
            st.success("Unlocked ✅")
        else:
            st.session_state["unlocked"] = False
            st.error("Wrong password ❌")
    if not st.session_state["unlocked"]:
        st.stop()

# ----------------------------
# Clipboard helper
# ----------------------------
def copy_button(text: str, label: str = "📋 Copy"):
    safe = text.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    html = f"""
    <button id="copybtn" style="
        padding:10px 14px; border-radius:10px; border:1px solid #ddd;
        background:white; cursor:pointer; font-weight:600;">
        {label}
    </button>
    <span id="copymsg" style="margin-left:10px; font-weight:600;"></span>
    <script>
    const btn = document.getElementById('copybtn');
    const msg = document.getElementById('copymsg');
    btn.onclick = async () => {{
        try {{
            await navigator.clipboard.writeText(`{safe}`);
            msg.textContent = "Copied ✅";
            setTimeout(() => msg.textContent = "", 2000);
        }} catch (e) {{
            msg.textContent = "Copy failed ❌";
            setTimeout(() => msg.textContent = "", 3000);
        }}
    }};
    </script>
    """
    components.html(html, height=60)

# ----------------------------
# Deterministic finance engine
# ----------------------------
def pmt(rate: float, nper: int, pv: float, fv: float = 0.0, when: int = 0) -> float:
    """
    Payment for fully-amortizing loan.
    rate: per-period rate (annual if nper in years).
    when: 0=end period, 1=begin period
    Returns payment (positive number means cash outflow from borrower).
    """
    if nper <= 0:
        raise ValueError("nper must be > 0")
    if abs(rate) < 1e-12:
        return (pv + fv) / nper
    factor = (1 + rate) ** nper
    payment = (rate * (pv * factor + fv)) / ((factor - 1) * (1 + rate * when))
    return payment

def irr(cashflows: List[float], guess: float = 0.10, max_iter: int = 200, tol: float = 1e-8) -> float:
    """
    Newton-Raphson IRR. cashflows[0] is time 0.
    """
    if not (any(cf < 0 for cf in cashflows) and any(cf > 0 for cf in cashflows)):
        raise ValueError("IRR requires at least one negative and one positive cashflow.")
    r = guess
    for _ in range(max_iter):
        npv_val = 0.0
        d_val = 0.0
        for t, cf in enumerate(cashflows):
            denom = (1 + r) ** t
            npv_val += cf / denom
            if t > 0:
                d_val -= t * cf / ((1 + r) ** (t + 1))
        if abs(npv_val) < tol:
            return r
        if abs(d_val) < 1e-18:
            break
        r = r - npv_val / d_val
    raise ValueError("IRR did not converge")

def triangular(min_v: float, mode_v: float, max_v: float) -> float:
    return random.triangular(min_v, max_v, mode_v)

@dataclass
class TriParam:
    low: float
    mode: float
    high: float

@dataclass
class Inputs:
    capex: float
    overrun_pct: float
    debt_ratio: float
    equity_ratio: float
    interest_rate: float
    debt_tenor_years: int
    project_life_years: int
    construction_years: int
    ebitda: float
    om_pct_of_capex: float
    tax_rate: float
    reserves_pct_of_ebitda: float

@dataclass
class ScenarioResult:
    name: str
    total_capex: float
    debt: float
    equity: float
    annual_debt_service: float
    annual_cfads: float
    dscr: float
    equity_irr: Optional[float]
    notes: str

def compute_cfads(inputs: Inputs, total_capex: float) -> float:
    o_and_m = inputs.om_pct_of_capex * total_capex
    taxable_base = max(0.0, inputs.ebitda - o_and_m)
    taxes = inputs.tax_rate * taxable_base
    reserves = inputs.reserves_pct_of_ebitda * inputs.ebitda
    return inputs.ebitda - o_and_m - taxes - reserves

def equity_cashflows_simple(inputs: Inputs, debt_principal: float, equity_principal: float, annual_cfads: float) -> List[float]:
    """
    Simple levered equity cashflow model:
    - Construction years: equity paid per phasing; debt drawn per phasing; IDC capitalized (simplified).
    - Operations: yearly CF to equity = CFADS - debt_service (during tenor), then CFADS after debt maturity.
    This is intentionally simple and universal; you can refine later.
    """
    # CAPEX phasing (fixed, universal default)
    # Note: keep deterministic (you can make it a distribution in MC if desired).
    phasing = [0.30, 0.40, 0.30]
    if inputs.construction_years != 3:
        # Generic fallback: equal phasing
        phasing = [1.0 / inputs.construction_years] * inputs.construction_years

    # Equity and debt draws
    equity_draws = [-equity_principal * p for p in phasing]
    # Operating years after construction
    ops_years = inputs.project_life_years
    # Debt service annual (level PMT)
    debt_service = pmt(inputs.interest_rate, inputs.debt_tenor_years, debt_principal)

    equity_cfs = []
    # time 0..construction_years-1
    equity_cfs.extend(equity_draws)

    # ops years (start right after construction)
    for y in range(1, ops_years + 1):
        if y <= inputs.debt_tenor_years:
            equity_cfs.append(annual_cfads - debt_service)
        else:
            equity_cfs.append(annual_cfads)

    return equity_cfs

def run_scenarios(inputs: Inputs) -> Tuple[ScenarioResult, ScenarioResult]:
    base_debt = inputs.capex * inputs.debt_ratio
    base_equity = inputs.capex * inputs.equity_ratio
    overrun_amt = inputs.capex * inputs.overrun_pct
    total_capex = inputs.capex + overrun_amt

    # Case A: overrun fully equity-funded; debt capped at base_debt
    debt_a = base_debt
    equity_a = base_equity + overrun_amt

    # Case B: incremental pro-rata on overrun
    debt_b = base_debt + inputs.debt_ratio * overrun_amt
    equity_b = base_equity + inputs.equity_ratio * overrun_amt

    # Debt service
    ds_a = pmt(inputs.interest_rate, inputs.debt_tenor_years, debt_a)
    ds_b = pmt(inputs.interest_rate, inputs.debt_tenor_years, debt_b)

    # CFADS (assumes EBITDA is steady-state; same CFADS used in both cases on total CAPEX after overrun)
    cfads = compute_cfads(inputs, total_capex)

    dscr_a = cfads / ds_a if ds_a > 0 else float("inf")
    dscr_b = cfads / ds_b if ds_b > 0 else float("inf")

    irr_a = None
    irr_b = None
    notes = "IRR computed with simple levered cashflow model (construction phasing + level debt service)."
    try:
        cf_a = equity_cashflows_simple(inputs, debt_a, equity_a, cfads)
        irr_a = irr(cf_a, guess=0.08)
    except Exception:
        irr_a = None
    try:
        cf_b = equity_cashflows_simple(inputs, debt_b, equity_b, cfads)
        irr_b = irr(cf_b, guess=0.08)
    except Exception:
        irr_b = None

    a = ScenarioResult(
        name="Case A (Debt capped, overrun equity-funded)",
        total_capex=total_capex,
        debt=debt_a,
        equity=equity_a,
        annual_debt_service=ds_a,
        annual_cfads=cfads,
        dscr=dscr_a,
        equity_irr=irr_a,
        notes=notes
    )
    b = ScenarioResult(
        name="Case B (Pro-rata 60/40 on overrun)",
        total_capex=total_capex,
        debt=debt_b,
        equity=equity_b,
        annual_debt_service=ds_b,
        annual_cfads=cfads,
        dscr=dscr_b,
        equity_irr=irr_b,
        notes=notes
    )
    return a, b

# ----------------------------
# Monte Carlo
# ----------------------------
@dataclass
class MCConfig:
    n: int
    overrun_pct: TriParam
    om_pct: TriParam
    tax_rate: TriParam
    reserves_pct: TriParam
    interest_rate: TriParam

def mc_run(base: Inputs, mc: MCConfig) -> Dict[str, Any]:
    """
    Runs MC and returns percentiles + probabilities for both cases.
    """
    results = {
        "A_IRR": [],
        "B_IRR": [],
        "A_DSCR": [],
        "B_DSCR": [],
        "samples": 0
    }

    for _ in range(mc.n):
        sampled = Inputs(
            capex=base.capex,
            overrun_pct=triangular(mc.overrun_pct.low, mc.overrun_pct.mode, mc.overrun_pct.high),
            debt_ratio=base.debt_ratio,
            equity_ratio=base.equity_ratio,
            interest_rate=triangular(mc.interest_rate.low, mc.interest_rate.mode, mc.interest_rate.high),
            debt_tenor_years=base.debt_tenor_years,
            project_life_years=base.project_life_years,
            construction_years=base.construction_years,
            ebitda=base.ebitda,
            om_pct_of_capex=triangular(mc.om_pct.low, mc.om_pct.mode, mc.om_pct.high),
            tax_rate=triangular(mc.tax_rate.low, mc.tax_rate.mode, mc.tax_rate.high),
            reserves_pct_of_ebitda=triangular(mc.reserves_pct.low, mc.reserves_pct.mode, mc.reserves_pct.high),
        )
        a, b = run_scenarios(sampled)

        if a.equity_irr is not None and b.equity_irr is not None:
            results["A_IRR"].append(a.equity_irr)
            results["B_IRR"].append(b.equity_irr)
            results["A_DSCR"].append(a.dscr)
            results["B_DSCR"].append(b.dscr)
            results["samples"] += 1

    def pct(vals, p):
        if not vals:
            return None
        vs = sorted(vals)
        k = int(round((p/100) * (len(vs)-1)))
        return vs[k]

    out = {}
    for key in ["A_IRR","B_IRR","A_DSCR","B_DSCR"]:
        vals = results[key]
        out[key] = {
            "P10": pct(vals, 10),
            "P50": pct(vals, 50),
            "P90": pct(vals, 90),
            "P95": pct(vals, 95),
        }

    out["samples_used"] = results["samples"]
    return out

# ----------------------------
# LLM calls (narrative only)
# ----------------------------
def call_openai(model_name: str, system: str, user_text: str, temperature=0.2) -> str:
    if not openai_client:
        return "❌ OpenAI not configured"
    resp = openai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_text}],
        temperature=temperature,
    )
    return resp.choices[0].message.content

def call_claude(model_name: str, system: str, user_text: str) -> str:
    if not anthropic_client:
        return "❌ Claude not configured"
    msg = anthropic_client.messages.create(
        model=model_name,
        max_tokens=1500,
        temperature=0.2,
        system=system,
        messages=[{"role": "user", "content": user_text}],
    )
    out = ""
    for block in msg.content:
        if hasattr(block, "text"):
            out += block.text
    return out.strip()

def call_grok(model_name: str, system: str, user_text: str) -> str:
    if not XAI_API_KEY:
        return "❌ Grok not configured"
    last_err = None
    for attempt in range(3):
        try:
            r = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": model_name,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_text}],
                    "temperature": 0.2,
                },
                timeout=140,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            last_err = f"❌ Grok Error ({r.status_code}): {r.text}"
        except Exception as e:
            last_err = f"❌ Grok Error: {str(e)}"
        time.sleep(1.5 * (attempt + 1))
    return last_err or "❌ Grok Error: unknown"

# ----------------------------
# UI: models
# ----------------------------
st.markdown("### Models")
available_leaders = []
if OPENAI_API_KEY:
    available_leaders.append("OpenAI")
if ANTHROPIC_API_KEY:
    available_leaders.append("Claude")
if XAI_API_KEY:
    available_leaders.append("Grok")
if not available_leaders:
    st.error("No model keys configured.")
    st.stop()

leader = st.selectbox("Leader (Master)", available_leaders, index=0)

c1, c2, c3 = st.columns(3)
with c1:
    openai_model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o"], index=0, disabled=(not OPENAI_API_KEY))
with c2:
    claude_model = st.selectbox("Claude model", ["claude-sonnet-4-20250514","claude-opus-4-20250514","claude-haiku-4-20250514"], index=0, disabled=(not ANTHROPIC_API_KEY))
with c3:
    grok_model = st.selectbox("Grok model", ["grok-4-fast","grok-4"], index=0, disabled=(not XAI_API_KEY))

def leader_call(system: str, user_text: str, temperature=0.2):
    if leader == "OpenAI":
        return call_openai(openai_model, system, user_text, temperature=temperature)
    if leader == "Claude":
        return call_claude(claude_model, system, user_text)
    return call_grok(grok_model, system, user_text)

# Critics toggles
st.markdown("### Critics")
use_claude = st.checkbox("Use Claude as critic", value=bool(ANTHROPIC_API_KEY))
use_grok = st.checkbox("Use Grok as critic", value=bool(XAI_API_KEY))

# ----------------------------
# Universal Inputs (manual UI)
# ----------------------------
st.markdown("## Universal Ledger Inputs (manual)")

colA, colB, colC = st.columns(3)
with colA:
    capex = st.number_input("CAPEX (USD)", value=3_200_000_000.0, step=10_000_000.0, format="%.0f")
    ebitda = st.number_input("Annual EBITDA (USD)", value=420_000_000.0, step=1_000_000.0, format="%.0f")
    overrun_pct_mode = st.number_input("Overrun % (mode)", value=0.15, step=0.01, format="%.3f")
with colB:
    debt_ratio = st.number_input("Debt ratio", value=0.60, step=0.01, format="%.2f")
    equity_ratio = st.number_input("Equity ratio", value=0.40, step=0.01, format="%.2f")
    interest_rate_mode = st.number_input("Debt interest rate (mode)", value=0.065, step=0.001, format="%.4f")
with colC:
    debt_tenor = st.number_input("Debt tenor (years)", value=20, step=1)
    proj_life = st.number_input("Project life (years)", value=25, step=1)
    constr_years = st.number_input("Construction period (years)", value=3, step=1)

# Basic validation
if abs((debt_ratio + equity_ratio) - 1.0) > 1e-6:
    st.warning("Debt ratio + Equity ratio should equal 1.0 for consistent scenario arithmetic.")

# ----------------------------
# Assumption distributions (Triangular defaults) for Monte Carlo
# ----------------------------
st.markdown("## Monte Carlo (10,000) — Triangular defaults")

mc_enabled = st.checkbox("Enable Monte Carlo", value=True)
mc_n = st.number_input("Iterations", value=10_000, step=1_000)

c4, c5, c6 = st.columns(3)
with c4:
    st.caption("Overrun % (Triangular)")
    overrun_low = st.number_input("Overrun low", value=max(0.0, overrun_pct_mode - 0.10), step=0.01, format="%.3f")
    overrun_mode = st.number_input("Overrun mode", value=overrun_pct_mode, step=0.01, format="%.3f")
    overrun_high = st.number_input("Overrun high", value=min(1.0, overrun_pct_mode + 0.15), step=0.01, format="%.3f")
with c5:
    st.caption("O&M % of CAPEX (Triangular)")
    om_low = st.number_input("O&M low", value=0.01, step=0.005, format="%.3f")
    om_mode = st.number_input("O&M mode", value=0.025, step=0.005, format="%.3f")
    om_high = st.number_input("O&M high", value=0.04, step=0.005, format="%.3f")
with c6:
    st.caption("Tax / Reserves / Interest (Triangular)")
    tax_low = st.number_input("Tax low", value=0.0, step=0.01, format="%.3f")
    tax_mode = st.number_input("Tax mode", value=0.20, step=0.01, format="%.3f")
    tax_high = st.number_input("Tax high", value=0.30, step=0.01, format="%.3f")

    res_low = st.number_input("Reserves low", value=0.0, step=0.005, format="%.3f")
    res_mode = st.number_input("Reserves mode", value=0.02, step=0.005, format="%.3f")
    res_high = st.number_input("Reserves high", value=0.05, step=0.005, format="%.3f")

    ir_low = st.number_input("Interest low", value=max(0.0, interest_rate_mode - 0.02), step=0.001, format="%.4f")
    ir_mode = st.number_input("Interest mode", value=interest_rate_mode, step=0.001, format="%.4f")
    ir_high = st.number_input("Interest high", value=interest_rate_mode + 0.02, step=0.001, format="%.4f")

# ----------------------------
# Prompt
# ----------------------------
st.markdown("## Prompt (for the narrative loop)")
prompt = st.text_area("Your question / task", height=200)

iterations = st.number_input("Critique iterations (Master→Critics→Master)", value=2, step=1, min_value=1, max_value=6)

run = st.button("Run Nemexis v10.3.6")

# ----------------------------
# Engine: deterministic ledger (point-estimate)
# ----------------------------
def build_point_inputs() -> Inputs:
    return Inputs(
        capex=float(capex),
        overrun_pct=float(overrun_mode),
        debt_ratio=float(debt_ratio),
        equity_ratio=float(equity_ratio),
        interest_rate=float(ir_mode),
        debt_tenor_years=int(debt_tenor),
        project_life_years=int(proj_life),
        construction_years=int(constr_years),
        ebitda=float(ebitda),
        om_pct_of_capex=float(om_mode),
        tax_rate=float(tax_mode),
        reserves_pct_of_ebitda=float(res_mode),
    )

def ledger_to_markdown(a: ScenarioResult, b: ScenarioResult) -> str:
    def usd(x): return f"${x:,.0f}"
    def pct(x): return f"{100*x:,.2f}%"

    lines = []
    lines.append("### Deterministic Ledger (point estimate)")
    lines.append(f"- Total CAPEX (incl. overrun): **{usd(a.total_capex)}**")
    lines.append("")
    lines.append("| Metric | Case A (Debt capped) | Case B (Pro-rata) |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Debt | {usd(a.debt)} | {usd(b.debt)} |")
    lines.append(f"| Equity | {usd(a.equity)} | {usd(b.equity)} |")
    lines.append(f"| Annual debt service (PMT) | {usd(a.annual_debt_service)} | {usd(b.annual_debt_service)} |")
    lines.append(f"| Annual CFADS | {usd(a.annual_cfads)} | {usd(b.annual_cfads)} |")
    lines.append(f"| DSCR (CFADS/PMT) | {a.dscr:,.2f}x | {b.dscr:,.2f}x |")
    lines.append(f"| Equity IRR (simple model) | {pct(a.equity_irr) if a.equity_irr is not None else 'n/a'} | {pct(b.equity_irr) if b.equity_irr is not None else 'n/a'} |")
    lines.append("")
    lines.append(f"Notes: {a.notes}")
    return "\n".join(lines)

# ----------------------------
# Narrative loop (Master + Critics)
# ----------------------------
MASTER_SYSTEM = (
    "You are Nemexis Master. You produce the best possible decision-grade memo in plain language. "
    "Do not output JSON or do heavy arithmetic. Reference the provided Ledger values as source of truth."
)

CRITIC_SYSTEM_TECH = (
    "You are a senior technical reviewer. Critique logic, technical risks, and missing assumptions. "
    "Be precise and actionable. Do not rewrite the full memo."
)

CRITIC_SYSTEM_FIN = (
    "You are a project finance reviewer. Critique financial logic, covenants/DSCR, cashflow timing, and missing inputs. "
    "Do not rewrite the full memo."
)

CRITIC_SYSTEM_LEGAL = (
    "You are a contracts/commercial reviewer. Focus on risk allocation: LDs/caps, change orders, force majeure, termination, insurance. "
    "Do not rewrite the full memo."
)

CRITIC_SYSTEM_RISK = (
    "You are a risk/governance reviewer. Flag overconfidence, missing validation steps, and propose an IC decision checklist. "
    "Do not rewrite the full memo."
)

def run_critics(master_text: str) -> Dict[str, str]:
    critiques = {}

    critic_input = f"MASTER MEMO:\n{master_text}"

    if use_claude and ANTHROPIC_API_KEY:
        critiques["Claude-Technical"] = call_claude(claude_model, CRITIC_SYSTEM_TECH, critic_input)
        critiques["Claude-Finance"] = call_claude(claude_model, CRITIC_SYSTEM_FIN, critic_input)
        critiques["Claude-Legal"] = call_claude(claude_model, CRITIC_SYSTEM_LEGAL, critic_input)
        critiques["Claude-Risk"] = call_claude(claude_model, CRITIC_SYSTEM_RISK, critic_input)

    if use_grok and XAI_API_KEY:
        critiques["Grok-Technical"] = call_grok(grok_model, CRITIC_SYSTEM_TECH, critic_input)
        critiques["Grok-Finance"] = call_grok(grok_model, CRITIC_SYSTEM_FIN, critic_input)
        critiques["Grok-Legal"] = call_grok(grok_model, CRITIC_SYSTEM_LEGAL, critic_input)
        critiques["Grok-Risk"] = call_grok(grok_model, CRITIC_SYSTEM_RISK, critic_input)

    return critiques

def master_revise(prev: str, critiques: Dict[str, str], ledger_md: str) -> str:
    critique_blob = "\n\n".join([f"### {k}\n{v}" for k, v in critiques.items()])
    user_text = f"""
You are revising the memo.
Rules:
- Incorporate valid critiques.
- Keep memo concise and IC-ready.
- Do NOT output math derivations.
- Use the Ledger section as the only numeric source of truth.

LEDGER:
{ledger_md}

PREVIOUS MEMO:
{prev}

CRITIQUES:
{critique_blob}

Return the revised memo only.
""".strip()
    return leader_call(MASTER_SYSTEM, user_text, temperature=0.2)

# ----------------------------
# RUN
# ----------------------------
if run:
    if not prompt.strip():
        st.error("Paste a prompt/question in the text box.")
        st.stop()

    # Deterministic base ledger
    base_inputs = build_point_inputs()
    try:
        case_a, case_b = run_scenarios(base_inputs)
    except Exception as e:
        st.error(f"Ledger compute failed: {e}")
        st.stop()

    ledger_md = ledger_to_markdown(case_a, case_b)
    st.markdown("## 1) Deterministic Ledger (source of truth)")
    st.markdown(ledger_md)

    # Monte Carlo
    mc_summary = None
    if mc_enabled:
        st.markdown("## 2) Monte Carlo 10,000 (triangular)")
        mc_cfg = MCConfig(
            n=int(mc_n),
            overrun_pct=TriParam(overrun_low, overrun_mode, overrun_high),
            om_pct=TriParam(om_low, om_mode, om_high),
            tax_rate=TriParam(tax_low, tax_mode, tax_high),
            reserves_pct=TriParam(res_low, res_mode, res_high),
            interest_rate=TriParam(ir_low, ir_mode, ir_high),
        )
        with st.spinner("Running Monte Carlo..."):
            mc_summary = mc_run(base_inputs, mc_cfg)

        st.write(f"Samples used: {mc_summary['samples_used']:,} (non-convergent IRR paths are dropped).")
        st.markdown("### IRR percentiles")
        st.table({
            "Case A IRR": mc_summary["A_IRR"],
            "Case B IRR": mc_summary["B_IRR"],
        })
        st.markdown("### DSCR percentiles")
        st.table({
            "Case A DSCR": mc_summary["A_DSCR"],
            "Case B DSCR": mc_summary["B_DSCR"],
        })

    # Master initial
    st.markdown("## 3) Iterative narrative loop (Master ↔ Critics)")
    base_user = f"""
TASK:
{prompt.strip()}

You must use this Ledger as numeric truth:
{ledger_md}

If Monte Carlo summary is present, reference it as uncertainty/tail risk.
""".strip()

    if mc_summary:
        base_user += "\n\nMONTE CARLO SUMMARY:\n" + json.dumps(mc_summary, indent=2)

    with st.spinner("Master drafting v1..."):
        memo = leader_call(MASTER_SYSTEM, base_user, temperature=0.2)

    st.markdown("### Master v1")
    st.write(memo)

    history = []
    history.append(("Master v1", memo))

    # Iterations
    for i in range(1, int(iterations) + 1):
        st.divider()
        st.markdown(f"### Iteration {i}")

        with st.spinner("Critics reviewing..."):
            critiques = run_critics(memo)

        if critiques:
            st.markdown("#### Critiques")
            for k, v in critiques.items():
                with st.expander(k, expanded=False):
                    st.write(v)
        else:
            st.info("No critics enabled/configured — skipping critique stage.")

        with st.spinner("Master integrating critiques..."):
            memo = master_revise(memo, critiques, ledger_md)

        st.markdown(f"#### Master v{i+1}")
        st.write(memo)
        history.append((f"Master v{i+1}", memo))

    # Final output (always)
    st.divider()
    st.markdown("## 4) Final Deliverable (always produced)")

    final_package = []
    final_package.append("# Nemexis Output (v10.3.6)")
    final_package.append(f"Generated: {datetime.datetime.now()}")
    final_package.append(f"Leader: {leader}")
    final_package.append("")
    final_package.append("## Deterministic Ledger")
    final_package.append(ledger_md)
    if mc_summary:
        final_package.append("")
        final_package.append("## Monte Carlo Summary (10,000, triangular)")
        final_package.append("```json\n" + json.dumps(mc_summary, indent=2) + "\n```")
    final_package.append("")
    final_package.append("## Final Memo")
    final_package.append(memo)
    final_package.append("")
    final_package.append("## Memo History")
    for title, text in history:
        final_package.append(f"### {title}\n{text}\n")

    export_text = "\n".join(final_package)

    st.markdown("### Export")
    copy_button(export_text, "📋 Copy Everything")
    st.download_button("⬇️ Download Markdown", export_text, file_name="nemexis_output.md", mime="text/markdown")
    st.text_area("All output", value=export_text, height=320)
