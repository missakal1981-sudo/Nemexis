import os
import time
import json
import math
import random
import datetime
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components

from openai import OpenAI
import anthropic
import requests

# ============================================================
# Nemexis v10.3.7 — Universal + Robust
#
# Fixes from v10.3.6:
# 1) "Unknown" workflow (no more forcing users to enter 0):
#    - Per input: Known / Assume (Triangular) / Block
#    - STRICT mode: Block prevents metric computation (transparent)
#    - ASSUMPTION mode: Assume uses triangular draws; ledger uses mode
#
# 2) Monte Carlo now NEVER returns all-null just because IRR fails:
#    - DSCR always tracked if computable
#    - IRR tracked only when convergent
#    - Convergence rate reported
#    - IRR uses robust bisection (not Newton)
#
# 3) Ledger-locked memo:
#    - LLM forbidden to introduce numbers (no digits)
#    - We enforce via sanitizer: replace digit runs with "[#]"
#    - Final memo always consistent with deterministic ledger/MC tables
#
# 4) Always produces final output (no JSON schema dependence).
# ============================================================

# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="Nemexis v10.3.7", layout="wide")
st.title("Nemexis v10.3.7 — Universal Reliability (Unknown workflow + MC 10k + Ledger-locked memo)")

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
    rate: per-period (annual if nper in years)
    Returns positive payment (cash outflow).
    """
    if nper <= 0:
        raise ValueError("nper must be > 0")
    if abs(rate) < 1e-12:
        return (pv + fv) / nper
    factor = (1 + rate) ** nper
    payment = (rate * (pv * factor + fv)) / ((factor - 1) * (1 + rate * when))
    return payment

def npv(rate: float, cashflows: List[float]) -> float:
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))

def irr_bisect(cashflows: List[float], lo: float = -0.9, hi: float = 2.0, tol: float = 1e-7, max_iter: int = 200) -> Optional[float]:
    """
    Robust IRR via bisection: finds r such that NPV(r)=0.
    Returns None if no sign change.
    """
    if not (any(cf < 0 for cf in cashflows) and any(cf > 0 for cf in cashflows)):
        return None

    def f(r): return npv(r, cashflows)

    flo = f(lo)
    fhi = f(hi)
    if math.isnan(flo) or math.isnan(fhi) or math.isinf(flo) or math.isinf(fhi):
        return None
    if flo == 0:
        return lo
    if fhi == 0:
        return hi
    # need sign change
    if flo * fhi > 0:
        return None

    a, b = lo, hi
    fa, fb = flo, fhi
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol:
            return m
        # preserve bracket
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)

def triangular(low: float, mode: float, high: float) -> float:
    return random.triangular(low, high, mode)

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
    total_capex: Optional[float]
    debt: Optional[float]
    equity: Optional[float]
    annual_debt_service: Optional[float]
    annual_cfads: Optional[float]
    dscr: Optional[float]
    equity_irr: Optional[float]
    notes: str

def compute_cfads(ebitda: float, total_capex: float, om_pct: float, tax_rate: float, reserves_pct: float) -> float:
    o_and_m = om_pct * total_capex
    taxable_base = max(0.0, ebitda - o_and_m)
    taxes = tax_rate * taxable_base
    reserves = reserves_pct * ebitda
    return ebitda - o_and_m - taxes - reserves

def equity_cashflows_simple(inputs: Inputs, debt_principal: float, equity_principal: float, annual_cfads: float) -> List[float]:
    # deterministic phasing: if construction is 3 years, 30/40/30 else equal
    if inputs.construction_years == 3:
        phasing = [0.30, 0.40, 0.30]
    else:
        phasing = [1.0 / inputs.construction_years] * inputs.construction_years

    equity_draws = [-equity_principal * p for p in phasing]
    debt_service = pmt(inputs.interest_rate, inputs.debt_tenor_years, debt_principal)

    cfs = []
    cfs.extend(equity_draws)
    for y in range(1, inputs.project_life_years + 1):
        if y <= inputs.debt_tenor_years:
            cfs.append(annual_cfads - debt_service)
        else:
            cfs.append(annual_cfads)
    return cfs

def run_scenarios(inputs: Inputs) -> Tuple[ScenarioResult, ScenarioResult]:
    base_debt = inputs.capex * inputs.debt_ratio
    base_equity = inputs.capex * inputs.equity_ratio
    overrun_amt = inputs.capex * inputs.overrun_pct
    total_capex = inputs.capex + overrun_amt

    # Case A: debt capped at base; overrun is equity-funded
    debt_a = base_debt
    equity_a = base_equity + overrun_amt

    # Case B: incremental pro-rata on overrun
    debt_b = base_debt + inputs.debt_ratio * overrun_amt
    equity_b = base_equity + inputs.equity_ratio * overrun_amt

    ds_a = pmt(inputs.interest_rate, inputs.debt_tenor_years, debt_a)
    ds_b = pmt(inputs.interest_rate, inputs.debt_tenor_years, debt_b)

    cfads = compute_cfads(inputs.ebitda, total_capex, inputs.om_pct_of_capex, inputs.tax_rate, inputs.reserves_pct_of_ebitda)

    dscr_a = cfads / ds_a if ds_a > 0 else None
    dscr_b = cfads / ds_b if ds_b > 0 else None

    irr_a = None
    irr_b = None
    notes = "IRR computed with robust bisection on a simple levered cashflow model."

    # IRR bisection robust
    try:
        irr_a = irr_bisect(equity_cashflows_simple(inputs, debt_a, equity_a, cfads))
    except Exception:
        irr_a = None
    try:
        irr_b = irr_bisect(equity_cashflows_simple(inputs, debt_b, equity_b, cfads))
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
        name="Case B (Pro-rata on overrun)",
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
    A_IRR, B_IRR = [], []
    A_DSCR, B_DSCR = [], []

    irr_a_ok = 0
    irr_b_ok = 0
    dscr_ok = 0

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

        # DSCR always recorded if computable
        if a.dscr is not None and b.dscr is not None and not (math.isnan(a.dscr) or math.isnan(b.dscr)):
            A_DSCR.append(a.dscr)
            B_DSCR.append(b.dscr)
            dscr_ok += 1

        # IRR recorded only if converged
        if a.equity_irr is not None and not math.isnan(a.equity_irr):
            A_IRR.append(a.equity_irr)
            irr_a_ok += 1
        if b.equity_irr is not None and not math.isnan(b.equity_irr):
            B_IRR.append(b.equity_irr)
            irr_b_ok += 1

    def pct(vals, p):
        if not vals:
            return None
        vs = sorted(vals)
        k = int(round((p / 100) * (len(vs) - 1)))
        return vs[k]

    out = {
        "A_IRR": { "P10": pct(A_IRR,10), "P50": pct(A_IRR,50), "P90": pct(A_IRR,90), "P95": pct(A_IRR,95) },
        "B_IRR": { "P10": pct(B_IRR,10), "P50": pct(B_IRR,50), "P90": pct(B_IRR,90), "P95": pct(B_IRR,95) },
        "A_DSCR": { "P10": pct(A_DSCR,10), "P50": pct(A_DSCR,50), "P90": pct(A_DSCR,90), "P95": pct(A_DSCR,95) },
        "B_DSCR": { "P10": pct(B_DSCR,10), "P50": pct(B_DSCR,50), "P90": pct(B_DSCR,90), "P95": pct(B_DSCR,95) },
        "counts": {
            "iterations": mc.n,
            "dscr_samples": dscr_ok,
            "irr_a_samples": irr_a_ok,
            "irr_b_samples": irr_b_ok
        }
    }
    return out

# ----------------------------
# Unknown workflow helpers
# ----------------------------
MODE_KNOWN = "Known"
MODE_ASSUME = "Assume (Triangular)"
MODE_BLOCK = "Block"

def known_assume_block_ui(label: str, default_value: float, step: float, fmt: str,
                          tri_defaults: Tuple[float,float,float]) -> Tuple[str, float, TriParam]:
    """
    Returns (mode, known_value, tri_param).
    If mode is Block, known_value is still returned but ignored.
    """
    st.markdown(f"**{label}**")
    mode = st.selectbox(f"{label} mode", [MODE_KNOWN, MODE_ASSUME, MODE_BLOCK], index=0, key=f"{label}_mode")
    known_value = st.number_input(f"{label} value", value=float(default_value), step=float(step), format=fmt, key=f"{label}_val")
    low0, mode0, high0 = tri_defaults
    c1,c2,c3 = st.columns(3)
    with c1:
        low = st.number_input(f"{label} low", value=float(low0), step=float(step), format=fmt, key=f"{label}_low")
    with c2:
        modev = st.number_input(f"{label} mode", value=float(mode0), step=float(step), format=fmt, key=f"{label}_modev")
    with c3:
        high = st.number_input(f"{label} high", value=float(high0), step=float(step), format=fmt, key=f"{label}_high")

    # sanitize triangular ordering
    low_s = min(low, modev, high)
    high_s = max(low, modev, high)
    mode_s = min(max(modev, low_s), high_s)

    return mode, known_value, TriParam(low_s, mode_s, high_s)

# ----------------------------
# Models (narrative only)
# ----------------------------
def call_openai(model_name: str, system: str, user_text: str, temperature=0.2) -> str:
    if not openai_client:
        return "❌ OpenAI not configured"
    resp = openai_client.chat.completions.create(
        model=model_name,
        messages=[{"role":"system","content":system},{"role":"user","content":user_text}],
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
        messages=[{"role":"user","content":user_text}],
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
                headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type":"application/json"},
                json={
                    "model": model_name,
                    "messages": [{"role":"system","content":system},{"role":"user","content":user_text}],
                    "temperature": 0.2,
                },
                timeout=140,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            last_err = f"❌ Grok Error ({r.status_code}): {r.text}"
        except Exception as e:
            last_err = f"❌ Grok Error: {str(e)}"
        time.sleep(1.5*(attempt+1))
    return last_err or "❌ Grok Error: unknown"

# ----------------------------
# UI: model selection
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

c1,c2,c3 = st.columns(3)
with c1:
    openai_model = st.selectbox("OpenAI model", ["gpt-4o-mini","gpt-4o"], index=0, disabled=(not OPENAI_API_KEY))
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
# Universal Ledger Inputs with Unknown workflow
# ----------------------------
st.markdown("## Universal Ledger Inputs (Known / Assume / Block)")

left, right = st.columns(2)

with left:
    capex_mode, capex_val, capex_tri = known_assume_block_ui(
        "CAPEX (USD)", 3_200_000_000.0, 10_000_000.0, "%.0f",
        (2_500_000_000.0, 3_200_000_000.0, 4_200_000_000.0)
    )
    ebitda_mode, ebitda_val, ebitda_tri = known_assume_block_ui(
        "Annual EBITDA (USD)", 420_000_000.0, 1_000_000.0, "%.0f",
        (250_000_000.0, 420_000_000.0, 600_000_000.0)
    )
    overrun_mode_sel, overrun_val, overrun_tri = known_assume_block_ui(
        "Overrun %", 0.15, 0.01, "%.3f",
        (0.00, 0.15, 0.30)
    )

with right:
    ir_mode_sel, ir_val, ir_tri = known_assume_block_ui(
        "Interest rate (annual)", 0.065, 0.001, "%.4f",
        (0.03, 0.065, 0.10)
    )
    om_mode_sel, om_val, om_tri = known_assume_block_ui(
        "O&M % of CAPEX (annual)", 0.025, 0.005, "%.4f",
        (0.01, 0.025, 0.04)
    )
    tax_mode_sel, tax_val, tax_tri = known_assume_block_ui(
        "Tax rate (effective)", 0.20, 0.01, "%.3f",
        (0.00, 0.20, 0.30)
    )

st.markdown("### Additional controls")
c4,c5,c6 = st.columns(3)
with c4:
    reserves_mode_sel, reserves_val, reserves_tri = known_assume_block_ui(
        "Reserves % of EBITDA", 0.02, 0.005, "%.4f",
        (0.00, 0.02, 0.05)
    )
with c5:
    debt_ratio = st.number_input("Debt ratio", value=0.60, step=0.01, format="%.2f")
    equity_ratio = st.number_input("Equity ratio", value=0.40, step=0.01, format="%.2f")
with c6:
    debt_tenor = st.number_input("Debt tenor (years)", value=20, step=1, min_value=1)
    proj_life = st.number_input("Project life (years)", value=25, step=1, min_value=1)
    constr_years = st.number_input("Construction (years)", value=3, step=1, min_value=0)

if abs((debt_ratio + equity_ratio) - 1.0) > 1e-6:
    st.warning("Debt ratio + Equity ratio should equal 1.0 for consistent scenario math.")

# ----------------------------
# Prompt
# ----------------------------
st.markdown("## Prompt (narrative loop)")
prompt = st.text_area("Your question / task", height=200)

iterations = st.number_input("Critique iterations (Master→Critics→Master)", value=2, step=1, min_value=1, max_value=6)

mode_run = st.selectbox("Run mode", ["STRICT (block if missing)", "ASSUMPTION (fill gaps + MC)"], index=1)

mc_enabled = st.checkbox("Run Monte Carlo 10,000", value=True)
mc_n = st.number_input("Monte Carlo iterations", value=10_000, step=1_000, min_value=1000)

run = st.button("Run Nemexis v10.3.7")

# ----------------------------
# Build point inputs / distributions
# ----------------------------
def resolve_value(mode: str, known: float, tri: TriParam, allow_assume: bool) -> Optional[float]:
    if mode == MODE_BLOCK:
        return None
    if mode == MODE_KNOWN:
        return float(known)
    # assume
    if allow_assume:
        return float(tri.mode)
    return None

def build_point_inputs(allow_assume: bool) -> Optional[Inputs]:
    # Required for even basic scenario math: capex, overrun, ratios
    capex_v = resolve_value(capex_mode, capex_val, capex_tri, allow_assume)
    overrun_v = resolve_value(overrun_mode_sel, overrun_val, overrun_tri, allow_assume)
    if capex_v is None or overrun_v is None:
        return None

    # For DSCR/IRR we need: ebitda, om, tax, reserves, interest, tenor, life, construction
    ebitda_v = resolve_value(ebitda_mode, ebitda_val, ebitda_tri, allow_assume)
    ir_v = resolve_value(ir_mode_sel, ir_val, ir_tri, allow_assume)
    om_v = resolve_value(om_mode_sel, om_val, om_tri, allow_assume)
    tax_v = resolve_value(tax_mode_sel, tax_val, tax_tri, allow_assume)
    res_v = resolve_value(reserves_mode_sel, reserves_val, reserves_tri, allow_assume)

    # If missing, we still create Inputs but set to 0 and later mark metrics n/a
    # (we never tell user to put 0; this is internal safe default)
    ebitda_v = float(ebitda_v) if ebitda_v is not None else 0.0
    ir_v = float(ir_v) if ir_v is not None else 0.0
    om_v = float(om_v) if om_v is not None else 0.0
    tax_v = float(tax_v) if tax_v is not None else 0.0
    res_v = float(res_v) if res_v is not None else 0.0

    return Inputs(
        capex=capex_v,
        overrun_pct=overrun_v,
        debt_ratio=float(debt_ratio),
        equity_ratio=float(equity_ratio),
        interest_rate=ir_v,
        debt_tenor_years=int(debt_tenor),
        project_life_years=int(proj_life),
        construction_years=int(constr_years),
        ebitda=ebitda_v,
        om_pct_of_capex=om_v,
        tax_rate=tax_v,
        reserves_pct_of_ebitda=res_v
    )

def can_compute_cash_metrics(allow_assume: bool) -> bool:
    # If any of these are blocked in STRICT, we won't compute DSCR/IRR/MC
    needed = [ebitda_mode, ir_mode_sel, om_mode_sel, tax_mode_sel, reserves_mode_sel]
    if allow_assume:
        # in assumption mode, "assume" is ok
        return all(m != MODE_BLOCK for m in needed)
    else:
        # strict: must be known (not assume, not block)
        return all(m == MODE_KNOWN for m in needed)

def ledger_markdown(a: ScenarioResult, b: ScenarioResult) -> str:
    def usd(x): return "n/a" if x is None else f"${x:,.0f}"
    def ratio(x): return "n/a" if x is None else f"{x:,.2f}x"
    def pct(x): return "n/a" if x is None else f"{100*x:,.2f}%"

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
    lines.append(f"| DSCR (CFADS/PMT) | {ratio(a.dscr)} | {ratio(b.dscr)} |")
    lines.append(f"| Equity IRR (simple model) | {pct(a.equity_irr)} | {pct(b.equity_irr)} |")
    lines.append("")
    lines.append(f"Notes: {a.notes}")
    return "\n".join(lines)

def sanitize_no_numbers(text: str) -> str:
    """
    Enforce ledger-locked memo: replace any digit runs with [#].
    This prevents the memo from inventing new numbers.
    """
    return re.sub(r"\d+([.,]\d+)*", "[#]", text)

# ----------------------------
# Narrative loop prompts (ledger-locked)
# ----------------------------
MASTER_SYSTEM = (
    "You are Nemexis Master. You write a decision-grade memo.\n"
    "CRITICAL RULE: Do NOT introduce any numbers, digits, percentages, currency figures, or dates.\n"
    "If you need to refer to numbers, refer to the ledger fields by name only, e.g., 'Case B DSCR (see ledger)'."
)

CRITIC_SYSTEM_TECH = (
    "You are a senior technical reviewer. Critique technical logic, failure modes, and missing constraints. "
    "Be actionable. Do not rewrite the full memo."
)
CRITIC_SYSTEM_FIN = (
    "You are a project finance reviewer. Critique financing logic, DSCR, covenant headroom, and missing inputs. "
    "Be actionable. Do not rewrite the full memo."
)
CRITIC_SYSTEM_LEGAL = (
    "You are a contracts/commercial reviewer. Focus on LDs/caps, change orders, force majeure, termination, insurance. "
    "Be actionable. Do not rewrite the full memo."
)
CRITIC_SYSTEM_RISK = (
    "You are a risk/governance reviewer. Flag overconfidence, missing validation, and propose an IC checklist. "
    "Be actionable. Do not rewrite the full memo."
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

def master_revise(prev: str, critiques: Dict[str, str], ledger_md: str, mc_summary: Optional[Dict[str,Any]]) -> str:
    critique_blob = "\n\n".join([f"### {k}\n{v}" for k, v in critiques.items()])
    mc_blob = json.dumps(mc_summary, indent=2) if mc_summary else "(none)"
    user_text = f"""
You are revising the memo.
Rules:
- Incorporate valid critiques.
- Keep it concise and IC-ready.
- DO NOT introduce any numbers/digits. Reference ledger fields by name only.
- Use the Ledger + Monte Carlo summary as factual anchors (but do not quote numbers).

LEDGER:
{ledger_md}

MONTE CARLO SUMMARY:
{mc_blob}

PREVIOUS MEMO:
{prev}

CRITIQUES:
{critique_blob}

Return the revised memo only.
""".strip()
    out = leader_call(MASTER_SYSTEM, user_text, temperature=0.2)
    return sanitize_no_numbers(out)

# ----------------------------
# RUN
# ----------------------------
if run:
    if not prompt.strip():
        st.error("Paste a prompt/question in the text box.")
        st.stop()

    allow_assume = mode_run.startswith("ASSUMPTION")

    # Build deterministic point inputs
    point_inputs = build_point_inputs(allow_assume=allow_assume)
    if point_inputs is None:
        st.error("Blocking: CAPEX and/or Overrun % are not available. Set them to Known or Assume.")
        st.stop()

    # Determine if cash metrics are computable
    cash_ok = can_compute_cash_metrics(allow_assume=allow_assume)

    # Ledger compute
    st.divider()
    st.markdown("## 1) Deterministic Ledger (source of truth)")

    if cash_ok:
        case_a, case_b = run_scenarios(point_inputs)
    else:
        # compute only scenario capital structure + debt service if possible
        base_debt = point_inputs.capex * point_inputs.debt_ratio
        base_equity = point_inputs.capex * point_inputs.equity_ratio
        overrun_amt = point_inputs.capex * point_inputs.overrun_pct
        total_capex = point_inputs.capex + overrun_amt
        debt_a = base_debt
        eq_a = base_equity + overrun_amt
        debt_b = base_debt + point_inputs.debt_ratio * overrun_amt
        eq_b = base_equity + point_inputs.equity_ratio * overrun_amt

        ds_a = pmt(point_inputs.interest_rate, point_inputs.debt_tenor_years, debt_a) if (ir_mode_sel != MODE_BLOCK) else None
        ds_b = pmt(point_inputs.interest_rate, point_inputs.debt_tenor_years, debt_b) if (ir_mode_sel != MODE_BLOCK) else None

        case_a = ScenarioResult(
            name="Case A",
            total_capex=total_capex,
            debt=debt_a,
            equity=eq_a,
            annual_debt_service=ds_a,
            annual_cfads=None,
            dscr=None,
            equity_irr=None,
            notes="Cash metrics blocked in STRICT unless EBITDA/O&M/Tax/Reserves/Rate are Known. Use ASSUMPTION mode to fill."
        )
        case_b = ScenarioResult(
            name="Case B",
            total_capex=total_capex,
            debt=debt_b,
            equity=eq_b,
            annual_debt_service=ds_b,
            annual_cfads=None,
            dscr=None,
            equity_irr=None,
            notes=case_a.notes
        )

    ledger_md = ledger_markdown(case_a, case_b)
    st.markdown(ledger_md)

    # Monte Carlo
    mc_summary = None
    if mc_enabled:
        st.markdown("## 2) Monte Carlo (10,000, triangular)")

        if not allow_assume:
            st.info("Monte Carlo is only available in ASSUMPTION mode (STRICT blocks assumptions).")
        elif not cash_ok:
            st.warning("Monte Carlo needs EBITDA, interest rate, O&M%, tax%, reserves% to be Known or Assume (not Block).")
        else:
            mc_cfg = MCConfig(
                n=int(mc_n),
                overrun_pct=overrun_tri,
                om_pct=om_tri,
                tax_rate=tax_tri,
                reserves_pct=reserves_tri,
                interest_rate=ir_tri
            )
            with st.spinner("Running Monte Carlo..."):
                mc_summary = mc_run(point_inputs, mc_cfg)

            st.write("Counts:", mc_summary["counts"])

            st.markdown("### IRR percentiles (only converged samples)")
            st.table({
                "Case A IRR": mc_summary["A_IRR"],
                "Case B IRR": mc_summary["B_IRR"],
            })

            st.markdown("### DSCR percentiles")
            st.table({
                "Case A DSCR": mc_summary["A_DSCR"],
                "Case B DSCR": mc_summary["B_DSCR"],
            })

    # Narrative loop
    st.markdown("## 3) Iterative narrative loop (Master ↔ Critics)")

    base_user = f"""
TASK:
{prompt.strip()}

LEDGER (numeric truth; do not quote numbers in memo):
{ledger_md}

MONTE CARLO SUMMARY (if present; do not quote numbers in memo):
{json.dumps(mc_summary, indent=2) if mc_summary else "(none)"}

Write an IC-ready memo with:
- decision recommendation (Proceed / Proceed w/ Mitigation / Defer)
- top technical + financial + contractual risks
- validation checklist / next steps
Again: do not write any digits; refer to ledger fields by name.
""".strip()

    with st.spinner("Master drafting v1..."):
        memo = leader_call(MASTER_SYSTEM, base_user, temperature=0.2)
    memo = sanitize_no_numbers(memo)

    st.markdown("### Master v1")
    st.write(memo)

    history = [("Master v1", memo)]

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
            memo = master_revise(memo, critiques, ledger_md, mc_summary)

        st.markdown(f"#### Master v{i+1}")
        st.write(memo)
        history.append((f"Master v{i+1}", memo))

    # Final output package (always)
    st.divider()
    st.markdown("## 4) Final Deliverable (always produced)")

    export = []
    export.append("# Nemexis Output (v10.3.7)")
    export.append(f"Generated: {datetime.datetime.now()}")
    export.append(f"Leader: {leader}")
    export.append("")
    export.append("## Deterministic Ledger")
    export.append(ledger_md)
    if mc_summary:
        export.append("")
        export.append("## Monte Carlo Summary (triangular)")
        export.append("```json\n" + json.dumps(mc_summary, indent=2) + "\n```")
    export.append("")
    export.append("## Final Memo (ledger-locked: no numbers)")
    export.append(memo)
    export.append("")
    export.append("## Memo History")
    for title, txt in history:
        export.append(f"### {title}\n{txt}\n")

    export_text = "\n".join(export)

    st.markdown("### Export")
    copy_button(export_text, "📋 Copy Everything")
    st.download_button("⬇️ Download Markdown", export_text, file_name="nemexis_output.md", mime="text/markdown")
    st.text_area("All output", value=export_text, height=320)
