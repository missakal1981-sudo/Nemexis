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
# Nemexis v10.3.8 — Best Universal Build
#
# Key upgrades vs v10.3.7:
# 1) Auto-parse prompt -> prefill Known values (CAPEX/EBITDA/rates/ratios/tenors/IRR).
# 2) Calibration mode: anchor base-case IRR to the user's stated base IRR.
#    - Engine scales CFADS by a multiplier so base scenario IRR matches base IRR.
#    - Then applies same multiplier to stressed scenarios (universal bridge to "real model").
# 3) Monte Carlo outputs include probabilities:
#    - Prob(IRR < target) for each case
#    - Prob(DSCR < covenant) for each case
# 4) Ledger-first memo:
#    - LLM is not allowed to compute numbers.
#    - Quantified results printed by the engine (tables + probabilities).
# ============================================================

# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="Nemexis v10.3.8", layout="wide")
st.title("Nemexis v10.3.8 — Universal Reliability (Auto-Parse + Calibration + MC 10k)")

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
# Finance engine
# ----------------------------
def pmt(rate: float, nper: int, pv: float) -> float:
    if nper <= 0:
        raise ValueError("nper must be > 0")
    if abs(rate) < 1e-12:
        return pv / nper
    factor = (1 + rate) ** nper
    return (rate * pv * factor) / (factor - 1)

def npv(rate: float, cashflows: List[float]) -> float:
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))

def irr_bisect(cashflows: List[float], lo: float = -0.90, hi: float = 2.00, tol: float = 1e-8, max_iter: int = 200) -> Optional[float]:
    if not (any(cf < 0 for cf in cashflows) and any(cf > 0 for cf in cashflows)):
        return None

    def f(r): return npv(r, cashflows)

    flo, fhi = f(lo), f(hi)
    if math.isnan(flo) or math.isnan(fhi) or math.isinf(flo) or math.isinf(fhi):
        return None
    if flo == 0: return lo
    if fhi == 0: return hi
    if flo * fhi > 0:
        return None

    a, b = lo, hi
    fa, fb = flo, fhi
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol:
            return m
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
    total_capex: float
    debt: float
    equity: float
    annual_debt_service: float
    annual_cfads: float
    dscr: float
    equity_irr: Optional[float]
    notes: str

MODE_KNOWN = "Known"
MODE_ASSUME = "Assume (Triangular)"
MODE_BLOCK = "Block"

# ----------------------------
# Cashflow model (simple but stable)
# ----------------------------
def compute_cfads(ebitda: float, total_capex: float, om_pct: float, tax_rate: float, reserves_pct: float) -> float:
    o_and_m = om_pct * total_capex
    taxable_base = max(0.0, ebitda - o_and_m)
    taxes = tax_rate * taxable_base
    reserves = reserves_pct * ebitda
    return ebitda - o_and_m - taxes - reserves

def equity_cashflows_simple(inputs: Inputs, debt_principal: float, equity_principal: float, annual_cfads: float) -> List[float]:
    # deterministic phasing
    if inputs.construction_years == 3:
        phasing = [0.30, 0.40, 0.30]
    elif inputs.construction_years > 0:
        phasing = [1.0 / inputs.construction_years] * inputs.construction_years
    else:
        phasing = []

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

# ----------------------------
# Scenarios (Case A capped debt, Case B pro-rata on overrun)
# ----------------------------
def run_scenarios(inputs: Inputs, cfads_multiplier: float = 1.0) -> Tuple[ScenarioResult, ScenarioResult]:
    base_debt = inputs.capex * inputs.debt_ratio
    base_equity = inputs.capex * inputs.equity_ratio
    overrun_amt = inputs.capex * inputs.overrun_pct
    total_capex = inputs.capex + overrun_amt

    # Case A: debt capped at base
    debt_a = base_debt
    equity_a = base_equity + overrun_amt

    # Case B: pro-rata incremental
    debt_b = base_debt + inputs.debt_ratio * overrun_amt
    equity_b = base_equity + inputs.equity_ratio * overrun_amt

    ds_a = pmt(inputs.interest_rate, inputs.debt_tenor_years, debt_a)
    ds_b = pmt(inputs.interest_rate, inputs.debt_tenor_years, debt_b)

    cfads = compute_cfads(inputs.ebitda, total_capex, inputs.om_pct_of_capex, inputs.tax_rate, inputs.reserves_pct_of_ebitda)
    cfads *= cfads_multiplier

    dscr_a = cfads / ds_a if ds_a > 0 else float("inf")
    dscr_b = cfads / ds_b if ds_b > 0 else float("inf")

    irr_a = irr_bisect(equity_cashflows_simple(inputs, debt_a, equity_a, cfads))
    irr_b = irr_bisect(equity_cashflows_simple(inputs, debt_b, equity_b, cfads))

    note = "IRR via bisection; CFADS multiplier may be calibrated to base case IRR."

    a = ScenarioResult(
        name="Case A (Debt capped, overrun equity-funded)",
        total_capex=total_capex, debt=debt_a, equity=equity_a,
        annual_debt_service=ds_a, annual_cfads=cfads, dscr=dscr_a, equity_irr=irr_a, notes=note
    )
    b = ScenarioResult(
        name="Case B (Pro-rata on overrun)",
        total_capex=total_capex, debt=debt_b, equity=equity_b,
        annual_debt_service=ds_b, annual_cfads=cfads, dscr=dscr_b, equity_irr=irr_b, notes=note
    )
    return a, b

# ----------------------------
# Calibration: choose CFADS multiplier so BASE scenario IRR matches base_case_IRR
# ----------------------------
def calibrate_cfads_multiplier(base_inputs: Inputs, target_base_irr: float) -> float:
    """
    Calibrate multiplier k so that base scenario (overrun_pct=0) has equity IRR ~ target_base_irr.
    We calibrate on Case "base" using the base debt/equity ratios.
    """
    if target_base_irr <= -0.5 or target_base_irr >= 2.0:
        return 1.0

    # build a copy with overrun=0
    bi = Inputs(
        capex=base_inputs.capex,
        overrun_pct=0.0,
        debt_ratio=base_inputs.debt_ratio,
        equity_ratio=base_inputs.equity_ratio,
        interest_rate=base_inputs.interest_rate,
        debt_tenor_years=base_inputs.debt_tenor_years,
        project_life_years=base_inputs.project_life_years,
        construction_years=base_inputs.construction_years,
        ebitda=base_inputs.ebitda,
        om_pct_of_capex=base_inputs.om_pct_of_capex,
        tax_rate=base_inputs.tax_rate,
        reserves_pct_of_ebitda=base_inputs.reserves_pct_of_ebitda,
    )

    # objective: IRR(k) - target = 0
    # k scales CFADS (proxy for tax credits, sculpting, refin, etc.)
    def irr_for_k(k: float) -> Optional[float]:
        a, _ = run_scenarios(bi, cfads_multiplier=k)  # Case A == base when overrun=0
        return a.equity_irr

    # Bracket k
    k_lo, k_hi = 0.1, 10.0
    irr_lo = irr_for_k(k_lo)
    irr_hi = irr_for_k(k_hi)
    if irr_lo is None or irr_hi is None:
        return 1.0
    # Ensure monotonic-ish: if not bracketed, return 1.0
    if (irr_lo - target_base_irr) * (irr_hi - target_base_irr) > 0:
        return 1.0

    for _ in range(60):
        k_mid = 0.5 * (k_lo + k_hi)
        irr_mid = irr_for_k(k_mid)
        if irr_mid is None:
            k_lo = k_mid
            continue
        if abs(irr_mid - target_base_irr) < 1e-5:
            return k_mid
        if (irr_lo - target_base_irr) * (irr_mid - target_base_irr) <= 0:
            k_hi, irr_hi = k_mid, irr_mid
        else:
            k_lo, irr_lo = k_mid, irr_mid
    return 0.5 * (k_lo + k_hi)

# ----------------------------
# Monte Carlo with probabilities
# ----------------------------
@dataclass
class MCConfig:
    n: int
    overrun_pct: TriParam
    om_pct: TriParam
    tax_rate: TriParam
    reserves_pct: TriParam
    interest_rate: TriParam

def mc_run(base: Inputs, mc: MCConfig, cfads_multiplier: float, target_irr: float, covenant_dscr: float) -> Dict[str, Any]:
    A_IRR, B_IRR = [], []
    A_DSCR, B_DSCR = [], []

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
        a, b = run_scenarios(sampled, cfads_multiplier=cfads_multiplier)

        A_DSCR.append(a.dscr)
        B_DSCR.append(b.dscr)
        if a.equity_irr is not None:
            A_IRR.append(a.equity_irr)
        if b.equity_irr is not None:
            B_IRR.append(b.equity_irr)

    def pct(vals, p):
        if not vals:
            return None
        vs = sorted(vals)
        k = int(round((p / 100) * (len(vs) - 1)))
        return vs[k]

    def prob_lt(vals, threshold):
        if not vals:
            return None
        return sum(1 for v in vals if v < threshold) / len(vals)

    out = {
        "A_IRR": { "P10": pct(A_IRR,10), "P50": pct(A_IRR,50), "P90": pct(A_IRR,90), "P95": pct(A_IRR,95) },
        "B_IRR": { "P10": pct(B_IRR,10), "P50": pct(B_IRR,50), "P90": pct(B_IRR,90), "P95": pct(B_IRR,95) },
        "A_DSCR": { "P10": pct(A_DSCR,10), "P50": pct(A_DSCR,50), "P90": pct(A_DSCR,90), "P95": pct(A_DSCR,95) },
        "B_DSCR": { "P10": pct(B_DSCR,10), "P50": pct(B_DSCR,50), "P90": pct(B_DSCR,90), "P95": pct(B_DSCR,95) },
        "Prob": {
            "A_IRR_lt_target": prob_lt(A_IRR, target_irr),
            "B_IRR_lt_target": prob_lt(B_IRR, target_irr),
            "A_DSCR_lt_covenant": prob_lt(A_DSCR, covenant_dscr),
            "B_DSCR_lt_covenant": prob_lt(B_DSCR, covenant_dscr),
        },
        "counts": {
            "iterations": mc.n,
            "irr_a_samples": len(A_IRR),
            "irr_b_samples": len(B_IRR),
            "dscr_samples": len(A_DSCR),
        }
    }
    return out

# ----------------------------
# Prompt auto-parser
# ----------------------------
def parse_money(text: str) -> Optional[float]:
    """
    Parses $3.2B, 420M, 1,920,000,000, etc.
    Returns float USD.
    """
    if not text:
        return None
    t = text.replace(",", "")
    m = re.search(r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*([bBmMkK]?)", t)
    if not m:
        return None
    num = float(m.group(1))
    suf = m.group(2).lower()
    mult = 1.0
    if suf == "k": mult = 1e3
    if suf == "m": mult = 1e6
    if suf == "b": mult = 1e9
    return num * mult

def parse_pct(text: str) -> Optional[float]:
    if not text:
        return None
    t = text.strip()
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", t)
    if m:
        return float(m.group(1)) / 100.0
    # also accept 0.15
    m2 = re.search(r"\b(0\.[0-9]+)\b", t)
    if m2:
        v = float(m2.group(1))
        if 0 < v < 1:
            return v
    return None

def autoparse_prompt(p: str) -> Dict[str, Any]:
    """
    Best-effort extraction from the prompt.
    """
    out = {}
    lines = p.splitlines()
    for ln in lines:
        low = ln.lower()

        if "capex" in low:
            v = parse_money(ln)
            if v: out["capex"] = v

        if "ebitda" in low:
            v = parse_money(ln)
            if v: out["ebitda"] = v

        if "interest" in low or "rate" in low:
            v = parse_pct(ln)
            if v: out["interest_rate"] = v

        if "debt tenor" in low or "tenor" in low:
            m = re.search(r"(\d+)\s*(year|yr)", low)
            if m: out["debt_tenor"] = int(m.group(1))

        if "project life" in low or "life" in low:
            m = re.search(r"(\d+)\s*(year|yr)", low)
            if m: out["proj_life"] = int(m.group(1))

        if "construction" in low and "year" in low:
            m = re.search(r"(\d+)\s*(year|yr)", low)
            if m: out["constr_years"] = int(m.group(1))

        if "overrun" in low:
            v = parse_pct(ln)
            if v: out["overrun_pct"] = v

        if "target" in low and "irr" in low:
            v = parse_pct(ln)
            if v: out["target_irr"] = v

        if "base case" in low and "irr" in low:
            v = parse_pct(ln)
            if v: out["base_irr"] = v

        if "dscr" in low:
            m = re.search(r"(\d+(?:\.\d+)?)\s*x", low)
            if m: out["dscr"] = float(m.group(1))

        if "60%" in low and "debt" in low:
            out["debt_ratio"] = 0.60
            out["equity_ratio"] = 0.40
        if "40%" in low and "equity" in low:
            out["equity_ratio"] = 0.40

    return out

# ----------------------------
# LLM calls (narrative only)
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
        max_tokens=1400,
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
if OPENAI_API_KEY: available_leaders.append("OpenAI")
if ANTHROPIC_API_KEY: available_leaders.append("Claude")
if XAI_API_KEY: available_leaders.append("Grok")
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
# Prompt box + auto parse
# ----------------------------
st.markdown("## Prompt")
prompt = st.text_area("Paste your prompt / problem statement", height=200)

ap_col1, ap_col2 = st.columns([1,2])
with ap_col1:
    auto_parse_btn = st.button("Auto-parse prompt → fill Known")
with ap_col2:
    st.caption("Tip: include lines like 'CAPEX: $3.2B', 'Annual EBITDA: $420M', 'Debt: 60% at 6.5%', 'Debt tenor: 20 years'.")

# Store defaults in session_state keys
def ss_set(key, value):
    st.session_state[key] = value

# We'll bind keys by using fixed widget keys below.
if auto_parse_btn and prompt.strip():
    parsed = autoparse_prompt(prompt)
    if "capex" in parsed:
        ss_set("CAPEX_val", float(parsed["capex"]))
        ss_set("CAPEX_mode", MODE_KNOWN)
    if "ebitda" in parsed:
        ss_set("EBITDA_val", float(parsed["ebitda"]))
        ss_set("EBITDA_mode", MODE_KNOWN)
    if "interest_rate" in parsed:
        ss_set("IR_val", float(parsed["interest_rate"]))
        ss_set("IR_mode", MODE_KNOWN)
    if "overrun_pct" in parsed:
        ss_set("OVR_val", float(parsed["overrun_pct"]))
        ss_set("OVR_mode", MODE_KNOWN)
    if "debt_ratio" in parsed:
        ss_set("DEBT_RATIO", float(parsed["debt_ratio"]))
    if "equity_ratio" in parsed:
        ss_set("EQUITY_RATIO", float(parsed["equity_ratio"]))
    if "debt_tenor" in parsed:
        ss_set("TENOR", int(parsed["debt_tenor"]))
    if "proj_life" in parsed:
        ss_set("LIFE", int(parsed["proj_life"]))
    if "constr_years" in parsed:
        ss_set("CONSTR", int(parsed["constr_years"]))
    if "target_irr" in parsed:
        ss_set("TARGET_IRR", float(parsed["target_irr"]))
    if "base_irr" in parsed:
        ss_set("BASE_IRR", float(parsed["base_irr"]))
    if "dscr" in parsed:
        ss_set("COV_DSCR", float(parsed["dscr"]))
    st.success(f"Parsed and applied: {parsed}")

# ----------------------------
# Universal inputs with Known/Assume/Block
# ----------------------------
st.markdown("## Universal Inputs (Known / Assume / Block)")

def tri_ui(prefix: str, label: str, default_known: float, tri_default: Tuple[float,float,float], step: float, fmt: str):
    mode_key = f"{prefix}_mode"
    val_key = f"{prefix}_val"
    low_key = f"{prefix}_low"
    modev_key = f"{prefix}_modev"
    high_key = f"{prefix}_high"

    if mode_key not in st.session_state:
        st.session_state[mode_key] = MODE_ASSUME
    if val_key not in st.session_state:
        st.session_state[val_key] = float(default_known)
    if low_key not in st.session_state:
        st.session_state[low_key] = float(tri_default[0])
    if modev_key not in st.session_state:
        st.session_state[modev_key] = float(tri_default[1])
    if high_key not in st.session_state:
        st.session_state[high_key] = float(tri_default[2])

    st.markdown(f"**{label}**")
    mode = st.selectbox("Mode", [MODE_KNOWN, MODE_ASSUME, MODE_BLOCK], key=mode_key)
    known_val = st.number_input("Value", key=val_key, step=step, format=fmt)

    c1,c2,c3 = st.columns(3)
    with c1:
        low = st.number_input("Low", key=low_key, step=step, format=fmt)
    with c2:
        modev = st.number_input("Mode", key=modev_key, step=step, format=fmt)
    with c3:
        high = st.number_input("High", key=high_key, step=step, format=fmt)

    lo = min(low, modev, high)
    hi = max(low, modev, high)
    md = min(max(modev, lo), hi)
    return mode, float(known_val), TriParam(lo, md, hi)

colL, colR = st.columns(2)
with colL:
    capex_mode, capex_known, capex_tri = tri_ui("CAPEX", "CAPEX (USD)", 3_200_000_000.0, (2_500_000_000.0, 3_200_000_000.0, 4_200_000_000.0), 10_000_000.0, "%.0f")
    ebitda_mode, ebitda_known, ebitda_tri = tri_ui("EBITDA", "Annual EBITDA (USD)", 420_000_000.0, (250_000_000.0, 420_000_000.0, 600_000_000.0), 1_000_000.0, "%.0f")
    overrun_mode, overrun_known, overrun_tri = tri_ui("OVR", "Overrun %", 0.15, (0.00, 0.15, 0.30), 0.01, "%.3f")

with colR:
    ir_mode, ir_known, ir_tri = tri_ui("IR", "Interest rate (annual)", 0.065, (0.03, 0.065, 0.10), 0.001, "%.4f")
    om_mode, om_known, om_tri = tri_ui("OM", "O&M % of CAPEX (annual)", 0.025, (0.01, 0.025, 0.04), 0.005, "%.4f")
    tax_mode, tax_known, tax_tri = tri_ui("TAX", "Tax rate (effective)", 0.20, (0.00, 0.20, 0.30), 0.01, "%.3f")

res_mode, res_known, res_tri = tri_ui("RES", "Reserves % of EBITDA", 0.02, (0.00, 0.02, 0.05), 0.005, "%.4f")

st.markdown("### Structure inputs")
c7,c8,c9 = st.columns(3)
with c7:
    debt_ratio = st.number_input("Debt ratio", key="DEBT_RATIO", value=st.session_state.get("DEBT_RATIO", 0.60), step=0.01, format="%.2f")
with c8:
    equity_ratio = st.number_input("Equity ratio", key="EQUITY_RATIO", value=st.session_state.get("EQUITY_RATIO", 0.40), step=0.01, format="%.2f")
with c9:
    covenant_dscr = st.number_input("Covenant DSCR", key="COV_DSCR", value=st.session_state.get("COV_DSCR", 1.35), step=0.01, format="%.2f")

c10,c11,c12 = st.columns(3)
with c10:
    debt_tenor = st.number_input("Debt tenor (years)", key="TENOR", value=st.session_state.get("TENOR", 20), step=1, min_value=1)
with c11:
    proj_life = st.number_input("Project life (years)", key="LIFE", value=st.session_state.get("LIFE", 25), step=1, min_value=1)
with c12:
    constr_years = st.number_input("Construction (years)", key="CONSTR", value=st.session_state.get("CONSTR", 3), step=1, min_value=0)

st.markdown("### IRR anchors (optional but powerful)")
c13,c14 = st.columns(2)
with c13:
    target_irr = st.number_input("Target equity IRR", key="TARGET_IRR", value=st.session_state.get("TARGET_IRR", 0.08), step=0.005, format="%.3f")
with c14:
    base_case_irr = st.number_input("Base case equity IRR (anchor)", key="BASE_IRR", value=st.session_state.get("BASE_IRR", 0.087), step=0.005, format="%.3f")

calibration_on = st.checkbox("Calibration ON (anchor base-case IRR to the value above)", value=True)

# ----------------------------
# Run controls
# ----------------------------
st.markdown("## Run")
mode_run = st.selectbox("Run mode", ["STRICT (block if missing)", "ASSUMPTION (fill gaps + MC)"], index=1)
mc_enabled = st.checkbox("Run Monte Carlo 10,000", value=True)
mc_n = st.number_input("Monte Carlo iterations", value=10_000, step=1_000, min_value=1000)
iterations = st.number_input("Critique iterations (Master↔Critics)", value=2, step=1, min_value=1, max_value=6)

run = st.button("Run Nemexis v10.3.8")

# ----------------------------
# Resolve point inputs
# ----------------------------
def resolve_point(mode: str, known: float, tri: TriParam, allow_assume: bool) -> Optional[float]:
    if mode == MODE_BLOCK:
        return None
    if mode == MODE_KNOWN:
        return float(known)
    return float(tri.mode) if allow_assume else None

def build_point_inputs(allow_assume: bool) -> Optional[Inputs]:
    capex_v = resolve_point(capex_mode, capex_known, capex_tri, allow_assume)
    ovr_v = resolve_point(overrun_mode, overrun_known, overrun_tri, allow_assume)
    if capex_v is None or ovr_v is None:
        return None

    ebitda_v = resolve_point(ebitda_mode, ebitda_known, ebitda_tri, allow_assume)
    ir_v = resolve_point(ir_mode, ir_known, ir_tri, allow_assume)
    om_v = resolve_point(om_mode, om_known, om_tri, allow_assume)
    tax_v = resolve_point(tax_mode, tax_known, tax_tri, allow_assume)
    res_v = resolve_point(res_mode, res_known, res_tri, allow_assume)

    # STRICT can still output scenario arithmetic even if cash metrics blocked
    ebitda_v = float(ebitda_v) if ebitda_v is not None else 0.0
    ir_v = float(ir_v) if ir_v is not None else 0.0
    om_v = float(om_v) if om_v is not None else 0.0
    tax_v = float(tax_v) if tax_v is not None else 0.0
    res_v = float(res_v) if res_v is not None else 0.0

    return Inputs(
        capex=capex_v,
        overrun_pct=ovr_v,
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

def cash_metrics_available(allow_assume: bool) -> bool:
    if allow_assume:
        return all(m != MODE_BLOCK for m in [ebitda_mode, ir_mode, om_mode, tax_mode, res_mode])
    return all(m == MODE_KNOWN for m in [ebitda_mode, ir_mode, om_mode, tax_mode, res_mode])

def usd(x): return f"${x:,.0f}"
def pct(x): return f"{100*x:,.2f}%"
def ratio(x): return f"{x:,.2f}x"

# ----------------------------
# LLM prompts (no math)
# ----------------------------
MASTER_SYSTEM = (
    "You are Nemexis Master. Write an IC-ready memo.\n"
    "Rules:\n"
    "- Do NOT compute any numbers.\n"
    "- Do NOT introduce any new numeric claims.\n"
    "- You may reference the 'Quant Results' section by name.\n"
    "- Keep it decision-grade and concise."
)

CRITIC_SYSTEM_TECH = "You are a senior technical reviewer. Critique technical logic and failure modes. Be actionable."
CRITIC_SYSTEM_FIN = "You are a project finance reviewer. Critique DSCR/IRR framing, covenants, and missing inputs. Be actionable."
CRITIC_SYSTEM_LEGAL = "You are a contracts reviewer. Critique LDs/caps, change orders, force majeure, termination, insurance. Be actionable."
CRITIC_SYSTEM_RISK = "You are a risk/governance reviewer. Critique overconfidence, missing validation, and propose IC checklist. Be actionable."

def run_critics(master_text: str) -> Dict[str,str]:
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

def master_revise(prev: str, critiques: Dict[str,str], quant_text: str) -> str:
    blob = "\n\n".join([f"### {k}\n{v}" for k,v in critiques.items()])
    user_text = f"""
Revise the memo.
- Integrate valid critique.
- Do not compute numbers.
- Do not introduce new numeric claims.
- You may point to the Quant Results section for numeric support.

QUANT RESULTS (authoritative):
{quant_text}

PREVIOUS MEMO:
{prev}

CRITIQUES:
{blob}

Return revised memo only.
""".strip()
    return leader_call(MASTER_SYSTEM, user_text, temperature=0.2)

# ----------------------------
# RUN
# ----------------------------
if run:
    allow_assume = mode_run.startswith("ASSUMPTION")
    if not prompt.strip():
        st.error("Paste your prompt.")
        st.stop()

    point_inputs = build_point_inputs(allow_assume=allow_assume)
    if point_inputs is None:
        st.error("Blocking: CAPEX and/or Overrun % are not available. Set them to Known or Assume.")
        st.stop()

    cash_ok = cash_metrics_available(allow_assume=allow_assume)

    # Calibration multiplier
    cfads_mult = 1.0
    if calibration_on and cash_ok:
        try:
            cfads_mult = calibrate_cfads_multiplier(point_inputs, float(base_case_irr))
        except Exception:
            cfads_mult = 1.0

    st.divider()
    st.markdown("## 1) Deterministic Ledger (source of truth)")

    if cash_ok:
        a, b = run_scenarios(point_inputs, cfads_multiplier=cfads_mult)
    else:
        # scenario arithmetic only
        base_debt = point_inputs.capex * point_inputs.debt_ratio
        base_equity = point_inputs.capex * point_inputs.equity_ratio
        overrun_amt = point_inputs.capex * point_inputs.overrun_pct
        total_capex = point_inputs.capex + overrun_amt
        debt_a = base_debt
        equity_a = base_equity + overrun_amt
        debt_b = base_debt + point_inputs.debt_ratio * overrun_amt
        equity_b = base_equity + point_inputs.equity_ratio * overrun_amt
        ds_a = pmt(point_inputs.interest_rate, point_inputs.debt_tenor_years, debt_a) if (ir_mode != MODE_BLOCK and point_inputs.interest_rate > 0) else float("nan")
        ds_b = pmt(point_inputs.interest_rate, point_inputs.debt_tenor_years, debt_b) if (ir_mode != MODE_BLOCK and point_inputs.interest_rate > 0) else float("nan")
        a = ScenarioResult("Case A", total_capex, debt_a, equity_a, ds_a, float("nan"), float("nan"), None,
                           "Cash metrics blocked in STRICT unless EBITDA/O&M/Tax/Reserves/Rate are Known.")
        b = ScenarioResult("Case B", total_capex, debt_b, equity_b, ds_b, float("nan"), float("nan"), None,
                           a.notes)

    # Ledger table
    st.markdown(f"- Total CAPEX (incl. overrun): **{usd(a.total_capex)}**")
    st.markdown("")
    st.markdown("| Metric | Case A (Debt capped) | Case B (Pro-rata) |")
    st.markdown("|---|---:|---:|")
    st.markdown(f"| Debt | {usd(a.debt)} | {usd(b.debt)} |")
    st.markdown(f"| Equity | {usd(a.equity)} | {usd(b.equity)} |")
    st.markdown(f"| Annual debt service (PMT) | {usd(a.annual_debt_service) if cash_ok else 'n/a'} | {usd(b.annual_debt_service) if cash_ok else 'n/a'} |")
    st.markdown(f"| Annual CFADS | {usd(a.annual_cfads) if cash_ok else 'n/a'} | {usd(b.annual_cfads) if cash_ok else 'n/a'} |")
    st.markdown(f"| DSCR (CFADS/PMT) | {ratio(a.dscr) if cash_ok else 'n/a'} | {ratio(b.dscr) if cash_ok else 'n/a'} |")
    st.markdown(f"| Equity IRR (simple, calibrated if ON) | {pct(a.equity_irr) if a.equity_irr is not None else 'n/a'} | {pct(b.equity_irr) if b.equity_irr is not None else 'n/a'} |")
    st.caption(f"Notes: {a.notes}  |  CFADS multiplier (calibration): {cfads_mult:,.3f}")

    # Monte Carlo
    mc_summary = None
    if mc_enabled:
        st.divider()
        st.markdown("## 2) Monte Carlo (10,000, triangular)")

        if not allow_assume:
            st.info("Monte Carlo is only available in ASSUMPTION mode.")
        elif not cash_ok:
            st.warning("Monte Carlo needs EBITDA, rate, O&M, tax, reserves to be Known or Assume (not Block).")
        else:
            mc_cfg = MCConfig(
                n=int(mc_n),
                overrun_pct=overrun_tri,
                om_pct=om_tri,
                tax_rate=tax_tri,
                reserves_pct=res_tri,
                interest_rate=ir_tri
            )
            with st.spinner("Running Monte Carlo..."):
                mc_summary = mc_run(point_inputs, mc_cfg, cfads_multiplier=cfads_mult, target_irr=float(target_irr), covenant_dscr=float(covenant_dscr))

            st.write("Counts:", mc_summary["counts"])

            st.markdown("### IRR percentiles")
            st.table({"Case A IRR": mc_summary["A_IRR"], "Case B IRR": mc_summary["B_IRR"]})

            st.markdown("### DSCR percentiles")
            st.table({"Case A DSCR": mc_summary["A_DSCR"], "Case B DSCR": mc_summary["B_DSCR"]})

            st.markdown("### Probabilities")
            st.table(mc_summary["Prob"])

    # Quant text (engine-generated; authoritative)
    quant_lines = []
    quant_lines.append("Quant Results (engine-generated, authoritative):")
    quant_lines.append(f"- Deterministic: Case A DSCR={ratio(a.dscr) if cash_ok else 'n/a'}, Case B DSCR={ratio(b.dscr) if cash_ok else 'n/a'}")
    quant_lines.append(f"- Deterministic: Case A IRR={pct(a.equity_irr) if a.equity_irr is not None else 'n/a'}, Case B IRR={pct(b.equity_irr) if b.equity_irr is not None else 'n/a'}")
    if mc_summary:
        quant_lines.append(f"- Monte Carlo: Prob(IRR < target): Case A={mc_summary['Prob']['A_IRR_lt_target']}, Case B={mc_summary['Prob']['B_IRR_lt_target']}")
        quant_lines.append(f"- Monte Carlo: Prob(DSCR < covenant): Case A={mc_summary['Prob']['A_DSCR_lt_covenant']}, Case B={mc_summary['Prob']['B_DSCR_lt_covenant']}")
    quant_text = "\n".join(quant_lines)

    # Narrative loop
    st.divider()
    st.markdown("## 3) Iterative narrative loop (Master ↔ Critics)")

    base_user = f"""
TASK:
{prompt.strip()}

Use the ledger and Quant Results, but do NOT compute numbers and do NOT introduce new numeric claims.

{quant_text}

Write:
- Recommendation: Proceed / Proceed with Mitigation / Defer
- Why, with reference to Case A/Case B DSCR and IRR behavior (by name)
- Top technical drivers and financial transmission
- Contractual mitigants/gaps
- Next validation steps checklist
""".strip()

    with st.spinner("Master drafting v1..."):
        memo = leader_call(MASTER_SYSTEM, base_user, temperature=0.2)

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
            for k,v in critiques.items():
                with st.expander(k, expanded=False):
                    st.write(v)

        with st.spinner("Master integrating critiques..."):
            memo = master_revise(memo, critiques, quant_text)

        st.markdown(f"#### Master v{i+1}")
        st.write(memo)
        history.append((f"Master v{i+1}", memo))

    # Final output
    st.divider()
    st.markdown("## 4) Final Deliverable (always produced)")

    export = []
    export.append("# Nemexis Output (v10.3.8)")
    export.append(f"Generated: {datetime.datetime.now()}")
    export.append(f"Leader: {leader}")
    export.append("")
    export.append("## Deterministic Ledger")
    export.append(f"- Total CAPEX (incl. overrun): {usd(a.total_capex)}")
    export.append(f"- Case A: Debt {usd(a.debt)}, Equity {usd(a.equity)}, DSCR {ratio(a.dscr) if cash_ok else 'n/a'}, IRR {pct(a.equity_irr) if a.equity_irr is not None else 'n/a'}")
    export.append(f"- Case B: Debt {usd(b.debt)}, Equity {usd(b.equity)}, DSCR {ratio(b.dscr) if cash_ok else 'n/a'}, IRR {pct(b.equity_irr) if b.equity_irr is not None else 'n/a'}")
    export.append(f"- CFADS multiplier (calibration): {cfads_mult:,.3f}")
    if mc_summary:
        export.append("")
        export.append("## Monte Carlo Summary (triangular)")
        export.append("```json\n" + json.dumps(mc_summary, indent=2) + "\n```")
    export.append("")
    export.append("## Quant Results")
    export.append(quant_text)
    export.append("")
    export.append("## Final Memo")
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
