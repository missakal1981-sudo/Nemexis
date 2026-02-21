import os
import time
import json
import math
import datetime
import streamlit as st
from openai import OpenAI
import anthropic
import requests
import streamlit.components.v1 as components
from simpleeval import SimpleEval

# =====================================================
# Nemexis v5 + Deterministic Math Verifier (Option A)
# Master: OpenAI
# Critics: Claude + Grok
# Flow: Rev0 -> Critiques -> Rev1 -> Guardrail Check -> Rev1b
# Add-on: Master emits Math Claims JSON, app verifies deterministically
# =====================================================

st.set_page_config(page_title="Nemexis", layout="wide")
st.title("Nemexis — Universal Reliability Engine (v5 + Math Verifier)")

# ----------------------------
# Password Gate (optional)
# ----------------------------
APP_PASSWORD = os.getenv("NEMEXIS_PASSWORD", "").strip()
if APP_PASSWORD:
    entered = st.text_input("Password", type="password")
    if entered != APP_PASSWORD:
        st.stop()

# ----------------------------
# Load Keys
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
XAI_API_KEY = os.getenv("XAI_API_KEY", "").strip()

missing = []
if not OPENAI_API_KEY:
    missing.append("OPENAI_API_KEY")
if not ANTHROPIC_API_KEY:
    missing.append("ANTHROPIC_API_KEY")
if not XAI_API_KEY:
    missing.append("XAI_API_KEY")

if missing:
    st.error(f"Missing secrets: {', '.join(missing)} (Manage app → Settings → Secrets)")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ----------------------------
# Clipboard helper
# ----------------------------
def copy_to_clipboard_button(text: str, button_label: str = "📋 Copy all output"):
    safe = text.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    html = f"""
    <button id="copybtn" style="
        padding:10px 14px; border-radius:10px; border:1px solid #ddd;
        background:white; cursor:pointer; font-weight:600;">
        {button_label}
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
            msg.textContent = "Copy failed (browser blocked) ❌";
            setTimeout(() => msg.textContent = "", 3000);
        }}
    }};
    </script>
    """
    components.html(html, height=60)

# ----------------------------
# Deterministic finance helpers
# ----------------------------
def pmt(rate, nper, pv, fv=0.0, when=0):
    rate = float(rate)
    nper = int(nper)
    pv = float(pv)
    fv = float(fv)
    when = int(when)

    if nper <= 0:
        raise ValueError("nper must be > 0")
    if rate == 0:
        return -(pv + fv) / nper

    factor = (1 + rate) ** nper
    payment = -(rate * (pv * factor + fv)) / ((factor - 1) * (1 + rate * when))
    return payment

def npv(rate, cashflows):
    rate = float(rate)
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows, start=1))

def irr(cashflows, guess=0.1, max_iter=100, tol=1e-7):
    if not (any(cf < 0 for cf in cashflows) and any(cf > 0 for cf in cashflows)):
        raise ValueError("IRR requires at least one negative and one positive cashflow.")

    r = float(guess)
    for _ in range(max_iter):
        npv_val = 0.0
        d_val = 0.0
        for t, cf in enumerate(cashflows):
            npv_val += cf / ((1 + r) ** t)
            if t > 0:
                d_val -= t * cf / ((1 + r) ** (t + 1))
        if abs(npv_val) < tol:
            return r
        if d_val == 0:
            break
        r = r - npv_val / d_val
    raise ValueError("IRR did not converge")

# ----------------------------
# Math claims extraction + verification
# ----------------------------
def extract_json_block(text: str) -> str | None:
    """
    Extract the first ```json ... ``` block.
    """
    marker = "```json"
    start = text.find(marker)
    if start == -1:
        return None
    start = text.find("\n", start)
    if start == -1:
        return None
    end = text.find("```", start + 1)
    if end == -1:
        return None
    return text[start:end].strip()

def verify_math_claims(claims_json: dict, rel_tol=1e-6, abs_tol=1.0):
    """
    claims_json format:
    { "claims": [ {id, type, expr, inputs, expected, units}, ... ] }
    types: arithmetic | pmt | npv | irr
    - arithmetic uses expr + inputs
    - pmt/npv/irr use inputs dict as kwargs
    """
    results = []
    for c in claims_json.get("claims", []):
        cid = c.get("id", "")
        ctype = c.get("type", "arithmetic")
        expr = c.get("expr", "")
        inputs = c.get("inputs", {}) or {}
        expected = c.get("expected", None)
        units = c.get("units", "")

        row = {
            "id": cid,
            "type": ctype,
            "units": units,
            "ok": False,
            "expected": expected,
            "computed": None,
            "error": None,
        }

        try:
            if ctype == "arithmetic":
                se = SimpleEval()
                se.names = dict(inputs)
                computed = se.eval(expr)
            elif ctype == "pmt":
                computed = pmt(**inputs)
            elif ctype == "npv":
                computed = npv(**inputs)
            elif ctype == "irr":
                computed = irr(**inputs)
            else:
                raise ValueError(f"Unknown claim type: {ctype}")

            row["computed"] = computed

            if expected is None:
                # No comparison requested (allowed but not ideal)
                row["ok"] = True
            else:
                row["ok"] = math.isclose(float(computed), float(expected), rel_tol=rel_tol, abs_tol=abs_tol)

        except Exception as e:
            row["error"] = str(e)

        results.append(row)

    return results

def render_math_verification(label: str, text: str):
    """
    Show math verification table for the first JSON block found.
    """
    st.markdown(f"### Math Verification — {label}")

    json_block = extract_json_block(text)
    if not json_block:
        st.info("No Math Claims JSON block found.")
        return None

    try:
        claims = json.loads(json_block)
    except Exception as e:
        st.error(f"Could not parse Math Claims JSON: {e}")
        st.code(json_block)
        return None

    results = verify_math_claims(claims)
    st.table(results)

    # Determine mismatch severity
    mismatches = [
        r for r in results
        if r["error"] is None and r["expected"] is not None and r["ok"] is False
    ]
    errors = [r for r in results if r["error"] is not None]

    if errors:
        st.error("Math verifier encountered errors. Treat computed numbers as untrusted until fixed.")
    elif mismatches:
        st.error("Math mismatch detected. Treat computed numbers as untrusted until corrected.")
    else:
        st.success("All declared math claims verified ✅")

    return {"claims": claims, "results": results}

# ----------------------------
# UI
# ----------------------------
st.markdown("### Inputs")
prompt = st.text_area("User Prompt", height=180)
context = st.text_area("Context (optional)", height=150)

st.markdown("### Settings")
master_model = st.selectbox("Master model (OpenAI)", ["gpt-4o-mini", "gpt-4o"], index=0)
claude_model = st.selectbox(
    "Claude judge model",
    ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-4-20250514"],
    index=0,
)
grok_model = st.selectbox("Grok judge model", ["grok-4-fast", "grok-4"], index=0)

mode = st.radio(
    "Assumptions mode",
    ["Strict (no numeric assumptions, no derived numbers)", "Bounding (allow ranges + scenarios)"],
    index=0,
)
STRICT_MODE = mode.startswith("Strict")

run = st.button("Run Nemexis")

# ----------------------------
# Orbits
# ----------------------------
ORBITS = [
    {"name": "Technical Engineering", "judge_mandate":
        "You are a senior offshore/industrial engineering reviewer. Critique technical correctness/realism, "
        "failure modes, unit sanity checks, schedule realism, and technical-financial coupling mistakes. "
        "Do NOT rewrite; provide targeted fixes."
    },
    {"name": "Economics / Project Finance", "judge_mandate":
        "You are a project finance reviewer. Critique cash-flow logic, timing, covenants/DSCR, sensitivities, "
        "and math sanity checks. Do NOT invent numbers. Do NOT rewrite; provide targeted fixes."
    },
    {"name": "Contract / Legal (Commercial)", "judge_mandate":
        "You are a contracts/commercial reviewer. Critique entitlement/pass-through, LDs/caps, change orders, "
        "termination triggers, compliance/approvals, and what clauses must be checked. "
        "Do NOT rewrite; provide targeted fixes."
    },
    {"name": "Risk / Governance", "judge_mandate":
        "You are a risk officer. Critique ambiguity, overconfidence, missing assumptions, validation gaps. "
        "Produce mini risk register + what changes the conclusion. Do NOT rewrite; provide targeted fixes."
    },
]

# ----------------------------
# Guardrails (Master)
# ----------------------------
INPUT_LOCK_RULES = """
INPUT LOCK:
- Include 'Inputs Used (Verbatim)' listing user inputs exactly.
- Include 'Assumptions Added by Master' for anything not provided.
- Do NOT silently change user inputs.
""".strip()

CLAIMS_AUDIT_RULES = """
CLAIMS AUDIT:
- Include 'Claims Audit'.
- Every numeric claim tagged: [Known] [Computed] [Assumed] [Unknown]
- [Computed] only if you show enough calculation path to replicate.
- Do NOT mark IRR/DSCR as [Computed] unless you actually compute it from explicit cashflows or provided schedules.
""".strip()

CFADS_GUARDRAIL = """
CFADS / DSCR GUARDRAIL (STRICT):
- DSCR is based on CFADS, not EBITDA.
- If only EBITDA is provided, DSCR headroom and debt capacity are [Unknown] unless CFADS bridge is provided.
- You may NOT compute debt service or max debt capacity from EBITDA/DSCR in Strict mode.
""".strip()

CAPITAL_STRUCTURE_GUARDRAIL = """
CAPITAL STRUCTURE GUARDRAIL (STRICT):
- In Strict mode, you may NOT assert post-shock debt/equity funding as fact unless provided.
- You MUST present 'Financing Treatment of Shock/Overrun' with two cases:
  Case A: Debt capped; overrun equity-funded.
  Case B: Debt upsized requires lender consent + CFADS/DSCR headroom (Unknown without model).
""".strip()

STRICT_RULES = """
STRICT MODE:
- No new numeric assumptions (tax rate, O&M %, capacity factor, probabilities, escalation, etc.).
- No derived numeric outputs unless underlying inputs are explicitly provided.
- If Blocking inputs exist, Confidence must be LOW.
- Do not use words like "likely" for threshold crossings when key blocking inputs are missing.
""".strip()

BOUNDING_RULES = """
BOUNDING MODE:
- You may add numeric assumptions only as ranges + scenario table.
- All added numeric assumptions tagged [Assumed] with brief basis.
""".strip()

# ----------------------------
# Output format (Master must output Math Claims JSON)
# ----------------------------
MASTER_SYSTEM = "You are Nemexis Master. Produce a decision-grade draft. Follow guardrails strictly."

MASTER_FORMAT = """
Return in this exact structure:

## Inputs Used (Verbatim)

## Assumptions Added by Master

## Financing Treatment of Shock/Overrun (MANDATORY)

## Executive Answer

## Calculations / Logic
- Only compute what is computable from given inputs

## Key Risks (ranked)

## Contract / Legal Checks

## Missing Inputs
### Blocking
### Non-blocking

## Claims Audit
- Every numeric claim tagged [Known] [Computed] [Assumed] [Unknown]

## Math Claims (JSON)
Provide a single JSON block enclosed exactly like this:

```json
{
  "claims": [
    {
      "id": "capex_overrun",
      "type": "arithmetic",
      "expr": "CAPEX*0.15",
      "inputs": {"CAPEX": 3200000000},
      "expected": 480000000,
      "units": "USD"
    }
  ]
}
