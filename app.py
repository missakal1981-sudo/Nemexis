import os
import time
import json
import math
import datetime
import re
import streamlit as st
from openai import OpenAI
import anthropic
import requests
import streamlit.components.v1 as components
from simpleeval import SimpleEval

# =====================================================
# Nemexis v10.3.4 — Universal Hard Gates
#
# Fixes the remaining "Math OK but nonsense" problem by adding:
# 1) REQUIRED ID CONTRACT:
#    - each required id must have correct type + required keys
#    - prevents pmt_case_a being emitted as arithmetic, etc.
# 2) TIME BASIS COHERENCE:
#    - PMT claims must match selected time basis (units + nper scale)
# 3) SCENARIO BALANCE + ALLOCATION CHECKS (universal):
#    Uses user-selected debt_ratio/equity_ratio (default 60/40) to validate:
#    - base_capex = base_debt + base_equity
#    - incremental_allocation:
#        Case A: debt_case_a = base_debt; equity_case_a = base_equity + overrun
#        Case B: debt_case_b = base_debt + debt_ratio*overrun
#                equity_case_b = base_equity + equity_ratio*overrun
#    - total_rebalance:
#        debt_case_b = debt_ratio*(base_capex + overrun)
#        equity_case_b = equity_ratio*(base_capex + overrun)
#    - capped_variable:
#        enforces that at least one of debt_case_a/base_debt or equity_case_a/base_equity matches exactly
#
# 4) STILL UNIVERSAL:
#    - supports human numeric notations in EXPR (3.2B, 60%) via normalizer
#    - but requires JSON numeric inputs to be plain numbers (no "B", no "%")
# =====================================================

st.set_page_config(page_title="Nemexis v10.3.4", layout="wide")
st.title("Nemexis v10.3.4 — Universal Reliability Engine (Typed Claims + Scenario Gates)")

# ----------------------------
# Secrets
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
XAI_API_KEY = os.getenv("XAI_API_KEY", "").strip()
APP_PASSWORD = os.getenv("NEMEXIS_PASSWORD", "").strip()

if not OPENAI_API_KEY and not ANTHROPIC_API_KEY and not XAI_API_KEY:
    st.error("No model keys found. Add at least one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY.")
    st.stop()

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
# Universal numeric normalizer for expr (NOT for JSON numbers)
# ----------------------------
SUFFIX_MULT = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}

def normalize_expr(expr: str) -> str:
    if expr is None:
        return ""
    s = str(expr)
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    s = re.sub(r"(\d+(\.\d+)?)\s*%", lambda m: str(float(m.group(1)) / 100.0), s)

    def repl_suffix(m):
        num = float(m.group(1))
        suf = m.group(3).lower().strip()
        mult = SUFFIX_MULT.get(suf, 1.0)
        return f"({num}*{mult})"

    s = re.sub(r"(\d+(\.\d+)?)(\s*[kKmMbBtT])\b", repl_suffix, s)
    return s

def normalize_number(x):
    if isinstance(x, (int, float)):
        return x
    if x is None:
        return x
    s = str(x).strip().replace(",", "")
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except:
            return x
    m = re.fullmatch(r"(\d+(\.\d+)?)([kKmMbBtT])", s)
    if m:
        num = float(m.group(1))
        mult = SUFFIX_MULT[m.group(3).lower()]
        return num * mult
    try:
        return float(s) if "." in s else int(s)
    except:
        return x

# ----------------------------
# REQUIRED CLAIM CONTRACT (universal)
# ----------------------------
REQUIRED_CONTRACT = {
    "capex_overrun": {"type": "arithmetic", "units_kind": "usd"},
    "base_debt": {"type": "arithmetic", "units_kind": "usd"},
    "base_equity": {"type": "arithmetic", "units_kind": "usd"},
    "debt_case_a": {"type": "arithmetic", "units_kind": "usd"},
    "equity_case_a": {"type": "arithmetic", "units_kind": "usd"},
    "debt_case_b": {"type": "arithmetic", "units_kind": "usd"},
    "equity_case_b": {"type": "arithmetic", "units_kind": "usd"},
    "pmt_case_a": {"type": "pmt", "units_kind": "pmt"},
    "pmt_case_b": {"type": "pmt", "units_kind": "pmt"},
}

REQUIRED_IDS = set(REQUIRED_CONTRACT.keys())

# ----------------------------
# Math Claims parsing + auto-repair (expected/units), but NOT type for required IDs
# ----------------------------
def extract_json_block(text: str) -> str | None:
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

def strip_json_comments(s: str) -> str:
    return re.sub(r"//.*", "", s)

def _tolerances_for_units(units: str):
    u = (units or "").lower()
    if "usd" in u:
        return 1e-6, 1_000_000.0
    if "ratio" in u:
        return 1e-6, 1e-6
    if "%" in u or "percent" in u:
        return 1e-6, 1e-4
    return 1e-6, 1.0

def _default_units_for_claim(claim_id: str, claim_type: str, time_basis: str):
    if claim_type == "pmt":
        return "USD_per_year" if time_basis == "annual" else "USD_per_month"
    return "USD"

def _compute_claim_value(claim: dict):
    ctype = claim.get("type")
    inputs = claim.get("inputs") or {}
    if ctype == "arithmetic":
        expr = normalize_expr(claim.get("expr", ""))
        se = SimpleEval()
        se.names = dict(inputs)
        return se.eval(expr)
    if ctype == "pmt":
        return pmt(**inputs)
    if ctype == "npv":
        return npv(**inputs)
    if ctype == "irr":
        return irr(**inputs)
    raise ValueError(f"Unknown claim type: {ctype}")

def coerce_claim(c: dict, time_basis: str, idx: int):
    fixed = dict(c)

    # accept {id, value} as arithmetic ONLY for NON-required ids
    if "value" in fixed and ("expr" not in fixed and "inputs" not in fixed):
        if fixed.get("id") in REQUIRED_IDS:
            raise ValueError("Required id cannot be value-only; must provide proper type/expr/inputs")
        fixed.setdefault("type", "arithmetic")
        fixed["expr"] = "value"
        fixed["inputs"] = {"value": normalize_number(fixed["value"])}

    # infer arithmetic only if type missing AND id is not required
    if "type" not in fixed and "expr" in fixed and fixed.get("id") not in REQUIRED_IDS:
        fixed["type"] = "arithmetic"

    if "id" not in fixed or not str(fixed["id"]).strip():
        fixed["id"] = f"claim_{idx}"

    # inputs must exist
    if "inputs" not in fixed or fixed["inputs"] is None:
        fixed["inputs"] = {}

    if not isinstance(fixed["inputs"], dict):
        raise ValueError("inputs must be an object")

    # normalize inputs numbers
    fixed["inputs"] = {k: normalize_number(v) for k, v in fixed["inputs"].items()}

    # ensure type
    if "type" not in fixed or fixed["type"] is None:
        raise ValueError("missing type")
    fixed["type"] = str(fixed["type"]).strip().lower()

    # arithmetic needs expr
    if fixed["type"] == "arithmetic":
        if "expr" not in fixed or fixed["expr"] is None:
            raise ValueError("arithmetic missing expr")
        fixed["expr"] = normalize_expr(fixed["expr"])

    # pmt schema
    if fixed["type"] == "pmt":
        req = {"rate", "nper", "pv"}
        if not req.issubset(set(fixed["inputs"].keys())):
            raise ValueError(f"pmt inputs must include {sorted(list(req))}")

    # units default
    if "units" not in fixed or not str(fixed["units"]).strip():
        fixed["units"] = _default_units_for_claim(fixed["id"], fixed["type"], time_basis)

    return fixed

def auto_repair_math_claims(raw: dict, time_basis: str):
    if not isinstance(raw, dict):
        return None, "Math Claims root is not an object"
    claims = raw.get("claims")
    if not isinstance(claims, list):
        return None, "Math Claims missing 'claims' list"

    repaired = {"claims": []}
    for i, c in enumerate(claims):
        if not isinstance(c, dict):
            return None, f"Claim #{i} is not an object"
        try:
            fixed = coerce_claim(c, time_basis=time_basis, idx=i)
        except Exception as e:
            return None, f"Claim #{i} invalid: {e}"

        # expected auto-fill
        if "expected" not in fixed:
            try:
                val = _compute_claim_value(fixed)
                if "usd" in (fixed.get("units") or "").lower():
                    fixed["expected"] = float(round(float(val) / 1000.0) * 1000.0)
                else:
                    fixed["expected"] = float(val)
            except Exception as e:
                return None, f"Could not compute expected for {fixed['id']}: {e}"

        repaired["claims"].append(fixed)

    return repaired, None

def verify_math_claims(repaired: dict):
    results = []
    for c in repaired.get("claims", []):
        cid = c.get("id", "")
        ctype = c.get("type")
        expected = c.get("expected", None)
        units = c.get("units", "")

        row = {"id": cid, "type": ctype, "units": units, "ok": False, "expected": expected, "computed": None, "error": None}
        try:
            computed = _compute_claim_value(c)
            row["computed"] = computed
            if expected is None:
                row["ok"] = True
            else:
                rel_tol, abs_tol = _tolerances_for_units(units)
                row["ok"] = math.isclose(float(computed), float(expected), rel_tol=rel_tol, abs_tol=abs_tol)
        except Exception as e:
            row["error"] = str(e)

        results.append(row)

    return results

# ----------------------------
# Universal semantic gates
# ----------------------------
def required_ids_gate(repaired: dict):
    ids = {c.get("id") for c in repaired.get("claims", [])}
    missing = sorted(list(REQUIRED_IDS - ids))
    if missing:
        return False, f"Missing required claim IDs: {missing}"
    return True, "OK"

def required_contract_gate(repaired: dict, time_basis: str):
    # Build map
    cmap = {c["id"]: c for c in repaired.get("claims", []) if "id" in c}
    for rid, spec in REQUIRED_CONTRACT.items():
        c = cmap.get(rid)
        if c is None:
            return False, f"Missing required claim {rid}"
        if c.get("type") != spec["type"]:
            return False, f"Required claim {rid} must be type={spec['type']}, got {c.get('type')}"
        # Units kind checks
        units = (c.get("units") or "").lower()
        if spec["units_kind"] == "usd":
            if "usd" not in units or ("per_" in units):
                # allow plain USD only
                if units not in ("usd",):
                    return False, f"Required claim {rid} must have units=USD, got {c.get('units')}"
        if spec["units_kind"] == "pmt":
            if time_basis == "annual" and "year" not in units:
                return False, f"{rid} must be USD_per_year in annual mode, got {c.get('units')}"
            if time_basis == "monthly" and "month" not in units:
                return False, f"{rid} must be USD_per_month in monthly mode, got {c.get('units')}"
            # PMT inputs coherence
            inp = c.get("inputs") or {}
            nper = int(inp.get("nper"))
            rate = float(inp.get("rate"))
            if time_basis == "annual":
                if nper > 60:
                    return False, f"{rid} nper={nper} looks monthly; annual mode expects years"
                if rate <= 0 or rate > 1:
                    return False, f"{rid} rate={rate} invalid for annual rate"
            else:
                if nper < 60:
                    return False, f"{rid} nper={nper} looks annual; monthly mode expects months"
                if rate <= 0 or rate > 1:
                    return False, f"{rid} rate={rate} invalid"

    return True, "OK"

def scenario_balance_gate(repaired: dict, allocation_model: str, debt_ratio: float, equity_ratio: float, abs_tol_usd: float = 1_000_000.0):
    """
    Universal scenario gate using required claims:
    - base_capex = base_debt + base_equity
    - incremental_allocation:
      A: debt_case_a = base_debt; equity_case_a = base_equity + overrun
      B: debt_case_b = base_debt + debt_ratio*overrun; equity_case_b = base_equity + equity_ratio*overrun
    - total_rebalance:
      B: debt_case_b = debt_ratio*(base_capex+overrun); equity_case_b = equity_ratio*(base_capex+overrun)
      A (debt capped) still: debt_case_a=base_debt; equity_case_a=base_equity+overrun
    - capped_variable:
      Must satisfy base_capex + overrun == debt_case_x + equity_case_x for both cases,
      and at least one of (debt_case_a==base_debt) or (equity_case_a==base_equity) holds.
    """
    cmap = {c["id"]: c for c in repaired.get("claims", []) if "id" in c}
    def v(cid):
        return float(cmap[cid]["expected"])

    base_debt = v("base_debt")
    base_equity = v("base_equity")
    overrun = v("capex_overrun")

    debt_a = v("debt_case_a")
    eq_a = v("equity_case_a")
    debt_b = v("debt_case_b")
    eq_b = v("equity_case_b")

    base_capex = base_debt + base_equity
    total_capex = base_capex + overrun

    def close(a, b):
        return abs(a - b) <= abs_tol_usd

    # universal identity: both cases must fund total capex after overrun
    if not close(debt_a + eq_a, total_capex):
        return False, f"Scenario identity fails Case A: debt+equity={debt_a+eq_a} vs total_capex={total_capex}"
    if not close(debt_b + eq_b, total_capex):
        return False, f"Scenario identity fails Case B: debt+equity={debt_b+eq_b} vs total_capex={total_capex}"

    if allocation_model == "incremental_allocation":
        if not close(debt_a, base_debt):
            return False, f"Case A debt must equal base_debt under incremental_allocation"
        if not close(eq_a, base_equity + overrun):
            return False, f"Case A equity must equal base_equity+overrun under incremental_allocation"
        if not close(debt_b, base_debt + debt_ratio * overrun):
            return False, f"Case B debt must equal base_debt + debt_ratio*overrun under incremental_allocation"
        if not close(eq_b, base_equity + equity_ratio * overrun):
            return False, f"Case B equity must equal base_equity + equity_ratio*overrun under incremental_allocation"

    elif allocation_model == "total_rebalance":
        if not close(debt_a, base_debt):
            return False, f"Case A debt must equal base_debt under total_rebalance (debt capped case)"
        if not close(eq_a, base_equity + overrun):
            return False, f"Case A equity must equal base_equity+overrun under total_rebalance (debt capped case)"
        if not close(debt_b, debt_ratio * total_capex):
            return False, f"Case B debt must equal debt_ratio*total_capex under total_rebalance"
        if not close(eq_b, equity_ratio * total_capex):
            return False, f"Case B equity must equal equity_ratio*total_capex under total_rebalance"

    else:  # capped_variable
        if not (close(debt_a, base_debt) or close(eq_a, base_equity)):
            return False, f"capped_variable requires either debt_case_a==base_debt or equity_case_a==base_equity"
    return True, "OK"

# ----------------------------
# Math gate
# ----------------------------
def math_gate_ok(text: str, time_basis: str, allocation_model: str, debt_ratio: float, equity_ratio: float):
    block = extract_json_block(text)
    if not block:
        return False, "No fenced ```json``` block found in Math Claims section"
    block = strip_json_comments(block)
    try:
        raw = json.loads(block)
    except Exception as e:
        return False, f"Math Claims JSON parse error: {e}"

    repaired, err = auto_repair_math_claims(raw, time_basis=time_basis)
    if err:
        return False, f"Math Claims auto-repair failed: {err}"

    ok, msg = required_ids_gate(repaired)
    if not ok:
        return False, msg

    ok, msg = required_contract_gate(repaired, time_basis=time_basis)
    if not ok:
        return False, msg

    ok, msg = scenario_balance_gate(repaired, allocation_model=allocation_model, debt_ratio=debt_ratio, equity_ratio=equity_ratio)
    if not ok:
        return False, msg

    results = verify_math_claims(repaired)
    for r in results:
        if r["error"] is not None:
            return False, f"Math verifier error in {r['id']}: {r['error']}"
        if r["expected"] is not None and r["ok"] is False:
            return False, f"Math mismatch in {r['id']}"
    return True, results

def render_math_table(text: str, title: str, time_basis: str, allocation_model: str, debt_ratio: float, equity_ratio: float):
    st.markdown(f"### Math Verification — {title}")
    ok, info = math_gate_ok(text, time_basis=time_basis, allocation_model=allocation_model, debt_ratio=debt_ratio, equity_ratio=equity_ratio)
    if ok and isinstance(info, list):
        st.table(info)
        st.success("Math verified ✅")
    else:
        st.error(f"Math verification failed: {info}")
    return ok, info

# ----------------------------
# Blocking parsing
# ----------------------------
def parse_blocking_items(text: str):
    m = re.search(r"## Missing Inputs.*?### Blocking(.*?)(### Non-blocking|## Claims Audit|## Confidence|## Math Claims|$)", text, flags=re.S | re.I)
    if not m:
        return ["(Blocking section not found)"]
    block = m.group(1).strip()
    if not block:
        return []
    if re.fullmatch(r"(?is)\s*none\s*", block):
        return []
    bullets = re.findall(r"^\s*-\s+(.*)$", block, flags=re.M)
    nums = re.findall(r"^\s*\d+\.\s+(.*)$", block, flags=re.M)
    items = [x.strip() for x in (bullets + nums) if x.strip()]
    if len(items) == 1 and items[0].strip().lower() == "none":
        return []
    return items

# ----------------------------
# Model calls
# ----------------------------
def call_openai(model_name: str, system: str, user_text: str, temperature=0.2) -> str:
    resp = openai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_text}],
        temperature=temperature,
    )
    return resp.choices[0].message.content

def call_claude(model_name: str, system: str, user_text: str) -> str:
    msg = anthropic_client.messages.create(
        model=model_name,
        max_tokens=1600,
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
# Leader UI
# ----------------------------
available_leaders = []
if OPENAI_API_KEY:
    available_leaders.append("OpenAI")
if ANTHROPIC_API_KEY:
    available_leaders.append("Claude")
if XAI_API_KEY:
    available_leaders.append("Grok")

st.markdown("### Leader")
leader = st.selectbox("Leader (drives the draft)", available_leaders, index=0)

col1, col2, col3 = st.columns(3)
with col1:
    openai_model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o"], index=0, disabled=(not OPENAI_API_KEY))
with col2:
    claude_model = st.selectbox("Claude model", ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-4-20250514"], index=0, disabled=(not ANTHROPIC_API_KEY))
with col3:
    grok_model = st.selectbox("Grok model", ["grok-4-fast", "grok-4"], index=0, disabled=(not XAI_API_KEY))

def leader_call(system: str, user_text: str, temperature=0.2):
    if leader == "OpenAI":
        return call_openai(openai_model, system, user_text, temperature=temperature)
    if leader == "Claude":
        return call_claude(claude_model, system, user_text)
    return call_grok(grok_model, system, user_text)

# ----------------------------
# Universal controls
# ----------------------------
st.markdown("### Universal Controls")
time_basis = st.selectbox("Time basis for PMT claims", ["annual", "monthly"], index=0)
allocation_model = st.selectbox("Allocation model", ["incremental_allocation", "total_rebalance", "capped_variable"], index=0)

st.markdown("### Funding Ratios (universal)")
debt_ratio = st.number_input("Debt ratio", min_value=0.0, max_value=1.0, value=0.60, step=0.01)
equity_ratio = st.number_input("Equity ratio", min_value=0.0, max_value=1.0, value=0.40, step=0.01)
if abs((debt_ratio + equity_ratio) - 1.0) > 1e-9:
    st.warning("Debt ratio + Equity ratio should equal 1.0 for meaningful scenario checks.")

# ----------------------------
# Problem inputs
# ----------------------------
st.markdown("### Problem")
user_prompt = st.text_area("Prompt", height=220)
user_context = st.text_area("Context (optional)", height=140)
blocker_input = st.text_area("Provide missing inputs (optional)", height=120)

DEFAULT_ASSUMPTION_PACK = "\n".join([
    "Finance assumptions (use ranges; tag each [Assumed]):",
    "- CAPEX phasing: Year1 30%, Year2 40%, Year3 30%",
    "- IDC: capitalized at debt rate",
    "- Debt amortization: if sculpting schedule missing, use level PMT",
    "- O&M: 1–4% of CAPEX per year (use sensitivity)",
    "- Taxes: 0–30% effective (use sensitivity)",
    "- Reserves: 0–5% of EBITDA",
    "- CFADS bridge: CFADS = EBITDA - O&M - taxes - reserves",
    "",
    "Technical assumptions (use ranges; tag each [Assumed]):",
    "- Availability: 92–98%",
    "- Capacity factor: 40–55%",
    "- Delay range: 0–12 months",
    "- Failure modes: (i) geotech/foundations, (ii) logistics/weather, (iii) grid/interconnection",
])

st.markdown("### Assumption Pack (used only in ASSUMPTION)")
assumption_pack = st.text_area("Assumption Pack", value=DEFAULT_ASSUMPTION_PACK, height=220)

colA, colB = st.columns(2)
with colA:
    run_strict = st.button("Run STRICT")
with colB:
    run_assume = st.button("Run ASSUMPTION")

max_iters = st.slider("ASSUMPTION iterations", 2, 10, 6)

# ----------------------------
# Prompt template
# ----------------------------
MASTER_FORMAT = "\n".join([
    "Return in this exact structure:",
    "## Mode Banner",
    "## Scenario Definitions (MANDATORY)",
    "- Time basis: must equal TIME_BASIS_SELECTED.",
    "- Allocation model: must equal ALLOCATION_MODEL_SELECTED.",
    "- Use DEBT_RATIO and EQUITY_RATIO provided.",
    "## Inputs Used (Verbatim)",
    "## Assumptions Added by Master",
    "## Executive Answer",
    "## Calculations / Logic",
    "- No inline JSON. No code blocks except the final Math Claims block.",
    "## Key Risks (ranked)",
    "## Contract / Legal Checks",
    "## Missing Inputs",
    "### Blocking",
    "### Non-blocking",
    "## Claims Audit",
    "## Math Claims (JSON)",
    "Provide exactly ONE fenced JSON block with schema {\"claims\":[...]} that includes ALL required IDs:",
    f"{sorted(list(REQUIRED_IDS))}",
    "",
    "Hard requirements for required IDs:",
    "- base_debt/base_equity/capex_overrun/debt_case_a/equity_case_a/debt_case_b/equity_case_b must be arithmetic claims (type='arithmetic') with expr+inputs+expected+units='USD'.",
    "- pmt_case_a/pmt_case_b must be pmt claims (type='pmt') with inputs rate,nper,pv and units matching TIME_BASIS_SELECTED.",
    "- JSON numeric inputs must be plain numbers (no 'B', no '%', no expressions like 0.065/12).",
    "",
    "## Confidence",
])

MASTER_SYSTEM_STRICT = "\n".join([
    "You are Nemexis Master (STRICT).",
    "- Do NOT add numeric assumptions.",
    "- If blocking inputs exist, do NOT state threshold outcomes as likely; say cannot determine.",
    "- Provide one fenced Math Claims JSON block meeting all required claim constraints.",
])

MASTER_SYSTEM_ASSUME = "\n".join([
    "You are Nemexis Master (ASSUMPTION MODE).",
    "- Use Assumption Pack to fill gaps; tag each [Assumed].",
    "- Provide one fenced Math Claims JSON block meeting all required claim constraints.",
    "- JSON numeric inputs must be plain numbers (no B, no %, no 0.065/12 expressions).",
    "- In ASSUMPTION mode, Blocking should be NONE unless truly impossible.",
])

def build_user_text():
    txt = user_prompt.strip()
    if user_context.strip():
        txt += "\n\n---\nContext:\n" + user_context.strip()
    if blocker_input.strip():
        txt += "\n\n---\nUser-provided missing inputs:\n" + blocker_input.strip()
    return txt

def run_master_strict():
    txt = build_user_text()
    master_input = "\n\n".join([
        "MODE: STRICT",
        f"TIME_BASIS_SELECTED: {time_basis}",
        f"ALLOCATION_MODEL_SELECTED: {allocation_model}",
        f"DEBT_RATIO: {debt_ratio}",
        f"EQUITY_RATIO: {equity_ratio}",
        "USER PROMPT:",
        txt,
        "OUTPUT FORMAT:",
        MASTER_FORMAT
    ])
    return leader_call(MASTER_SYSTEM_STRICT, master_input, temperature=0.15)

def run_master_assume(feedback: str):
    txt = build_user_text()
    master_input = "\n\n".join([
        "MODE: ASSUMPTION",
        f"TIME_BASIS_SELECTED: {time_basis}",
        f"ALLOCATION_MODEL_SELECTED: {allocation_model}",
        f"DEBT_RATIO: {debt_ratio}",
        f"EQUITY_RATIO: {equity_ratio}",
        "USER PROMPT:",
        txt,
        "ASSUMPTION PACK:",
        assumption_pack.strip(),
        feedback.strip() if feedback else "",
        "OUTPUT FORMAT:",
        MASTER_FORMAT
    ])
    return leader_call(MASTER_SYSTEM_ASSUME, master_input, temperature=0.25)

def ensure_fenced_json(draft_fn, max_tries=2):
    last = None
    for _ in range(max_tries):
        last = draft_fn()
        if extract_json_block(last) is not None:
            return last, None
    return last, "Missing fenced ```json``` block after retries"

# ----------------------------
# STRICT
# ----------------------------
if run_strict:
    if not user_prompt.strip():
        st.error("Please paste a prompt.")
        st.stop()
    st.divider()
    st.markdown("## STRICT Result")
    draft, fence_err = ensure_fenced_json(run_master_strict, max_tries=2)
    if fence_err:
        st.error(fence_err)
    render_math_table(draft, "STRICT", time_basis=time_basis, allocation_model=allocation_model, debt_ratio=debt_ratio, equity_ratio=equity_ratio)
    st.write(draft)

# ----------------------------
# ASSUMPTION iterative + best-valid + synthesis
# ----------------------------
if run_assume:
    if not user_prompt.strip():
        st.error("Please paste a prompt.")
        st.stop()

    st.divider()
    st.markdown("## ASSUMPTION (iterative)")

    history = []
    best_valid = None
    best_valid_iter = None
    best_valid_math = None

    for it in range(1, max_iters + 1):
        st.markdown(f"### Iteration {it}")

        feedback = ""
        if history:
            prev = history[-1]
            feedback = "\n".join([
                "PREVIOUS ITERATION MUST-FIX:",
                f"- Math OK: {prev['math_ok']}",
                f"- Blocking inputs: {prev['blockers']}",
                f"- Math failure reason: {prev['math_fail_reason']}",
                "",
                "Hard rules:",
                "- Must include ALL required claim IDs with correct types/units.",
                "- JSON numeric inputs must be plain numbers (no B, no %, no 0.065/12).",
                "- Scenario balance must satisfy funding identity checks.",
                "- In ASSUMPTION mode, Blocking should be NONE.",
            ])

        def draft_call():
            return run_master_assume(feedback)

        draft, fence_err = ensure_fenced_json(draft_call, max_tries=2)
        if fence_err:
            st.error(f"[Iter {it}] {fence_err}")

        st.write(draft)

        ok_math, math_info = math_gate_ok(draft, time_basis=time_basis, allocation_model=allocation_model, debt_ratio=debt_ratio, equity_ratio=equity_ratio)
        if ok_math:
            st.success("Math verified ✅")
            st.table(math_info)
        else:
            st.error(f"Math failed: {math_info}")

        blockers = parse_blocking_items(draft)
        blockers_empty = (len(blockers) == 0)

        history.append({
            "iter": it,
            "draft": draft,
            "math_ok": ok_math,
            "math_fail_reason": None if ok_math else str(math_info),
            "blockers": blockers,
            "blockers_empty": blockers_empty,
        })

        if ok_math and blockers_empty:
            best_valid = draft
            best_valid_iter = it
            best_valid_math = math_info
            break

        if ok_math and best_valid is None:
            best_valid = draft
            best_valid_iter = it
            best_valid_math = math_info

    st.divider()
    st.markdown("## Best Valid Iteration")
    if best_valid is None:
        st.error("No math-valid iteration produced. Increase iterations or simplify.")
        st.stop()

    st.write(f"Selected iteration: {best_valid_iter}")
    st.write(best_valid)

    st.divider()
    st.markdown("## Final Consolidated Response (Synthesis)")

    SYNTH_SYSTEM = "\n".join([
        "You are Nemexis Final Synthesizer.",
        "Given a best-valid draft, produce a clean consolidated memo.",
        "Rules:",
        "- Preserve Scenario Definitions exactly.",
        "- Do not introduce new numeric assumptions.",
        "- Provide one fenced Math Claims JSON block including ALL required IDs with correct types.",
        "- No other JSON or code blocks.",
    ])

    SYNTH_FORMAT = "\n".join([
        "## Final IC Memo (Consolidated)",
        "- 10 bullets max",
        "",
        "## Scenario Definitions",
        "",
        "## Assumptions Register",
        "",
        "## Case A vs Case B",
        "",
        "## Top 3 Technical Drivers",
        "",
        "## Contractual Mitigants / Gaps",
        "",
        "## Recommendation",
        "",
        "## Math Claims (JSON)",
    ])

    synth_input = "\n\n".join([
        "BEST VALID DRAFT:",
        best_valid,
        "OUTPUT FORMAT:",
        SYNTH_FORMAT
    ])

    final_memo = None
    final_math_table = None
    last_err = None

    for attempt in range(1, 3):
        with st.spinner(f"Synthesizing final memo (attempt {attempt})..."):
            memo = leader_call(SYNTH_SYSTEM, synth_input, temperature=0.2)
        ok_math, math_info = math_gate_ok(memo, time_basis=time_basis, allocation_model=allocation_model, debt_ratio=debt_ratio, equity_ratio=equity_ratio)
        if ok_math:
            final_memo = memo
            final_math_table = math_info
            break
        last_err = math_info

    if final_memo is None:
        st.error(f"Final synthesis could not pass math verification. Last error: {last_err}")
        final_memo = best_valid
        final_math_table = best_valid_math if isinstance(best_valid_math, list) else []

    st.write(final_memo)
    st.markdown("### Final memo — Math Verification")
    if isinstance(final_math_table, list):
        st.table(final_math_table)
    st.success("Final consolidated memo is math-verified ✅")

    export_text = f"""# Nemexis Export
Generated: {datetime.datetime.now()}
Leader: {leader}
Mode: ASSUMPTION + Final Synthesis (v10.3.4)

## UNIVERSAL CONTROLS
Time basis: {time_basis}
Allocation model: {allocation_model}
Debt ratio: {debt_ratio}
Equity ratio: {equity_ratio}

## USER PROMPT
{user_prompt.strip()}

## CONTEXT
{user_context.strip() if user_context.strip() else "(none)"}

## USER PROVIDED INPUTS
{blocker_input.strip() if blocker_input.strip() else "(none)"}

## ASSUMPTION PACK
{assumption_pack.strip()}

---

## BEST VALID ITERATION (#{best_valid_iter})
{best_valid}

---

## FINAL CONSOLIDATED RESPONSE
{final_memo}

---

## HISTORY
"""
    for h in history:
        export_text += f"\n\n### Iteration {h['iter']}\n"
        export_text += f"- Math OK: {h['math_ok']}\n"
        export_text += f"- Blocking: {h['blockers']}\n"
        export_text += f"- Math fail reason: {h['math_fail_reason']}\n"
        export_text += "\n---\n"
        export_text += h["draft"]

    st.divider()
    st.markdown("## Export")
    copy_to_clipboard_button(export_text, "📋 Copy Everything")
    st.download_button("⬇️ Download Markdown", export_text, file_name="nemexis_output.md", mime="text/markdown")
    st.text_area("All output (Cmd/Ctrl+A then Copy)", value=export_text, height=320)
