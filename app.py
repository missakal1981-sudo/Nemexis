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
# Nemexis v10.3 — Universal fixes (still domain-agnostic)
#
# Adds:
# 1) Scenario Definitions (required, explicit)
# 2) Universal time-basis control (annual vs monthly)
# 3) Universal allocation model control (incremental vs total rebalance vs capped)
# 4) Deterministic Math Claims verification with:
#    - schema enforcement
#    - unit/time-basis coherence checks
#    - currency tolerances (+/- $1M)
# 5) Best-valid iteration selection + final synthesis (math-verified)
# 6) Password UX button + feedback + session state
# 7) Leader switching (OpenAI / Claude / Grok)
#
# NOTE: Finance functions supported deterministically: pmt/npv/irr + arithmetic.
# =====================================================

st.set_page_config(page_title="Nemexis v10.3", layout="wide")
st.title("Nemexis v10.3 — Universal Reliability Engine (Scenarios + Time Basis + Verified Math)")

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
# Password UX (button + state)
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
# Math Claims parsing/verification
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

def validate_math_claims_schema(claims_json: dict):
    if not isinstance(claims_json, dict):
        return False, "Math Claims JSON is not an object"
    if "claims" not in claims_json or not isinstance(claims_json["claims"], list):
        return False, "Math Claims JSON must contain 'claims' as a list"

    for i, c in enumerate(claims_json["claims"]):
        if not isinstance(c, dict):
            return False, f"Claim #{i} is not an object"
        for k in ["id", "type", "inputs", "expected", "units"]:
            if k not in c:
                return False, f"Claim #{i} missing key '{k}'"
        ctype = c["type"]
        if ctype not in ["arithmetic", "pmt", "npv", "irr"]:
            return False, f"Claim #{i} invalid type '{ctype}'"

        if ctype == "arithmetic":
            if "expr" not in c:
                return False, f"Arithmetic claim #{i} missing 'expr'"
            if not isinstance(c["inputs"], dict):
                return False, f"Arithmetic claim #{i} inputs must be object"
        elif ctype == "pmt":
            req = {"rate", "nper", "pv"}
            if not req.issubset(set(c["inputs"].keys())):
                return False, f"PMT claim #{i} inputs must include {sorted(list(req))}"
        elif ctype == "npv":
            req = {"rate", "cashflows"}
            if not req.issubset(set(c["inputs"].keys())):
                return False, f"NPV claim #{i} inputs must include {sorted(list(req))}"
        elif ctype == "irr":
            req = {"cashflows"}
            if not req.issubset(set(c["inputs"].keys())):
                return False, f"IRR claim #{i} inputs must include {sorted(list(req))}"

    return True, "OK"

def _tolerances_for_units(units: str):
    u = (units or "").lower()
    if "usd" in u:
        return 1e-6, 1_000_000.0  # rounding ok to ~$1M
    if "ratio" in u:
        return 1e-6, 1e-6
    if "%" in u or "percent" in u:
        return 1e-6, 1e-4
    return 1e-6, 1.0

def verify_math_claims(claims_json: dict):
    results = []
    for c in claims_json.get("claims", []):
        cid = c.get("id", "")
        ctype = c.get("type", "arithmetic")
        expr = c.get("expr", "")
        inputs = c.get("inputs", {}) or {}
        expected = c.get("expected", None)
        units = c.get("units", "")

        row = {"id": cid, "type": ctype, "units": units, "ok": False, "expected": expected, "computed": None, "error": None}
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
                raise ValueError("Unknown claim type")

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

def _time_basis_from_claim(claim: dict) -> str | None:
    """Infer time basis from claim inputs for PMT, or from units."""
    units = (claim.get("units") or "").lower()
    if "per_month" in units or "/month" in units or "usd_per_month" in units:
        return "monthly"
    if "per_year" in units or "/year" in units or "usd_per_year" in units:
        return "annual"
    # If PMT, infer from nper (12*years often) and rate magnitude
    if claim.get("type") == "pmt":
        nper = claim.get("inputs", {}).get("nper")
        rate = claim.get("inputs", {}).get("rate")
        try:
            nper = int(nper)
            rate = float(rate)
        except Exception:
            return None
        if nper >= 120:  # likely monthly
            return "monthly"
        # otherwise annual-ish
        return "annual"
    return None

def _units_match_time_basis(units: str, time_basis: str) -> bool:
    u = (units or "").lower()
    if time_basis == "monthly":
        return ("month" in u) or ("per_month" in u) or ("usd/month" in u)
    if time_basis == "annual":
        return ("year" in u) or ("per_year" in u) or ("usd/year" in u) or ("annual" in u)
    return True

def math_gate_ok(text: str, expected_time_basis: str | None = None):
    """
    Math gate:
    - fenced json exists
    - schema ok
    - deterministic verification ok
    - universal unit/time-basis coherence: if expected_time_basis provided, PMT claims must match units
    """
    block = extract_json_block(text)
    if not block:
        return False, "No fenced ```json``` block found in Math Claims section"
    try:
        claims = json.loads(block)
    except Exception as e:
        return False, f"Math Claims JSON parse error: {e}"

    ok_schema, msg = validate_math_claims_schema(claims)
    if not ok_schema:
        return False, f"Math Claims schema error: {msg}"

    # unit/time-basis coherence check (universal)
    if expected_time_basis:
        for c in claims.get("claims", []):
            if c.get("type") == "pmt":
                tb = _time_basis_from_claim(c) or expected_time_basis
                if not _units_match_time_basis(c.get("units", ""), tb):
                    return False, f"Units/time-basis mismatch in {c.get('id')}: units={c.get('units')} expected_time_basis={tb}"

    results = verify_math_claims(claims)
    for r in results:
        if r["error"] is not None:
            return False, f"Math verifier error in {r['id']}: {r['error']}"
        if r["expected"] is not None and r["ok"] is False:
            return False, f"Math mismatch in {r['id']}"
    return True, results

def render_math_table(text: str, title: str, expected_time_basis: str | None):
    st.markdown(f"### Math Verification — {title}")
    ok, info = math_gate_ok(text, expected_time_basis=expected_time_basis)
    if ok and isinstance(info, list):
        st.table(info)
        st.success("Math verified ✅")
    else:
        st.error(f"Math verification failed: {info}")
    return ok, info

# ----------------------------
# Blocking parsing (treat "None" as empty)
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
# Leader selection
# ----------------------------
available_leaders = []
if OPENAI_API_KEY:
    available_leaders.append("OpenAI")
if ANTHROPIC_API_KEY:
    available_leaders.append("Claude")
if XAI_API_KEY:
    available_leaders.append("Grok")

st.markdown("### Leader / Critics")
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
# Universal UI controls (no domain hardcoding)
# ----------------------------
st.markdown("### Universal Controls")
time_basis = st.selectbox("Time basis for periodic claims", ["annual", "monthly"], index=0)
allocation_model = st.selectbox(
    "Allocation model for shock/overrun scenarios",
    ["incremental_allocation", "total_rebalance", "capped_variable"],
    index=0,
    help="incremental_allocation: apply ratios to delta only; total_rebalance: apply ratios to total; capped_variable: keep one fixed and allocate remainder"
)

# ----------------------------
# UI: Problem
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

max_iters = st.slider("ASSUMPTION iterations", 2, 8, 4)

# ----------------------------
# Master prompts: Universal Scenario Definitions + Units rules
# ----------------------------
MASTER_FORMAT = "\n".join([
    "Return in this exact structure:",
    "",
    "## Mode Banner",
    "",
    "## Scenario Definitions (MANDATORY)",
    "- Time basis: annual OR monthly (must match the selection below).",
    "- Allocation model: incremental_allocation OR total_rebalance OR capped_variable (must match selection below).",
    "- Define Case A and Case B explicitly using that allocation model (no ambiguity).",
    "",
    "## Inputs Used (Verbatim)",
    "- Bullet list of key-value facts only.",
    "",
    "## Assumptions Added by Master",
    "",
    "## Executive Answer",
    "",
    "## Calculations / Logic",
    "- NO inline JSON examples. Do not include any code blocks except the single Math Claims JSON block.",
    "",
    "## Key Risks (ranked)",
    "",
    "## Contract / Legal Checks",
    "",
    "## Missing Inputs",
    "### Blocking",
    "### Non-blocking",
    "",
    "## Claims Audit",
    "",
    "## Math Claims (JSON)",
    "Provide exactly ONE fenced JSON block (no other JSON anywhere).",
    "Schema:",
    "```json",
    "{",
    '  "claims": [',
    "    {",
    '      "id": "capex_overrun",',
    '      "type": "arithmetic",',
    '      "expr": "CAPEX*0.15",',
    '      "inputs": {"CAPEX": 3200000000},',
    '      "expected": 480000000,',
    '      "units": "USD"',
    "    }",
    "  ]",
    "}",
    "```",
    "",
    "Rules:",
    "- For pmt: type='pmt' and inputs keys rate,nper,pv (no expr).",
    "- rate must match time basis (annual rate if annual; monthly rate if monthly).",
    "- nper must match time basis (years if annual; months if monthly).",
    "- units must match time basis: USD_per_year for annual; USD_per_month for monthly.",
    "- Do not call PMT() inside arithmetic expr.",
    "",
    "## Confidence",
])

MASTER_SYSTEM_STRICT = "\n".join([
    "You are Nemexis Master (STRICT).",
    "- Do NOT add numeric assumptions.",
    "- If blocking inputs exist, do NOT state threshold outcomes as likely; say cannot determine.",
    "- Any computed number must be backed by ONE fenced Math Claims JSON block (valid schema).",
    "- Scenario Definitions must match the selected time basis and allocation model.",
])

MASTER_SYSTEM_ASSUME = "\n".join([
    "You are Nemexis Master (ASSUMPTION MODE).",
    "- Use the Assumption Pack to fill missing values; tag each [Assumed] (prefer ranges + sensitivity).",
    "- Scenario Definitions must match selected time basis and allocation model.",
    "- Provide PMT as type=pmt with correct time basis (rate,nper) and matching units.",
    "- Provide only one fenced Math Claims JSON block.",
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

    ok_math, info = math_gate_ok(draft, expected_time_basis=time_basis)
    render_math_table(draft, "STRICT", expected_time_basis=time_basis)

    st.write(draft)

    blockers = parse_blocking_items(draft)
    st.markdown("### Blocking inputs")
    st.write(blockers if blockers else ["None"])

# ----------------------------
# ASSUMPTION: iterative + best-valid + final synthesis
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
                "- One fenced Math Claims JSON block only.",
                "- PMT must be type=pmt with correct time basis (rate,nper) and matching units.",
                "- Scenario Definitions must match TIME_BASIS_SELECTED and ALLOCATION_MODEL_SELECTED.",
                "- In ASSUMPTION mode, Blocking should be NONE (fill gaps).",
            ])

        def draft_call():
            return run_master_assume(feedback)

        draft, fence_err = ensure_fenced_json(draft_call, max_tries=2)
        if fence_err:
            st.error(f"[Iter {it}] {fence_err}")

        st.write(draft)

        ok_math, math_info = math_gate_ok(draft, expected_time_basis=time_basis)
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
        st.error("No math-valid iteration produced. Increase iterations or simplify claims.")
        st.stop()

    st.write(f"Selected iteration: {best_valid_iter}")
    st.write(best_valid)

    # ----------------------------
    # FINAL SYNTHESIS STEP (must pass math verification)
    # ----------------------------
    st.divider()
    st.markdown("## Final Consolidated Response (Synthesis)")

    SYNTH_SYSTEM = "\n".join([
        "You are Nemexis Final Synthesizer.",
        "You will be given a best-valid draft that already passed math verification.",
        "Your job: produce a clean consolidated IC-ready memo.",
        "Rules:",
        "- Preserve Scenario Definitions (time basis and allocation model) exactly.",
        "- Do not introduce new numeric assumptions beyond those already in the draft.",
        "- Keep exactly one fenced Math Claims JSON block at the end (schema-compliant).",
        "- No other JSON blocks, no inline code blocks.",
        "- All computed numbers must appear in Math Claims and match the prose.",
        "- Ensure units match time basis: annual=>USD_per_year, monthly=>USD_per_month for PMT claims.",
    ])

    SYNTH_FORMAT = "\n".join([
        "Return in this exact structure:",
        "",
        "## Final IC Memo (Consolidated)",
        "- 10 bullets max, decision-grade",
        "",
        "## Scenario Definitions",
        "",
        "## Assumptions Register",
        "- list assumptions (from the draft) in clean bullets",
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
        "(single fenced block, schema compliant)",
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

        ok_math, math_info = math_gate_ok(memo, expected_time_basis=time_basis)
        if ok_math:
            final_memo = memo
            final_math_table = math_info
            break
        last_err = math_info

    if final_memo is None:
        st.error(f"Final synthesis could not pass math verification. Last error: {last_err}")
        st.write("Showing best-valid iteration instead:")
        st.write(best_valid)
        final_memo = best_valid
        final_math_table = best_valid_math if isinstance(best_valid_math, list) else []

    st.write(final_memo)
    st.markdown("### Final memo — Math Verification")
    if isinstance(final_math_table, list):
        st.table(final_math_table)
    st.success("Final consolidated memo is math-verified ✅")

    # ----------------------------
    # EXPORT
    # ----------------------------
    export_text = f"""# Nemexis Export
Generated: {datetime.datetime.now()}
Leader: {leader}
Mode: ASSUMPTION + Final Synthesis (v10.3)

## UNIVERSAL CONTROLS
Time basis: {time_basis}
Allocation model: {allocation_model}

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
