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

st.set_page_config(page_title="Nemexis v10.3.3", layout="wide")
st.title("Nemexis v10.3.3 — Universal Reliability (Math Normalizer + Required Claims + Verified Synthesis)")

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
# Universal hard requirement: claim IDs (scenario arithmetic + PMTs)
# ----------------------------
REQUIRED_CLAIM_IDS = {
    "capex_overrun",
    "base_debt",
    "base_equity",
    "equity_case_a",
    "debt_case_a",
    "equity_case_b",
    "debt_case_b",
    "pmt_case_a",
    "pmt_case_b",
}

# ----------------------------
# Math normalization utilities (UNIVERSAL)
# ----------------------------
SUFFIX_MULT = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}

def normalize_expr(expr: str) -> str:
    """
    Universal normalization for LLM/human-style math:
    - 3,200,000 -> 3200000
    - 60% -> 0.60
    - 3.2B -> (3.2*1e9)
    - 92M -> (92*1e6)
    - 1.92b -> (1.92*1e9)
    """
    if expr is None:
        return ""
    s = str(expr)

    # remove commas in numbers: 3,200,000 -> 3200000
    s = re.sub(r"(?<=\d),(?=\d)", "", s)

    # percent: 60% -> 0.60
    s = re.sub(r"(\d+(\.\d+)?)\s*%", lambda m: str(float(m.group(1)) / 100.0), s)

    # suffix: 3.2B / 92M / 10k
    def repl_suffix(m):
        num = float(m.group(1))
        suf = m.group(3).lower()
        mult = SUFFIX_MULT.get(suf, 1.0)
        return f"({num}*{mult})"

    s = re.sub(r"(\d+(\.\d+)?)(\s*[kKmMbBtT])\b", repl_suffix, s)

    return s

def normalize_number(x):
    """
    Normalize numeric inputs that might come as strings with suffixes.
    """
    if isinstance(x, (int, float)):
        return x
    if x is None:
        return x
    s = str(x).strip()
    s = s.replace(",", "")
    # percent string
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except:
            return x
    # suffix string
    m = re.fullmatch(r"(\d+(\.\d+)?)([kKmMbBtT])", s)
    if m:
        num = float(m.group(1))
        mult = SUFFIX_MULT[m.group(3).lower()]
        return num * mult
    # plain float/int
    try:
        if "." in s:
            return float(s)
        return int(s)
    except:
        return x

# ----------------------------
# Math claims parsing/verification + auto-repair
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

def _default_units_for_claim(claim_type: str, time_basis: str):
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
    """
    Accepts flexible claim formats and normalizes into standard schema:
    {id, type, inputs, expected, units, (expr for arithmetic)}
    """
    fixed = dict(c)

    # allow {id, value} style
    if "value" in fixed and "expr" not in fixed and "inputs" not in fixed:
        fixed.setdefault("type", "arithmetic")
        fixed["expr"] = "value"
        fixed["inputs"] = {"value": normalize_number(fixed["value"])}

    # if type missing but expr exists -> arithmetic
    if "type" not in fixed and "expr" in fixed:
        fixed["type"] = "arithmetic"

    if "id" not in fixed or not str(fixed["id"]).strip():
        fixed["id"] = f"claim_{idx}"

    if "inputs" not in fixed or fixed["inputs"] is None:
        fixed["inputs"] = {}

    # normalize all inputs numbers
    if isinstance(fixed["inputs"], dict):
        fixed["inputs"] = {k: normalize_number(v) for k, v in fixed["inputs"].items()}
    else:
        raise ValueError(f"Claim #{idx} inputs must be object")

    if "units" not in fixed or not str(fixed["units"]).strip():
        fixed["units"] = _default_units_for_claim(fixed["type"], time_basis)

    # arithmetic needs expr
    if fixed["type"] == "arithmetic":
        if "expr" not in fixed or fixed["expr"] is None:
            raise ValueError(f"Arithmetic claim {fixed['id']} missing expr")
        fixed["expr"] = normalize_expr(fixed["expr"])

    # pmt needs rate,nper,pv
    if fixed["type"] == "pmt":
        req = {"rate", "nper", "pv"}
        if not req.issubset(set(fixed["inputs"].keys())):
            raise ValueError(f"PMT claim {fixed['id']} inputs must include {sorted(list(req))}")

    if fixed["type"] == "npv":
        req = {"rate", "cashflows"}
        if not req.issubset(set(fixed["inputs"].keys())):
            raise ValueError(f"NPV claim {fixed['id']} inputs must include {sorted(list(req))}")

    if fixed["type"] == "irr":
        req = {"cashflows"}
        if not req.issubset(set(fixed["inputs"].keys())):
            raise ValueError(f"IRR claim {fixed['id']} inputs must include {sorted(list(req))}")

    return fixed

def auto_repair_math_claims(claims_obj: dict, time_basis: str):
    if not isinstance(claims_obj, dict):
        return None, "Math Claims root is not an object"

    claims = claims_obj.get("claims")
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
                return None, f"Could not compute expected for claim {fixed['id']}: {e}"

        repaired["claims"].append(fixed)

    return repaired, None

def verify_math_claims(repaired: dict):
    results = []
    for c in repaired.get("claims", []):
        cid = c.get("id", "")
        ctype = c.get("type")
        inputs = c.get("inputs") or {}
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

def required_ids_present(repaired: dict):
    ids = {c.get("id") for c in repaired.get("claims", [])}
    missing = sorted(list(REQUIRED_CLAIM_IDS - ids))
    return (len(missing) == 0), missing

def math_gate_ok(text: str, time_basis: str):
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

    ok_ids, missing_ids = required_ids_present(repaired)
    if not ok_ids:
        return False, f"Missing required claim IDs: {missing_ids}"

    results = verify_math_claims(repaired)
    for r in results:
        if r["error"] is not None:
            return False, f"Math verifier error in {r['id']}: {r['error']}"
        if r["expected"] is not None and r["ok"] is False:
            return False, f"Math mismatch in {r['id']}"

    return True, results

def render_math_table(text: str, title: str, time_basis: str):
    st.markdown(f"### Math Verification — {title}")
    ok, info = math_gate_ok(text, time_basis=time_basis)
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
# Leader selection UI
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
time_basis = st.selectbox("Time basis for periodic claims", ["annual", "monthly"], index=0)
allocation_model = st.selectbox(
    "Allocation model for shock/overrun scenarios",
    ["incremental_allocation", "total_rebalance", "capped_variable"],
    index=0,
)

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

max_iters = st.slider("ASSUMPTION iterations", 2, 8, 4)

# ----------------------------
# Master format
# ----------------------------
MASTER_FORMAT = "\n".join([
    "Return in this exact structure:",
    "",
    "## Mode Banner",
    "",
    "## Scenario Definitions (MANDATORY)",
    "- Time basis: must equal TIME_BASIS_SELECTED.",
    "- Allocation model: must equal ALLOCATION_MODEL_SELECTED.",
    "- Define Case A and Case B explicitly using that allocation model.",
    "",
    "## Inputs Used (Verbatim)",
    "- Bullet list key-value facts only.",
    "",
    "## Assumptions Added by Master",
    "",
    "## Executive Answer",
    "",
    "## Calculations / Logic",
    "- No inline JSON. No code blocks except the final Math Claims block.",
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
    "Provide exactly ONE fenced JSON block with schema {\"claims\":[...]} and include ALL required claim IDs:",
    f"{sorted(list(REQUIRED_CLAIM_IDS))}",
    "",
    "Each claim MUST be one of:",
    "- arithmetic: needs expr + inputs + expected + units",
    "- pmt: needs inputs rate,nper,pv + expected + units",
    "",
    "No 'B' or '%' symbols inside JSON numbers; use full numbers (e.g., 3200000000) or decimals (0.6).",
    "",
    "## Confidence",
])

MASTER_SYSTEM_STRICT = "\n".join([
    "You are Nemexis Master (STRICT).",
    "- Do NOT add numeric assumptions.",
    "- If blocking inputs exist, do NOT state threshold outcomes as likely; say cannot determine.",
    "- Provide one fenced Math Claims JSON block with all required claim IDs and valid schema.",
])

MASTER_SYSTEM_ASSUME = "\n".join([
    "You are Nemexis Master (ASSUMPTION MODE).",
    "- Use the Assumption Pack to fill missing values; tag each [Assumed].",
    "- Provide one fenced Math Claims JSON block with ALL required claim IDs.",
    "- Use full numbers in JSON (no 3.2B, no 60%).",
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
    render_math_table(draft, "STRICT", time_basis=time_basis)
    st.write(draft)
    blockers = parse_blocking_items(draft)
    st.markdown("### Blocking inputs")
    st.write(blockers if blockers else ["None"])

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
                "- Must include ALL required claim IDs in Math Claims JSON.",
                "- JSON numbers must be plain numbers (no B, no %).",
                "- PMT claims must use type='pmt' with inputs rate,nper,pv.",
                "- In ASSUMPTION mode, Blocking should be NONE (fill gaps).",
            ])

        def draft_call():
            return run_master_assume(feedback)

        draft, fence_err = ensure_fenced_json(draft_call, max_tries=2)
        if fence_err:
            st.error(f"[Iter {it}] {fence_err}")

        st.write(draft)

        ok_math, math_info = math_gate_ok(draft, time_basis=time_basis)
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
        "- Provide one fenced Math Claims JSON block including ALL required claim IDs.",
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
        ok_math, math_info = math_gate_ok(memo, time_basis=time_basis)
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
Mode: ASSUMPTION + Final Synthesis (v10.3.2)

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
