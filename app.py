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

st.set_page_config(page_title="Nemexis v10", layout="wide")
st.title("Nemexis v10 — STRICT/ASSUMPTION + Multi-Leader + Deterministic Math (Fixed)")

# ----------------------------
# Load secrets
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
# Math Claims schema enforcement
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
            required = {"rate", "nper", "pv"}
            if not required.issubset(set(c["inputs"].keys())):
                return False, f"PMT claim #{i} inputs must include {sorted(list(required))}"
        elif ctype == "npv":
            required = {"rate", "cashflows"}
            if not required.issubset(set(c["inputs"].keys())):
                return False, f"NPV claim #{i} inputs must include {sorted(list(required))}"
        elif ctype == "irr":
            required = {"cashflows"}
            if not required.issubset(set(c["inputs"].keys())):
                return False, f"IRR claim #{i} inputs must include {sorted(list(required))}"
    return True, "OK"

def verify_math_claims(claims_json: dict, rel_tol=1e-6, abs_tol=1.0):
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
                raise ValueError(f"Unknown claim type: {ctype}")

            row["computed"] = computed

            if expected is None:
                row["ok"] = True
            else:
                row["ok"] = math.isclose(float(computed), float(expected), rel_tol=rel_tol, abs_tol=abs_tol)

        except Exception as e:
            row["error"] = str(e)

        results.append(row)

    return results

def math_gate_ok(text: str):
    block = extract_json_block(text)
    if not block:
        return False, "No ```json``` block found in Math Claims section"
    try:
        claims = json.loads(block)
    except Exception as e:
        return False, f"Math Claims JSON parse error: {e}"

    ok_schema, msg = validate_math_claims_schema(claims)
    if not ok_schema:
        return False, f"Math Claims schema error: {msg}"

    results = verify_math_claims(claims)
    for r in results:
        if r["error"] is not None:
            return False, f"Math verifier error in {r['id']}: {r['error']}"
        if r["expected"] is not None and r["ok"] is False:
            return False, f"Math mismatch in {r['id']}"
    return True, results

def render_math_table(text: str, title: str):
    st.markdown(f"### Math Verification — {title}")
    ok, info = math_gate_ok(text)
    if ok is True and isinstance(info, list):
        st.table(info)
        st.success("Math verified ✅")
    else:
        st.error(f"Math verification failed: {info}")
    return ok, info

# ----------------------------
# Blocking inputs parsing
# ----------------------------
def parse_blocking_items(text: str):
    m = re.search(r"## Missing Inputs.*?### Blocking(.*?)(### Non-blocking|## Claims Audit|## Confidence|## Math Claims|$)", text, flags=re.S | re.I)
    if not m:
        return ["(Blocking section not found)"]
    block = m.group(1).strip()
    if not block:
        return []
    bullets = re.findall(r"^\s*-\s+(.*)$", block, flags=re.M)
    nums = re.findall(r"^\s*\d+\.\s+(.*)$", block, flags=re.M)
    items = [x.strip() for x in (bullets + nums) if x.strip()]
    if not items and re.search(r"\bnone\b", block, flags=re.I):
        return []
    return items if items else ["(No bullet items found under Blocking)"]

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
# Assumption pack
# ----------------------------
DEFAULT_ASSUMPTION_PACK = "\n".join([
    "Finance assumptions (use ranges; tag each [Assumed]):",
    "- CAPEX phasing: Year1 30%, Year2 40%, Year3 30%",
    "- IDC: capitalized at debt rate",
    "- Debt amortization: if sculpting schedule missing, use level PMT",
    "- O&M: 1–4% of CAPEX per year (use sensitivity)",
    "- Taxes: 0–30% effective (use sensitivity)",
    "- CFADS bridge: CFADS = EBITDA - O&M - taxes - reserves (define reserves if used)",
    "",
    "Technical assumptions (use ranges; tag each [Assumed]):",
    "- Availability: 92–98%",
    "- Capacity factor: 40–55% (if required for revenue cross-check)",
    "- Delay range: 0–12 months (if modeling schedule impacts)",
    "- Top failure modes for offshore wind: (i) geotech/foundations, (ii) marine logistics/weather/vessels, (iii) cables/interconnection",
])

st.markdown("### Problem")
user_prompt = st.text_area("Prompt", height=220)
user_context = st.text_area("Context (optional)", height=140)
blocker_input = st.text_area("Provide missing inputs (optional)", height=120)

st.markdown("### Assumption Pack (used only in ASSUMPTION)")
assumption_pack = st.text_area("Assumption Pack", value=DEFAULT_ASSUMPTION_PACK, height=220)

colA, colB = st.columns(2)
with colA:
    run_strict = st.button("Run STRICT")
with colB:
    run_assume = st.button("Run ASSUMPTION")

max_iters = st.slider("ASSUMPTION iterations", 1, 6, 3)

# ----------------------------
# Master prompts
# ----------------------------
MASTER_FORMAT = "\n".join([
    "Return in this exact structure:",
    "",
    "## Mode Banner",
    "",
    "## Inputs Used (Verbatim)",
    "- List bullet key-value items ONLY (do not paste the whole prompt).",
    "",
    "## Assumptions Added by Master",
    "",
    "## Financing Treatment of Shock/Overrun (MANDATORY)",
    "- Case A (Debt capped): debt stays at base; overrun equity-funded (equity increases by full overrun).",
    "- Case B (Pro-rata 60/40): incremental overrun funded 60/40 (debt +60%*overrun, equity +40%*overrun).",
    "",
    "## Executive Answer",
    "",
    "## Calculations / Logic",
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
    "Provide exactly one JSON block with this schema:",
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
    "PMT claim must use inputs keys: rate (decimal), nper, pv.",
    "Example:",
    "```json",
    "{",
    '  "claims": [',
    "    {",
    '      "id": "debt_service",',
    '      "type": "pmt",',
    '      "inputs": {"rate": 0.065, "nper": 20, "pv": 1920000000},',
    '      "expected": -175000000,',
    '      "units": "USD_per_year"',
    "    }",
    "  ]",
    "}",
    "```",
    "",
    "## Confidence",
])

MASTER_SYSTEM_STRICT = "\n".join([
    "You are Nemexis Master (STRICT).",
    "Rules:",
    "- Do NOT add numeric assumptions.",
    "- If blocking inputs exist, do NOT state threshold outcomes as likely; say cannot determine.",
    "- Any computed number must be backed by Math Claims JSON (schema must be valid).",
])

MASTER_SYSTEM_ASSUME = "\n".join([
    "You are Nemexis Master (ASSUMPTION MODE).",
    "Rules:",
    "- Use the Assumption Pack to fill missing values; tag each [Assumed]. Prefer ranges + sensitivity.",
    "- Case A definition: debt stays at base; overrun equity-funded (equity = base equity + overrun).",
    "- Case B definition: incremental overrun funded 60/40 (debt += 0.6*overrun; equity += 0.4*overrun).",
    "- Any computed number must be backed by Math Claims JSON with VALID schema.",
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
    master_input = "\n\n".join(["MODE: STRICT", "USER PROMPT:", txt, "OUTPUT FORMAT:", MASTER_FORMAT])
    return leader_call(MASTER_SYSTEM_STRICT, master_input, temperature=0.15)

def run_master_assume(feedback: str):
    txt = build_user_text()
    master_input = "\n\n".join([
        "MODE: ASSUMPTION",
        "USER PROMPT:",
        txt,
        "ASSUMPTION PACK:",
        assumption_pack.strip(),
        feedback.strip() if feedback else "",
        "OUTPUT FORMAT:",
        MASTER_FORMAT
    ])
    return leader_call(MASTER_SYSTEM_ASSUME, master_input, temperature=0.25)

# ----------------------------
# STRICT run (auto retry once if math fails)
# ----------------------------
if run_strict:
    if not user_prompt.strip():
        st.error("Please paste a prompt.")
        st.stop()

    st.divider()
    st.markdown("## STRICT Result")

    draft = run_master_strict()
    ok_math, _ = render_math_table(draft, "STRICT (attempt 1)")
    if not ok_math:
        st.warning("STRICT failed math schema/verification. Auto-retrying once…")
        draft = run_master_strict()
        ok_math, _ = render_math_table(draft, "STRICT (attempt 2)")

    st.write(draft)
    blockers = parse_blocking_items(draft)
    st.markdown("### Blocking inputs")
    st.write(blockers if blockers else ["None"])

# ----------------------------
# ASSUMPTION iterative (schema enforced)
# ----------------------------
if run_assume:
    if not user_prompt.strip():
        st.error("Please paste a prompt.")
        st.stop()

    st.divider()
    st.markdown("## ASSUMPTION Result (iterative)")

    history = []
    final = None

    for it in range(1, max_iters + 1):
        st.markdown(f"### Iteration {it}")

        feedback = ""
        if history:
            prev = history[-1]
            feedback = "\n".join([
                "PREVIOUS ITERATION MUST-FIX:",
                f"- Math OK: {prev['math_ok']}",
                f"- Blocking inputs: {prev['blockers']}",
                f"- Issues: {prev['issues']}",
                "",
                "Rules:",
                "- Math Claims JSON must be valid schema and match prose.",
                "- Fix Case A / Case B definitions if wrong.",
                "- In ASSUMPTION mode, Blocking should be NONE (fill gaps explicitly).",
            ])

        draft = run_master_assume(feedback)
        st.write(draft)

        ok_math, _ = render_math_table(draft, f"ASSUMPTION iter {it}")
        blockers = parse_blocking_items(draft)

        issues = []
        if blockers:
            issues.append("Blocking inputs remain")
        if not ok_math:
            issues.append("Math schema/verification failed")

        history.append({
            "iter": it,
            "draft": draft,
            "math_ok": ok_math,
            "blockers": blockers,
            "issues": issues
        })

        blockers_empty = (len(blockers) == 0)
        if ok_math and blockers_empty:
            st.success("✅ Converged (math verified + no blocking inputs).")
            final = draft
            break

        final = draft

    st.divider()
    st.markdown("## Final ASSUMPTION Draft")
    st.write(final)

    export_text = f"""# Nemexis Export
Generated: {datetime.datetime.now()}
Leader: {leader}
Mode: ASSUMPTION

## USER PROMPT
{user_prompt.strip()}

## CONTEXT
{user_context.strip() if user_context.strip() else "(none)"}

## USER PROVIDED BLOCKER INPUTS
{blocker_input.strip() if blocker_input.strip() else "(none)"}

## ASSUMPTION PACK
{assumption_pack.strip()}

---

## FINAL DRAFT
{final}

---

## HISTORY
"""
    for h in history:
        export_text += f"\n\n### Iteration {h['iter']}\n"
        export_text += f"- Math OK: {h['math_ok']}\n"
        export_text += f"- Blocking: {h['blockers']}\n"
        export_text += f"- Issues: {h['issues']}\n"
        export_text += "\n---\n"
        export_text += h["draft"]

    st.divider()
    st.markdown("## Export")
    copy_to_clipboard_button(export_text, "📋 Copy Everything")
    st.download_button("⬇️ Download Markdown", export_text, file_name="nemexis_output.md", mime="text/markdown")
    st.text_area("All output (Cmd/Ctrl+A then Copy)", value=export_text, height=320)
