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
# Nemexis v9 — STRICT + ASSUMPTION + Leader Switching + Deterministic Math
#
# Key features:
# - Password unlock button w/ feedback + session persistence
# - Two buttons: Run STRICT / Run ASSUMPTION
# - Leader provider selectable: OpenAI / Claude / Grok (if keys exist)
# - Critics are the remaining providers
# - Assumption Pack (Finance + Technical) editable in UI for assumption mode
# - Deterministic math verifier for Math Claims JSON (schema enforced)
# =====================================================

st.set_page_config(page_title="Nemexis v9", layout="wide")
st.title("Nemexis v9 — Reliability Engine (STRICT + ASSUMPTION + Multi-Leader)")

# ----------------------------
# Load Secrets / Keys
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
XAI_API_KEY = os.getenv("XAI_API_KEY", "").strip()
APP_PASSWORD = os.getenv("NEMEXIS_PASSWORD", "").strip()

if not OPENAI_API_KEY and not ANTHROPIC_API_KEY and not XAI_API_KEY:
    st.error("No model keys found. Add at least one key in Streamlit Secrets (OPENAI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY).")
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
# Math claims extraction + strict schema verification
# ----------------------------
REQUIRED_CLAIM_KEYS = {"id", "type", "units", "expected"}

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
    """
    Expected format:
    {
      "claims": [
        {"id":..., "type":"arithmetic|pmt|npv|irr", "expr":..., "inputs":..., "expected":..., "units":...},
        ...
      ]
    }
    For arithmetic: requires expr + inputs
    For pmt/npv/irr: requires inputs dict with correct args
    """
    if not isinstance(claims_json, dict):
        return False, "Math Claims JSON is not an object"

    if "claims" not in claims_json:
        return False, "Math Claims JSON must contain top-level key 'claims'"

    if not isinstance(claims_json["claims"], list):
        return False, "'claims' must be a list"

    for i, c in enumerate(claims_json["claims"]):
        if not isinstance(c, dict):
            return False, f"Claim #{i} is not an object"
        missing = REQUIRED_CLAIM_KEYS - set(c.keys())
        if missing:
            return False, f"Claim #{i} missing keys: {sorted(list(missing))}"

        ctype = c.get("type", "")
        if ctype not in ("arithmetic", "pmt", "npv", "irr"):
            return False, f"Claim #{i} has invalid type: {ctype}"

        if ctype == "arithmetic":
            if "expr" not in c or "inputs" not in c:
                return False, f"Arithmetic claim #{i} must include 'expr' and 'inputs'"
            if not isinstance(c["inputs"], dict):
                return False, f"Arithmetic claim #{i} 'inputs' must be an object"
        else:
            if "inputs" not in c:
                return False, f"{ctype} claim #{i} must include 'inputs'"
            if not isinstance(c["inputs"], dict):
                return False, f"{ctype} claim #{i} 'inputs' must be an object"

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
        return False, "No ```json``` block found for Math Claims"
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
# Reviewer rubric JSON prompt
# ----------------------------
RUBRIC_PROMPT = "\n".join([
    "You are a reviewer. Output ONLY a JSON object in this schema:",
    "{",
    '  "overall_score_0_5": 0,',
    '  "critical_must_fix": [],',
    '  "notes": ""',
    "}",
    "Rules:",
    "- overall_score_0_5 is 0..5",
    "- critical_must_fix is a list (empty if none)",
    "- notes 1–3 sentences",
    "- Do not include any other text"
])

def parse_reviewer_json(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "No JSON object found"
    try:
        return json.loads(text[start:end+1]), None
    except Exception as e:
        return None, str(e)

# ----------------------------
# Model calls by provider
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
# Leader / Critics selection UI
# ----------------------------
available_leaders = []
if OPENAI_API_KEY:
    available_leaders.append("OpenAI")
if ANTHROPIC_API_KEY:
    available_leaders.append("Claude")
if XAI_API_KEY:
    available_leaders.append("Grok")

st.markdown("### Leader / Critic configuration")
leader = st.selectbox("Leader (drives the draft)", available_leaders, index=0)

# Leader model selections by provider
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    openai_model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o"], index=0, disabled=(not OPENAI_API_KEY))
with col_m2:
    claude_model = st.selectbox("Claude model", ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-4-20250514"], index=0, disabled=(not ANTHROPIC_API_KEY))
with col_m3:
    grok_model = st.selectbox("Grok model", ["grok-4-fast", "grok-4"], index=0, disabled=(not XAI_API_KEY))

def leader_call(system: str, user_text: str, temperature=0.2):
    if leader == "OpenAI":
        return call_openai(openai_model, system, user_text, temperature=temperature)
    if leader == "Claude":
        return call_claude(claude_model, system, user_text)
    return call_grok(grok_model, system, user_text)

def critic_calls(system: str, user_text: str):
    # critics are the remaining configured providers (max 2)
    outputs = {}
    if leader != "OpenAI" and OPENAI_API_KEY:
        outputs["OpenAI"] = call_openai(openai_model, system, user_text, temperature=0.2)
    if leader != "Claude" and ANTHROPIC_API_KEY:
        outputs["Claude"] = call_claude(claude_model, system, user_text)
    if leader != "Grok" and XAI_API_KEY:
        outputs["Grok"] = call_grok(grok_model, system, user_text)
    return outputs

# ----------------------------
# Assumption Pack (Finance + Technical)
# ----------------------------
DEFAULT_ASSUMPTION_PACK = "\n".join([
    "ASSUMPTION PACK (Finance + Technical) — editable",
    "",
    "Finance defaults (tag all as [Assumed]):",
    "- CAPEX phasing over construction (example): Year1 30%, Year2 40%, Year3 30%",
    "- Interest during construction (IDC): assume capitalized, rate = debt interest rate unless specified",
    "- Debt amortization: if sculpting schedule not provided, assume level payment (PMT) over tenor",
    "- CFADS bridge: CFADS = EBITDA - O&M - taxes - reserves (if missing, assume ranges and clearly label)",
    "- O&M: if missing, assume a range (e.g., 1–4% of CAPEX per year) and show sensitivity",
    "- Taxes: if missing, assume a range (e.g., 0–30% effective) and show sensitivity",
    "",
    "Technical defaults (tag all as [Assumed]):",
    "- Availability: assume 92–98% (range)",
    "- Capacity factor / performance: if needed, assume a range appropriate to the technology and disclose uncertainty",
    "- Schedule risk: assume potential delay range (e.g., 0–12 months) and show effect on cost/time only (no hidden cashflows)",
    "- Key technical failure modes: choose top 3 that plausibly drive cost/schedule for the stated asset type; if asset type is unclear, ask for it",
])

# ----------------------------
# Master prompt templates
# ----------------------------
MASTER_FORMAT = "\n".join([
    "Return in this exact structure:",
    "",
    "## Mode Banner",
    "- STRICT or ASSUMPTION",
    "",
    "## Inputs Used (Verbatim)",
    "",
    "## Assumptions Added by Master",
    "",
    "## Financing Treatment of Shock/Overrun (MANDATORY)",
    "- Case A (Debt capped): scenario",
    "- Case B (Debt upsized): scenario",
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
    "Provide a single JSON block in this schema:",
    "",
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
    "- Use type=arithmetic with expr+inputs+expected+units",
    "- Use type=pmt/npv/irr only when inputs are complete",
    "- expected MUST match what you state in the prose",
    "",
    "## Confidence",
])

MASTER_SYSTEM_STRICT = "\n".join([
    "You are Nemexis Master (STRICT).",
    "Rules:",
    "- Do NOT invent numeric inputs.",
    "- If blocking inputs remain, do NOT claim threshold outcomes as likely. Say cannot determine.",
    "- Any [Computed] number must be backed by a Math Claim and must verify.",
    "- If you cannot provide Math Claims in the required schema, do not label any number [Computed].",
])

MASTER_SYSTEM_ASSUME = "\n".join([
    "You are Nemexis Master (ASSUMPTION MODE).",
    "Rules:",
    "- You MAY fill gaps using the Assumption Pack, but every assumed number must be tagged [Assumed] and listed.",
    "- Prefer ranges + show sensitivity rather than single point values.",
    "- Any [Computed] number must be backed by a Math Claim and must verify.",
    "- Your Math Claims JSON MUST follow the required schema exactly.",
    "- In ASSUMPTION mode, Blocking inputs should be driven to NONE by assumptions (unless truly impossible).",
])

# ----------------------------
# User prompt sanitation warning (optional)
# ----------------------------
def looks_polluted(text: str) -> bool:
    # If many unrelated short lines without bullets, warn
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) > 40:
        return True
    junk_markers = ["Personal account", "Miguel Sanchez", "Hustle Business", "litigation", "News research request"]
    return any(m.lower() in text.lower() for m in junk_markers)

# ----------------------------
# UI Inputs
# ----------------------------
st.markdown("### Problem")
user_prompt = st.text_area("Prompt", height=220, placeholder="Paste the question / task here (clean).")
user_context = st.text_area("Context (optional)", height=140, placeholder="Paste data, excerpts, constraints…")

st.markdown("### Two-mode workflow")
col_a, col_b = st.columns(2)
with col_a:
    run_strict = st.button("Run STRICT")
with col_b:
    run_assume = st.button("Run ASSUMPTION")

st.markdown("### Assumption Pack (used only in ASSUMPTION mode)")
assumption_pack = st.text_area("Assumption Pack", value=DEFAULT_ASSUMPTION_PACK, height=260)

max_iters = st.slider("ASSUMPTION iterations", 1, 6, 3)

# ----------------------------
# Review rubrics
# ----------------------------
def reviewer_rubrics(draft: str):
    critics = critic_calls("You are a strict reviewer.", draft + "\n\n" + RUBRIC_PROMPT)
    parsed = {}
    for k, raw in critics.items():
        obj, err = parse_reviewer_json(raw)
        if obj is None:
            obj = {"overall_score_0_5": 0, "critical_must_fix": ["Could not parse JSON"], "notes": err or ""}
        parsed[k] = obj
    return parsed

# ----------------------------
# Run functions
# ----------------------------
def build_user_text():
    txt = user_prompt.strip()
    if user_context.strip():
        txt += "\n\n---\nContext:\n" + user_context.strip()
    return txt

def strict_run():
    txt = build_user_text()
    if looks_polluted(txt):
        st.warning("Your prompt looks polluted with unrelated text. Clean it for best results.")

    master_input = "\n\n".join([
        "MODE: STRICT",
        "USER PROMPT:",
        txt,
        "OUTPUT FORMAT:",
        MASTER_FORMAT
    ])

    draft = leader_call(MASTER_SYSTEM_STRICT, master_input, temperature=0.15)
    return draft

def assume_run(prev_feedback: str):
    txt = build_user_text()
    if looks_polluted(txt):
        st.warning("Your prompt looks polluted with unrelated text. Clean it for best results.")

    master_input = "\n\n".join([
        "MODE: ASSUMPTION",
        "USER PROMPT:",
        txt,
        "ASSUMPTION PACK:",
        assumption_pack.strip(),
        prev_feedback.strip() if prev_feedback else "",
        "OUTPUT FORMAT:",
        MASTER_FORMAT
    ])

    draft = leader_call(MASTER_SYSTEM_ASSUME, master_input, temperature=0.25)
    return draft

# ----------------------------
# Execute STRICT
# ----------------------------
if run_strict:
    if not user_prompt.strip():
        st.error("Please paste a prompt.")
        st.stop()

    st.divider()
    st.markdown("## STRICT Result")

    draft = strict_run()
    st.write(draft)

    ok_math, math_info = render_math_table(draft, "STRICT")
    blockers = parse_blocking_items(draft)
    st.markdown("### Blocking inputs")
    st.write(blockers if blockers else ["None"])

    rubrics = reviewer_rubrics(draft)
    st.markdown("### Critic rubrics")
    st.json(rubrics)

    if blockers:
        st.info("STRICT found blockers. Either provide missing inputs (in Context) and re-run STRICT, or run ASSUMPTION mode.")

# ----------------------------
# Execute ASSUMPTION (iterative)
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
                f"- Critic must-fix items: {prev['must_fix']}",
                "",
                "Rules:",
                "- Fix Math Claims JSON schema if wrong.",
                "- Ensure all computed numbers are backed by Math Claims.",
                "- In ASSUMPTION mode, Blocking should be NONE (fill gaps with explicit assumptions).",
            ])

        draft = assume_run(feedback)
        st.write(draft)

        ok_math, _ = render_math_table(draft, f"ASSUMPTION iter {it}")
        blockers = parse_blocking_items(draft)

        rubrics = reviewer_rubrics(draft)
        st.markdown("#### Critic rubrics")
        st.json(rubrics)

        must_fix = []
        for r in rubrics.values():
            must_fix.extend(r.get("critical_must_fix", []))

        history.append({
            "iter": it,
            "draft": draft,
            "math_ok": ok_math,
            "blockers": blockers,
            "rubrics": rubrics,
            "must_fix": must_fix
        })

        blockers_empty = (len(blockers) == 0)
        no_critical = (len(must_fix) == 0)

        if ok_math and blockers_empty and no_critical:
            st.success("✅ Converged in ASSUMPTION mode.")
            final = draft
            break

        final = draft

    st.divider()
    st.markdown("## Final ASSUMPTION Draft")
    st.write(final)

    export_text = f"""# Nemexis Export
Generated: {datetime.datetime.now()}
Leader: {leader}
Mode: ASSUMPTION (iterative)

## USER PROMPT
{user_prompt.strip()}

## CONTEXT
{user_context.strip() if user_context.strip() else "(none)"}

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
        export_text += f"- Must-fix: {h['must_fix']}\n"
        export_text += "\n---\n"
        export_text += h["draft"]

    st.divider()
    st.markdown("## Export")
    copy_to_clipboard_button(export_text, "📋 Copy Everything")
    st.download_button("⬇️ Download Markdown", export_text, file_name="nemexis_output.md", mime="text/markdown")
    st.text_area("All output (Cmd/Ctrl+A then Copy)", value=export_text, height=320)
