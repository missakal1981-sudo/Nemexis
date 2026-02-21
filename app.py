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

st.set_page_config(page_title="Nemexis", layout="wide")
st.title("Nemexis — Universal Reliability Engine (v6 Safe + Math Verifier)")

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

def render_math_verification(label: str, text: str):
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

    mismatches = [r for r in results if r["error"] is None and r["expected"] is not None and r["ok"] is False]
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

ORBITS = [
    {"name": "Technical Engineering", "judge_mandate":
        "You are a senior engineering reviewer. Critique technical correctness/realism, failure modes, unit sanity checks, schedule realism."
    },
    {"name": "Economics / Project Finance", "judge_mandate":
        "You are a project finance reviewer. Critique cash-flow logic, timing, covenants/DSCR, sensitivities, and math sanity checks. Do NOT invent numbers."
    },
    {"name": "Contract / Legal (Commercial)", "judge_mandate":
        "You are a contracts/commercial reviewer. Critique entitlement/pass-through, LDs/caps, change orders, termination triggers, compliance/approvals."
    },
    {"name": "Risk / Governance", "judge_mandate":
        "You are a risk officer. Critique ambiguity, overconfidence, missing assumptions, validation gaps. Provide a mini risk register."
    },
]

# Guardrails (strings built safely)
INPUT_LOCK_RULES = "\n".join([
    "INPUT LOCK:",
    "- Include 'Inputs Used (Verbatim)' listing user inputs exactly.",
    "- Include 'Assumptions Added by Master' for anything not provided.",
    "- Do NOT silently change user inputs.",
])

CLAIMS_AUDIT_RULES = "\n".join([
    "CLAIMS AUDIT:",
    "- Include 'Claims Audit'.",
    "- Every numeric claim tagged: [Known] [Computed] [Assumed] [Unknown]",
    "- [Computed] only if you show enough calculation path to replicate.",
    "- Do NOT mark IRR/DSCR as [Computed] unless you compute from explicit cashflows/schedules.",
])

CFADS_GUARDRAIL = "\n".join([
    "CFADS / DSCR GUARDRAIL (STRICT):",
    "- DSCR is based on CFADS, not EBITDA.",
    "- If only EBITDA is provided, DSCR headroom and debt capacity are [Unknown] unless CFADS bridge provided.",
    "- You may NOT compute debt service or max debt capacity from EBITDA/DSCR in Strict mode.",
])

CAPITAL_STRUCTURE_GUARDRAIL = "\n".join([
    "CAPITAL STRUCTURE GUARDRAIL (STRICT):",
    "- In Strict mode, you may NOT assert post-shock debt/equity funding as fact unless provided.",
    "- You MUST present 'Financing Treatment of Shock/Overrun' with Case A and Case B.",
])

STRICT_RULES = "\n".join([
    "STRICT MODE:",
    "- No new numeric assumptions (tax rate, O&M %, capacity factor, probabilities, escalation, etc.).",
    "- No derived numeric outputs unless underlying inputs are explicitly provided.",
    "- If Blocking inputs exist, Confidence must be LOW.",
    "- Avoid words like 'likely' for threshold crossings when key blocking inputs are missing.",
])

BOUNDING_RULES = "\n".join([
    "BOUNDING MODE:",
    "- You may add numeric assumptions only as ranges + scenario table.",
    "- All added numeric assumptions tagged [Assumed] with brief basis.",
])

MASTER_SYSTEM = "You are Nemexis Master. Produce a decision-grade draft. Follow guardrails strictly."

MASTER_FORMAT_LINES = [
    "Return in this exact structure:",
    "",
    "## Inputs Used (Verbatim)",
    "",
    "## Assumptions Added by Master",
    "",
    "## Financing Treatment of Shock/Overrun (MANDATORY)",
    "",
    "## Executive Answer",
    "",
    "## Calculations / Logic",
    "- Only compute what is computable from given inputs",
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
    "- Every numeric claim tagged [Known] [Computed] [Assumed] [Unknown]",
    "",
    "## Math Claims (JSON)",
    "Provide a single JSON block enclosed with triple backticks. Example:",
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
    "Rules for Math Claims:",
    "- Include ONLY claims you can compute from provided inputs.",
    "- Do NOT include DSCR/CFADS/IRR unless you have explicit cashflows/schedules.",
    "- expected must match what you state in prose.",
    "",
    "## Confidence",
]
MASTER_FORMAT = "\n".join(MASTER_FORMAT_LINES)

JUDGE_FORMAT = "\n".join([
    "Return critique:",
    "## Summary Verdict",
    "## Critical Issues (must-fix)",
    "## Corrections / Fixes",
    "## Missing Inputs (required)",
    "## Questions for the Team",
    "## Guardrail Violations Detected",
    "## Confidence in Master Draft",
])

GUARDRAIL_CHECK_SYSTEM = "\n".join([
    "You are Nemexis Guardrail Checker.",
    "Detect violations of:",
    "- Strict/Bounding rules",
    "- Input Lock",
    "- CFADS/DSCR misuse",
    "- Funding mix asserted as fact in STRICT",
    "- Missing Inputs not split Blocking/Non-blocking",
    "- Confidence not LOW when Blocking inputs exist",
    "- Math Claims JSON missing or inconsistent with prose",
    "Output exactly:",
    "## Violations Found",
    "- bullet list (or 'None')",
    "",
    "## Required Corrections",
    "- bullet list (or 'None')",
    "",
    "## Should we regenerate?",
    "- Yes/No",
])

MASTER_FIX_SYSTEM = "\n".join([
    "You are Nemexis Master. Rewrite the draft to remove violations.",
    "Rules:",
    "- Do not add numeric assumptions in STRICT.",
    "- Do not compute DSCR/CFADS from EBITDA.",
    "- Do not assert funding mix as fact in STRICT; present Case A/Case B.",
    "- Ensure Blocking vs Non-blocking missing inputs.",
    "- Ensure Math Claims JSON exists and matches prose.",
    "Return corrected draft only.",
])

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
        max_tokens=1800,
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

if run:
    if not prompt.strip():
        st.error("Please enter a User Prompt.")
        st.stop()

    user_text = prompt.strip()
    if context.strip():
        user_text += "\n\n---\nContext:\n" + context.strip()

    mode_rules = STRICT_RULES if STRICT_MODE else BOUNDING_RULES

    st.divider()
    st.markdown("## Step 1 — Master Draft (Rev 0)")

    rev0_input = "\n\n".join([
        f"MODE: {'STRICT' if STRICT_MODE else 'BOUNDING'}",
        "USER PROMPT:",
        user_text,
        INPUT_LOCK_RULES,
        CLAIMS_AUDIT_RULES,
        CFADS_GUARDRAIL if STRICT_MODE else "",
        CAPITAL_STRUCTURE_GUARDRAIL if STRICT_MODE else "",
        mode_rules,
        "OUTPUT FORMAT:",
        MASTER_FORMAT
    ])

    with st.spinner("Master generating Rev 0..."):
        rev0 = call_openai(master_model, MASTER_SYSTEM, rev0_input, temperature=0.15)

    st.markdown("### Rev 0 (Master)")
    st.write(rev0)
    math0 = render_math_verification("Rev 0", rev0)

    st.divider()
    st.markdown("## Step 2 — Orbit Critiques (Claude + Grok)")

    critiques = []
    for orbit in ORBITS:
        orbit_name = orbit["name"]
        mandate = orbit["judge_mandate"]

        judge_input = "\n\n".join([
            f"MODE: {'STRICT' if STRICT_MODE else 'BOUNDING'}",
            f"ORBIT: {orbit_name}",
            "MASTER DRAFT (REV 0):",
            rev0,
            "CRITIQUE FORMAT:",
            JUDGE_FORMAT,
            "Also check: math structure (formulas/definitions), claims audit integrity, missing inputs classification."
        ])

        colA, colB = st.columns(2)
        with colA:
            st.markdown(f"#### Claude ({claude_model})")
            with st.spinner("Claude critiquing..."):
                claude_out = call_claude(claude_model, mandate, judge_input)
            st.write(claude_out)

        with colB:
            st.markdown(f"#### Grok ({grok_model})")
            with st.spinner("Grok critiquing..."):
                grok_out = call_grok(grok_model, mandate, judge_input)
            st.write(grok_out)

        critiques.append({"orbit": orbit_name, "claude": claude_out, "grok": grok_out})

    st.divider()
    st.markdown("## Step 3 — Master Integration (Rev 1)")

    critique_blob = ""
    for c in critiques:
        critique_blob += f"\n\n=== ORBIT: {c['orbit']} ===\n\n--- Claude ---\n{c['claude']}\n\n--- Grok ---\n{c['grok']}\n"

    rev1_input = "\n\n".join([
        f"MODE: {'STRICT' if STRICT_MODE else 'BOUNDING'}",
        "USER PROMPT:",
        user_text,
        INPUT_LOCK_RULES,
        CLAIMS_AUDIT_RULES,
        CFADS_GUARDRAIL if STRICT_MODE else "",
        CAPITAL_STRUCTURE_GUARDRAIL if STRICT_MODE else "",
        mode_rules,
        "REV 0:",
        rev0,
        "CRITIQUES:",
        critique_blob,
        "OUTPUT FORMAT:",
        MASTER_FORMAT
    ])

    with st.spinner("Master generating Rev 1..."):
        rev1 = call_openai(master_model, MASTER_SYSTEM, rev1_input, temperature=0.15)

    st.markdown("### Rev 1 (Master)")
    st.write(rev1)
    math1 = render_math_verification("Rev 1", rev1)

    st.divider()
    st.markdown("## Step 4 — Guardrail Check + Auto-fix (Rev 1b)")

    check_input = "\n\n".join([
        f"MODE: {'STRICT' if STRICT_MODE else 'BOUNDING'}",
        "REV 1:",
        rev1,
        "RULES:",
        mode_rules,
        "Check Math Claims JSON presence and consistency with prose."
    ])

    with st.spinner("Guardrail checker reviewing Rev 1..."):
        guardrail_report = call_openai("gpt-4o-mini", GUARDRAIL_CHECK_SYSTEM, check_input, temperature=0.0)

    st.markdown("### Guardrail Report")
    st.write(guardrail_report)

    rev1b = rev1
    if "Should we regenerate?\n- Yes" in guardrail_report:
        fix_input = "\n\n".join([
            f"MODE: {'STRICT' if STRICT_MODE else 'BOUNDING'}",
            "VIOLATIONS REPORT:",
            guardrail_report,
            "CURRENT REV 1:",
            rev1,
            "USER PROMPT:",
            user_text
        ])
        with st.spinner("Auto-fixing → generating Rev 1b..."):
            rev1b = call_openai(master_model, MASTER_FIX_SYSTEM, fix_input, temperature=0.1)

    st.markdown("### Rev 1b (Corrected)")
    st.write(rev1b)
    math1b = render_math_verification("Rev 1b", rev1b)

    export_text = f"""# Nemexis Export
Generated: {datetime.datetime.now()}
MODE: {'STRICT' if STRICT_MODE else 'BOUNDING'}

## USER PROMPT
{prompt.strip()}

## CONTEXT
{context.strip() if context.strip() else "(none)"}

---

# REV 0
{rev0}

---

# REV 1
{rev1}

---

# GUARDRAIL REPORT
{guardrail_report}

---

# REV 1b
{rev1b}

---

# CRITIQUES
"""
    for c in critiques:
        export_text += f"""

## ORBIT: {c['orbit']}

### Claude
{c['claude']}

### Grok
{c['grok']}
"""

    st.divider()
    st.markdown("## Export")
    copy_to_clipboard_button(export_text, "📋 Copy Everything")
    st.download_button("⬇️ Download as Markdown (.md)", export_text, file_name="nemexis_output.md")
    st.text_area("All output (Cmd/Ctrl+A then Copy)", value=export_text, height=320)
