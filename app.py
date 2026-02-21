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

# ============================
# Nemexis v7 — Accuracy-First Convergence
# Goal (Mode A): Iterate until 0 blocking inputs remain AND all gates pass.
# Gates:
#  1) Deterministic math claims verified
#  2) Guardrail checker: no violations
#  3) Blocking inputs: NONE
#  4) Claude & Grok: no critical must-fix items
#
# If cannot converge within max_iters, stop and report remaining blockers.
# ============================

st.set_page_config(page_title="Nemexis v7", layout="wide")
st.title("Nemexis v7 — Accuracy-First Reliability Engine (0 Blocking Inputs)")

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

def math_gate_ok(text: str):
    block = extract_json_block(text)
    if not block:
        return False, "No Math Claims JSON block"
    try:
        claims = json.loads(block)
    except Exception as e:
        return False, f"Math Claims JSON parse error: {e}"
    results = verify_math_claims(claims)
    # Any error or mismatch fails
    for r in results:
        if r["error"] is not None:
            return False, f"Math verifier error in claim {r['id']}: {r['error']}"
        if r["expected"] is not None and r["ok"] is False:
            return False, f"Math mismatch in claim {r['id']}"
    return True, results

# ----------------------------
# Parse Blocking Inputs from Master output
# ----------------------------
def parse_blocking_items(text: str):
    """
    Extract bullet lines under '## Missing Inputs' -> '### Blocking'
    Returns list of strings.
    """
    # Find section
    m = re.search(r"## Missing Inputs.*?### Blocking(.*?)(### Non-blocking|## Claims Audit|## Confidence|$)", text, flags=re.S | re.I)
    if not m:
        return ["(Missing Inputs/Blocking section not found)"]

    block = m.group(1).strip()
    if not block:
        return []

    # If explicitly says None
    if re.search(r"\bnone\b", block, flags=re.I):
        # still could have bullets; treat as none only if no bullets
        bullets = re.findall(r"^\s*-\s+(.*)$", block, flags=re.M)
        if len(bullets) == 0:
            return []

    bullets = re.findall(r"^\s*-\s+(.*)$", block, flags=re.M)
    # Also allow numbered items
    nums = re.findall(r"^\s*\d+\.\s+(.*)$", block, flags=re.M)
    items = [b.strip() for b in bullets + nums if b.strip()]
    return items if items else ["(No bullet items found under Blocking)"]

# ----------------------------
# Reviewer rubric (Claude/Grok) in JSON
# ----------------------------
RUBRIC_PROMPT = "\n".join([
    "You are a reviewer. Critique the draft and output ONLY a JSON object.",
    "Return exactly this JSON schema:",
    "{",
    '  "overall_score_0_5": 0,',
    '  "critical_must_fix": [],',
    '  "blocking_inputs_remaining": [],',
    '  "notes": ""',
    "}",
    "",
    "Rules:",
    "- overall_score_0_5 must be a number from 0 to 5.",
    "- critical_must_fix: list of short bullets. Empty list means no must-fix.",
    "- blocking_inputs_remaining: list of inputs that must be provided to reach a fully defensible answer.",
    "- notes: 1–3 sentences max.",
    "- Do not include any other text.",
])

def parse_reviewer_json(text: str):
    # Find first { ... } block heuristically
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "No JSON object found"
    try:
        obj = json.loads(text[start:end+1])
        return obj, None
    except Exception as e:
        return None, str(e)

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
# Universal Strict instructions to Master
# ----------------------------
MASTER_SYSTEM = "\n".join([
    "You are Nemexis Master (Accuracy-first).",
    "You must produce an answer that can be accepted by an Investment Committee / technical board.",
    "You must NOT invent missing numeric inputs.",
    "Any number labeled [Computed] MUST be declared in Math Claims JSON and must verify deterministically.",
    "If Blocking inputs remain, you must not claim threshold outcomes as 'likely'. You may say 'cannot determine'.",
])

MASTER_FORMAT = "\n".join([
    "Return in this exact structure:",
    "",
    "## Inputs Used (Verbatim)",
    "",
    "## Assumptions Added by Master",
    "",
    "## Financing Treatment of Shock/Overrun (MANDATORY)",
    "- Case A: Debt capped; overrun equity-funded.",
    "- Case B: Debt upsized requires lender consent and CFADS/DSCR headroom.",
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
    "Provide a single JSON block with triple backticks like:",
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
    "- Include ONLY claims computable from provided inputs.",
    "- Do NOT include DSCR/CFADS/IRR unless explicit cashflows/schedules are provided.",
    "",
    "## Confidence",
    "- Must be LOW if Blocking inputs exist.",
])

# ----------------------------
# UI
# ----------------------------
st.markdown("### Inputs")
prompt = st.text_area("User Prompt", height=180)
context = st.text_area("Context (optional)", height=150)

st.markdown("### Settings")
master_model = st.selectbox("Master model (OpenAI)", ["gpt-4o-mini", "gpt-4o"], index=0)
claude_model = st.selectbox(
    "Claude critic model",
    ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-4-20250514"],
    index=0,
)
grok_model = st.selectbox("Grok critic model", ["grok-4-fast", "grok-4"], index=0)

max_iters = st.slider("Max iterations", min_value=2, max_value=10, value=5)
run = st.button("Run Nemexis (iterate to 0 blocking inputs)")

# ----------------------------
# Main loop
# ----------------------------
if run:
    if not prompt.strip():
        st.error("Please enter a User Prompt.")
        st.stop()

    user_text = prompt.strip()
    if context.strip():
        user_text += "\n\n---\nContext:\n" + context.strip()

    history = []
    converged = False
    remaining_blockers = None
    final_draft = None
    final_reviews = None
    final_math = None

    st.divider()
    st.markdown("## Iterations")

    for it in range(1, max_iters + 1):
        st.markdown(f"### Iteration {it}")

        # Draft prompt includes last critiques if present
        last_feedback = ""
        if history:
            last = history[-1]
            last_feedback = "\n\n".join([
                "PREVIOUS ITERATION FEEDBACK (must address):",
                f"- Blocking inputs remaining: {last['blocking']}",
                f"- Claude critical must-fix: {last['claude']['critical_must_fix']}",
                f"- Grok critical must-fix: {last['grok']['critical_must_fix']}",
                "Fix any math-claim inconsistencies and ensure Math Claims JSON matches prose.",
            ])

        master_input = "\n\n".join([
            "USER PROMPT:",
            user_text,
            last_feedback,
            "OUTPUT FORMAT:",
            MASTER_FORMAT
        ])

        with st.spinner("Master drafting..."):
            draft = call_openai(master_model, MASTER_SYSTEM, master_input, temperature=0.15)

        st.write(draft)

        # Gate 1: parse blocking inputs
        blocking = parse_blocking_items(draft)

        # Gate 2: math claims verified
        math_ok, math_info = math_gate_ok(draft)

        # Reviewers: Claude + Grok rubric JSON
        with st.spinner("Claude scoring..."):
            claude_review_raw = call_claude(claude_model, "You are a strict reviewer.", draft + "\n\n" + RUBRIC_PROMPT)
        claude_obj, claude_err = parse_reviewer_json(claude_review_raw)
        if claude_obj is None:
            claude_obj = {"overall_score_0_5": 0, "critical_must_fix": ["Could not parse Claude JSON"], "blocking_inputs_remaining": [], "notes": claude_err or ""}

        with st.spinner("Grok scoring..."):
            grok_review_raw = call_grok(grok_model, "You are a strict reviewer.", draft + "\n\n" + RUBRIC_PROMPT)
        grok_obj, grok_err = parse_reviewer_json(grok_review_raw)
        if grok_obj is None:
            grok_obj = {"overall_score_0_5": 0, "critical_must_fix": ["Could not parse Grok JSON"], "blocking_inputs_remaining": [], "notes": grok_err or ""}

        st.markdown("**Claude rubric:**")
        st.json(claude_obj)
        st.markdown("**Grok rubric:**")
        st.json(grok_obj)

        # Convergence rule (Mode A)
        # - blocking must be empty or explicitly 'None'
        blocking_is_zero = (len(blocking) == 0) or (len(blocking) == 1 and "not found" not in blocking[0].lower() and "no bullet" not in blocking[0].lower() and blocking[0].strip().lower() == "none")
        # Also treat "(...not found)" as not converged
        if len(blocking) == 1 and blocking[0].startswith("("):
            blocking_is_zero = False

        no_critical = (len(claude_obj.get("critical_must_fix", [])) == 0) and (len(grok_obj.get("critical_must_fix", [])) == 0)

        # Store iteration record
        history.append({
            "iter": it,
            "draft": draft,
            "blocking": blocking,
            "math_ok": math_ok,
            "math_info": math_info,
            "claude": claude_obj,
            "grok": grok_obj,
        })

        # Display gate status
        st.markdown("#### Gate Status")
        st.write({
            "blocking_inputs": blocking,
            "math_verified": math_ok,
            "no_critical_must_fix": no_critical,
        })

        # Decide stop/continue
        if blocking_is_zero and math_ok and no_critical:
            converged = True
            final_draft = draft
            final_reviews = {"claude": claude_obj, "grok": grok_obj}
            final_math = math_info
            remaining_blockers = []
            break
        else:
            final_draft = draft
            final_reviews = {"claude": claude_obj, "grok": grok_obj}
            final_math = math_info
            remaining_blockers = blocking

            # If blocking inputs exist, we cannot “iterate them away” without user data.
            # So we stop early and tell the user what is needed.
            if not blocking_is_zero:
                st.warning("Cannot reach 0 blocking inputs without additional data. Stopping early to avoid meaningless looping.")
                break

    st.divider()
    st.markdown("## Final Result")

    if converged:
        st.success("✅ Converged: 0 blocking inputs + math verified + no critical must-fix.")
    else:
        st.error("❌ Not converged to 0 blocking inputs.")
        st.write("Remaining blocking inputs:", remaining_blockers)

    st.markdown("### Final Draft")
    st.write(final_draft)

    st.markdown("### Final Math Verification")
    if isinstance(final_math, list):
        st.table(final_math)
    else:
        st.write(final_math)

    st.markdown("### Final Reviewer Rubrics")
    st.json(final_reviews)

    # Export
    export_text = f"""# Nemexis Export
Generated: {datetime.datetime.now()}
Mode: Accuracy-first (0 Blocking Inputs)

## USER PROMPT
{prompt.strip()}

## CONTEXT
{context.strip() if context.strip() else "(none)"}

## CONVERGED
{converged}

## REMAINING BLOCKERS
{remaining_blockers}

---

## FINAL DRAFT
{final_draft}

---

## FINAL REVIEWS
{json.dumps(final_reviews, indent=2)}

---

## HISTORY
"""
    for h in history:
        export_text += f"\n\n### Iteration {h['iter']}\n"
        export_text += f"- Blocking: {h['blocking']}\n"
        export_text += f"- Math OK: {h['math_ok']}\n"
        export_text += f"- Claude critical: {h['claude'].get('critical_must_fix', [])}\n"
        export_text += f"- Grok critical: {h['grok'].get('critical_must_fix', [])}\n"
        export_text += "\n---\n"
        export_text += h["draft"]

    st.divider()
    st.markdown("## Export")
    copy_to_clipboard_button(export_text, "📋 Copy Everything")
    st.download_button("⬇️ Download Markdown", export_text, file_name="nemexis_output.md", mime="text/markdown")
    st.text_area("All output (Cmd/Ctrl+A then Copy)", value=export_text, height=320)
