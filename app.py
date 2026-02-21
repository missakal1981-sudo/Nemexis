import os
import time
import streamlit as st
from openai import OpenAI
import anthropic
import requests
import streamlit.components.v1 as components

st.set_page_config(page_title="Nemexis", layout="wide")
st.title("Nemexis — Reliability Engine (Iterative + Guardrails v3)")

# ----------------------------
# Password Gate
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
# Copy-to-clipboard helper
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
# UI
# ----------------------------
st.markdown("### Inputs")
prompt = st.text_area("User Prompt", height=160, placeholder="What are you trying to solve / decide?")
context = st.text_area("Context (optional)", height=160, placeholder="Paste numbers, excerpts, constraints, assumptions...")

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
run = st.button("Run Nemexis (Rev 0 → Critiques → Rev 1 → Auto-fix)")

# ----------------------------
# Orbits
# ----------------------------
ORBITS = [
    {"name": "Technical Engineering", "judge_mandate":
        "You are a senior offshore/industrial engineering reviewer. "
        "Critique technical correctness/realism, failure modes, unit sanity checks, schedule realism, "
        "and technical-financial coupling mistakes. Do NOT rewrite; provide targeted fixes."
    },
    {"name": "Economics / Project Finance", "judge_mandate":
        "You are a project finance reviewer. Critique cash-flow logic, timing, covenants/DSCR, sensitivities, "
        "and math sanity checks. Do NOT invent numbers. Do NOT rewrite; provide targeted fixes."
    },
    {"name": "Contract / Legal (Commercial)", "judge_mandate":
        "You are a contracts/commercial reviewer (delivery + procurement). Critique entitlement/pass-through, "
        "LDs/caps, change orders, termination triggers, compliance/approvals, and what clauses must be checked. "
        "Do NOT rewrite; provide targeted fixes."
    },
    {"name": "Risk / Governance", "judge_mandate":
        "You are a risk officer. Critique ambiguity, overconfidence, missing assumptions, validation gaps. "
        "Produce mini risk register + what changes the conclusion. Do NOT rewrite; provide targeted fixes."
    },
]

# ----------------------------
# Guardrail Rules
# ----------------------------
INPUT_LOCK_RULES = """
INPUT LOCK:
- Include 'Inputs Used (Verbatim)' listing user inputs exactly.
- Include 'Assumptions Added by Master' for anything not provided.
- Do NOT silently change user inputs.
""".strip()

CLAIMS_AUDIT_RULES = """
CLAIMS AUDIT:
- Include a 'Claims Audit' section.
- Every numeric claim tagged: [Computed] [Estimated] [Assumed] [Unknown]
- [Computed] only if you show enough calculation path to replicate.
- Do NOT mark IRR as [Computed] unless you actually compute it from the cashflow logic you present.
""".strip()

PPA_CONSISTENCY_RULE = """
PPA CONSISTENCY:
- If PPA is fixed price, do NOT list 'PPA price volatility' as a risk unless you identify repricing/merchant exposure.
""".strip()

STRICT_RULES = """
STRICT MODE (very important):
- You may NOT introduce new numeric assumptions (tax rate, O&M %, capacity factor, probabilities, escalation, etc.).
- You may NOT introduce derived numeric outputs (e.g., Debt Service = EBITDA/DSCR) unless the underlying quantity is explicitly provided (CFADS) and method defined.
- Treat DSCR as a covenant/constraint, NOT as a calculator for debt service.
- If required numbers are missing, write [Unknown] and list them under Missing Inputs.
- If you cannot compute the key decision metric (IRR below/above 8%), Confidence must be LOW.
""".strip()

BOUNDING_RULES = """
BOUNDING MODE:
- You may add numeric assumptions only as ranges + scenario table (Low/Base/High).
- All added numeric assumptions tagged [Assumed] with brief basis.
- Separate Scenario Outputs from Facts.
""".strip()

NO_INVENTION_STRUCTURES = """
NO INVENTED STRUCTURES:
- Do not invent interest-only debt, ignore construction period, or fabricate sculpted debt service schedule.
- If a schedule is needed, say so and list required inputs.
""".strip()

# ----------------------------
# Output formats
# ----------------------------
MASTER_REV0_SYSTEM = "You are Nemexis Master (OpenAI). Produce Rev 0 that is decision-grade. Follow guardrails strictly."
MASTER_FORMAT = """
Return in this exact structure:

## Inputs Used (Verbatim)

## Assumptions Added by Master
(Strict mode: usually 'None' — instead list Unknowns)

## Executive Answer
(5 bullets max)

## Calculations / Logic
- Respect construction period
- Respect DSCR sculpting (do not fabricate schedule)
- If key metric cannot be computed, say so

## Key Risks (ranked)

## Contract / Legal Checks

## Missing Inputs (required)

## Claims Audit
(tag each numeric claim)

## Confidence
""".strip()

JUDGE_FORMAT = """
Return critique:

## Summary Verdict
## Critical Issues (must-fix)
## Corrections / Fixes
## Missing Inputs (required)
## Questions for the Team
## Guardrail Violations Detected
## Confidence in Master Draft
""".strip()

MASTER_REV1_SYSTEM = "You are Nemexis Master (OpenAI) creating Rev 1. Integrate critiques; accept/reject with reasons; follow guardrails."
MASTER_REV1_FORMAT = """
Return:

# Changelog (Rev 0 → Rev 1)
- Accepted critiques
- Rejected critiques (with reason)
- Key edits applied

# Rev 1 (Updated Deliverable)
(use same structure as Rev 0)
""".strip()

GUARDRAIL_CHECK_SYSTEM = """
You are Nemexis Guardrail Checker.
Your job: read Rev 1 and detect violations of:
- Strict mode rules (if strict)
- Input Lock
- Claims Audit (Computed without path)
- DSCR misuse (e.g., using EBITDA/DSCR as debt service)
- PPA consistency

Output exactly:

## Violations Found
- bullet list (or 'None')

## Required Corrections
- bullet list (or 'None')

## Should we regenerate Rev 1?
- Yes/No
""".strip()

MASTER_FIX_SYSTEM = """
You are Nemexis Master. You must correct Rev 1 to eliminate the listed violations.
Rules:
- Do not introduce new numeric assumptions in Strict mode.
- Remove derived numbers not supported by inputs.
- Downgrade confidence to Low if key metric cannot be computed.
Return corrected Rev 1 only (no extra commentary).
""".strip()

# ----------------------------
# Model call helpers
# ----------------------------
def call_openai(model_name: str, system: str, user_text: str) -> str:
    resp = openai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_text}],
        temperature=0.2,
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
                timeout=120,  # increased
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            last_err = f"❌ Grok Error ({r.status_code}): {r.text}"
        except Exception as e:
            last_err = f"❌ Grok Error: {str(e)}"
        time.sleep(1.5 * (attempt + 1))
    return last_err or "❌ Grok Error: unknown"

# ----------------------------
# Run
# ----------------------------
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

    with st.spinner("Master generating Rev 0..."):
        rev0_input = (
            f"MODE: {'STRICT' if STRICT_MODE else 'BOUNDING'}\n\n"
            f"USER PROMPT:\n{user_text}\n\n"
            f"{INPUT_LOCK_RULES}\n\n{CLAIMS_AUDIT_RULES}\n\n{PPA_CONSISTENCY_RULE}\n\n"
            f"{mode_rules}\n\n{NO_INVENTION_STRUCTURES}\n\n"
            f"OUTPUT FORMAT:\n{MASTER_FORMAT}"
        )
        rev0 = call_openai(master_model, MASTER_REV0_SYSTEM, rev0_input)

    st.markdown("### Rev 0 (Master)")
    st.write(rev0)

    st.divider()
    st.markdown("## Step 2 — Orbit Critiques (Claude + Grok)")

    critiques = []
    for orbit in ORBITS:
        orbit_name = orbit["name"]
        mandate = orbit["judge_mandate"]
        st.markdown(f"### Orbit: {orbit_name}")

        judge_input = (
            f"MODE: {'STRICT' if STRICT_MODE else 'BOUNDING'}\n\n"
            f"ORBIT: {orbit_name}\n\n"
            f"MASTER DRAFT (REV 0):\n{rev0}\n\n"
            f"CRITIQUE FORMAT:\n{JUDGE_FORMAT}\n\n"
            f"Explicitly check for guardrail violations (Strict rules, DSCR misuse, Claims Audit issues)."
        )

        colA, colB = st.columns(2)
        with colA:
            st.markdown(f"#### Claude ({claude_model})")
            with st.spinner("Claude critiquing..."):
                try:
                    claude_out = call_claude(claude_model, mandate, judge_input)
                except Exception as e:
                    claude_out = f"❌ Claude Error: {str(e)}"
            st.write(claude_out)

        with colB:
            st.markdown(f"#### Grok ({grok_model})")
            with st.spinner("Grok critiquing..."):
                grok_out = call_grok(grok_model, mandate, judge_input)
            st.write(grok_out)

        critiques.append({"orbit": orbit_name, "claude": claude_out, "grok": grok_out})

    st.divider()
    st.markdown("## Step 3 — Master Integration (Rev 1)")

    with st.spinner("Master integrating critiques into Rev 1..."):
        critique_blob = ""
        for c in critiques:
            critique_blob += f"\n\n=== ORBIT: {c['orbit']} ===\n\n--- Claude ---\n{c['claude']}\n\n--- Grok ---\n{c['grok']}\n"

        rev1_input = (
            f"MODE: {'STRICT' if STRICT_MODE else 'BOUNDING'}\n\n"
            f"USER PROMPT:\n{user_text}\n\n"
            f"{INPUT_LOCK_RULES}\n\n{CLAIMS_AUDIT_RULES}\n\n{PPA_CONSISTENCY_RULE}\n\n"
            f"{mode_rules}\n\n{NO_INVENTION_STRUCTURES}\n\n"
            f"REV 0:\n{rev0}\n\nCRITIQUES:\n{critique_blob}\n\n"
            f"OUTPUT FORMAT:\n{MASTER_REV1_FORMAT}"
        )
        rev1 = call_openai(master_model, MASTER_REV1_SYSTEM, rev1_input)

    st.markdown("### Rev 1 (Master)")
    st.write(rev1)

    # ----------------------------
    # Step 4 — Automatic Guardrail Check + Fix pass
    # ----------------------------
    st.divider()
    st.markdown("## Step 4 — Guardrail Check (Auto-fix if needed)")

    with st.spinner("Checking Rev 1 for guardrail violations..."):
        check_input = (
            f"MODE: {'STRICT' if STRICT_MODE else 'BOUNDING'}\n\n"
            f"STRICT RULES:\n{STRICT_RULES if STRICT_MODE else '(not strict)'}\n\n"
            f"REV 1:\n{rev1}"
        )
        guardrail_report = call_openai("gpt-4o-mini", GUARDRAIL_CHECK_SYSTEM, check_input)

    st.markdown("### Guardrail Report")
    st.write(guardrail_report)

    rev1_fixed = rev1
    if "## Should we regenerate Rev 1?\n- Yes" in guardrail_report or "Should we regenerate Rev 1?\n- Yes" in guardrail_report:
        with st.spinner("Auto-fixing Rev 1 (producing Rev 1b)..."):
            fix_input = (
                f"MODE: {'STRICT' if STRICT_MODE else 'BOUNDING'}\n\n"
                f"VIOLATIONS REPORT:\n{guardrail_report}\n\n"
                f"CURRENT REV 1:\n{rev1}\n\n"
                f"USER PROMPT:\n{user_text}\n\n"
                f"Remember: in STRICT mode do not add numeric assumptions, do not misuse DSCR, "
                f"and do not label Computed without a replicable path."
            )
            rev1_fixed = call_openai(master_model, MASTER_FIX_SYSTEM, fix_input)

        st.markdown("### Rev 1b (Corrected by Auto-fix)")
        st.write(rev1_fixed)

    # ----------------------------
    # EXPORT
    # ----------------------------
    export_text = f"""# Nemexis Export

## MODE
{'STRICT' if STRICT_MODE else 'BOUNDING'}

## USER PROMPT
{prompt.strip()}

## CONTEXT
{context.strip() if context.strip() else "(none)"}

---

# REV 0 (Master)
{rev0}

---

# ORBIT CRITIQUES (Claude + Grok)
"""
    for c in critiques:
        export_text += f"""

## ORBIT: {c['orbit']}

### Claude
{c['claude']}

### Grok
{c['grok']}
"""
    export_text += f"""

---

# REV 1 (Master)
{rev1}

---

# GUARDRAIL REPORT
{guardrail_report}

---

# REV 1b (Corrected)
{rev1_fixed}
"""

    st.divider()
    st.markdown("## Export")
    copy_to_clipboard_button(export_text, "📋 Copy All (Rev0+Critiques+Rev1+Rev1b)")
    st.download_button(
        label="⬇️ Download as Markdown (.md)",
        data=export_text,
        file_name="nemexis_output.md",
        mime="text/markdown",
    )
    st.text_area("All output (easy Cmd/Ctrl+A then Copy)", value=export_text, height=320)
