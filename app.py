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
# Nemexis v8 — Two-button workflow
# - STRICT: no numeric assumptions; stops if blockers exist
# - ASSUMPTION: fills gaps with explicit assumptions (tagged), iterates to convergence
# Deterministic math verifier: any [Computed] number must appear as a Math Claim and verify.
# ============================

st.set_page_config(page_title="Nemexis v8", layout="wide")
st.title("Nemexis v8 — Reliability Engine (STRICT + ASSUMPTION)")

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
    for r in results:
        if r["error"] is not None:
            return False, f"Math verifier error in claim {r['id']}: {r['error']}"
        if r["expected"] is not None and r["ok"] is False:
            return False, f"Math mismatch in claim {r['id']}"
    return True, results

def render_math_table(text: str, title: str):
    st.markdown(f"### Math Verification — {title}")
    ok, info = math_gate_ok(text)
    if ok is True and isinstance(info, list):
        st.table(info)
        st.success("All declared math claims verified ✅")
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
    # bullets or numbered
    bullets = re.findall(r"^\s*-\s+(.*)$", block, flags=re.M)
    nums = re.findall(r"^\s*\d+\.\s+(.*)$", block, flags=re.M)
    items = [x.strip() for x in (bullets + nums) if x.strip()]
    # treat "None" as empty if no bullets
    if not items and re.search(r"\bnone\b", block, flags=re.I):
        return []
    return items if items else ["(No bullet items found under Blocking)"]

# ----------------------------
# Reviewer rubric JSON prompt
# ----------------------------
RUBRIC_PROMPT = "\n".join([
    "You are a reviewer. Critique the draft and output ONLY a JSON object.",
    "Return exactly this JSON schema:",
    "{",
    '  "overall_score_0_5": 0,',
    '  "critical_must_fix": [],',
    '  "notes": ""',
    "}",
    "",
    "Rules:",
    "- overall_score_0_5 must be a number from 0 to 5.",
    "- critical_must_fix: list of short bullets. Empty list means no must-fix.",
    "- notes: 1–3 sentences max.",
    "- Do not include any other text.",
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
# Master instructions
# ----------------------------
MASTER_SYSTEM_STRICT = "\n".join([
    "You are Nemexis Master (STRICT).",
    "Rules:",
    "- Do NOT invent missing numeric inputs.",
    "- If blocking inputs remain, do NOT use 'likely' for threshold crossings; say 'cannot determine'.",
    "- Any [Computed] number MUST be declared in Math Claims JSON and must verify deterministically.",
    "- DSCR is based on CFADS, not EBITDA.",
    "- Present Financing Treatment with Case A and Case B as scenarios (not facts).",
])

MASTER_SYSTEM_ASSUMPTION = "\n".join([
    "You are Nemexis Master (ASSUMPTION MODE).",
    "Rules:",
    "- You may fill gaps, but every filled gap must be explicitly listed under 'Assumptions Added by Master' and tagged [Assumed].",
    "- Prefer ranges and scenario table when uncertain.",
    "- Any [Computed] number MUST be declared in Math Claims JSON and must verify deterministically.",
    "- Do not contradict user-provided inputs.",
    "- Separate Facts (Known) vs Assumptions vs Computed.",
])

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
    "- Case A: Debt capped; overrun equity-funded (scenario).",
    "- Case B: Debt upsized requires lender consent + headroom (scenario).",
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
    "Provide a single JSON block with triple backticks.",
    "",
    "Rules for Math Claims:",
    "- Include ONLY claims computable from provided inputs + explicitly stated assumptions.",
    "- If you compute a number, include it here with expected matching the prose.",
    "- Use arithmetic / pmt / npv / irr only when inputs are complete.",
    "",
    "## Confidence",
])

# ----------------------------
# UI
# ----------------------------
st.markdown("### Inputs")
prompt = st.text_area("User Prompt", height=180)
context = st.text_area("Context (optional)", height=150)

st.markdown("### Models")
master_model = st.selectbox("Master (OpenAI)", ["gpt-4o-mini", "gpt-4o"], index=0)
claude_model = st.selectbox("Claude", ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-4-20250514"], index=0)
grok_model = st.selectbox("Grok", ["grok-4-fast", "grok-4"], index=0)

st.markdown("### Controls")
col_run1, col_run2 = st.columns(2)
with col_run1:
    run_strict = st.button("Run STRICT (Audit)")
with col_run2:
    run_assume = st.button("Run ASSUMPTION (Fill gaps)")

max_iters = st.slider("Max iterations (ASSUMPTION mode)", 2, 8, 4)

# Persistent “blocker fill” box
st.markdown("### Provide Missing Inputs (optional)")
st.caption("If STRICT reports blockers, paste answers here and re-run STRICT.")
blocker_input = st.text_area("Blocker Inputs (paste O&M, CFADS bridge, debt schedule, CAPEX phasing, etc.)", height=140)

# ----------------------------
# Run logic
# ----------------------------
def run_engine(mode_name: str):
    if not prompt.strip():
        st.error("Please enter a prompt.")
        return None

    user_text = prompt.strip()
    if context.strip():
        user_text += "\n\n---\nContext:\n" + context.strip()
    if blocker_input.strip():
        user_text += "\n\n---\nUser-provided answers to missing inputs:\n" + blocker_input.strip()

    return user_text

def reviewer_rubrics(draft: str):
    # Claude
    claude_raw = call_claude(claude_model, "You are a strict reviewer.", draft + "\n\n" + RUBRIC_PROMPT)
    claude_obj, err = parse_reviewer_json(claude_raw)
    if claude_obj is None:
        claude_obj = {"overall_score_0_5": 0, "critical_must_fix": ["Could not parse Claude JSON"], "notes": err or ""}

    # Grok
    grok_raw = call_grok(grok_model, "You are a strict reviewer.", draft + "\n\n" + RUBRIC_PROMPT)
    grok_obj, err2 = parse_reviewer_json(grok_raw)
    if grok_obj is None:
        grok_obj = {"overall_score_0_5": 0, "critical_must_fix": ["Could not parse Grok JSON"], "notes": err2 or ""}

    return claude_obj, grok_obj

# ----------------------------
# STRICT run
# ----------------------------
if run_strict:
    user_text = run_engine("STRICT")
    if user_text is None:
        st.stop()

    st.divider()
    st.markdown("## STRICT Output")

    master_input = "\n\n".join([
        "MODE: STRICT",
        "USER PROMPT:",
        user_text,
        "OUTPUT FORMAT:",
        MASTER_FORMAT
    ])

    with st.spinner("Master (STRICT) drafting..."):
        draft = call_openai(master_model, MASTER_SYSTEM_STRICT, master_input, temperature=0.15)

    st.write(draft)
    ok_math, info = render_math_table(draft, "STRICT")
    blockers = parse_blocking_items(draft)

    st.markdown("### Blocking inputs detected")
    st.write(blockers if blockers else ["None"])

    claude_obj, grok_obj = reviewer_rubrics(draft)
    st.markdown("### Claude rubric")
    st.json(claude_obj)
    st.markdown("### Grok rubric")
    st.json(grok_obj)

    if blockers:
        st.warning("STRICT found blocking inputs. Provide them in the box above, then re-run STRICT, or switch to ASSUMPTION mode.")

# ----------------------------
# ASSUMPTION run (iterative)
# ----------------------------
if run_assume:
    user_text = run_engine("ASSUMPTION")
    if user_text is None:
        st.stop()

    st.divider()
    st.markdown("## ASSUMPTION Output (Iterative)")

    history = []
    final_draft = None

    for it in range(1, max_iters + 1):
        st.markdown(f"### Iteration {it}")

        feedback = ""
        if history:
            prev = history[-1]
            feedback = "\n\n".join([
                "PREVIOUS ITERATION FEEDBACK (must address):",
                f"- Blocking inputs remaining: {prev['blockers']}",
                f"- Claude must-fix: {prev['claude']['critical_must_fix']}",
                f"- Grok must-fix: {prev['grok']['critical_must_fix']}",
                "- Ensure Math Claims JSON matches all [Computed] numbers.",
                "- Ensure assumptions are explicitly listed and labeled [Assumed].",
            ])

        master_input = "\n\n".join([
            "MODE: ASSUMPTION",
            "USER PROMPT:",
            user_text,
            feedback,
            "OUTPUT FORMAT:",
            MASTER_FORMAT
        ])

        with st.spinner("Master (ASSUMPTION) drafting..."):
            draft = call_openai(master_model, MASTER_SYSTEM_ASSUMPTION, master_input, temperature=0.25)

        st.write(draft)
        ok_math, _ = render_math_table(draft, f"ASSUMPTION Iter {it}")

        blockers = parse_blocking_items(draft)
        claude_obj, grok_obj = reviewer_rubrics(draft)

        st.markdown("**Claude rubric**")
        st.json(claude_obj)
        st.markdown("**Grok rubric**")
        st.json(grok_obj)

        no_critical = (len(claude_obj.get("critical_must_fix", [])) == 0 and len(grok_obj.get("critical_must_fix", [])) == 0)

        history.append({
            "iter": it,
            "draft": draft,
            "ok_math": ok_math,
            "blockers": blockers,
            "claude": claude_obj,
            "grok": grok_obj
        })

        # Convergence for assumption mode:
        # - math verified
        # - no critical must-fix
        # - blockers empty or explicitly "None"
        blockers_empty = (len(blockers) == 0)
        if ok_math and no_critical and blockers_empty:
            st.success("✅ Converged in ASSUMPTION mode.")
            final_draft = draft
            break
        else:
            final_draft = draft

    st.divider()
    st.markdown("## Final ASSUMPTION Draft")
    st.write(final_draft)

    export_text = f"""# Nemexis Export
Generated: {datetime.datetime.now()}
MODE: ASSUMPTION (iterative)

## USER PROMPT
{prompt.strip()}

## CONTEXT
{context.strip() if context.strip() else "(none)"}

## USER PROVIDED BLOCKER INPUTS
{blocker_input.strip() if blocker_input.strip() else "(none)"}

---

## FINAL DRAFT
{final_draft}

---

## HISTORY
"""
    for h in history:
        export_text += f"\n\n### Iteration {h['iter']}\n"
        export_text += f"- Math OK: {h['ok_math']}\n"
        export_text += f"- Blocking: {h['blockers']}\n"
        export_text += f"- Claude must-fix: {h['claude'].get('critical_must_fix', [])}\n"
        export_text += f"- Grok must-fix: {h['grok'].get('critical_must_fix', [])}\n"
        export_text += "\n---\n"
        export_text += h["draft"]

    st.divider()
    st.markdown("## Export")
    copy_to_clipboard_button(export_text, "📋 Copy Everything (ASSUMPTION)")
    st.download_button("⬇️ Download Markdown", export_text, file_name="nemexis_output.md", mime="text/markdown")
    st.text_area("All output (Cmd/Ctrl+A then Copy)", value=export_text, height=320)
