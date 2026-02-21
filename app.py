import os
import json
import streamlit as st
from openai import OpenAI
import anthropic
import requests

# ============================
# Nemexis — Iterative Moderated Flow
# Master: OpenAI
# Judges: Claude + Grok
# Orbits: Engineering, Finance, Contract/Legal, Risk/Governance
# Iterations: Rev 0 -> Critiques -> Rev 1 (+ Changelog)
# ============================

st.set_page_config(page_title="Nemexis", layout="wide")
st.title("Nemexis — Reliability Engine (Iterative Moderated Flow)")

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
# UI Controls
# ----------------------------
st.markdown("### Inputs")
prompt = st.text_area("User Prompt", height=160, placeholder="What are you trying to solve / decide?")
context = st.text_area("Context (optional)", height=160, placeholder="Paste numbers, excerpts, constraints, assumptions...")

st.markdown("### Settings")
master_model = st.selectbox(
    "Master model (OpenAI)",
    ["gpt-4o-mini", "gpt-4o"],
    index=0,
    help="gpt-4o is stronger but costs more. Start with mini."
)

claude_model = st.selectbox(
    "Claude judge model",
    ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-4-20250514"],
    index=0
)

grok_model = st.selectbox(
    "Grok judge model",
    ["grok-4-fast", "grok-4"],
    index=0
)

run = st.button("Run Nemexis (Rev 0 → Critiques → Rev 1)")

# ----------------------------
# Orbits (your day-to-day cast)
# ----------------------------
ORBITS = [
    {
        "name": "Technical Engineering",
        "judge_mandate": (
            "You are a senior offshore/industrial engineering reviewer. "
            "Your job is to critique the Master draft for technical correctness and realism. "
            "Focus on physics/constraints, failure modes, engineering assumptions, unit sanity checks, schedule realism, "
            "and any technical-financial coupling mistakes. "
            "Do NOT rewrite the whole answer. Provide targeted critique and actionable fixes."
        ),
    },
    {
        "name": "Economics / Project Finance",
        "judge_mandate": (
            "You are a project finance / infrastructure investment reviewer. "
            "Critique the Master draft for economic and financial rigor. "
            "Focus on cash-flow logic, timing, leverage/covenants, sensitivities, missing inputs, and math sanity checks. "
            "Do NOT invent numbers. If required inputs are missing, explicitly list them. "
            "Do NOT rewrite the whole answer. Provide targeted critique and actionable fixes."
        ),
    },
    {
        "name": "Contract / Legal (Commercial)",
        "judge_mandate": (
            "You are a contracts/commercial & legal reviewer (project delivery + procurement). "
            "Critique the Master draft for contractual/legal blind spots. "
            "Focus on entitlement/pass-through, LDs/caps, change orders, termination triggers, compliance/approvals, "
            "tariff/classification nuances (if relevant), and what clauses/documents must be checked. "
            "Not legal advice; this is issue-spotting and decision-support. "
            "Do NOT rewrite the whole answer. Provide targeted critique and actionable fixes."
        ),
    },
    {
        "name": "Risk / Governance",
        "judge_mandate": (
            "You are a risk officer focused on decision defensibility. "
            "Critique the Master draft for ambiguity, overconfidence, missing assumptions, and validation gaps. "
            "Produce a mini risk register (top risks), confidence assessment, and what would change the conclusion. "
            "Do NOT rewrite the whole answer. Provide targeted critique and actionable fixes."
        ),
    },
]

# ----------------------------
# Core Prompt Templates
# ----------------------------

MASTER_REV0_SYSTEM = (
    "You are Nemexis Master (OpenAI). Your job is to produce the best possible first draft (Rev 0) "
    "that a professional could take into an IC/SteerCo/board discussion. "
    "Be precise, structured, and conservative with claims. "
    "If a critical input is missing, list it rather than guessing."
)

MASTER_OUTPUT_FORMAT = """
Return Rev 0 in this exact structure:

## Executive Answer
- 5 bullets maximum, decision-oriented

## Key Assumptions
- Bullet list

## Calculations / Logic
- Show key steps, do not invent missing numbers
- If you need inputs, list them as placeholders

## Key Risks (ranked)
- 5–10 bullets, ranked

## What to Validate Next
- concrete checks / documents / data needed

## Confidence
- Low / Medium / High + 1–2 lines why
""".strip()

JUDGE_CRITIQUE_FORMAT = """
Return critique in this structure (do not rewrite the whole answer):

## Summary Verdict
- 1–3 bullets on overall quality

## Critical Issues (must-fix)
- Bullet list (with short rationale)

## Corrections / Fixes
- Bullet list of specific edits or calculations to change

## Missing Inputs (required)
- Bullet list

## Questions for the Team
- Bullet list of questions to resolve uncertainty

## Confidence in Master Draft
- Low / Medium / High + why
""".strip()

MASTER_REV1_SYSTEM = (
    "You are Nemexis Master (OpenAI) creating Rev 1. "
    "You will receive Rev 0 plus critiques from multiple judges across four orbits. "
    "Your job is to: (1) accept/reject critique items with reasoning, "
    "(2) correct errors, (3) tighten assumptions, (4) produce a stronger Rev 1. "
    "Do NOT blindly merge everything. Keep it coherent and decision-grade."
)

MASTER_REV1_OUTPUT_FORMAT = """
Return in this exact structure:

# Changelog (Rev 0 → Rev 1)
- Accepted critiques: bullet list
- Rejected critiques (with reason): bullet list
- Key edits applied: bullet list

# Rev 1 (Updated Deliverable)
(use the same structure as Rev 0)
""".strip()

# ----------------------------
# Model Call Helpers
# ----------------------------

def call_openai(model_name: str, system: str, user_text: str) -> str:
    resp = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        temperature=0.2,
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
    r = requests.post(
        "https://api.x.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.2,
        },
        timeout=60,
    )
    if r.status_code != 200:
        return f"❌ Grok Error ({r.status_code}): {r.text}"
    return r.json()["choices"][0]["message"]["content"]


# ----------------------------
# Main Run
# ----------------------------
if run:
    if not prompt.strip():
        st.error("Please enter a User Prompt.")
        st.stop()

    user_text = prompt.strip()
    if context.strip():
        user_text += "\n\n---\nContext:\n" + context.strip()

    st.divider()
    st.markdown("## Step 1 — Master Draft (Rev 0)")

    with st.spinner("Master (OpenAI) generating Rev 0..."):
        rev0_input = f"USER PROMPT:\n{user_text}\n\nOUTPUT FORMAT:\n{MASTER_OUTPUT_FORMAT}"
        rev0 = call_openai(master_model, MASTER_REV0_SYSTEM, rev0_input)

    st.markdown("### Rev 0 (Master)")
    st.write(rev0)

    st.divider()
    st.markdown("## Step 2 — Orbit Critiques (Claude + Grok)")

    critiques = []  # store structured critiques for Rev1 integration

    for orbit in ORBITS:
        orbit_name = orbit["name"]
        mandate = orbit["judge_mandate"]

        st.markdown(f"### Orbit: {orbit_name}")

        judge_input = (
            f"ORBIT:\n{orbit_name}\n\n"
            f"YOUR MANDATE:\n{mandate}\n\n"
            f"MASTER DRAFT (REV 0):\n{rev0}\n\n"
            f"CRITIQUE FORMAT:\n{JUDGE_CRITIQUE_FORMAT}"
        )

        colA, colB = st.columns(2)

        with colA:
            st.markdown(f"#### Claude ({claude_model})")
            with st.spinner(f"Claude critiquing: {orbit_name}..."):
                try:
                    claude_out = call_claude(claude_model, mandate, judge_input)
                except Exception as e:
                    claude_out = f"❌ Claude Error: {str(e)}"
            st.write(claude_out)

        with colB:
            st.markdown(f"#### Grok ({grok_model})")
            with st.spinner(f"Grok critiquing: {orbit_name}..."):
                try:
                    grok_out = call_grok(grok_model, mandate, judge_input)
                except Exception as e:
                    grok_out = f"❌ Grok Error: {str(e)}"
            st.write(grok_out)

        critiques.append(
            {
                "orbit": orbit_name,
                "claude": claude_out,
                "grok": grok_out,
            }
        )

    st.divider()
    st.markdown("## Step 3 — Master Integration (Rev 1 + Changelog)")

    with st.spinner("Master (OpenAI) integrating critiques into Rev 1..."):
        critique_blob = ""
        for c in critiques:
            critique_blob += f"\n\n=== ORBIT: {c['orbit']} ===\n\n--- Claude Critique ---\n{c['claude']}\n\n--- Grok Critique ---\n{c['grok']}\n"

        rev1_input = (
            f"USER PROMPT:\n{user_text}\n\n"
            f"MASTER REV 0:\n{rev0}\n\n"
            f"CRITIQUES:\n{critique_blob}\n\n"
            f"OUTPUT FORMAT:\n{MASTER_REV1_OUTPUT_FORMAT}\n\n"
            f"Remember: accept/reject with reasons, correct errors, tighten assumptions, no blind merging."
        )

        rev1 = call_openai(master_model, MASTER_REV1_SYSTEM, rev1_input)

    st.markdown("### Rev 1 (Master, with Changelog)")
    st.write(rev1)

    st.divider()
    st.markdown("## What you have now")
    st.markdown(
        "- **Rev 0**: Master initial deliverable\n"
        "- **Orbit critiques**: Claude + Grok challenged Rev 0 across Engineering/Finance/Legal/Risk\n"
        "- **Rev 1**: Master integrated critiques with an explicit changelog\n\n"
        "Next upgrade (optional): run the orbit critics again on Rev 1 to produce Rev 2."
    )
