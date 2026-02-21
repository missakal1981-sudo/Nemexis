import os
import streamlit as st
from openai import OpenAI
import anthropic
import requests
import datetime

# =====================================================
# Nemexis v3 — Universal Reliability Engine
# Master: OpenAI
# Critics: Claude + Grok
# Layers:
#   1. STRICT (Audit / Epistemic Safe)
#   2. DIRECTIONAL OVERLAY (Executive-Grade, Bounded)
# =====================================================

st.set_page_config(page_title="Nemexis v3", layout="wide")
st.title("Nemexis v3 — Universal Reliability Engine")

# ===============================
# Password Gate (optional)
# ===============================
APP_PASSWORD = os.getenv("NEMEXIS_PASSWORD", "").strip()
if APP_PASSWORD:
    entered = st.text_input("Password", type="password")
    if entered != APP_PASSWORD:
        st.stop()

# ===============================
# Load API Keys
# ===============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
XAI_API_KEY = os.getenv("XAI_API_KEY", "").strip()

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)

anthropic_client = None
if ANTHROPIC_API_KEY:
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ===============================
# UI Controls
# ===============================
mode = st.selectbox(
    "Mode",
    ["STRICT (Audit Only)", "STRICT + Directional Overlay (Recommended)"],
    index=1
)

prompt = st.text_area("User Prompt", height=200)
context = st.text_area("Additional Context (optional)", height=150)

run = st.button("Run Nemexis")

# ===============================
# System Prompts
# ===============================

STRICT_SYSTEM = """
You are Nemexis STRICT Layer.

Rules:
- Do NOT invent missing numbers.
- If calculation requires missing inputs, explicitly state "Cannot compute".
- Label every numeric item as:
  [Known] [Computed] [Assumed] [Unknown]
- Separate Blocking vs Non-Blocking missing inputs.
- No directional speculation beyond input logic.
"""

STRICT_FORMAT = """
Return in this structure:

## Inputs Used (Verbatim)

## Assumptions Added (Explicit)

## Executive Answer (Strict)

## Calculations / Logic

## Key Risks (Ranked)

## Missing Inputs
### Blocking
### Non-Blocking

## Claims Audit
(Label every numeric item)

## Confidence
"""

OVERLAY_SYSTEM = """
You are Nemexis Directional Overlay Layer.

You are NOT allowed to:
- Introduce new hard numeric assumptions
- Contradict the STRICT layer

You ARE allowed to:
- Provide bounded directional reasoning
- Provide sensitivity framing
- Provide IF-THEN scenarios
- Provide executive IC framing
- Provide risk-adjusted decision narrative

All reasoning must:
- Be consistent with STRICT output
- Clearly label assumptions
- Use ranges if needed
- Never fabricate CFADS/DSCR values

Your goal:
Produce an IC-ready executive brief superior to a single-model memo.
"""

OVERLAY_FORMAT = """
# Directional Executive Overlay (Non-Binding)

## Decision Framing

## Downside Case (Bounded)

## Base Case Interpretation

## What Could Flip the Conclusion

## IC-Ready Summary (10 bullets max)
"""

# ===============================
# Model Call Helpers
# ===============================

def call_openai(system, user_text, temperature=0.2):
    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content


# ===============================
# Run Engine
# ===============================
if run:

    if not prompt.strip():
        st.error("Please enter a prompt.")
        st.stop()

    full_input = prompt
    if context.strip():
        full_input += "\n\n---\nContext:\n" + context.strip()

    st.divider()
    st.markdown("## STRICT Layer (Audit-Grade)")

    strict_output = call_openai(
        STRICT_SYSTEM,
        full_input + "\n\nFORMAT:\n" + STRICT_FORMAT,
        temperature=0.1
    )

    st.write(strict_output)

    final_output = strict_output

    # ===============================
    # Directional Overlay
    # ===============================
    if mode == "STRICT + Directional Overlay (Recommended)":

        st.divider()
        st.markdown("## Directional Overlay (Executive Layer)")

        overlay_input = f"""
STRICT OUTPUT:
{strict_output}

USER PROMPT:
{prompt}

FORMAT:
{OVERLAY_FORMAT}
"""

        overlay_output = call_openai(
            OVERLAY_SYSTEM,
            overlay_input,
            temperature=0.4
        )

        st.write(overlay_output)

        final_output = strict_output + "\n\n" + overlay_output

    # ===============================
    # Copy / Export Section
    # ===============================
    st.divider()
    st.markdown("## Export")

    export_text = f"""
Nemexis v3 Export
Generated: {datetime.datetime.now()}

{final_output}
"""

    st.download_button(
        label="Download Full Output (.txt)",
        data=export_text,
        file_name="nemexis_output.txt",
        mime="text/plain"
    )

    st.code(export_text)
