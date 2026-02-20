import os
import streamlit as st
from openai import OpenAI
import anthropic
from google import genai

st.set_page_config(page_title="Nemexis", layout="wide")
st.title("Nemexis — Engineering + Finance Reliability Engine (3-model)")

# --- Password gate ---
APP_PASSWORD = os.getenv("NEMEXIS_PASSWORD", "")
if APP_PASSWORD:
    entered = st.text_input("Password", type="password")
    if entered != APP_PASSWORD:
        st.stop()

# --- API keys from Streamlit Secrets ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

missing = []
if not OPENAI_API_KEY:
    missing.append("OPENAI_API_KEY")
if not ANTHROPIC_API_KEY:
    missing.append("ANTHROPIC_API_KEY")
if not GEMINI_API_KEY:
    missing.append("GEMINI_API_KEY")

if missing:
    st.error("Missing secrets: " + ", ".join(missing) + ". Add them in Manage app → Settings → Secrets.")
    st.stop()

prompt = st.text_area("Your Prompt", height=180, placeholder="Paste your engineering+finance question here...")
context = st.text_area("Optional Context", height=180, placeholder="Paste assumptions, numbers, excerpts…")

run = st.button("Run Nemexis (3 models)")

ROLE_SYSTEMS = {
    "Technical Reviewer (Engineering)": (
        "You are a senior engineering reviewer. Challenge technical assumptions, identify failure modes, "
        "spot missing constraints, and propose validation checks. Output: (1) Key assumptions "
        "(2) Critical risks (3) What to verify (4) Improvements."
    ),
    "Finance Reviewer (Project Finance / CFA)": (
        "You are a project finance / infrastructure investment reviewer. Show calculations step-by-step and include "
        "at least one order-of-magnitude sanity check. Quantify sensitivities and identify missing inputs. "
        "Output: (1) Inputs used (2) Calculations (3) Sensitivities (4) Missing data (5) IC takeaways."
    ),
    "Risk & Governance Reviewer": (
        "You are a risk officer focused on decision defensibility. Flag uncertainty, ambiguous claims, missing sources, "
        "and areas requiring validation. Output: (1) Highest-risk claims (2) Confidence assessment "
        "(3) Mitigations (4) What would change your mind."
    ),
}

SYNTHESIS_SYSTEM = (
    "You are Nemexis Synthesizer. Combine all outputs into Version 2. "
    "Rules: (a) Preserve disagreements explicitly (b) List assumptions clearly "
    "(c) Provide a validation plan (d) Provide a final recommendation with confidence level (Low/Med/High) and why."
)

# ---------- Provider calls ----------

def call_openai(system: str, user_text: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

def call_claude(system: str, user_text: str) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=1200,
        temperature=0.2,
        system=system,
        messages=[{"role": "user", "content": user_text}],
    )
    # Claude returns content blocks
    out_parts = []
    for blk in msg.content:
        if getattr(blk, "type", None) == "text":
            out_parts.append(blk.text)
    return "\n".join(out_parts).strip()

def call_gemini(system: str, user_text: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"{system}\n\nUSER:\n{user_text}",
    )
    return (resp.text or "").strip()

def run_three_models(system: str, user_text: str) -> dict:
    # Sequential for MVP simplicity; we can parallelize later
    return {
        "OpenAI": call_openai(system, user_text),
        "Claude": call_claude(system, user_text),
        "Gemini": call_gemini(system, user_text),
    }

# ---------- Run ----------
if run:
    if not prompt.strip():
        st.error("Please enter a prompt.")
        st.stop()

    user_text = prompt.strip()
    if context.strip():
        user_text += "\n\n---\nContext:\n" + context.strip()

    st.divider()
    st.subheader("Three-model outputs by role")

    all_outputs = {}

    for role_name, role_system in ROLE_SYSTEMS.items():
        st.markdown(f"## {role_name}")
        with st.spinner(f"Running {role_name} across OpenAI + Claude + Gemini..."):
            outs = run_three_models(role_system, user_text)
            all_outputs[role_name] = outs

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### OpenAI")
            st.write(outs["OpenAI"])
        with c2:
            st.markdown("### Claude")
            st.write(outs["Claude"])
        with c3:
            st.markdown("### Gemini")
            st.write(outs["Gemini"])

    st.divider()
    st.subheader("Nemexis — Version 2 (Synthesis)")

    combined = ""
    for role_name, outs in all_outputs.items():
        combined += f"\n\n## {role_name}\n"
        for provider, text in outs.items():
            combined += f"\n### {provider}\n{text}\n"

    synthesis_input = f"USER PROMPT:\n{user_text}\n\nTHREE-MODEL OUTPUTS:\n{combined}"
    v2 = call_openai(SYNTHESIS_SYSTEM, synthesis_input)
    st.write(v2)
