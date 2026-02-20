import os
import streamlit as st
from openai import OpenAI
import anthropic

st.set_page_config(page_title="Nemexis", layout="wide")
st.title("Nemexis — Engineering + Finance Reliability Engine (OpenAI + Claude)")

# ----------------------------
# Password Gate
# ----------------------------
APP_PASSWORD = os.getenv("NEMEXIS_PASSWORD", "").strip()
if APP_PASSWORD:
    entered = st.text_input("Password", type="password")
    if entered != APP_PASSWORD:
        st.stop()

# ----------------------------
# Load API Keys
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()

missing = []
if not OPENAI_API_KEY:
    missing.append("OPENAI_API_KEY")
if not ANTHROPIC_API_KEY:
    missing.append("ANTHROPIC_API_KEY")

if missing:
    st.error(f"Missing secrets: {', '.join(missing)}")
    st.stop()

prompt = st.text_area("Your Prompt", height=180)
context = st.text_area("Optional Context", height=180)
run = st.button("Run Nemexis (2 Models)")

# ----------------------------
# Role Prompts
# ----------------------------
ROLE_SYSTEMS = {
    "Technical Reviewer": (
        "You are a senior engineering reviewer. Challenge assumptions, identify failure modes, "
        "spot missing constraints, and propose validation checks."
    ),
    "Finance Reviewer": (
        "You are a project finance reviewer. Show calculations step-by-step, include a sanity check, "
        "quantify sensitivities, and identify missing inputs."
    ),
    "Risk Reviewer": (
        "You are a risk officer. Identify high-risk claims, uncertainty areas, and validation requirements."
    ),
}

SYNTHESIS_SYSTEM = (
    "You are Nemexis Synthesizer. Combine all reviewer outputs into Version 2. "
    "Preserve disagreements, list assumptions, provide validation plan, and assign confidence level."
)

# ----------------------------
# Provider Calls
# ----------------------------
def call_openai(system, user_text):
    try:
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
    except Exception as e:
        return f"❌ OpenAI Error: {str(e)}"

def call_claude(system, user_text):
    """
    Claude model naming is sometimes account-specific.
    We use a more compatible Sonnet model name.
    """
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1200,
            temperature=0.2,
            system=system,
            messages=[{"role": "user", "content": user_text}],
        )
        return "\n".join([block.text for block in msg.content if hasattr(block, "text")]).strip()
    except Exception as e:
        return f"❌ Claude Error: {str(e)}"

# ----------------------------
# Execution
# ----------------------------
if run:
    if not prompt.strip():
        st.error("Please enter a prompt.")
        st.stop()

    user_text = prompt.strip()
    if context.strip():
        user_text += "\n\nContext:\n" + context.strip()

    st.divider()
    all_outputs = {}

    for role_name, role_prompt in ROLE_SYSTEMS.items():
        st.subheader(role_name)
        with st.spinner("Running OpenAI + Claude..."):
            openai_out = call_openai(role_prompt, user_text)
            claude_out = call_claude(role_prompt, user_text)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### OpenAI")
            st.write(openai_out)
        with col2:
            st.markdown("### Claude")
            st.write(claude_out)

        all_outputs[role_name] = {"OpenAI": openai_out, "Claude": claude_out}

    st.divider()
    st.subheader("Nemexis — Version 2 (Synthesis)")

    combined_text = ""
    for role, providers in all_outputs.items():
        combined_text += f"\n\n## {role}\n"
        for provider, text in providers.items():
            combined_text += f"\n### {provider}\n{text}\n"

    synthesis = call_openai(SYNTHESIS_SYSTEM, combined_text)
    st.write(synthesis)
