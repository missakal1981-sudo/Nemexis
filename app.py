import os
import streamlit as st
from openai import OpenAI
import anthropic
import requests

st.set_page_config(page_title="Nemexis", layout="wide")
st.title("Nemexis — Engineering + Finance Reliability Engine (3 Models)")

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
    st.error(f"Missing secrets: {', '.join(missing)}")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

prompt = st.text_area("Your Prompt", height=180)
context = st.text_area("Optional Context", height=180)
run = st.button("Run Nemexis (OpenAI + Claude + Grok)")

ROLE_SYSTEMS = {
    "Technical Reviewer": "You are a senior engineering reviewer. Challenge assumptions and identify failure modes.",
    "Finance Reviewer": "You are a project finance reviewer. Show calculations step-by-step and include sanity checks.",
    "Risk Reviewer": "You are a risk officer. Identify high-risk claims and validation requirements.",
}

SYNTHESIS_SYSTEM = "You are Nemexis Synthesizer. Combine all outputs into Version 2 with explicit disagreements."

# ----------------------------
# Model Calls
# ----------------------------

def call_openai(system, user_text):
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user_text}],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ OpenAI Error: {str(e)}"


def call_claude(system, user_text):
    try:
        msg = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            temperature=0.2,
            system=system,
            messages=[{"role": "user", "content": user_text}],
        )
        return "".join(block.text for block in msg.content if hasattr(block, "text"))
    except Exception as e:
        return f"❌ Claude Error: {str(e)}"


def call_grok(system, user_text):
    try:
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "grok-4-fast",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
                ],
                "temperature": 0.2,
            },
        )

        if response.status_code != 200:
            return f"❌ Grok Error: {response.text}"

        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        return f"❌ Grok Error: {str(e)}"


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

    all_outputs = {}

    for role_name, role_prompt in ROLE_SYSTEMS.items():
        st.subheader(role_name)

        with st.spinner("Running models..."):
            openai_out = call_openai(role_prompt, user_text)
            claude_out = call_claude(role_prompt, user_text)
            grok_out = call_grok(role_prompt, user_text)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### OpenAI")
            st.write(openai_out)

        with col2:
            st.markdown("### Claude")
            st.write(claude_out)

        with col3:
            st.markdown("### Grok")
            st.write(grok_out)

        all_outputs[role_name] = {
            "OpenAI": openai_out,
            "Claude": claude_out,
            "Grok": grok_out,
        }

    st.divider()
    st.subheader("Nemexis — Version 2 (Synthesis)")

    combined = ""
    for role, providers in all_outputs.items():
        combined += f"\n\n## {role}\n"
        for provider, text in providers.items():
            combined += f"\n### {provider}\n{text}\n"

    synthesis = call_openai(SYNTHESIS_SYSTEM, combined)
    st.write(synthesis)
