import os
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Nemexis", layout="wide")
st.title("Nemexis — Engineering + Finance Reliability Engine")
# --- Simple password gate (MVP privacy) ---
APP_PASSWORD = os.getenv("NEMEXIS_PASSWORD", "")

if APP_PASSWORD:
    entered = st.text_input("Password", type="password")
    if entered != APP_PASSWORD:
        st.stop()

# Read key from Streamlit Cloud Secrets
api_key = os.getenv("OPENAI_API_KEY", "")
api_key = api_key.encode("ascii", "ignore").decode().strip()

if not api_key:
    st.error("API key not detected.")
    st.stop()

prompt = st.text_area("Your Prompt", height=180, placeholder="Paste your engineering+finance question here...")
context = st.text_area("Optional Context", height=180, placeholder="Paste assumptions, numbers, excerpts…")

run = st.button("Run Nemexis")

ROLE_SYSTEMS = {
    "Technical Reviewer (Engineering)": (
        "You are a senior engineering reviewer. Challenge technical assumptions, identify failure modes, "
        "spot missing constraints, and propose validation checks. Output: (1) Key assumptions "
        "(2) Critical risks (3) What to verify (4) Improvements."
    ),
    "Finance Reviewer (Project Finance / CFA)": (
        "You are a project finance / infrastructure investment reviewer. Challenge economic logic, quantify sensitivities, "
        "identify missing drivers, and summarize IC-ready takeaways. Output: (1) Key drivers (2) Sensitivities "
        "(3) Missing data (4) IC takeaways."
    ),
    "Risk & Governance Reviewer": (
        "You are a risk officer focused on decision defensibility. Flag uncertainty, ambiguous claims, missing sources, "
        "and areas requiring validation. Output: (1) Highest-risk claims (2) Confidence assessment "
        "(3) Mitigations (4) What would change your mind."
    ),
}

SYNTHESIS_SYSTEM = (
    "You are Nemexis Synthesizer. Combine reviewer outputs into Version 2. "
    "Rules: (a) Preserve disagreements explicitly (b) List assumptions clearly "
    "(c) Provide a validation plan (d) Provide a final recommendation with confidence level (Low/Med/High) and why."
)

def call_model(client: OpenAI, system: str, user_text: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

if run:
    if not api_key:
        st.error("Missing OPENAI_API_KEY. Add it in Streamlit Cloud → App settings → Secrets.")
        st.stop()

    if not prompt.strip():
        st.error("Please enter a prompt.")
        st.stop()

    client = OpenAI(api_key=api_key)

    user_text = prompt.strip()
    if context.strip():
        user_text += "\n\n---\nContext:\n" + context.strip()

    cols = st.columns(3)
    reviewer_outputs = {}

    for i, (role_name, role_system) in enumerate(ROLE_SYSTEMS.items()):
        with st.spinner(f"Running {role_name}..."):
            out = call_model(client, role_system, user_text)
            reviewer_outputs[role_name] = out
            with cols[i]:
                st.subheader(role_name)
                st.write(out)

    with st.spinner("Synthesizing Version 2..."):
        combined = "\n\n".join([f"## {k}\n{v}" for k, v in reviewer_outputs.items()])
        synthesis_input = f"USER PROMPT:\n{user_text}\n\nREVIEWER OUTPUTS:\n{combined}"
        v2 = call_model(client, SYNTHESIS_SYSTEM, synthesis_input)

    st.divider()
    st.subheader("Nemexis — Version 2 (Synthesis)")
    st.write(v2)
