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
    st.error(f"Missing secrets: {', '.join(missing)}  (Manage app → Settings → Secrets)")
    st.stop()

# ----------------------------
# UI
# ----------------------------
prompt = st.text_area("Your Prompt", height=180, placeholder="Paste your engineering+finance question here...")
context = st.text_area("Optional Context", height=180, placeholder="Paste assumptions, numbers, excerpts…")
run = st.button("Run Nemexis (2 Models)")

# ----------------------------
# Role Prompts
# ----------------------------
ROLE_SYSTEMS = {
    "Technical Reviewer": (
        "You are a senior engineering reviewer. Challenge assumptions, identify failure modes, "
        "spot missing constraints, and propose validation checks. "
        "Output: (1) Key assumptions (2) Critical risks (3) What to verify (4) Improvements."
    ),
    "Finance Reviewer": (
        "You are a project finance reviewer. Show calculations step-by-step, include at least one sanity check "
        "(order-of-magnitude), quantify sensitivities, and identify missing inputs. "
        "Output: (1) Inputs used (2) Calculations (3) Sensitivities (4) Missing data (5) IC takeaways."
    ),
    "Risk Reviewer": (
        "You are a risk officer. Identify high-risk claims, uncertainty areas, and validation requirements. "
        "Output: (1) Highest-risk claims (2) Confidence assessment (3) Mitigations (4) What would change your mind."
    ),
}

SYNTHESIS_SYSTEM = (
    "You are Nemexis Synthesizer. Combine all reviewer outputs into Version 2. "
    "Rules: (a) Preserve disagreements explicitly (b) List assumptions clearly "
    "(c) Provide a validation plan (d) Provide a final recommendation with confidence level (Low/Med/High) and why."
)

# ----------------------------
# Provider Calls
# ----------------------------

def call_openai(system: str, user_text: str) -> str:
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

# Claude: try multiple model names (Anthropic accounts differ)
CLAUDE_MODEL_CANDIDATES = [
    # Most common current names (try these first)
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    # Some accounts use these aliases
    "claude-3-5-sonnet",
    "claude-3-5-haiku",
    # Older but widely available in some orgs
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]

def _claude_try_model(client: anthropic.Anthropic, model_name: str, system: str, user_text: str) -> str:
    msg = client.messages.create(
        model=model_name,
        max_tokens=1200,
        temperature=0.2,
        system=system,
        messages=[{"role": "user", "content": user_text}],
    )
    parts = []
    for blk in msg.content:
        if hasattr(blk, "text"):
            parts.append(blk.text)
    return "\n".join(parts).strip()

def call_claude(system: str, user_text: str) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    last_err = None
    for model_name in CLAUDE_MODEL_CANDIDATES:
        try:
            out = _claude_try_model(client, model_name, system, user_text)
            # include the model name so you know which one worked
            return f"✅ Claude model: {model_name}\n\n{out}"
        except Exception as e:
            last_err = e
            # keep trying next model
            continue

    # If none worked, show a clean actionable message
    return (
        "❌ Claude Error: None of the Claude model names worked for your Anthropic account.\n\n"
        "What to do:\n"
        "1) Go to https://console.anthropic.com → (left) Models (or Docs) and check which model IDs your account supports.\n"
        "2) If your account is new/limited, you may only have certain models enabled.\n"
        "3) Paste the supported model ID into CLAUDE_MODEL_CANDIDATES (top of app.py).\n\n"
        f"Last error seen: {str(last_err)}"
    )

# ----------------------------
# Execution
# ----------------------------

if run:
    if not prompt.strip():
        st.error("Please enter a prompt.")
        st.stop()

    user_text = prompt.strip()
    if context.strip():
        user_text += "\n\n---\nContext:\n" + context.strip()

    st.divider()
    all_outputs = {}

    for role_name, role_prompt in ROLE_SYSTEMS.items():
        st.subheader(role_name)

        with st.spinner("Running OpenAI + Claude..."):
            openai_out = call_openai(role_prompt, user_text)
            claude_out = call_claude(role_prompt, user_text)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### OpenAI")
            st.write(openai_out)
        with c2:
            st.markdown("### Claude")
            st.write(claude_out)

        all_outputs[role_name] = {"OpenAI": openai_out, "Claude": claude_out}

    st.divider()
    st.subheader("Nemexis — Version 2 (Synthesis)")

    combined = ""
    for role, providers in all_outputs.items():
        combined += f"\n\n## {role}\n"
        for provider, text in providers.items():
            combined += f"\n### {provider}\n{text}\n"

    v2 = call_openai(SYNTHESIS_SYSTEM, f"USER PROMPT:\n{user_text}\n\nTWO-MODEL OUTPUTS:\n{combined}")
    st.write(v2)
