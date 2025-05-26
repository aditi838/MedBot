import streamlit as st
import requests
import uuid
from ai_agent import system_prompt

st.set_page_config(page_title="MedBot Chat", layout="wide")
st.title("🩺 MedBot – Healthcare Assistant")

# ── Session Init ────────────────────────────────────────────
state = st.session_state
if "session_id" not in state:
    state.session_id = str(uuid.uuid4())
if "messages" not in state:
    state.messages = []
if "grounding_scores" not in state:
    # parallel list: None for user messages, floats for assistant replies
    state.grounding_scores = []
if "user_metadata" not in state:
    state.user_metadata = {"age": None, "gender": None}
if "last_action" not in state:
    state.last_action = None

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox(
        "Model:",
        ["llama3-70b-8192", "llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b"]
    )
    provider     = "Groq"
    allow_search = st.checkbox("Allow Web Search", value=True)

    st.subheader("Patient Info (optional)")
    with st.form("patient_info_form"):
        age_input    = st.number_input(
            "Age", min_value=0, max_value=120,
            value=state.user_metadata.get("age") or 0
        )
        gender_input = st.selectbox(
            "Gender", ["", "Male", "Female", "Other"],
            index=["", "Male", "Female", "Other"]
                  .index(state.user_metadata.get("gender") or "")
        )
        submitted = st.form_submit_button("Submit Info")
    if submitted:
        state.user_metadata["age"]    = age_input or None
        state.user_metadata["gender"] = gender_input or None
        state.last_action = "metadata"
        state.messages.append("Patient info submitted.")
        state.grounding_scores.append(None)
        st.rerun()

# ── Chat Input ───────────────────────────────────────────────
if user_input := st.chat_input("Describe symptoms or ask a question..."):
    state.messages.append(user_input)
    state.grounding_scores.append(None)
    state.last_action = "chat"

# ── Backend Call ─────────────────────────────────────────────
if state.last_action:
    state.last_action = None
    with st.spinner("MedBot is thinking..."):
        payload = {
            "session_id":     state.session_id,
            "model_name":     selected_model,
            "model_provider": provider,
            "messages":       state.messages,
            "allow_search":   allow_search,
            "user_metadata":  state.user_metadata
        }
        try:
            r = requests.post("http://127.0.0.1:9999/chat", json=payload)
            r.raise_for_status()
            data  = r.json()
            reply = data.get("response", "No response.")
            score = data.get("grounding_score", None)
        except Exception as e:
            reply = f"Error: {e}"
            score = None

        state.messages.append(reply)
        state.grounding_scores.append(score)

# ── Render Chat History ──────────────────────────────────────
for idx, msg in enumerate(state.messages):
    role = "user" if idx % 2 == 0 else "assistant"
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        container = st.chat_message("assistant")
        score = state.grounding_scores[idx]
        if score is not None:
            # inline markdown + raw HTML span for tooltip
            md = (
                msg
                + "  \n"  # markdown line break
                + f"<span style='font-size:16px;cursor:help;' title='Grounding score: {score:.2f}'>ℹ️</span>"
            )
            container.markdown(md, unsafe_allow_html=True)
        else:
            container.write(msg)
