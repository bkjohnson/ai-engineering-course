"""Streamlit chat UI backed by the FastAPI /chat streaming endpoint.

Run the API first (`uvicorn main:app --reload`), then:

    streamlit run streamlit_app.py

Point at a deployed API with the API_BASE_URL environment variable.
"""

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECONDS = 300

st.set_page_config(page_title="Ask Claude", page_icon="✳️", layout="centered")

st.title("✳️ Ask Claude")
st.caption("Helpful. Dryly witty. Streaming.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown(f"**API:** `{API_BASE_URL}`")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Replay the conversation so far.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def stream_reply(messages: list[dict]):
    """Yield text chunks from the API's /ask endpoint in streaming mode."""
    *history, latest = messages
    with requests.post(
        f"{API_BASE_URL}/ask",
        json={"question": latest["content"], "history": history, "stream": True},
        stream=True,
        timeout=REQUEST_TIMEOUT_SECONDS,
    ) as response:
        response.raise_for_status()
        response.encoding = "utf-8"
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk


if question := st.chat_input("Ask a question…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(stream_reply(st.session_state.messages))
        except requests.RequestException as exc:
            st.error(f"Couldn't reach the API at {API_BASE_URL}: {exc}")
            # Drop the unanswered user turn so history stays consistent.
            st.session_state.messages.pop()
        else:
            st.session_state.messages.append({"role": "assistant", "content": answer})
