import streamlit as st
import pandas as pd
import requests as http

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Data Agent", page_icon="🤖", layout="wide")

# Session state
if "session_id"   not in st.session_state: st.session_state.session_id   = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "show_sql"     not in st.session_state: st.session_state.show_sql     = True
if "show_qid"     not in st.session_state: st.session_state.show_qid     = False

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def post_query(question: str) -> dict:
    payload = {"question": question, "session_id": st.session_state.session_id}
    resp    = http.post(f"{API_BASE}/query/conversation", json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    st.session_state.session_id = data["session_id"]
    return data["result"]

def fetch_history(limit: int = 10) -> list:
    resp = http.get(f"{API_BASE}/history", params={"limit": limit}, timeout=10)
    resp.raise_for_status()
    return resp.json()["history"]

def clear_cache_api() -> str:
    return http.post(f"{API_BASE}/cache/clear", timeout=10).json().get("message","Done")

def refresh_schema_api() -> str:
    return http.post(f"{API_BASE}/schema/refresh", timeout=120).json().get("message","Done")

def clear_conversation_api() -> None:
    if st.session_state.session_id:
        http.delete(f"{API_BASE}/query/conversation/{st.session_state.session_id}", timeout=10)
    st.session_state.session_id   = None
    st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# Result renderer — uses standardised response shape
# ---------------------------------------------------------------------------

def render_result(result: dict) -> None:
    if not result.get("success"):
        code = result.get("error_code", "ERROR")
        msg  = result.get("error_message", "Something went wrong.")
        st.error(f"**[{code}]** {msg}")
        if result.get("detail"):
            with st.expander("Details"):
                st.code(result["detail"])
        return

    # Query ID (optional display)
    if st.session_state.show_qid and result.get("query_id"):
        st.caption(f"🔑 Query ID: `{result['query_id']}`")

    # Meta row — uses standardised fields
    conf       = result.get("confidence", 0.0)
    latency    = result.get("latency_ms", 0)
    cached_tag = "⚡ cached" if result.get("cached") else f"⏱ {latency} ms"
    conf_color = "green" if conf >= 0.8 else "orange" if conf >= 0.5 else "red"
    data       = result.get("data", {})
    trunc_note = f" *(showing {len(data.get('rows',[]))} of {data.get('row_count',0)})*" \
                 if data.get("truncated") else ""

    col1, col2, col3 = st.columns([2, 1, 1])
    col1.markdown(f"**Rows:** {data.get('row_count',0)}{trunc_note}")
    col2.markdown(f"**Confidence:** :{conf_color}[{conf:.0%}]")
    col3.markdown(f"**{cached_tag}**")

    # SQL expander
    if st.session_state.show_sql:
        with st.expander("🔍 Generated SQL", expanded=False):
            st.code(result.get("sql",""), language="sql")
            if result.get("assumptions") and result["assumptions"] != "none":
                st.warning(f"⚠️ Assumptions: {result['assumptions']}")

    # Data table
    rows = data.get("rows", [])
    if rows:
        df = pd.DataFrame(rows, columns=data.get("columns",[]))
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No rows returned.")

    # Insight — now at "answer" key (standardised)
    answer = result.get("answer") or result.get("insight","")
    if answer:
        st.markdown(f"> 💡 {answer}")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")
    st.session_state.show_sql = st.toggle("Show SQL",      value=True)
    st.session_state.show_qid = st.toggle("Show Query ID", value=False)
    st.divider()

    st.subheader("Actions")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        clear_conversation_api()
        st.success("Conversation cleared.")
    if st.button("🔄 Refresh schema",      use_container_width=True):
        with st.spinner("Rebuilding schema embeddings…"):
            st.success(refresh_schema_api())
    if st.button("🧹 Clear result cache",  use_container_width=True):
        st.success(clear_cache_api())

    st.divider()
    st.subheader("📜 Recent queries")
    try:
        for h in fetch_history(10):
            icon   = "✅" if h["success"] else "❌"
            cached = "⚡" if h.get("cached") else ""
            st.markdown(
                f"{icon}{cached} `{str(h.get('question',''))[:48]}`  \n"
                f"<small>{h.get('row_count',0)} rows · "
                f"{h.get('latency_ms',0)} ms · "
                f"conf {float(h.get('confidence') or 0):.0%}</small>",
                unsafe_allow_html=True
            )
    except Exception:
        st.caption("(History unavailable — is the API running?)")

# ---------------------------------------------------------------------------
# Main chat
# ---------------------------------------------------------------------------

st.title("🤖 Data Agent")
st.caption("Ask questions about your data in plain English.")

for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        render_result(turn["result"])

if question := st.chat_input("e.g. Top 5 products by revenue last month"):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = post_query(question)
            except http.exceptions.ConnectionError:
                st.error("Cannot connect to the API. Is `uvicorn app:app` running?")
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.stop()
        render_result(result)
        st.session_state.chat_history.append({"question": question, "result": result})
