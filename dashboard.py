import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8001"

st.set_page_config(page_title="Self-Correcting RAG Pipeline", layout="wide")

st.title("🔎 Self-Correcting RAG Pipeline")
st.caption("Retrieval + Generation + Self-Evaluation + Automatic Retry")

tab1, tab2 = st.tabs(["Ask a Question", "History"])

# ---------------- TAB 1: Live Query ----------------
with tab1:
    query = st.text_input("Ask a question about your document:")
    max_retries = st.slider("Max retries allowed", 0, 3, 1)

    if st.button("Ask", type="primary") and query:
        with st.spinner("Retrieving, generating, and self-evaluating..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"query": query, "max_retries": max_retries}
                )
                response.raise_for_status()
                result = response.json()

                # --- Answer ---
                st.subheader("Answer")
                if result.get("from_cache"):
                    st.caption("⚡ Served from cache (identical question asked recently)")
                st.write(result["answer"])

                # --- Verdict badge (uses pipeline-level display_verdict) ---
                verdict = result["display_verdict"]
                if verdict.startswith("Faithful"):
                    st.success(f"✅ Verdict: {verdict}")
                elif verdict == "Unverifiable":
                    st.warning(f"⚠️ Verdict: {verdict}")
                elif verdict == "Irrelevant":
                    st.info(f"ℹ️ Verdict: {verdict} (question not answerable from document)")
                else:
                    st.error(f"❌ Verdict: {verdict}")

                st.caption(f"Raw eval framework verdict: {result['final_verdict']} · Retries used: {result['retries_used']}")

                # --- Retry log: show what happened at each attempt ---
                if result.get("retry_log"):
                    with st.expander(f"📋 See all {len(result['retry_log'])} attempt(s)"):
                        for i, entry in enumerate(result["retry_log"]):
                            if entry.get("stage") == "retrieval":
                                st.markdown(f"**Attempt {i+1} — Retrieval Retry**")
                                st.write(f"Reason: {entry.get('reason')}")
                                st.write(f"Action: {entry.get('action')}")
                            else:
                                st.markdown(f"**Attempt {i+1} — Generation (verdict: {entry.get('verdict')})**")
                                st.write(entry.get("answer"))
                                st.caption(f"Cosine: {entry.get('cosine')} · BERTScore: {entry.get('bert_score')}")
                            st.divider()

                # --- Detailed breakdown ---
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Retrieval Evaluation**")
                    st.json({
                        "Context Relevance": result["context_relevance"],
                        "Context Recall": result["context_recall"]
                    })

                with col2:
                    st.markdown("**Generation Evaluation**")
                    st.json(result["full_generation_eval"])

            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the API: {e}")

# ---------------- TAB 2: History ----------------
with tab2:
    st.subheader("Past Queries")

    if st.button("Refresh History"):
        st.rerun()

    try:
        response = requests.get(f"{API_URL}/history", params={"limit": 100})
        response.raise_for_status()
        data = response.json()
        logs = data["logs"]

        if not logs:
            st.info("No queries logged yet. Ask something in the first tab!")
        else:
            df = pd.DataFrame(logs)

            # --- Verdict distribution chart (uses smarter display_verdict) ---
            st.markdown("**Verdict Distribution (Display Verdict)**")
            verdict_col = "display_verdict" if "display_verdict" in df.columns else "final_verdict"
            verdict_counts = df[verdict_col].value_counts()
            st.bar_chart(verdict_counts)

            # --- Table of past queries ---
            st.markdown("**Query Log**")
            display_cols = [
                "timestamp", "query", "display_verdict", "final_verdict",
                "cosine_score", "bert_score", "nli_label",
                "context_relevance_score", "context_recall_score",
                "retries_used"
            ]
            display_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[display_cols], use_container_width=True)

    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API: {e}")