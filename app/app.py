import os
import sys
import asyncio
import pandas as pd
import streamlit as st

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.rag_functions import hybrid_retrieve_and_answer

# ------------------------------
# Streamlit Page Config
# ------------------------------
st.set_page_config(page_title="NFL RAG Q&A", layout="wide")

tab1, tab2 = st.tabs(["Ask a Question", "Monitoring"])

with tab1:
    st.title("NFL Knowledge Assistant")
    st.markdown(
        "Ask questions about NFL weekly stats, player info, or injuries. "
        "The assistant combines knowledge graph and vector retrieval for accurate answers."
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    # ------------------------------
    # User Inputs
    # ------------------------------
    user_input = st.text_input("Enter your question:")

    # ------------------------------
    # Helper: Run Async Pipeline with Caching
    # ------------------------------
    @st.cache_data(show_spinner=False)
    def run_pipeline(question, strategy="facts-first"):
        return asyncio.run(hybrid_retrieve_and_answer(question, merge_strategy=strategy))

    # ------------------------------
    # Main Query Execution
    # ------------------------------
    if st.button("Submit") and user_input:
        with st.spinner("Generating answer..."):
            try:
                result = run_pipeline(user_input)

                st.session_state.history.append(
                    {"question": user_input, "answer": result["llm_answer"]}
                )

                if "query_log" not in st.session_state:
                    st.session_state.query_log = []
                st.session_state.query_log.append({
                    "question": user_input,
                    "elapsed": result["elapsed_seconds"],
                    "cypher_used": result["cypher"],
                    "vector_hits": len(result["vector_hits"]),
                    "graph_rows": len(result["graph_rows"])
                })
                
                # Display the answer
                st.markdown("### Answer")
                st.write(result["llm_answer"])

                # Expandable details
                with st.expander("View Retrieval & KG Details"):
                    st.markdown("**Cypher Query Used:**")
                    st.code(result["cypher"] or "None")

                    st.markdown("**Cypher Parameters:**")
                    st.json(result["cypher_params"] or {})

                    st.markdown("**Vector Hits:**")
                    st.json(result["vector_hits"] or [])

                    st.markdown("**Graph Rows:**")
                    st.json(result["graph_rows"] or [])

                    st.markdown("**Elapsed Time:**")
                    st.write(f"{result['elapsed_seconds']:.2f} seconds")

            except Exception as e:
                st.error(f"Error generating answer: {e}")
    if st.session_state.history:
        st.markdown("---")
        st.subheader("Conversation History")
        for i, item in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"**Q{i}:** {item["question"]}")
            st.markdown(f"**A{i}:** {item["answer"]}")

with tab2:
    st.header("Pipeline Monitoring")
    st.markdown("Track query statistics.")

    if "query_log" in st.session_state and st.session_state.query_log:
        df = pd.DataFrame(st.session_state.query_log)
        st.dataframe(df)
        st.markdown(f"**Average response time:** {df["elapsed"].mean():.2f} seconds")
    else:
        st.info("No queries submitted yet.")