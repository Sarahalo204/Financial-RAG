import streamlit as st
from rag import rag, retrieval


st.set_page_config(
    page_title="Financial RAG",
    page_icon="💰",
    layout="wide"
)

st.title("Financial RAG Chatbot")

st.write("Ask financial questions about company reports.")

query = st.text_input(
    "Enter your question:"
)

if st.button("Generate Answer"):

    if query.strip() != "":

        with st.spinner("Generating answer..."):

            docs = retrieval(query, k=3)
            
            answer = rag(query, docs)

        st.subheader("Retrieved Documents")

        for i, doc in enumerate(docs):

            st.write(f"### Document {i+1}")
            st.write(doc["text"])
            st.write(f"Score: {doc['score']}")

        st.subheader("Generated Answer")

        st.success(answer)

    else:
        st.warning("Please enter a question.")