import streamlit as st
from common.vectordatabase import (
    get_filename_list,
)

from common.chat_history import clear_chat_history, convert_df


def sidebar_content(model_name):
    with st.sidebar:
        with st.expander("**LLM Prompt and Vector DB Sandbox**"):

            st.write(
                f"The *{model_name}*-powered AI tool is designed to enhance the efficiency by creating scripts and providing answers to document-related questions. The tool uses **retrieval augmented generation (RAG)** to help Q&A."
            )
            st.write(
                "**AI's responses should not be relied upon as accurate or error-free.** The quality of the retrieved contexts and responses may depend on LLM algorithms, RAG parameters, and how questions are asked. Harness its power but **with accountability and responsibility**."
            )
            st.write(
                "Users are strongly advised to **evaluate for accuracy** when using the tool. Read the retrieved contexts to compare to AI's responses."
            )


def setup_rag_parameters():
    with st.sidebar:
        with st.expander("⚙️ RAG Parameters"):
            st.session_state.num_source = st.slider(
                "Top N sources to view:", min_value=4, max_value=20, value=5, step=1
            )
            st.session_state.flag_mmr = st.toggle(
                "Diversity search",
                value=True,
                help="Diversity search, i.e., Maximal Marginal Relevance (MMR) tries to reduce redundancy of fetched documents and increase diversity. 0 being the most diverse, 1 being the least diverse. 0.5 is a balanced state.",
            )
            st.session_state._lambda_mult = st.slider(
                "Diversity parameter (lambda):",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.25,
            )
            st.session_state.flag_similarity_out = st.toggle(
                "Output similarity score",
                value=False,
                help="The retrieval process may become slower due to the cosine similarity calculations. A similarity score of 100% indicates the highest level of similarity between the query and the retrieved chunk.",
            )


def setup_collection(collection_list):
    # if "collection_name" not in st.session_state:
    #     st.session_state["collection_name"] = None
    #     index = 0
    # else:
    #     try:
    #         index = collection_list.index(st.session_state.collection_name)
    #     except ValueError:
    #         index = 0

    with st.sidebar:
        collection_name = st.selectbox(
            label="Select your collection",
            options=collection_list,
        )

    return collection_name


def setup_document(index, collection_name, loaded_doc=None, file_name=None):
    document_list = get_filename_list(index, collection_name)
    if file_name is not None and loaded_doc is not None:
        try:
            index = document_list.index(file_name)
        except ValueError:
            index = 0
    else:
        index = 0

    with st.sidebar:

        document_name = st.selectbox(
            "Select your document from the collection",
            document_list,
            index=index,
        )

    return document_name


def clear_chat_button(msgs):
    with st.sidebar:
        col1, col2 = st.columns(2)
        with col1:
            st.button(
                label="Clear chat history",
                use_container_width=True,
                on_click=lambda: clear_chat_history(msgs),
                help="Retrievals use your conversation history, which will influence future outcomes. Clear history to start fresh on a new topic.",
            )
        with col2:
            st.download_button(
                label="Download chat history",
                help="Download chat history in CSV",
                data=convert_df(msgs),
                file_name="chat_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
