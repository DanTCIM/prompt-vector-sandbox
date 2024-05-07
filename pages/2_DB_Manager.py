import streamlit as st
from common.sidebar import (
    setup_collection,
    setup_document,
)

from common.utils import (
    load_json,
)
from common.vectordatabase import (
    index_name,
    remove_document,
    get_filename_list,
    doc_vector_count,
    add_collection_input,
    remove_collection_button,
)
from pinecone import Pinecone


def main():
    st.set_page_config(page_title="Vector Database Manager")
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    st.sidebar.title("Vector Database Manager")

    # Initialize the list from the JSON file
    if "collection_list" not in st.session_state:
        st.session_state["collection_list"] = load_json("./data/collection_list.json")
    st.session_state.collection_name = setup_collection(
        st.session_state.collection_list
    )

    index = pc.Index(index_name)
    st.write(index.describe_index_stats())

    document_name = setup_document(index, st.session_state.collection_name)
    document_list = get_filename_list(index, st.session_state.collection_name)
    doc_vector_count(index, st.session_state.collection_name, document_name)

    st.button(
        label=f":red[DELETE SELECTED DOCUMENT: **{document_name}** (cannot be undone)]",
        use_container_width=True,
        on_click=lambda: remove_document(
            index, st.session_state.collection_name, document_name
        ),
        help="Remove selected document from the vector database.",
    )

    st.write(
        f"Here is the list of documents in the selected collection: **{st.session_state.collection_name}**"
    )
    st.write(document_list)
    add_collection_input("./data/collection_list.json")
    st.sidebar.divider()
    remove_collection_button(
        "./data/collection_list.json", index, st.session_state.collection_name
    )


if __name__ == "__main__":
    main()
