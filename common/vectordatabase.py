import streamlit as st
import re
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from common.utils import (
    embeddings_model,
    save_json,
)


def load_document(uploaded_file):
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getvalue())
    if uploaded_file.name.endswith(".pdf"):
        return PDFMinerLoader(uploaded_file.name, concatenate_pages=True)
    elif uploaded_file.name.endswith(".docx"):
        return Docx2txtLoader(uploaded_file.name)
    elif uploaded_file.name.endswith(".txt"):
        return TextLoader(uploaded_file.name)
    else:
        raise ValueError(
            "Unsupported file format. Please upload a file in a supported format."
        )


def text_split_fn(loaded_doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )

    return text_splitter.split_documents(loaded_doc)


def file_uploader(collection_name):
    uploaded_file = st.file_uploader(
        label=f"Upload your own PDF, DOCX, or TXT file to collection **{collection_name}**.",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=False,
        help="Pictures or charts in the document are not recognized",
    )
    return uploaded_file


index_name = "sandbox"


@st.cache_resource
def setup_vectorstore(index_name, collection_name):
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    vectorstore = PineconeVectorStore(
        index=pc.Index(index_name),
        embedding=embeddings_model,
        namespace=collection_name,
    )
    return vectorstore


def get_filename_list(index, collection_name):
    namespace_name = collection_name
    response = index.query(
        vector=[0.0] * 3072,
        top_k=10000,
        include_metadata=True,
        namespace=namespace_name,
    )

    # Extract the distinct "source" names from the response
    sources = [result["metadata"]["source"] for result in response["matches"]]

    distinct_sources = list(set(sources))

    return distinct_sources


def remove_document(index, collection_name, document_name):
    if not document_name:
        st.error("Select a document to remove from the dropdown list")

    metadata_filter = {"source": document_name}
    namespace_name = collection_name

    try:
        # Query the index with the metadata filter
        results = index.query(
            vector=[0.0] * 3072,
            top_k=10000,
            include_metadata=True,
            filter=metadata_filter,
            namespace=namespace_name,
        )
        ids = [result.id for result in results["matches"]]
        if ids:
            index.delete(ids=ids, namespace=namespace_name)
        else:
            st.error(f"No document found with name {document_name}")

    except Exception as e:
        st.error("No document to remove from the vector database")


def setup_retriever(vectorstore, document_name, num_source, flag_mmr, _lambda_mult):
    search_kwargs = {"k": num_source}

    if document_name != "All":
        search_kwargs["filter"] = {"source": document_name}

    if flag_mmr:
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={**search_kwargs, "lambda_mult": _lambda_mult},
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    return retriever


def doc_vector_count(index, collection_name, document_name):
    if not document_name:
        st.error("Zero vector for the document.")

    metadata_filter = {"source": document_name}
    namespace_name = collection_name

    try:
        # Query the index with the metadata filter
        results = index.query(
            vector=[0.0] * 3072,
            top_k=10000,
            include_metadata=True,
            filter=metadata_filter,
            namespace=namespace_name,
        )
        ids = [result.id for result in results["matches"]]
        length = len(ids)
        st.write(f"Number of vectors in *{document_name}*: {length}")

    except Exception as e:
        pass


# Add an item to the list
def add_item_in_list(file_name, item):
    if item:
        if "collection_list" not in st.session_state:
            st.session_state.collection_list = (
                []
            )  # Initialize the collection_list if it doesn't exist
        if (
            item not in st.session_state.collection_list
        ):  # Check if the item is not already in the list
            st.session_state.collection_list.append(item)
            st.session_state.collection_list = list(
                set(st.session_state.collection_list)
            )  # Convert to set and back to list to remove duplicates
            save_json(
                file_name, st.session_state.collection_list
            )  # Save the updated list to the JSON file
            st.sidebar.success(f"**{item}** is added to the collection list.")
        else:
            st.sidebar.error(f"{item} already exists in the list.")


# Remove an item from the list
def remove_item_from_list(file_name, index, item):
    if item:
        file_list = get_filename_list(index, item)
        if file_list == []:
            st.session_state.collection_list.remove(item)
            save_json(
                file_name, st.session_state.collection_list
            )  # Save the updated list to the JSON file
            st.sidebar.success(
                f"**{item}** successfully removed from the collection list."
            )
        else:
            st.sidebar.warning(
                f"{item} has files and cannot remove. Remove all the documents from the collection first."
            )


def add_collection_input(file_name):
    with st.sidebar:
        new_item = st.text_input(
            "Create a new collection:",
        )
        safe_item = re.sub(r"[^a-zA-Z0-9]+", "-", new_item.rstrip())
        st.button(
            f"Add a new collection: **{safe_item}**",
            on_click=lambda: add_item_in_list(file_name, safe_item),
        )


# Button to remove items
def remove_collection_button(file_name, index, item):
    st.sidebar.button(
        f":red[DELETE COLLECTION: **{item}**]",
        on_click=lambda: remove_item_from_list(file_name, index, item),
    )
