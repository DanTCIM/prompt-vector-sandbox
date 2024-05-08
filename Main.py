import os
import re
import streamlit as st

from common.utils import (
    StreamHandler,
    PrintRetrievalHandler,
    DocProcessStreamHandler,
    load_json,
)
from common.prompt import (
    initialize_prompt,
    list_prompts,
    select_prompt,
    add_new_prompt,
    delete_prompt,
    save_prompt_description,
    save_prompt_text,
)

from common.vectordatabase import (
    index_name,
    load_document,
    text_split_fn,
    file_uploader,
    setup_vectorstore,
    setup_retriever,
    get_filename_list,
)

from common.sidebar import (
    sidebar_content,
    setup_rag_parameters,
    setup_collection,
    setup_document,
    clear_chat_button,
)

from common.chat_history import display_chat_history

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from pinecone import Pinecone

use_anthropic = True


def llm_setup(use_anthropic):

    if use_anthropic:
        model_name = "claude-3-sonnet-20240229"
    else:
        # model_name = "gpt-3.5-turbo"
        model_name = "gpt-4-turbo"  # gpt-4 seems to be slow
    return model_name


def setup_llm(use_anthropic, model_name):
    if use_anthropic:
        return ChatAnthropic(
            model_name=model_name,
            anthropic_api_key=st.secrets["ANTHROPIC_API_KEY"],
            temperature=0,
            streaming=True,
        )
    else:
        return ChatOpenAI(
            model_name=model_name,
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            temperature=0,
            streaming=True,
        )


def setup_qa_chain(llm, retriever, memory):
    return ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )


def doc_processor(llm, prompt, document, msgs):
    prompt_footer = """
    "{text}"
    """
    prompt_tmp = PromptTemplate.from_template(prompt + prompt_footer)
    process_llm_chain = LLMChain(llm=llm, prompt=prompt_tmp)
    stuff_chain = StuffDocumentsChain(
        llm_chain=process_llm_chain, document_variable_name="text"
    )
    # with st.spinner("Processing the document..."):
    #     processed = stuff_chain.invoke(document)["output_text"]
    #     msgs.add_ai_message(processed)

    container = st.empty()
    doc_process_stream_handler = DocProcessStreamHandler(container=container, msgs=msgs)
    response = stuff_chain.run(document, callbacks=[doc_process_stream_handler])
    doc_process_stream_handler.on_llm_end(response)


def prompt_manager(
    prompt_list,
):
    st.subheader("Here is the full list of prompts:")
    st.write(prompt_list)

    chosen_prompt = st.session_state.prompt_dictionary[st.session_state.prompt_name]

    st.subheader("Add a new prompt:")
    new_prompt_name_input = st.text_input(
        label="Add a new prompt",
    )
    new_prompt_name = re.sub(r"[^a-zA-Z0-9]+", "-", new_prompt_name_input.rstrip())
    st.button(
        label=f"Add a new prompt: **{new_prompt_name}**",
        on_click=lambda: add_new_prompt(new_prompt_name),
    )

    st.subheader(f"Edit prompt: {st.session_state.prompt_name}")

    prompt_description = st.text_input(
        label="Prompt description",
        value=chosen_prompt["description"],
    )
    st.button(
        label="Save prompt description",
        on_click=lambda: save_prompt_description(
            st.session_state.prompt_name, prompt_description
        ),
    )
    prompt_contents = st.text_area(
        label="Prompt contents",
        value=chosen_prompt["prompt"],
        height=700,
    )
    st.button(
        label="Save prompt texts",
        on_click=lambda: save_prompt_text(
            st.session_state.prompt_name, prompt_contents
        ),
    )

    st.divider()
    st.button(
        label=f":red[DELETE PROMPT: **{st.session_state.prompt_name}** (cannot be undone)]",
        on_click=lambda: delete_prompt(st.session_state.prompt_name),
    )


def main():
    # Start Streamlit session
    st.set_page_config(page_title="Prompt-VectorDB-Sandbox", page_icon="📖")

    # API Key Setup and Initialize Pinecone connection
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

    st.header("LLM Prompt and Vector DB Sandbox")
    st.write(
        "Click the button in the sidebar to process the **full document**.\n\n Use the chat to search for **specific areas in the document**."
    )

    model_name = llm_setup(use_anthropic)
    sidebar_content(model_name)

    # Initialize the list from the JSON file
    if "collection_list" not in st.session_state:
        st.session_state["collection_list"] = load_json("./data/collection_list.json")
    st.session_state.collection_name = setup_collection(
        st.session_state.collection_list
    )
    vectorstore = setup_vectorstore(index_name, st.session_state.collection_name)
    index = pc.Index(index_name)

    uploaded_files = file_uploader(st.session_state.collection_name)

    # Initialize session state variables
    if "curr_files" not in st.session_state:
        st.session_state.curr_files = []

    if "prev_files" not in st.session_state:
        st.session_state.prev_files = []

    if "loaded_docs" not in st.session_state:
        st.session_state.loaded_docs = []

    if uploaded_files:
        file_names = [
            uploaded_file.name for uploaded_file in uploaded_files
        ]  # Get the names of all uploaded files
        st.session_state.curr_files = file_names

    if (
        st.session_state.curr_files
        and st.session_state.curr_files != st.session_state.prev_files
    ):

        with st.spinner("Extracting text and converting to embeddings..."):
            st.session_state.loaded_docs = []
            for uploaded_file in uploaded_files:
                loader = load_document(uploaded_file)
                st.session_state.loaded_docs.append(loader.load())

                temp_list = get_filename_list(index, st.session_state.collection_name)
                file_exists = uploaded_file.name in temp_list
                if file_exists:
                    st.warning(
                        f"A document named '{uploaded_file.name}' already exists and was not uploaded to the vector database. The loaded document is ready for processing."
                    )
                else:
                    splits = text_split_fn(st.session_state.loaded_docs[-1])

                    vectorstore.add_documents(
                        documents=splits,
                        namespace=st.session_state.collection_name,
                    )
                    st.success(
                        f"'{uploaded_file.name}' is loaded and saved to the vector database. The loaded document is ready for processing."
                    )
                try:
                    os.remove(uploaded_file.name)
                except Exception as e:
                    pass

        st.session_state.prev_files = st.session_state.curr_files

    selected_doc = "All"

    selected_doc = setup_document(
        index=index,
        collection_name=st.session_state.collection_name,
        loaded_doc=(
            st.session_state.loaded_docs[0] if st.session_state.loaded_docs else None
        ),
        file_name=(
            st.session_state.loaded_docs[0] if st.session_state.loaded_docs else None
        ),
    )
    # except Exception as e:
    #     pass

    setup_rag_parameters()

    # Setup memory for contextual conversation
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=msgs,
        return_messages=True,
    )
    # Initialize the chat history
    if len(msgs.messages) == 0:
        msgs.add_ai_message("Ask questions from Vector Database!")

    # Show the chat history

    display_chat_history(msgs)

    # Retrieve and RAG chain
    # Create a retriever using the vector database as the search source
    retriever = setup_retriever(
        vectorstore=vectorstore,
        document_name=selected_doc,
        num_source=st.session_state.num_source,
        flag_mmr=st.session_state.flag_mmr,
        _lambda_mult=st.session_state._lambda_mult,
    )

    # Setup LLM and QA chain
    llm = setup_llm(use_anthropic, model_name)

    # Define Q&A chain
    qa_chain = setup_qa_chain(llm, retriever, memory)

    initialize_prompt()
    prompt_list = list_prompts(st.session_state.prompt_dictionary)
    prompt_name = select_prompt(prompt_list)

    if prompt_name != st.session_state.prompt_name:
        st.session_state.prompt_name = prompt_name

    st.sidebar.write(
        "➡️ "
        + st.session_state.prompt_dictionary[st.session_state.prompt_name][
            "description"
        ]
    )

    if st.session_state.loaded_docs:
        prompt = st.session_state.prompt_dictionary[st.session_state.prompt_name][
            "prompt"
        ]
        for i, loaded_doc in enumerate(st.session_state.loaded_docs):
            st.sidebar.button(
                label=f"Process the doc: {st.session_state.curr_files[i]}",
                use_container_width=True,
                on_click=lambda doc=loaded_doc: doc_processor(
                    llm,
                    prompt,
                    doc,
                    msgs,
                ),
                help="Processing the full document. Can take a while to complete.",
            )

        # combined_page_content = ""
        # for loaded_doc_list in st.session_state.loaded_docs:
        #     for doc in loaded_doc_list:
        #         combined_page_content += doc.page_content + "\n"

        # combined_doc = [
        #     Document(
        #         page_content=combined_page_content,
        #         metadata={"source": "Combined document"},
        #     )
        # ]

        # st.sidebar.button(
        #     label="Process all loaded documents",
        #     use_container_width=True,
        #     on_click=lambda doc=combined_doc: doc_processor(
        #         llm,
        #         prompt,
        #         doc,
        #         msgs,
        #     ),
        #     help="Processing all loaded documents as once. Can take a while to complete.",
        # )
    else:
        st.sidebar.write(
            "No document loaded. Please upload a document first to process."
        )

    # Ask the user for a question
    if user_query := st.chat_input(
        placeholder="What is your question on the document?"
    ):
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            retrieval_handler = PrintRetrievalHandler(
                st.container(),
                msgs,
                calculate_similarity=st.session_state.flag_similarity_out,
            )
            stream_handler = StreamHandler(st.empty())
            response = qa_chain.run(
                user_query, callbacks=[retrieval_handler, stream_handler]
            )

    clear_chat_button(msgs)


if __name__ == "__main__":
    main()
