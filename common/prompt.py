import streamlit as st
from common.utils import (
    load_json,
    save_json,
)


def list_prompts(dictionary):
    return list(dictionary.keys())


def initialize_prompt():
    if "prompt_dictionary" not in st.session_state:
        st.session_state["prompt_dictionary"] = load_json(
            "./data/prompt_dictionary.json"
        )


def select_prompt(prompt_list):
    if "prompt_name" not in st.session_state:
        st.session_state["prompt_name"] = prompt_list[0]
        # index = 0
    # else:
    #     try:
    #         index = prompt_list.index(st.session_state["prompt_name"])
    #     except ValueError:
    #         st.session_state["prompt_name"] = prompt_list[0]
    #         index = 0

    with st.sidebar:
        prompt_name = st.selectbox(
            label="Select your prompt",
            options=prompt_list,
            # index=index,
            key="unique_name",
        )

    return prompt_name


def save_prompt_description(prompt_name, description):
    prompt_dictionary = load_json("./data/prompt_dictionary.json")
    prompt_dictionary[prompt_name]["description"] = description
    st.session_state.prompt_dictionary[prompt_name]["description"] = description
    save_json("./data/prompt_dictionary.json", prompt_dictionary)
    st.success(f"Prompt description updated successfully.")


def save_prompt_text(prompt_name, prompt_text):
    prompt_dictionary = load_json("./data/prompt_dictionary.json")
    prompt_dictionary[prompt_name]["prompt"] = prompt_text
    st.session_state.prompt_dictionary[prompt_name]["prompt"] = prompt_text
    save_json("./data/prompt_dictionary.json", prompt_dictionary)
    st.success(f"Prompt text updated successfully.")


def add_new_prompt(prompt_name):
    prompt_dictionary = load_json("./data/prompt_dictionary.json")

    if prompt_name in prompt_dictionary:
        st.error(
            f"Prompt '{prompt_name}' already exists. Please choose a different name."
        )
    else:
        prompt_dictionary[prompt_name] = {
            "description": "New description",
            "prompt": "New prompt text",
        }
        st.session_state.prompt_dictionary[prompt_name] = prompt_dictionary[prompt_name]
        save_json("./data/prompt_dictionary.json", prompt_dictionary)
        st.session_state.prompt_name = prompt_name
        st.success(f"New prompt '{prompt_name}' added successfully.")


def delete_prompt(prompt_name):
    prompt_dictionary = load_json("./data/prompt_dictionary.json")
    prompt_dictionary.pop(prompt_name)
    st.session_state.prompt_dictionary.pop(prompt_name)
    save_json("./data/prompt_dictionary.json", prompt_dictionary)
    st.success(f"Prompt '{prompt_name}' deleted successfully.")


def update_prompt_description(prompt_name, description):
    with st.sidebar:
        new_item = st.text_input(
            "Create a new collection:",
        )
        safe_item = re.sub(r"[^a-zA-Z0-9]+", "-", new_item.rstrip())
        st.button(
            f"Add a new collection: **{safe_item}**",
            on_click=lambda: add_item_in_list(file_name, safe_item),
        )
