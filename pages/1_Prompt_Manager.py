import streamlit as st
import re
from common.prompt import (
    initialize_prompt,
    list_prompts,
    select_prompt,
    add_new_prompt,
    delete_prompt,
    save_prompt_description,
    save_prompt_text,
)
from common.utils import (
    download_json_file,
    upload_json_file,
)


def main():
    st.sidebar.title("Prompt Manager")

    initialize_prompt()
    prompt_list = list_prompts(st.session_state.prompt_dictionary)
    prompt_name = select_prompt(prompt_list)

    if prompt_name != st.session_state.prompt_name:
        st.session_state.prompt_name = prompt_name

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

    file_path = "./data/prompt_dictionary.json"
    download_json_file(file_path)
    upload_json_file(file_path)


if __name__ == "__main__":
    main()
