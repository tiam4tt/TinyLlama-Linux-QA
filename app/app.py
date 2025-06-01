import streamlit as st
from unsloth import FastLanguageModel
import torch

# Streamlit page configuration
st.set_page_config(page_title="TinyLlama-Linux-QA", page_icon=":penguin:")

# Constants
MODEL_PATH = "tiam4tt/TinyLlama-1.1B-chat.v1.0-linux-qna"
MAX_SEQ_LENGTH = 256
PROMPT = """Below is a question relating to the Linux operating system, paired with a paragraph describing further context. Write a short, simple, concise, and comprehensive response to the question.
### Question
{}
### Context
{}
### Response
{}"""


# Initialize session state
def initialize_session_state():
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.questions = []
        st.session_state.answers = []
        st.session_state.model = None
        st.session_state.tokenizer = None
        load_model()


# Load model and tokenizer
@st.cache_resource
def load_model():
    with st.spinner("Waking up the bot..."):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        return model, tokenizer


# Answer generation
def answer_question(question, device):
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer

    inputs = tokenizer(PROMPT.format(question, "", ""), return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_SEQ_LENGTH,
            do_sample=True,
            temperature=1.0,
            top_p=0.9
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


# Parse generated output
def parse_generated_output(text):
    parts = text.split("###")
    for part in parts:
        part = part.strip()
        if part.lower().startswith("response"):
            return part[len("Response") :].strip()
    return ""


# Main app
def main():
    initialize_session_state()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    st.header("TinyLlama4Linux")

    with st.sidebar:
        if st.button("Clear Conversation", icon=":material/delete:"):
            st.session_state.questions = []
            st.session_state.answers = []
            st.rerun()

    question = st.chat_input(placeholder="Ask anything about Linux")

    if question:
        # Append user question immediately to display it first
        st.session_state.questions.append(question)
        with st.chat_message("user"):
            st.write(question)
        # Process answer and append it
        with st.spinner("Thinking..."):
            raw_answer = answer_question(question, device)
            # print(raw_answer)
            parsed_answer = parse_generated_output(raw_answer)
            if parsed_answer:
                st.session_state.answers.append(parsed_answer)
        st.rerun()

    # Display conversation history
    for ques, ans in zip(st.session_state.questions, st.session_state.answers):
        with st.chat_message("user"):
            st.write(ques)
        with st.chat_message("assistant"):
            st.write(ans)


if __name__ == "__main__":
    main()
