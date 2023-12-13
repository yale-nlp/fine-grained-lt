import streamlit as st

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.title("Text Simplification Model")

@st.cache_data
def load(model_option):

    model_name_dict = {
        "bart": "facebook/bart-large",
        "bart_xsum": "facebook/bart-large-xsum",
        "flant5": "google/flan-t5-large",
        "flant5_base": "google/flan-t5-base",
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name_dict[model_option])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_dict[model_option])
    return tokenizer, model

@st.cache_data
def encode(text, _tokenizer):
    """This function takes a batch of samples,
    and tokenizes them into IDs for the model."""
    # Tokenize the Findings (the input)
    model_inputs = _tokenizer(
        [text], max_length=768, padding=True, truncation=True, return_tensors="pt"
    )
    return model_inputs

@st.cache_data
def predict(text, _model, _tokenizer):
    model_inputs = encode(text, _tokenizer)
    model_outputs = _model.generate(**model_inputs)
    return tokenizer.batch_decode(model_outputs)

# Get user input for model
model_option = st.selectbox(
    label = 'Model Selection',
    options = ["bart","bart_xsum","flant5","flant5_base"],
    index = 0
    )

# Get user input for text
st.text_input("Text to Simplify:", key="text")

# Load model and run inference
if st.button("Simplify!"):
    tokenizer, model = load(model_option)
    model_outputs = predict(st.session_state.text, model, tokenizer)
    model_outputs[0]