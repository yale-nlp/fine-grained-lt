import streamlit as st

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

dataset_example_dictionary = {
    "cochrane": [
        """A total of 38 studies involving 7843 children were included. Following educational intervention delivered to children, their parents or both, there was a significantly reduced risk of subsequent emergency department visits (RR 0.73, 95% CI 0.65 to 0.81, N = 3008) and hospital admissions (RR 0.79, 95% CI 0.69 to 0.92, N = 4019) compared with control. There were also fewer unscheduled doctor visits (RR 0.68, 95% CI 0.57 to 0.81, N = 1009). Very few data were available for other outcomes (FEV1, PEF, rescue medication use, quality of life or symptoms) and there was no statistically significant difference between education and control. Asthma education aimed at children and their carers who present to the emergency department for acute exacerbations can result in lower risk of future emergency department presentation and hospital admission. There remains uncertainty as to the long-term effect of education on other markers of asthma morbidity such as quality of life, symptoms and lung function. It remains unclear as to what type, duration and intensity of educational packages are the most effective in reducing acute care utilisation.""",
        """Five trials of MSP/RESA vaccine with 217 participants were included; all five reported on safety, and two on efficacy. No severe or systemic adverse effects were reported at doses of 13 to 15 \u00b5g of each antigen (39 to 45 \u00b5g total). One small efficacy trial with 17 non-immune participants with blood-stage parasites showed no reduction or delay in parasite growth rates after artificial challenge. In the second efficacy trial in 120 children aged five to nine years in Papua New Guinea, episodes of clinical malaria were not reduced, but MSP/RESA significantly reduced parasite density only in children who had not been pretreated with an antimalarial drug (sulfadoxine-pyrimethamine). Infections with the 3D7 parasite subtype of MSP2 (the variant included in the vaccine) were reduced (RR 0.38, 95% CI 0.26 to 0.57; 719 participants) while those with the other main subtype, FC27, were not (720 participants). The MSP/RESA (Combination B) vaccine shows promise as a way to reduce the severity of malaria episodes, but the effect of the vaccine is MSP2 variant-specific. Pretreatment for malaria during a vaccine trial makes the results difficult to interpret, particularly with the relatively small sample sizes of early trials. The results show that blood-stage vaccines may play a role and merit further development."""
        ],
    "medeasi": [
        """Intervention for obese adolescents should be focused on developing healthy eating and exercise habits rather than on losing a specific amount of weight.""",
        """The liver may be enlarged, hard, or tender; massive hepatomegaly with easily palpable nodules signifies advanced disease."""
    ]
}

model_dictionary = {
     "cochrane": {
          "baseline": "ljyflores/bart_xsum_cochrane_finetune",
          "ul":       "ljyflores/bart_xsum_cochrane_ul"
     },
     "medeasi": {
          "baseline": "ljyflores/bart_xsum_medeasi_finetune",
          "ul":       "ljyflores/bart_xsum_medeasi_ul"
     }
}

st.title("Text Simplification Model")

def load(dataset_name, model_variant_name):
    tokenizer = AutoTokenizer.from_pretrained(model_dictionary[dataset_name][model_variant_name])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dictionary[dataset_name][model_variant_name])
    return tokenizer, model

def encode(text, tokenizer):
    """This function takes a batch of samples,
    and tokenizes them into IDs for the model."""
    # Tokenize the Findings (the input)
    model_inputs = tokenizer(
        [text], padding=True, truncation=True, return_tensors="pt"
    )
    return model_inputs

def predict(text, model, tokenizer):
    model_inputs = encode(text, tokenizer)
    model_outputs = model.generate(**model_inputs, max_length=768)
    return tokenizer.batch_decode(model_outputs)

def clean(s):
    return s.replace("<s>","").replace("</s>","")

# Get user input for model
dataset_option = st.selectbox(
    label = 'Dataset Selection',
    options = ["cochrane", "medeasi"],
    index = 0
    )

st.subheader("Dataset Examples")
st.caption("Try out some of these examples by copying them into the space below!")
st.text(dataset_example_dictionary[dataset_option][0])
st.text(dataset_example_dictionary[dataset_option][1])

# Get user input for text
st.subheader("Input Text to Simplify")
st.text_input("Text to Simplify:", key="text")

# Load model and run inference
if st.button("Simplify!"):
    tokenizer_baseline, model_baseline = load(dataset_option, "baseline")
    model_outputs_baseline = predict(st.session_state.text, model_baseline, tokenizer_baseline)
    f"Baseline: {clean(model_outputs_baseline[0])}"

    tokenizer_ul, model_ul = load(dataset_option, "ul")
    model_outputs_ul = predict(st.session_state.text, model_ul, tokenizer_ul)
    f"Unlikelihood Learning: {clean(model_outputs_ul[0])}"
    