import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

MODEL_PATH = r"C:\Users\krish\Downloads\Media Bias Detection\transformer_bias_model\final_model"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model()

id2label = {
    0: "Left-Leaning",
    1: "Neutral",
    2: "Right-Leaning"
}

def predict_bias(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits.softmax(dim=-1).numpy()[0]
    pred_id = int(np.argmax(probs))
    pred_label = id2label[pred_id]
    confidence = float(probs[pred_id])

    return pred_label, confidence

st.title("📰 Media Bias Detection App")
st.write("Enter a news headline or short article text:")

text_input = st.text_area("Headline:", height=120)

if st.button("Predict Bias"):
    if len(text_input.strip()) == 0:
        st.warning("Please enter some text!")
    else:
        label, conf = predict_bias(text_input)
        st.success(f"Prediction: **{label}**  (Confidence: {conf:.3f})")
