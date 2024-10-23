import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load BioBERT for sequence classification
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = BertForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Function to classify blood pressure
def classify_bp(systolic, diastolic):
    input_text = f"My systolic pressure is {systolic} and diastolic pressure is {diastolic}."
    inputs = tokenizer(input_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    if predicted_class == 0:
        return "low blood pressure"
    elif predicted_class == 1:
        return "normal blood pressure"
    elif predicted_class == 2:
        return "high blood pressure"
    else:
        return "unknown condition"

# Function to provide health advice based on classification
def get_health_advice(bp_classification):
    if bp_classification == "low blood pressure":
        return ("Consider increasing your salt intake, drinking more fluids, and eating small, frequent meals. "
                "If symptoms persist, consult a healthcare provider.")
    elif bp_classification == "normal blood pressure":
        return ("Maintain a balanced diet, exercise regularly, and keep track of your blood pressure to ensure it remains normal.")
    elif bp_classification == "high blood pressure":
        return ("Reduce your salt intake, exercise regularly, manage stress, and consult a healthcare provider for further evaluation.")
    else:
        return "Consult a healthcare provider for a proper assessment."

# Streamlit app
st.title("Blood Pressure Measurement App")

# User input for blood pressure
systolic = st.number_input("Enter your systolic pressure (upper number):", min_value=0)
diastolic = st.number_input("Enter your diastolic pressure (lower number):", min_value=0)

if st.button("Submit"):
    if systolic and diastolic:
        bp_classification = classify_bp(systolic, diastolic)
        advice = get_health_advice(bp_classification)

        st.subheader("Blood Pressure Classification:")
        st.write(bp_classification)

        st.subheader("Health Advice:")
        st.write(advice)
    else:
        st.warning("Please enter valid blood pressure readings.")
