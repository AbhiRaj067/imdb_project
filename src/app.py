import streamlit as st
import joblib
import os
from preprocessing import clean_text

st.set_page_config(page_title="Sentiment Analysis", page_icon="🎭", layout="wide")
st.title("🎭 Movie Review Sentiment Analysis")

model_type = st.sidebar.selectbox("Choose a Model", ["Naive Bayes", "Logistic Regression"])

@st.cache_resource
def load_models():
    return joblib.load("models/nb_model.pkl"), joblib.load("models/lr_model.pkl")

try:
    nb_model, lr_model = load_models()
    models_loaded = True
except FileNotFoundError:
    st.error("Models not found. Train models first.")
    models_loaded = False

col1, col2 = st.columns([2, 1])
with col1:
    user_input = st.text_area("Enter your movie review here:")
    if st.button("Analyze Sentiment") and user_input and models_loaded:
        cleaned_input = clean_text(user_input)
        model = nb_model if model_type == "Naive Bayes" else lr_model
        prediction = model.predict([cleaned_input])[0]
        
        if prediction == 1:
            st.success("Sentiment: Positive")
            st.balloons()
        else:
            st.error("Sentiment: Negative")

with col2:
    if os.path.exists("data/class_distribution.png"):
        st.image("data/class_distribution.png", caption="Class Distribution")
    if os.path.exists("data/pos_wordcloud.png"):
        st.image("data/pos_wordcloud.png", caption="Positive Words")
