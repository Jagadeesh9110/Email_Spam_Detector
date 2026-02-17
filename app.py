import streamlit as st
import pickle
from preprocessing import preprocess_text

# --- Sidebar ---
st.sidebar.title("Model Information")
st.sidebar.write("Model: Linear SVM")
st.sidebar.write("Vectorizer: TF-IDF (3000 features)")
st.sidebar.write("Datasets: Enron + SMS Spam")

# --- Load the saved vectorizer and classifier ---
try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please run 'python train.py' first.")
    st.stop()

# --- Streamlit App Interface ---
st.title("📧 Email Spam Detector")
st.write(
    "Enter the text of an email below to classify it as either **Spam** or **Ham**."
)

# Example Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Load Spam Example"):
        st.session_state.email_input = "Subject: FREE money!!! Click here to claim your prize now."
with col2:
    if st.button("Load Ham Example"):
        st.session_state.email_input = "Subject: Meeting Reminder. Please join the team meeting tomorrow at 10 AM."

# Input Text Area
input_email = st.text_area("Email Text:", value=st.session_state.get('email_input', ''), height=200)

if st.button("Classify Email"):
    if input_email.strip() == "":
        st.warning("Please enter the email text to classify.")
    else:
        # 1. Preprocess the input text
        cleaned_text = preprocess_text(input_email)

        # 2. Vectorize the cleaned text
        text_vector = vectorizer.transform([cleaned_text])

        # 3. Predict using the classifier
        prediction = classifier.predict(text_vector)[0]
        probability = classifier.predict_proba(text_vector)[0]
        
        # 4. Display the result
        if prediction == 1:
            st.warning(f"Prediction: This email is **SPAM**. 🚨")
            st.write(f"Confidence: {probability[1]:.2%}")
        else:
            st.success(f"Prediction: This email is **HAM** (Not Spam). ✅")
            st.write(f"Confidence: {probability[0]:.2%}")
