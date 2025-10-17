import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# --- Load the saved vectorizer and classifier ---

try:
    with open('vectorizer.pkl','rb') as f:
        vectorizer =pickle.load(f)
    with open('classifier.pkl','rb') as f:
        classifier =pickle.Load(f)
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'vectorizer.pkl' and 'classifier.pkl' are in the same directory as this script.")
    st.stop()

ps = PorterStemmer()

def preprocess_text(text):
    text = re.sub(r'Subject: ', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    stemmed_words = [ps.stem(word) for word in words if word not in set(stopwords.words('english'))]
    return " ".join(stemmed_words)

# --- Streamlit App Interface ---
st.title("📧 Email Spam Detector")
st.write(
    "Enter the text of an email below to classify it as either **Spam** or **Ham**."
)

input_email = st.text_area("Email Text:", height=200)

if st.button("classify Email"):
    if input_email.strip() == "":
        st.warning("Please enter the email text to classify.")
    else:
        # 1. Preprocess the input text
        cleaned_text = preprocess_text(input_email)

        # 2. Vectorize the cleaned text
        text_vector = vectorizer.transform([cleaned_text])

        # 3. Predict using the classifier
        prediction = classifier.predict(text_vector)[0]

        # 4. Display the result
        if prediction == 1:
            st.warning("Prediction: This email is **SPAM**. 🚨")
        else:
            st.success("Prediction: This email is **HAM** (Not Spam). ✅")
    
