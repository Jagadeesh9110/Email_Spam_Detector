# 📧 Email Spam Detector

This project is a machine learning-based application designed to classify emails as either **"Spam"** or **"Ham"** (not spam). It uses a Support Vector Machine (SVM) model trained on the Enron email dataset to make predictions. The application is deployed as an interactive web app using Streamlit.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_URL_HERE)

---

##  Project Overview

The core of this project is a binary classification model. The process involves several key machine learning steps:
1.  **Data Loading & Cleaning:** The Enron email dataset is loaded, cleaned of missing values, and prepared for processing.
2.  **Text Preprocessing:** Email text is normalized by removing punctuation, converting to lowercase, removing common stop words, and applying stemming to reduce words to their root form.
3.  **Feature Extraction:** The cleaned text is converted into a numerical format using the **TF-IDF (Term Frequency-Inverse Document Frequency)** technique.
4.  **Model Training & Evaluation:** Multiple models (Naive Bayes, Logistic Regression, and SVM) were trained and evaluated. The SVM model was selected as the best performer with an accuracy of over 98%.
5.  **Deployment:** The trained TF-IDF vectorizer and SVM model were saved and deployed as a web application using Streamlit Cloud.

---

##  Features

* **Interactive Web Interface:** A simple and clean UI to paste email text and get an instant classification.
* **High Accuracy:** The underlying SVM model achieves over 98% accuracy on the test set.
* **Real-time Prediction:** Classifies new, unseen emails in real-time.

---

##  Technology Stack

* **Programming Language:** Python 3
* **Machine Learning Libraries:** Scikit-learn, NLTK
* **Data Handling:** Pandas, NumPy
* **Web Framework:** Streamlit
* **Deployment:** Streamlit Cloud, GitHub

---

##  How to Run the Project Locally

To run this application on your local machine, follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/Jagadeesh9110/email-spam-detector.git](https://github.com/Jagadeesh9110/email-spam-detector.git)
cd email-spam-detector