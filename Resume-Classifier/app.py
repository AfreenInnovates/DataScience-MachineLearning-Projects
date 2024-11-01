import streamlit as st
import pickle
import re
import PyPDF2  # For extracting text from PDFs

# Load the pre-trained models and vectorizers
knn_model_cv = pickle.load(open("Resume-Classifier/knn_model_cv.pkl", "rb"))
knn_model_tf = pickle.load(open("Resume-Classifier/knn_model_tf.pkl", "rb"))

# Load the Label Encoder for categories
category_le = pickle.load(open("Resume-Classifier/category_le.pkl", "rb"))

# Load the vectorizers
cv = pickle.load(open("Resume-Classifier/count_vectorizer.pkl", "rb"))
tf = pickle.load(open("Resume-Classifier/tfidf_vectorizer.pkl", "rb"))

category_mapping = {
    0: "Advocate", 1: "Arts", 2: "Automation Testing", 3: "Blockchain",
    4: "Business Analyst", 5: "Civil Engineer", 6: "Data Science", 7: "Database",
    8: "DevOps Engineer", 9: "DotNet Developer", 10: "ETL Developer", 11: "Electrical Engineering",
    12: "HR", 13: "Hadoop", 14: "Health and fitness", 15: "Java Developer",
    16: "Mechanical Engineer", 17: "Network Security Engineer", 18: "Operations Manager",
    19: "PMO", 20: "Python Developer", 21: "SAP Developer", 22: "Sales",
    23: "Testing", 24: "Web Designing"
}


def clean_resume(txt):
    clean_text = re.sub('http\S+\s', ' ', txt)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+\s', ' ', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text


# Function to extract text from uploaded PDF file
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


st.title("Resume Classification")

uploaded_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"])

model_type = st.selectbox("Select Model", ("CountVectorizer (KNN - 92% accurate)", "TfidfVectorizer (KNN - 97% accurate)"))

# Prediction button
if st.button("Classify Resume"):
    if uploaded_file:

        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            resume_text = str(uploaded_file.read(), "utf-8")

        cleaned_resume = clean_resume(resume_text)

        # Vectorize the input based on model selected
        if model_type == "CountVectorizer (KNN - 92% accurate)":
            input_vector = cv.transform([cleaned_resume]).toarray()
            model = knn_model_cv
        else:
            input_vector = tf.transform([cleaned_resume]).toarray()
            model = knn_model_tf

        predicted_category_index = model.predict(input_vector).item()

        predicted_category = category_mapping.get(predicted_category_index, "Unknown Category")

        st.write(f"Predicted Category: **{predicted_category}**")
    else:
        st.write("Please upload a PDF or TXT file.")
