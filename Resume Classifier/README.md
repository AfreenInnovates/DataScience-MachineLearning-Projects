# Resume Classification App

This project is a web-based application built using **Streamlit** that classifies resumes into predefined categories such as Data Science, Python Developer, Blockchain, and others. The project utilizes machine learning models for text classification, specifically KNN classifiers, with CountVectorizer and TfidfVectorizer for feature extraction. The primary goal is to predict the profession or category of the uploaded resume.

## Problem Statement

In today's fast-paced recruitment world, manually reviewing thousands of resumes to match them with job profiles is time-consuming and inefficient. This project aims to automate the process of categorizing resumes into relevant job roles using natural language processing (NLP) techniques and machine learning algorithms. The resume text is extracted, cleaned, vectorized, and classified into one of 25 predefined categories, providing an efficient solution for recruiters.

## Project Overview

This project uses **K-Nearest Neighbors (KNN)** classifiers trained on resume data. The application takes PDF or text-based resumes as input and predicts the category of the resume based on the job role the resume fits best.

## Video Demo

<div>
    <a href="https://www.loom.com/share/ea2094576f7445d0926cba8fb4e5a2d7">
      <p>Resume Classifier - Demo - Watch Video</p>
    </a>
    <a href="https://www.loom.com/share/ea2094576f7445d0926cba8fb4e5a2d7">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/ea2094576f7445d0926cba8fb4e5a2d7-e8171c4cd40a1aaa-full-play.gif">
    </a>
  </div>

## Key Features

- **Upload PDF or Text resumes**: Users can upload resumes in PDF or TXT format.
- **Model Selection**: Users can select between two models:
  - CountVectorizer with KNN (92% accuracy)
  - TfidfVectorizer with KNN (97% accuracy)
- **Resume Text Extraction**: The app extracts text from the uploaded PDF using `PyPDF2` and cleans the text using regex.
- **Text Vectorization**: The resume text is vectorized using either CountVectorizer or TfidfVectorizer.
- **Category Prediction**: The KNN model predicts the category, which is displayed as output.

## Libraries and Requirements

To run this project, you need to install the following libraries:

```bash
pip install streamlit, PyPDF2, scikit-learn
```

Additionally, the following files are needed, which contain pre-trained models and vectorizers:
- `knn_model_cv.pkl`: Pre-trained KNN model using CountVectorizer.
- `knn_model_tf.pkl`: Pre-trained KNN model using TfidfVectorizer.
- `count_vectorizer.pkl`: Pre-trained CountVectorizer for feature extraction.
- `tfidf_vectorizer.pkl`: Pre-trained TfidfVectorizer for feature extraction.
- `category_le.pkl`: Pre-trained LabelEncoder for decoding predicted categories.

### Installation and Running the Application

1. Clone the repository to your local machine.
2. Install the required libraries using the command above.
3. Run the Streamlit app using the following command:

```bash
streamlit run app.py
```

4. Upload a resume in PDF or TXT format, select the model, and click **Classify Resume** to see the predicted job category.

## Dataset

The dataset used to train the models was sourced from [https://raw.githubusercontent.com/611noorsaeed/Resume-Screening-App/refs/heads/main/UpdatedResumeDataSet.csv](https://raw.githubusercontent.com/611noorsaeed/Resume-Screening-App/refs/heads/main/UpdatedResumeDataSet.csv). 

## Code Explanation

### Main Components:

- **Loading Models**: The pre-trained KNN models and vectorizers are loaded using the `pickle` module.
- **Text Cleaning**: The `clean_resume` function uses regex patterns to clean the text, removing URLs, special characters, and unnecessary spaces.
- **PDF Text Extraction**: The `extract_text_from_pdf` function uses `PyPDF2` to extract text from the uploaded PDF file.
- **Model Prediction**: Depending on the selected model (CountVectorizer or TfidfVectorizer), the app vectorizes the cleaned resume text and predicts the job category using the KNN classifier.
- **Category Mapping**: A dictionary (`category_mapping`) maps the model's predicted numerical label to the corresponding job category name, such as Data Science, Blockchain, or Java Developer.

## Conclusion

This project simplifies the resume screening process by automating resume classification. It leverages machine learning to reduce manual effort, allowing recruiters to quickly filter resumes by job role, enhancing recruitment efficiency. 

The use of NLP and KNN classification makes this project a robust solution for resume categorization. The app also demonstrates the practical implementation of text vectorization and classification models in a real-world application.
