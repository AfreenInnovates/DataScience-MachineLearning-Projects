import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model
loaded_model = load_model('my_model.keras')

# Load the tokenizers
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_tokenizer.pkl', 'rb') as f:
    label_tokenizer = pickle.load(f)

# Define parameters
max_length = 200
padding_type = 'post'
trunc_type = 'post'

# Function to predict the category of the input text
def predict_text(text, model):
    # Tokenize and pad the input text
    text_sequence = tokenizer.texts_to_sequences([text])
    text_padded = pad_sequences(text_sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Predict the category
    prediction = model.predict(text_padded)
    highest_prob_index = np.argmax(prediction)

    # Get the labels from the tokenizer
    labels = list(label_tokenizer.word_index.keys())
    predicted_label = labels[highest_prob_index - 1]  # Adjust index if necessary

    return predicted_label

st.sidebar.markdown("### Note:")
st.sidebar.write("You can input any text related to sports, technology, business, politics, or entertainment, "
                 "and the model will classify it into one of these categories.")

# Text input
input_text = st.text_area("Enter your text here:", "")

# Classify button
if st.button("Classify"):
    if input_text:
        prediction = predict_text(input_text, loaded_model)
        st.success(f"The predicted category is: **{prediction}**")
    else:
        st.error("Please enter some text to classify.")