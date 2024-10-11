[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AfreenInnovates/DataScience-MachineLearning-Projects/blob/main/ASL%20Alphabet%20Recognition%20-%20CNN%20%26%20Tensorboard/asl_alphabet_recognition_(cnn_and_tensorboard).ipynb)


# **ASL Alphabet Recognition Using CNN**

## **Project Overview**

This project aims to build a Convolutional Neural Network (CNN) for recognizing the American Sign Language (ASL) alphabet from images. The model is trained on a dataset of hand gestures representing each letter in the ASL alphabet and is tested on unseen images to classify the corresponding letter.
The project includes steps for data preprocessing, model training, evaluation, and performance tracking using TensorBoard.

---

## **Dataset**

- **Source:** [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- The dataset contains labeled images of ASL hand signs for 26 letters and additional labels like "space" and "nothing."
- I split the dataset into **training**, **validation**, and **test** sets for effective model training and evaluation.

---

# **Problem Statement**

The goal of this project is to develop a machine learning model that can recognize and classify American Sign Language (ASL) alphabets from images. ASL is a vital means of communication for individuals who are deaf or hard of hearing, and automating ASL recognition can have significant real-world applications, including:
- Translating sign language into spoken language for wider communication.
- Improving accessibility for individuals with hearing impairments.
- Creating educational tools for learning ASL more effectively.

Given a dataset of images depicting hand gestures for each ASL letter, the task is to build and train a Convolutional Neural Network (CNN) that can:
- Accurately classify unseen hand gesture images into their corresponding ASL letters.
- Achieve high accuracy on both the validation and test datasets.
  
The model will be evaluated based on its **training accuracy**, **validation accuracy**, and **test accuracy** to ensure its generalization performance.

---

# **Model Performance**

## **Training Accuracy**

Throughout the training process, the model's performance is evaluated on both the training and validation datasets after every epoch. Here are the key observations from training:

- **Training Loss**: The loss decreases consistently as the model learns from the data.
- **Training Accuracy**: 
  - The model achieves high accuracy during training, indicating that it is effectively learning patterns in the dataset.
  - At the end of training, the training accuracy reaches **100%**, indicating strong learning on the training data.

## **Validation Accuracy**

Validation accuracy is a key indicator of how well the model generalizes to unseen data during the training process:

- **Validation Loss**: Similar to the training loss, the validation loss decreases during the early epochs but may fluctuate slightly as the model continues to train.
- **Validation Accuracy**: 
  - The model's validation accuracy starts at around **85%** in the first few epochs.
  - After completing training, the validation accuracy stabilizes at around **98%**, indicating good generalization without overfitting.

---

# **Summary of Accuracies**
- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~98%
- **Test Accuracy**: 100%

These metrics demonstrate that the CNN model is well-trained and performs accurately across both validation and test datasets.

# Conclusion

This project demonstrates how to build, train, and evaluate a CNN model for recognizing ASL alphabets using image data. With data augmentation and proper evaluation metrics, the model performs well, achieving a high accuracy on the test set. 

## Acknowledgements
- Dataset sourced from Kaggle ASL Alphabet Dataset.
- This project was built using PyTorch, Matplotlib, and TensorBoard.
- Learned experiment tracking from here: https://www.learnpytorch.io/07_pytorch_experiment_tracking/
