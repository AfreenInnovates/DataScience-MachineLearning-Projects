import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained MobileNetV2 model and modify it for 2 classes (cats and dogs)
weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model = model.to(device)

model_path = 'mobilenet_dogs_vs_cats.pth'

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    st.write("Model loaded successfully!")
else:
    st.error("Model file not found. Please check the file path.")
    st.stop()

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def imshow(inp, title=None):
    """Helper function to unnormalize and display image."""
    inp = inp.numpy().transpose((1, 2, 0))  # Convert from tensor to numpy
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  # Unnormalize
    inp = np.clip(inp, 0, 1)  # Clip to [0, 1] range
    st.image(inp, caption=title)


class_names = ['cat', 'dog']


def predict_image(image, model, class_names):
    # Transform the image
    image_tensor = data_transforms(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)

    return class_names[preds[0]]


# Streamlit App Interface
st.title("Dog vs Cat Classifier")

# Upload an image
uploaded_file = st.file_uploader("Upload an image of a cat or dog", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file).convert('RGB')

    # Show the image before prediction
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Classifying...")

    # Make prediction
    predicted_label = predict_image(image, model, class_names)

    st.write(f"Predicted Label: **{predicted_label}**")

    # Show image and prediction side by side
    imshow(data_transforms(image), title=f'Predicted: {predicted_label}')
