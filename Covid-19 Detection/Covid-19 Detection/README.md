[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AfreenInnovates/DataScience-MachineLearning-Projects/blob/main/Covid-19%20Detection/Covid-19%20Detection/covid19-detection-pytorch.ipynb)

# Problem Definition

The COVID-19 pandemic has had a profound impact on global health systems and economies. Early and accurate detection of COVID-19 is crucial for effective treatment, isolation, and containment of the virus. Traditional diagnostic methods, such as RT-PCR tests, are time-consuming and require specialized laboratory equipment, which may not be readily available in all settings. This project aims to develop an alternative diagnostic tool using deep learning techniques to analyze chest X-ray images for the detection of COVID-19. By leveraging image classification methods, this project seeks to provide a supplementary tool that can assist healthcare professionals in making informed decisions, thereby enhancing the overall response to the pandemic.

# COVID-19 Detection Using PyTorch

This project focuses on detecting COVID-19 using image classification techniques powered by PyTorch. The model classifies images into three categories: **Normal**, **COVID**, and **Viral Pneumonia**, using a custom convolutional neural network (CNN).

## Dataset

The dataset used in this project contains images organized into three categories:
- **Normal**
- **Covid**
- **Viral Pneumonia**

The images are stored in separate directories for training and testing.

## Requirements

Some key dependencies include:
- PyTorch
- torchvision
- opendatasets (for downloading datasets)
- tqdm (for progress tracking)

## Usage

1. **Dataset Setup**: The dataset is automatically downloaded and extracted into the project directory. You can customize the dataset path by modifying the `data_dir` variable in the notebook.
   
2. **Training the Model**: The notebook loads the dataset, defines a CNN model, and trains it to classify images. You can execute the notebook cells to run the training process.
   
3. **Evaluation**: Once the model is trained, it is evaluated on the test set, which contains images from the three categories.

## Directory Structure

```bash
├── covid19_detection_pytorch.ipynb  # Main notebook
├── data                             # Dataset directory (train/test)
├── models                           # Saved models
```

## How to Run

1. Clone the repository and navigate to the project directory:
    ```bash
    git clone <repository-url>
    cd <project-directory>
    ```

2. Run the notebook using Jupyter or Google Colab:
    ```bash
    jupyter notebook covid19_detection_pytorch.ipynb
    ```

## Results

The model classifies images into three categories: Normal, Covid, and Viral Pneumonia. You can visualize the results and metrics such as accuracy in the final cells of the notebook.

- Training Accuracy: 93.65%
- Validation/Test Accuracy: 80.30%

This project successfully developed a convolutional neural network model using PyTorch to classify chest X-ray images into Normal, COVID-19, and Viral Pneumonia categories. Achieving a training accuracy of 93.65% and a validation/test accuracy of 80.30%, the model demonstrates significant potential in assisting the detection of COVID-19. These results indicate that deep learning techniques can be effectively applied to medical imaging for rapid diagnostic purposes.
