[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AfreenInnovates/DataScience-MachineLearning-Projects/blob/main/Covid-19%20Detection/Covid-19%20Detection/covid19-detection-pytorch.ipynb)

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

The model classifies images into three categories: **Normal**, **Covid**, and **Viral Pneumonia**. You can visualize the results and metrics such as accuracy in the final cells of the notebook.

Training accuracy: 93.65 <br>
Validation/Test Accuracy: 80.303030
