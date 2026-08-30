# Alzheimer's Disease Prediction Using Hybrid Deep Learning

This project focuses on classifying **Alzheimer's disease stages** from **brain MRI images** using deep learning.

The final model combines **ResNet50** and **EfficientNetB3** with a **CBAM (Convolutional Block Attention Module)**. **Grad-CAM** was also used to understand which areas of the MRI images contributed to the model's predictions.

## Classes

The model classifies MRI images into four categories:

* **Mild Demented**
* **Moderate Demented**
* **Non Demented**
* **Very Mild Demented**

## Image Preprocessing

The MRI images are processed using the following steps:

* **Gaussian Blur**
* **CLAHE**
* **Resizing to 224 × 224**
* **Pixel normalization**

Gaussian Blur is applied before CLAHE to reduce noise before enhancing the image contrast.

## Model

The final model uses:

* **ResNet50**
* **EfficientNetB3**
* **CBAM attention module**
* **Feature fusion**
* **Final classification layer**

The combination of ResNet50 and EfficientNetB3 allows the model to learn features from both architectures. CBAM is used to help the model focus on important features.

## Explainability

**Grad-CAM (Gradient-weighted Class Activation Mapping)** was used to visualize the areas of the MRI images that influenced the model's predictions.

Grad-CAM visualizations were generated for the different Alzheimer's disease classes and are available in the **`explainability`** folder.

## Results

The final model was tested on **4,354 images**.

* **Accuracy:** 77.47%
* **Precision:** 81.95%
* **Recall:** 77.47%
* **F1 Score:** 77.32%
* **Balanced Accuracy:** 79.44%
* **Cohen's Kappa:** 70.03%
* **Matthews Correlation Coefficient:** 71.60%

### Class-wise Results

**Mild Demented**

* Precision: **83.83%**
* Recall: **85.95%**
* F1 Score: **84.87%**

**Moderate Demented**

* Precision: **99.28%**
* Recall: **99.69%**
* F1 Score: **99.48%**

**Non Demented**

* Precision: **90.81%**
* Recall: **50.20%**
* F1 Score: **64.65%**

**Very Mild Demented**

* Precision: **55.15%**
* Recall: **81.93%**
* F1 Score: **65.92%**

The **confusion matrix, ROC curve, accuracy graph, loss graph, and evaluation results** are available in the **`results`** folder.

## Streamlit Application

A **Streamlit application** was created to test the trained model.

The application allows the user to:

* Upload an MRI image
* Preprocess the image
* Predict the Alzheimer's disease stage
* Display prediction confidence
* Display the probability of each class

The application code is available in **`app.py`**.

## Project Structure

```text
alzheimers-disease-prediction/
│
├── README.md
├── app.py
├── train_hybrid_cbam.py
├── evaluate_final_hybrid.py
├── gradcam_all_classes.py
├── classification_report.txt
├── model_summary.txt
├── training_history.json
├── requirements.txt
│
├── models/
│   ├── hybrid_model.py
│   └── cbam.py
│
├── preprocessing/
│   └── image_preprocessing.py
│
├── results/
│   ├── accuracy_hybrid_cbam.png
│   ├── loss_hybrid_cbam.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── evaluation_results.txt
│
└── explainability/
    ├── GradCAM_MildDemented.png
    ├── GradCAM_ModerateDemented.png
    ├── GradCAM_NonDemented.png
    └── GradCAM_VeryMildDemented.png
```

## Technologies Used

* **Python 3.12.2**
* **TensorFlow 2.21.0**
* **Keras**
* **ResNet50**
* **EfficientNetB3**
* **CBAM**
* **OpenCV**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Streamlit**
* **Grad-CAM**

## Running the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

To run the Streamlit application:

```bash
streamlit run app.py
```

The trained model weights are not included in this repository because the final weights file is approximately **418 MB** and exceeds GitHub's normal file size limit.

## Dataset

The dataset contains MRI images belonging to the four Alzheimer's disease classes listed above.

The original dataset and augmented images are not included in this repository because of their size.

## Note

This project was developed as part of my **M.Tech Artificial Intelligence** coursework.

It is intended for **academic and research purposes** and should not be used as a medical diagnostic tool.
