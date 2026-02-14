# Breast Cancer Wisconsin (Diagnostic) - ML Classification

**BITS Pilani - ML Assignment 2** 

**Name:** Vipul Prakash Chaudhari 

**BITS-ID** 2025AA05991 

**Date:** 12 Feb 2026

---

## Problem Statement

Develop a complete machine learning solution for **binary classification** on the Breast Cancer Wisconsin (Diagnostic) dataset. The goal is to predict whether a tumor is **malignant (M)** or **benign (B)** based on 30 real-valued features computed from digitized images of fine needle aspirate (FNA) of breast mass.

This assignment demonstrates:

- Training and evaluation of 6 different ML models
- Comprehensive performance comparison using 6 metrics per model
- Deployment of an interactive Streamlit web application
- Best practices in ML pipeline design, validation, and deployment

---

## Dataset Description

**Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
**Source:** [Kaggle - UCI ML Repository](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
**Kaggle URL:** https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

### Key Characteristics:

- **Instances:** 569 samples
- **Features:** 30 real-valued features (all numeric)
- **Target Variable:** `diagnosis`
  - **M** = Malignant (positive class)
  - **B** = Benign (negative class)
- **Missing Values:** None
- **Class Distribution:**
  - Malignant: 212 (37.3%)
  - Benign: 357 (62.7%)

### Feature Groups:

The 30 features are computed for each cell nucleus and include:

1. **Mean values** (10 features): radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
2. **Standard error** (10 features): SE of the above measurements
3. **Worst values** (10 features): Mean of the three largest values

### Why This Dataset Meets Requirements:

* **Binary Classification:** Two classes (M/B) -Yes
* **≥500 Instances:** 569 samples - Yes
* **≥12 Features:** 30 numeric features-Yes
* **Clean Data:** No missing values
* **Real-world Application:** Medical diagnosis use case

---

## Models Implemented

Six machine learning models were trained and evaluated on the same dataset:

1. **Logistic Regression** (with StandardScaler)
2. **Decision Tree Classifier**
3. **K-Nearest Neighbors (KNN)** (with StandardScaler, k=5)
4. **Naive Bayes** (Gaussian)
5. **Random Forest Classifier** (100 estimators)
6. **XGBoost Classifier** (Gradient Boosting)

### Training Configuration:

- **Random State:** 42 (for reproducibility)
- **Train-Test Split:** 80-20 with stratification
- **Class Imbalance Handling:** `class_weight='balanced'` applied when imbalance ratio > 0.20
- **Preprocessing:**
  - Median imputation for missing values (if any)
  - StandardScaler for LR and KNN
  - No scaling for tree-based models (DT, RF, XGBoost)

---

## Model Comparison Table

| Model               | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression | 0.9737   | 0.9934 | 0.9535    | 0.9535 | 0.9535 | 0.9431 |
| Random Forest       | 0.9649   | 0.9923 | 0.9535    | 0.9302 | 0.9417 | 0.9229 |
| XGBoost             | 0.9649   | 0.9911 | 0.9318    | 0.9535 | 0.9425 | 0.9224 |
| KNN                 | 0.9649   | 0.9906 | 0.9535    | 0.9302 | 0.9417 | 0.9228 |
| Naive Bayes         | 0.9298   | 0.9832 | 0.9048    | 0.8837 | 0.8941 | 0.8466 |
| Decision Tree       | 0.9474   | 0.9244 | 0.9302    | 0.9070 | 0.9185 | 0.8851 |

**Notes:**

- Table shows test set performance (114 samples)
- Sorted by F1 Score (descending)
- All metrics rounded to 4 decimal places
- Metrics computed with **malignant (M) as positive class**

### Metric Definitions:

- **Accuracy:** Overall correctness
- **AUC:** Area Under ROC Curve (binary classification)
- **Precision:** True Positives / (True Positives + False Positives)
- **Recall:** True Positives / (True Positives + False Negatives)
- **F1:** Harmonic mean of Precision and Recall
- **MCC:** Matthews Correlation Coefficient (-1 to +1)

---

## Per-Model Observations

### 1. Logistic Regression **Best Overall**

- **Strengths:** Highest accuracy (97.37%), best AUC (0.9934), and top F1 score (0.9535)
- **Why it works:** Linear separability with StandardScaler normalization; well-suited for medical diagnosis with balanced precision-recall
- **Use case:** Production deployment for high-stakes medical prediction

### 2. Random Forest

- **Strengths:** Strong ensemble performance (F1: 0.9417), excellent AUC (0.9923), robust to outliers
- **Why it works:** Handles non-linear patterns via multiple decision trees; good generalization
- **Trade-off:** Slightly lower recall than Logistic Regression

### 3. XGBoost

- **Strengths:** Competitive F1 (0.9425), highest recall (0.9535), efficient gradient boosting
- **Why it works:** Advanced boosting ensures minimal false negatives (critical in cancer detection)
- **Use case:** When minimizing missed malignant cases is priority

### 4. KNN

- **Strengths:** Solid F1 (0.9417), simple interpretability, no training phase
- **Why it works:** Similar tumors cluster in feature space after scaling
- **Trade-off:** Slower prediction time; sensitive to k selection

### 5. Naive Bayes

- **Strengths:** Fast training/prediction, decent performance (F1: 0.8941)
- **Limitations:** Assumes feature independence (violated in correlated features)
- **Use case:** Baseline model or real-time applications with acceptable accuracy

### 6. Decision Tree

- **Strengths:** Fully interpretable rules, no preprocessing needed
- **Limitations:** Lowest AUC (0.9244), prone to overfitting on complex patterns
- **Use case:** When model explainability is mandatory (regulatory requirements)

### Key Insights:

- **Scaling matters:** LR and KNN benefit significantly from StandardScaler
- **Ensemble advantage:** RF and XGBoost show consistent performance across metrics
- **Recall priority:** For cancer detection, XGBoost's high recall (95.35%) minimizes false negatives
- **Production choice:** Logistic Regression offers best balance of performance, speed, and interpretability

