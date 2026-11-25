# UK Road Casualty Severity Classification Pipeline

A machine learning pipeline built in Python to analyze and classify road traffic collision severity using the UK Department for Transport (DfT) road casualty dataset.

---

## 📌 Project Overview
This project evaluates demographic, temporal, and environmental factors associated with road traffic accidents across the UK. By framing casualty outcome prediction as a machine learning classification task, the pipeline compares baseline statistical models against decision tree and ensemble approaches.

* **Primary Objective:** Predict collision severity (e.g., Fatal, Serious, Slight) based on road conditions, speed limits, vehicle types, and casualty demographics.
* **Key Analyses:** 
  * Exploratory data analysis (EDA) and bivariate associations between risk factors and casualty severity.
  * Multivariate classification pipelines incorporating data preprocessing, feature encoding, and class imbalance handling.
* **Core Script:** `Casualty_analysis.py`

---

## ⚙️ Methodology & Models Evaluated

The analysis compares three supervised learning algorithms:
1. **Logistic Regression:** Serves as the linear baseline for bivariate and multivariate benchmark classification.
2. **Decision Trees:** Captures non-linear decision boundaries and feature interactions.
3. **Random Forest:** Ensemble bagging technique used to minimize variance, mitigate overfitting, and extract feature importance rankings.

### Model Evaluation Metrics
Models are evaluated using:
* **Accuracy & Balanced Accuracy** (to account for class imbalance across severity tiers)
* **Precision, Recall, and F1-Score (Macro / Weighted)**
* **Confusion Matrices** for granular error analysis across collision categories

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.x
* **Data Manipulation & Analysis:** `pandas`, `numpy`
* **Machine Learning & Preprocessing:** `scikit-learn`
* **Visualization:** `matplotlib`, `seaborn`

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/GTH87/dft-casualty-classification.git
cd dft-casualty-classification