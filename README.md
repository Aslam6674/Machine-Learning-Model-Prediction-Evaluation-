# ❤️ Heart Disease Prediction — ML Classification Project

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?style=flat-square&logo=scikit-learn)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow?style=flat-square&logo=googlecolab)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

> A complete end-to-end machine learning pipeline that predicts the presence of heart disease in patients using clinical data. Three classification algorithms are trained, evaluated, and compared — with Random Forest emerging as the best-performing model.

---

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide. Early and accurate prediction can save lives. This project builds a binary classification model using the UCI Heart Disease dataset to predict whether a patient has heart disease (1) or not (0) based on 13 clinical features.

| Detail | Info |
|--------|------|
| **Task** | Binary Classification |
| **Dataset** | [Heart Disease Dataset — Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) |
| **Target** | `target` (1 = Disease, 0 = No Disease) |
| **Models** | Logistic Regression, Random Forest, KNN |
| **Best Model** | ✅ Random Forest |
| **Platform** | Google Colab |

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── heart_disease_prediction.ipynb   # Main Colab notebook (all 10 cells)
├── heart.csv                        # Dataset (download from Kaggle)
├── README.md                        # This file
│
└── outputs/                        
    ├── 01_distributions.png
    ├── 02_target_balance.png
    ├── 03_correlation_analysis.png
    ├── 04_model_comparison.png
    ├── 05_roc_curves.png
    ├── 06_confusion_matrices.png
    ├── 07_feature_importance.png
    └── 08_learning_curve.png
```

---

## 📊 Dataset Description

The dataset contains **303 patient records** with **14 columns** (13 features + 1 target).

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Age of patient (years) | Continuous |
| `sex` | Sex (1 = Male, 0 = Female) | Categorical |
| `cp` | Chest pain type (0–3) | Categorical |
| `trestbps` | Resting blood pressure (mmHg) | Continuous |
| `chol` | Serum cholesterol (mg/dl) | Continuous |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = True) | Categorical |
| `restecg` | Resting ECG results (0–2) | Categorical |
| `thalach` | Maximum heart rate achieved | Continuous |
| `exang` | Exercise-induced angina (1 = Yes) | Categorical |
| `oldpeak` | ST depression induced by exercise | Continuous |
| `slope` | Slope of peak exercise ST segment | Categorical |
| `ca` | Number of major vessels (0–3) | Continuous |
| `thal` | Thalassemia type (0–3) | Categorical |
| `target` | **Heart disease present** (1 = Yes, 0 = No) | **Target** |

---

## ⚙️ Setup & Usage

### 1. Clone / Download

```bash
git clone https://github.com/Aslam6674/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Get the Dataset

- Go to [Kaggle Dataset Page](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- Download `heart.csv`
- Place it in the project root folder

### 3. Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `heart_disease_prediction.ipynb`
3. Run **Cell 1** to install dependencies
4. Run **Cell 2** — upload `heart.csv` when prompted
5. Run all remaining cells in order (Cell 3 → Cell 10)

### 4. Dependencies

All installed automatically in Cell 1. Core libraries used:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
```

---

## 🔬 Methodology

### Step 1 — Load, Explore & Preprocess
- Loaded dataset and inspected shape, dtypes, and missing values
- Found **zero missing values** — no imputation needed
- Removed **duplicate rows**
- Identified categorical vs continuous features
- Applied **StandardScaler** to continuous features for distance-based models
- Kept outliers — they are clinically valid data points

### Step 2 — Feature Engineering & Selection
- Created 4 new interaction features:
  - `age_thalach` — Age × Max Heart Rate (cardiovascular stress proxy)
  - `bp_chol_ratio` — Blood Pressure ÷ Cholesterol (vascular risk ratio)
  - `angina_oldpeak` — Exercise Angina × ST Depression (severity combo)
  - `cp_exang` — Chest Pain Type × Exercise Angina (symptom combo)
- Computed correlation of all features with target
- Dropped features with `|correlation| < 0.05` as they carry no signal

### Step 3 — Train 3 Models
| Model | Why Chosen |
|-------|-----------|
| **Logistic Regression** | Strong linear baseline; interpretable; fast |
| **Random Forest** | Handles non-linearity; robust ensemble; gives feature importance |
| **K-Nearest Neighbors** | Non-parametric; captures local patterns; distance-based |

- **80/20 stratified train/test split** (`random_state=42`)
- **5-fold stratified cross-validation** to check generalization

### Step 4 — Evaluate & Compare
All models evaluated on: Accuracy, Precision, Recall, F1 Score, ROC-AUC, CV F1

### Step 5 — Best Model Analysis
- Confusion matrix comparison across all 3 models
- Feature importance plot (Random Forest)
- Learning curve to check for overfitting

---

## 📈 Results

### Model Comparison Table

| Rank | Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | CV F1 |
|------|-------|----------|-----------|--------|----------|---------|-------|
| 🥇 1 | **Random Forest** | ~0.934 | ~0.938 | ~0.938 | ~0.938 | ~0.983 | ~0.920 |
| 🥈 2 | Logistic Regression | ~0.902 | ~0.906 | ~0.906 | ~0.906 | ~0.965 | ~0.895 |
| 🥉 3 | K-Nearest Neighbors | ~0.885 | ~0.882 | ~0.906 | ~0.893 | ~0.950 | ~0.872 |

> *Exact values will vary slightly based on dataset version and random state.*

### Key Visualizations

- **Distributions by class** — How each feature differs between disease/no-disease patients
- **Correlation heatmap** — Feature relationships and target correlations
- **ROC curves** — All 3 models on a single plot for direct comparison
- **Confusion matrices** — Side-by-side for all 3 models
- **Feature importance** — Top predictors identified by Random Forest
- **Learning curve** — Confirms Random Forest generalizes well without overfitting

---

## 🏆 Best Model — Random Forest

Random Forest came out on top because it doesn't try to fit heart disease into a straight line — it builds hundreds of decision trees and combines them, which naturally handles the messy, non-linear relationships between features like chest pain, heart rate, and ST depression. Logistic Regression fell short because it can only draw a flat boundary between sick and healthy, which just isn't realistic for something as complex as heart disease. KNN struggled once we added engineered features — distances in high-dimensional space get noisy and the model loses its ability to find truly "similar" patients. What really locked in Random Forest as the winner was its Recall score, because in a medical context, missing an actual heart disease case is far more dangerous than a false alarm, and it missed the fewest. Honestly, if this model were ever used in a real clinic, Random Forest is the one you'd trust your diagnosis to.

### Top Predictive Features
1. `cp` — Chest pain type
2. `thalach` — Maximum heart rate
3. `ca` — Number of major vessels
4. `oldpeak` — ST depression
5. `thal` — Thalassemia type
6. `age_thalach` *(engineered)* — Age × Max HR
7. `exang` — Exercise-induced angina

---

## 📂 Notebook Structure

| Cell   | Purpose |
|--------|---------|
| Cell 1 | Install libraries & imports |
| Cell 2 | Load dataset (upload prompt) |
| Cell 3 | EDA — distributions, class balance |
| Cell 4 | Preprocessing — dedup, outliers, scaling decisions |
| Cell 5 | Feature engineering + correlation-based selection |
| Cell 6 | 80/20 stratified split + StandardScaler |
| Cell 7 | Train all 3 models with full metrics |
| Cell 8 | Comparison table + bar charts + ROC curves |
| Cell 9 | Confusion matrices + feature importance + learning curve |
| Cell 10 | Final ranked table + conclusion |

---

## 🙋 Author

**Mohammad Aslam**
- GitHub: [@Aslam6674](https://github.com/Aslam6674/Aslam6674)
- LinkedIn: [aslam-mohammad-863267361](https://www.linkedin.com/in/aslam-mohammad-863267361/)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

> ⭐ If you found this project helpful, consider starring the repo!
