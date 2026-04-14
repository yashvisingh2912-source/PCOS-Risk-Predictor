# 🧬 PCOS Risk Predictor | Explainable ML for Early Health Screening

A machine learning-based decision support system that predicts the likelihood of **Polycystic Ovary Syndrome (PCOS)** using symptom-based and clinical features.

This project combines **predictive modeling + explainable AI (XAI)** to support early awareness and risk stratification — not medical diagnosis.

🔗 **Live Demo:** [Add your Streamlit link here]

---

## 🎯 Problem Statement

PCOS is one of the most common endocrine disorders affecting women globally, yet it often remains undiagnosed in early stages due to:

- Non-specific symptoms  
- Lack of awareness  
- Delayed clinical evaluation  

Late detection can lead to serious long-term complications such as infertility, insulin resistance, and metabolic disorders.

👉 This project aims to build a **data-driven early screening system** that can:

- Identify potential PCOS risk patterns  
- Provide interpretable predictions  
- Encourage timely medical consultation  

---

## 📊 Dataset

- **Source:** Kaggle — PCOS clinical dataset from an Indian hospital study  
- **Size:** 539 rows, 42 features after cleaning  
- **Target:** PCOS diagnosis (Yes/No)  
- **Class distribution:** 67% No PCOS, 33% PCOS  

---

## 🧠 Approach Overview

This project follows a **two-model architecture**:

### 1️⃣ Symptom-Based Model
Designed for non-clinical users, leveraging:

- Lifestyle factors  
- Visible symptoms  
- Reproductive history  

### 2️⃣ Clinical Model
Designed for medical-grade structured inputs, including:

- Hormonal markers (FSH, LH, AMH, PRL, etc.)  
- Ultrasound metrics (follicle count, endometrium)  
- Metabolic indicators (BMI, RBS, BP, etc.)  

---

## 🤖 Model Development

Three algorithms were trained and evaluated:

| Model | PCOS Recall | PCOS Precision | F1 Score |
|------|-------------|----------------|----------|
| Random Forest | 0.84 | 0.70 | 0.76 |
| Logistic Regression | 0.89 | 0.61 | 0.72 |
| XGBoost | 0.84 | 0.69 | 0.76 |

**Final model: Logistic Regression** at decision threshold 0.45.

---

### Why Recall was prioritized over Accuracy

In medical screening problems, **accuracy is misleading due to class imbalance**.

A naive model predicting “No PCOS” for all cases achieves **67% accuracy**, but completely fails at detection.

- False negatives (missed PCOS cases) are more critical than false positives  
- Therefore, the model is optimized for **high recall on the PCOS class**

---

## 🧠 Explainable AI (SHAP Integration)

To ensure transparency and interpretability, this project uses **SHAP (SHapley Additive exPlanations)**.

### What SHAP provides:

- Feature-level contribution for each prediction  
- Clear separation of:
  - Risk-increasing factors 🔴  
  - Risk-decreasing factors 🟢  
- Local explanations (per user input)  

### Why it matters:

- Makes predictions interpretable  
- Builds trust in model output  
- Helps users understand *why* they are at risk  

---

## 🖥️ App Features

### 🔹 Symptom-Based Screening
- Uses lifestyle + visible symptoms  
- No medical tests required  
- Designed for early awareness  

### 🔹 Clinical Analysis
- Uses lab + hormonal + ultrasound data  
- Provides higher precision risk estimation  

### 🔹 Risk Stratification
- 🟢 Low Risk  
- 🟡 Moderate Risk  
- 🔴 High Risk  

Each category includes actionable guidance for next steps.

### 🔹 Explainability Dashboard
- SHAP-based feature impact visualization  
- Personalized explanation per prediction  

---

## ⚙️ Modeling Pipeline

- StandardScaler + Logistic Regression (Pipeline-based architecture)  
- class_weight = "balanced" to handle class imbalance  
- Separate models for:
  - Symptom-based prediction  
  - Clinical prediction  

This ensures:
- Better generalization  
- Domain separation (lay vs clinical inputs)  
- Improved interpretability  

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **ML:** scikit-learn  
- **Explainability:** SHAP  
- **Visualization:** Matplotlib  
- **Data Processing:** pandas, numpy  
- **Model Serialization:** joblib  

---

## 🎯 Key Highlights

- Real-world healthcare ML application  
- Dual-model architecture (symptom + clinical)  
- Strong focus on **recall-oriented medical modeling**  
- Integrated **Explainable AI (SHAP)**  
- Interactive Streamlit web application  
- End-to-end ML pipeline (training → deployment)  

---

## ⚠️ Disclaimer

This project is built for **educational and awareness purposes only**.

It is **not a medical diagnostic tool**.  
Always consult a qualified healthcare professional for medical advice or diagnosis.
