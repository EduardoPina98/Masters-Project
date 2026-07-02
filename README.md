# Master Degree Project
Machine Learning Models for Detecting Chronic Disease Progression Based on VO2max for Cardiovascular Diseases and Type 2 Diabetes using SmartWatches.

## 📌 Project Overview
This project explores the use of non-invasive health biomarkers, primarily **($VO2max$)**, alongside other physiological metrics collected via a modern consumer wearable, to build Machine Learning (ML) classification pipelines capable of predicting chronic disease progression risks.

### Key Highlights:
- **Wearable Ingestion:** Real-world tracking data gathered via a **Garmin Venu 2 Plus** smartwatch and extracted based on an open source project https://github.com/cyberjunky/python-garminconnect/tree/0.2.26 
- **Data Augmentation:** Developed fully synthetic benchmark datasets (via OpenAI ChatGPT) to evaluate models across diverse risk distributions when clinical data was limited to evaluate a fully synthetic dataset versus a dataset derived from real world data.
- **Biomarkers Analysed:** ($VO2max$), Heart Rate Variability (HRV), Resting Heart Rate (RHR), Sleep/Nocturnal recovery metrics, Body Mass Index (BMI), Respiration Rate ($RR$) and Blood Oxygen Saturation ($SpO_2$).

---

## ⚙️ Methodology & Machine Learning Pipeline
The analytical framework implemented in this project consists of four main phases:

1. **Pre-Processing & Standardisation:** Data parsing, cleaning and feature engineering (e.g., calculating Basal Metabolic Rate via the Mifflin-St. Jeor equation). Features are normalised using `RobustScaler` to handle outliers.
2. **Feature Selection:** Applied **Recursive Feature Elimination (RFE)** to isolate highly correlated and impactful physiological non-invasive features.
3. **Model Training & Hyperparameter Tuning:**
   - **Algorithms Used:** Support Vector Machine for Classification Problems(SVM/SVC) and Multiclass Logistic Regression Classifiers.
   - **Optimization:** Hyperparameter tuning using `GridSearchCV` paired with `TimeSeriesSplit` cross-validation strategy (5, 7, and 10 folds) to preserve temporal dependencies.
4. **Evaluation:** Performance benchmarks focusing on `Balanced Accuracy`, `Balanced F1-Score`, `Balanced Precision`, and `Balanced Recall` due to their relatively unbalanced datasets.

---

## 📊 Results & Key Findings

The models were rigorously benchmarked across multiple risk profiles (Healthy, Sedentary, and High Chronic Disease risk profiles). Below is a summary of the performance insights derived during the evaluation phase:

### 1. Feature Importance (RFE Outcomes)
The Recursive Feature Elimination (RFE) process successfully isolated a lean, impactful subset of biometric indicators. The features providing the highest predictive power for both CVD and T2D progression include:
*   **Primary Predictor:** $VO_2\text{max}$ (Cardiorespiratory fitness level).
*   **Autonomic & Recovery Indicators:** Heart Rate Variability (HRV) and Resting Heart Rate (RHR).
*   **Sleep Quality:** Sleep duration and nocturnal heart rate irregularities (suboptimal recovery markers).
*   **Physiological & Metabolic Brackets:** Body Mass Index (BMI), Respiration Rate ($RR$), and Blood Oxygen Saturation ($SpO_2$) trends.

### 2. Dataset Benchmarking: Fully Synthetic vs. Smartwatch Related Data
A core contribution of this thesis was benchmarking model performance across different data origins to analyse how well the classifiers generalise to genuine user habits:
* **Fully Synthetic Datasets (Unrelated to User Data):** These datasets feature highly controlled variability, allowing the models to map relationships quickly during training. However, they lack the messy, multi layered nuances of human habits.
* **Smartwatch Related Datasets:** Built around the underlying physiological structures and real world tracking metrics derived from the Garmin Venu 2 Plus, these datasets provided superior realistic context.

### 3. Model Performance Summary
*   **Cross-Validation Rigour:** Splitting the datasets into 5, 7, and 10 folds via `TimeSeriesSplit` ensured that temporal biometric habits were not disrupted, eliminating data leakage risks.
*   **Top Performer:** The **Support Vector Classifier (SVC/SVM)** consistently showed robust generalisation when optimised using `RobustScaler`, proving highly effective at distinguishing subtle physiological deviations between sedentary profiles and early-stage chronic disease profiles.
*   **Logistic Regression:** Provided clear interpretability by weighting key clinical risk boundaries. The model effectively flagged significant statistical risks associated with sharp drops in daytime $SpO_2$ (below 92%) and irregular sleeping heart rate spikes that exceeded the traditional nocturnal recovery baseline.
* **Cardiovascular Disease (CVD) Models:** The SVC model trained on **smartwatch related data** outperformed the fully synthetic baseline, boosting the *Balanced F1-Score* from **85.2% up to 91.3%** and *Balanced Accuracy* to **90.8%**. Balanced Precision also jumped to 91.2%, signalling a significant reduction in false positives.
* **Type 2 Diabetes (T2D) Models:** A similar trend was observed for T2D prediction, where the smartwatch derived dataset achieved a higher *Balanced F1-Score* of **89.9%** (compared to 87.9% on the purely synthetic set).
* **Generalization vs. Control:** While the purely synthetic data provided a reliable initialisation benchmark, the real world physiological structures embedded in the smartwatch based profiles proved significantly better at helping the models generalize during final test phases.
