# Hospital Readmission Prediction

## Summary
This project leverages electronic health record (EHR) data to predict 30‑day hospital readmissions, with a target classification accuracy of **80%**. We combine thorough data preprocessing, feature engineering, class‑imbalance handling, and hyperparameter‑tuned machine learning models to deliver a robust readmission predictor.

## 1. Dataset
- **Source:** Patient admissions data including demographics, lab results, diagnoses, medications, and readmission labels  
- **Records:** ~100 K encounters; minority class (readmitted) ~10%  
- **Features:**  
  - **Demographics:** age, gender, race  
  - **Clinical:** lab values (e.g., serum measurements), number of diagnoses, medication flags  
  - **Administrative:** insurance/payer codes  

## 2. Data Preprocessing
1. **Missing‑value strategy**  
   - **Route A (Labeling):** Mark NaNs as “Unknown” for categorical fields  
   - **Route B (Dropping):** Remove columns with > 30% missing—ultimately dropped `max_serum` and `A1C` after confirming no class‑based bias in missingness  
2. **Encoding**  
   - **Race:** one‑hot encoding (dropped reference category “Caucasian”)  
   - **Age:** mapped age ranges to ordinal integers  
   - **Gender & Medications:** binary encoding (Yes = 1 / No = 0)  
3. **Outlier Handling & Scaling**  
   - Identified outliers via proximity‑based distance measures; removed extreme values  
   - Standardized numeric features to zero mean and unit variance  
4. **Duplicate & Bias Reduction**  
   - Ensured each patient contributes only first (non‑readmitted) and last (readmitted) encounter to avoid overrepresentation  

## 3. Feature Selection & Engineering
- **Permutation Importance:** Identified “number of diagnoses” and key lab values as top predictors  
- **RFECV (Recursive Feature Elimination with CV):** Selected optimal subset of ~35 features  
- **Imbalance Handling:**  
  - Applied **SMOTE** & **SMOTE‑ENN** to synthesize minority samples and clean overlaps  
  - Introduced a random feature baseline to drop features with lower importance  

## 4. Modeling & Hyperparameter Tuning

| Model                       | Key Details                                                                                                                                                                                       | Performance                  |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------|
| **Logistic Regression**     | Baseline; L2 regularization                                                                                                                                                                        | CV Accuracy ~ 61%            |
| **Random Forest**           | 200 trees; depth tuned                                                                                                                                                                             | CV Accuracy ~ 64%            |
| **XGBoost + SMOTE‑ENN**     | RandomizedSearchCV (50 candidates, 5‑fold CV) → subsample=0.5, reg_lambda=0.5, reg_alpha=0.3, n_estimators=800, max_depth=10, learning_rate=0.01, colsample_bytree=0.9, min_child_weight=5         | **Best CV Accuracy: 66.83%** |
| **MedBERT (Transformer)**   | Early stopping at epoch 34 (best_epoch = 21); learning‑rate scheduler to decay step size                                                                                                          | Validation AUC: 0.6645       |

> **Current milestone:** ~67% CV accuracy and AUC 0.6645; **goal:** 80% classification accuracy

## 5. Results & Metrics
- **Best Cross‑Validation Accuracy (XGBoost):** 66.83%  
- **Validation AUC (MedBERT):** 0.6645  
- **Top Predictors:**  
  1. Number of diagnoses  
  2. Select lab values (e.g., serum metrics)  
  3. Medication flags (e.g., insulin, diuretics)  

## 6. Next Steps
1. **Threshold Optimization:** Tune classification threshold to maximize AUC on hold‑out set  
2. **Advanced Architectures:** Experiment with deeper transformer‑based models and fine‑tuning strategies  
3. **Automated Tuning:** Evaluate AutoML platforms (H2O, Google AutoML) vs. manual hyperparameter searches  
4. **Feature Analysis:** Visualize feature distributions, examine multicollinearity, and apply dimensionality reduction  
5. **Stakeholder Reporting:** Build a Power BI dashboard summarizing feature importances and model performance  
6. **Deployment:** Serve best model via FastAPI and containerize for reproducible testing  

## Conclusion
Through rigorous preprocessing, imbalance correction, and iterative model tuning, this pipeline demonstrates a clear path toward an **80%** readmission‑prediction accuracy. The project—documented in this repository—serves as a foundation for deploying clinically actionable ML solutions in healthcare.
