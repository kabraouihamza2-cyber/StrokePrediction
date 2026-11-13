# 🧠 Predictive Modeling for Stroke Risk

> *Full methodology, code output, and in-depth analysis: [`report/full_report.md`](report/full_report.md)*

An end-to-end machine learning pipeline that predicts stroke risk from patient medical records. The system is designed around one clinical principle: **it is safer to send a healthy patient for extra tests than to send a sick patient home**. Every design decision — from how missing data is handled to which metric optimizes the model — flows from that principle.

---

## 🎯 Objective & Clinical Context

Stroke is the **second leading cause of death globally** (WHO). Early detection is critical, but it is often missed because symptoms can be subtle and clinical datasets are heavily imbalanced — roughly 95% of patients in this dataset did not have a stroke.

This creates a dangerous trap called the **Accuracy Paradox**: a dumb model that always predicts "No Stroke" achieves 95% accuracy while catching **0% of actual strokes**. This project is specifically engineered to escape that trap.

**Primary metric: Recall (Sensitivity)**  
**Optimization metric: F2-Score** (weights Recall 2× more than Precision)

The AI is built as a **decision-support tool for doctors** — it flags risk, it does not make final diagnoses.

---

## 📂 Dataset

**Source:** [Stroke Prediction Dataset — Kaggle (fedesoriano)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data)

- 5,110 anonymized patient records
- 12 clinical features: age, hypertension, heart disease, average glucose level, BMI, smoking status, work type, gender, marital status, residence type, stroke (target)
- Class imbalance: **~5% stroke**, ~95% no stroke
- Selected for realistic challenges: missing values, outliers, inconsistencies — not a pre-cleaned toy dataset
- Fully anonymized — GDPR/ethics compliant

> **Dataset selection rationale:** Two prior candidates were rejected — one included only female patients (gender bias), the other was already fully preprocessed (nothing to solve).

---

## 🔬 Full Pipeline

### Phase 1 — Data Cleaning

- **Missing BMI values:** Could not be computed directly (no height/weight). Rows were NOT dropped — stroke data is too rare to waste. Solution: **Conditional Median Imputation** — BMI was estimated per age group × work type combination, preserving internal clinical variability.
- **Outlier handling:** Standard IQR clipping was rejected. Medical-range clipping was applied instead — an extreme BMI or glucose reading is a *real clinical warning sign*, not noise.
- **Duplicates:** None detected.
- **Membership constraints:** Invalid category values and capitalization inconsistencies standardized. The small "Other" gender category (~<3% of data) was removed as it was not clinically actionable in this binary context.
- **ID column** dropped — it is a non-informative identifier that could introduce artificial patterns.

---

### Phase 2 — Feature Engineering

| Feature | Description |
|---|---|
| `Glucose_Status` | Glucose binned by WHO/ADA thresholds: Normal (≤140), Prediabetic (140–200), Diabetic (>200). Assumption: random blood glucose context (documented). |
| `risk_factors_count` | Composite 0–5 score: sum of hypertension + heart disease + high glucose (≥140) + smoking (current/former) + overweight (BMI ≥25). |
| `age_bmi_interaction` | `age × bmi` — captures how BMI risk amplifies with age. |
| `age_hypertension_interaction` | `age × hypertension` — captures how hypertension risk compounds over lifetime. |
| `age_group` | Life-stage buckets (child, teen, adult, senior). |

---

### Phase 3 — Encoding

- **Binary encoding** (0/1): `gender`, `ever_married`, `Residence_type`
- **One-Hot Encoding** (with `drop_first=True` to prevent the Dummy Variable Trap): `work_type`, `smoking_status`, `Glucose_Status`, `age_group`
- Full code book created for reproducibility and transparency.

**Memory optimization applied after encoding:** `float32` / `int32` → **~48% memory reduction** (997.9 KB → 518.9 KB).

---

### Phase 4 — Feature Selection

A multi-step strategy was used — a feature was only dropped when **multiple methods agreed**:

| Step | Method | Purpose |
|---|---|---|
| 1 | **Variance Filtering** | Remove zero or near-zero variance features (>95% same value) |
| 2 | **Correlation Matrix** | Remove highly correlated pairs (r > 0.9) to fix multicollinearity |
| 3 | **Chi-Square Test** (p < 0.05) | Test statistical dependency between categorical features and stroke |
| 4 | **RFECV** (Random Forest estimator, ROC-AUC scoring) | Recursively eliminate weakest features using cross-validation |

**Features removed and why:**

| Feature | Reason |
|---|---|
| `age_hypertension_interaction` | r = 0.97 with hypertension — pure redundancy |
| `age_bmi_interaction` | r = 0.91 with age |
| `age_group_child` | Highly correlated with `work_type_children` |
| `gender` | Chi-Square p > 0.05; removed to prevent bias despite high RFE rank |
| `Residence_type` | Chi-Square p > 0.05, low RFE rank |
| `smoking_status_smokes` | p > 0.05 (too many "Unknown" entries); signal already in `risk_factors_count` |
| `work_type_Private` | High p-value, redundant |
| `age_group_senior` | Severe multicollinearity with continuous `age` — detected via SHAP post-training |

> **Note on `gender`:** It ranked #1 in RFECV but failed the Chi-Square test. It was dropped to prioritize clinically valid signals and avoid demographic bias.

> **PCA was explicitly rejected** — it would create a "black box" that doctors cannot interpret. Explainability was mandatory.

---

### Phase 5 — Handling Class Imbalance

Three strategies were compared head-to-head:

| Strategy | Stroke Recall | Stroke Precision | Notes |
|---|---|---|---|
| Baseline (no handling) | ~0–4% | — | Accuracy Paradox in full effect |
| Random UnderSampling | 84% | 12% | Deletes real patient data |
| SMOTE Oversampling | 84% | 13% | Generates synthetic patients |
| **Class Weighting** ✅ | **84%** | **13%** | Mathematically penalizes missed strokes; keeps 100% real data |

**Winner: Class Weighting** (`class_weight='balanced'`). Same recall as SMOTE, but no data fabrication and no data deletion. In a medical context, data integrity is paramount.

---

### Phase 6 — Model Training & Hyperparameter Tuning

**Setup:**
- 80% train / 20% test — **stratified split** to preserve the 5% stroke ratio in both sets
- **Stratified 5-Fold Cross-Validation** throughout
- All models trained inside **Scikit-learn pipelines** (scaling learned only from train set → prevents data leakage)
- Optimization metric: **F2-Score**

**6 models evaluated at baseline (default params):**

| Model | Accuracy | Stroke Recall |
|---|---|---|
| Logistic Regression | 95% | 0% ← Accuracy Paradox |
| KNN | 94% | 4% |
| Decision Tree | 91% | 14% |
| Random Forest | 95% | 4% |
| AdaBoost | 95% | 0% |
| Bagging | 94% | 6% |

All baseline models failed clinically. Hyperparameter tuning with `GridSearchCV` was then applied to all 6 with class weighting enabled.

**Tuning winner (Round 1):** Logistic Regression — `C=10`, `penalty=L1 (Lasso)`, `class_weight=balanced`

---

### Phase 7 — Explainability (SHAP) + Model Correction

After training, SHAP analysis revealed a critical bug: the engineered `age_group` bins (child/teen/senior) were **severely multicollinear** with continuous `age`. Logistic Regression compensated by giving `age_group_child` an artificially large positive weight — causing the model to incorrectly suggest children are at high stroke risk.

**Fix:** All categorical age group features were removed. Only continuous `age` was retained.

After retraining:

**Final Model:** Logistic Regression — `C=0.1`, `penalty=L2 (Ridge)`, `class_weight=balanced`

The model became medically coherent. SHAP confirmed the top predictors are:
1. **Age** — highest driver
2. **risk_factors_count** — engineered composite feature
3. **avg_glucose_level / Glucose_Status_Diabetic**
4. **hypertension**
5. **heart_disease**

---

### Phase 8 — Dashboard

Built with **Streamlit** — 2 pages:
- Evaluation metrics & model explanation visualizations (SHAP plots, confusion matrix)
- Interactive patient risk prediction interface

---

## 📊 Final Results

### Performance at Default Threshold (0.5)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| No Stroke | 0.99 | 0.72 | 0.84 |
| **Stroke** | **0.13** | **0.82** | **0.23** |

→ The model catches **82 out of every 100 stroke patients**.

### High-Sensitivity Mode (Threshold = 0.3)

| Class | Precision | Recall |
|---|---|---|
| Stroke | 0.09 | **0.84** |

→ Catches 2% more strokes, but significantly more false alarms and 59% overall accuracy. Available as an optional screening mode.

**Recommended operating threshold: 0.5** — strong recall, controlled false positives, stable behavior.

---

## ⚠️ Limitations

- Low precision (13%) — many healthy patients will be flagged. This is an **acceptable clinical trade-off** (extra tests vs. missed stroke), but requires medical follow-up infrastructure.
- Dataset does not specify whether glucose is fasting or random — a documented assumption was made.
- Results are dataset-specific; generalization to other hospital populations requires revalidation.
- The model is a screening tool — final diagnosis must always rest with a licensed physician.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.14 |
| Data manipulation | `pandas`, `numpy` |
| Machine learning | `scikit-learn`, `imbalanced-learn` |
| Explainability | `shap` |
| Visualization | `matplotlib`, `seaborn` |
| Dashboard | `streamlit` |

---

## 👥 Team & Contributions

| Member | Phase | Contributions |
|---|---|---|
| **EL MEHDI TERECH** | Weeks 1–2 (Team Leader) | EDA, data cleaning, feature engineering, Streamlit dashboard |
| **EL KABRAOUI HAMZA** | Weeks 1–2 | EDA, data cleaning, feature engineering, Gantt chart, presentation |
| **Salimi Achraf** | Weeks 3–4 (Domain Supervisor) | Feature selection, modeling, evaluation, explainability, code refactoring (1,100+ lines → modular system) |
| **MOHAMMED HAFIANE** | Weeks 3–4 (Domain Supervisor) | Feature selection, modeling, evaluation, explainability, presentation |

**Team rule:** Everyone is present during code execution. If a complex technique is used, the author teaches it to the team on the spot. No black boxes in the team, and no black boxes in the model.

---

## 🧭 Conclusion

This project demonstrates that a well-designed **Logistic Regression model**, guided by medical reasoning and ethical design principles, can achieve clinically meaningful stroke detection (82% recall) while remaining fully interpretable. The key lesson: in medical AI, **how you preprocess, what metric you optimize, and how you handle imbalance matters far more than which model you pick**.
