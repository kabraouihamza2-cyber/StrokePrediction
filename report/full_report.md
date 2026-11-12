# I. Introduction

## 1. Main Objective
The primary goal of this project is to build an **AI system that predicts if a patient is at risk of a stroke** using their medical data.

## 2. Why Recall Matters More Than Accuracy
In a medical project, "accuracy" is not enough because it's a matter of life and death .
* **Priority:** Our main focus is **Recall (Sensitivity)**. This means the model is designed to catch as many true stroke cases as possible.
* **Safety:** It is better to make a mistake by sending a healthy patient for more tests than to make the mistake of sending a sick person home. Catching the risk early ensures patients get the care they need to survive.

## 3. Supporting Doctors, Not Replacing Them
This AI is built to be a **tool for doctors**, not a replacement. 
* **Reducing Errors:** Doctors are human and can sometimes miss a diagnosis. This AI helps by acting as a second check to reduce those mistakes.
* **Decision Support:** Because AI can also make mistakes, it is used to help doctors make better decisions, not to make the final choice for them.

## 4. The Mission
We are here to address the problem of missed diagnoses. By using advanced machine learning to optimize recall, we aim to provide a solution that helps as many patients as possible avoid the risk of death. Our tool is a safety net designed to save precious human lives.

## 5. Summary of the Strategy:
* **Problem:** Many stroke cases are missed in early stages.
* **Solution:** An AI model that prioritizes finding every possible case (High Recall).
* **Result:** Fewer missed diagnoses and more lives saved.





---





# II. Team Organization & Soft Skills

## 1. Team Members & Roles:
Salimi Achraf - Advanced Modeling & Evaluation (Weeks 3-4), Domain Supervisor

MOHAMMED HAFIANE - Advanced Modeling & Evaluation (Weeks 3-4), Domain Supervisor

EL MEHDI TERECH - Data Preparation & Engineering (Weeks 1-2) (Team leader)

EL KABRAOUI HAMZA - Data Preparation & Engineering (Weeks 1-2)

Although the team leader did not initially possess strong domain knowledge in AI, he played a crucial organizational role. He ensured continuous progress by monitoring task completion, reminding members about deadlines, organizing working sessions, and maintaining team motivation. His coordination guaranteed that no member was left behind and that the project advanced consistently.

### 1.1 Roles & Responsibility
Tasks are divided clearly based on the project timeline:
* **Weeks 1-2 (Data Preparation):** El Mehdi Terech and EL KABRAOUI HAMZA are responsible for the fundamental work: EDA, Data Cleaning, and Feature Engineering.
* **Weeks 3-4 (Advanced Modeling):** Achraf Salimi and Mohammed Hafiane are responsible for the complex outputs: Feature Selection, Modeling, Hyperparameter Tuning, Evaluation, and Explainability.

Because Achraf and Mohammed with prior domain knowledge, they act as active supervisors across all 4 weeks, ensuring the clinical logic stays perfectly accurate throughout the entire pipeline.

#### Final Steps 
* Mehdi developed the dashboard.
* Hamza created the Gantt chart.
* Achraf worked on code refactoring with documentation.
* Hamza and Mohammed prepared the presentation.
* The report was written by everyone.


## 2. Communication & Teaching Strategy

The hardest part of AI is explaining it to others. Our team operates on a strict rule: **everyone is present when a task is being executed.** If a team member is coding, the others observe, provide help, and offer domain knowledge when needed. If someone uses a complex math trick, they have to teach the rest of the team how it works on the spot.

## 3. Inclusion & Ethical Choices
We need diverse perspectives. Decisions about dropping or keeping medical features will be debated as a team. We view AI as a *decision-support tool* to help doctors, not replace them. Our main priority is to make sure our code is fair to all patients.



---












# III. Project Management


## 1. Planning Strategy

To successfully mix technical work with our "soft skills" requirements, we will manage the project step-by-step:

**Gantt Chart:** We used a Gantt chart to track our schedule and connect our coding tasks with our
presentation goals.

**Strict Deadline:** We have a hard deadline to finish the entire project the week before exams, so strict
time management is required.

**Phases:** We broke the project down into clear phases so we don't get stuck.

**Meetings:** We had regular team check-ins to make sure everyone understands the complex AI
concepts so we can all explain them during the presentation.

Here is how we plan to execute the project:

| Phase | Core Goal | What We Will Do |
| :--- | :--- | :--- |
| **Phase 0: Team Training** | Learning the Basics | 	Not everyone on the team knows AI or Python. We scheduled teaching sessions to train everyone on the basics (Python & Pandas). Everyone must be present so that all team members understand the code and the logic. |
| **Phase 1: Initiation** | Setup & Strategy | We will define our goals, search for a medical dataset, and set our rules for "Ethical AI" (fairness, and transparency). |
| **Phase 2: EDA & Data Cleaning** | Getting Data Ready | • *Exploration (EDA):* We will explore the data to measure how imbalanced it is. <br> • *Missing Data:* We plan to use a smart method called **Conditional Imputation** to fill in missing BMI values so we don't delete patients. <br> • *Outliers:* We will use **Clipping (Winsorization)** to cap extreme values instead of dropping them, because extreme blood pressure or BMI is a real clinical warning sign. |
| **Phase 3: Feature Engineering** | Creating New Data | • We plan to create new features (like a combined "risk factor score"). <br> • We will group numeric data (like glucose) using real medical rules (like the WHO guidelines) instead of just using random math. |
| **Phase 4: Feature Selection** | Removing Noise | We will use statistical tests (like Chi-Square) to pick the best features. **We will not use PCA (Dimensionality Reduction)** because we want doctors to easily understand our final model. |
| **Phase 5: Model & Hyperparameter Training** | Fixing Imbalance | We will build our data pipelines. To handle the fact that there are very few stroke patients, **we will avoid creating fake data (like SMOTE)**. Instead, we will use **Cost-Sensitive Learning** (using the `class_weight` technique) to punish the model harder when it misses a stroke patient. We will also use **GridSearchCV** to test and optimize different models. |
| **Phase 6: Evaluation** | Proving it Works | We will stop falling for the **"Accuracy Paradox"** (where a model looks great but misses all the sick patients). We will use the **F2-Score** to force the AI to care twice as much about finding sick patients (Recall). We will use K-Fold cross-validation to prove our model is stable. |
| **Phase 7: Explainability** | Showing Why it Works | We will use **SHAP** graphs to explain exactly *why* the AI made its decisions, proving it followed real medical logic and not just random bias. |
| **Phase 8: Dashboard Development** | Visual Decision Support | We implemented an interactive model dashboard (added on **20/04/2026**, developed over **2 days**) to present evaluation metrics, model explanations, and predictions in a clear visual interface for easier interpretation by users and stakeholders. |




## 2. Timeline Overview

| Timeline | Tasks & Milestones | Focus Area |
| :--- | :--- | :--- |
| **Week 1** | Training, Dataset Selection, Project Setup | Upskilling the team and defining the scope of the project. |
| **Week 2** | EDA, Data Cleaning, Feature Engineering | Exploring and preparing the dataset to ensure high clinical quality. |
| **Week 3** | Modeling, Hyperparameter Tuning | Building the data pipelines and optimizing the machine learning model. |
| **Week 4** | Evaluation, Explainability, Presentation | Proving the model's value mathematically and visually. |
| **Later on (Final Phase)** | Model Dashboard Development & Integration | Creating an interactive dashboard to visualize model results, evaluation metrics, and explainability outputs. *(Developed on 20/04/2026 over 2 days)* |


## 3. Gantt Chart

| Week | Task                                         | Start  | End    | Duration | Focus Area                                    |
| :--- | :------------------------------------------- | :----- | :----- | :------- | :-------------------------------------------- |
| **1**| **Team Training & Project Setup**            | Mar 1  | Mar 5  | 5 days   | Team training, project setup                  |
| **1**| **Dataset Search & Problem Definition**      | Mar 6  | Mar 7  | 2 days   | Dataset selection, problem definition         |
| **2**| **EDA & Data Understanding**                 | Mar 8  | Mar 10 | 3 days   | Exploring and understanding data              |
| **2**| **Data Cleaning & Missing Value Handling**   | Mar 11 | Mar 12 | 2 days   | Cleaning and preparing data                   |
| **2**| **Feature Engineering**                      | Mar 13 | Mar 14 | 2 days   | Creating and refining features                |
| **3**| **Feature Selection**                        | Mar 15 | Mar 16 | 2 days   | Selecting important features                  |
| **3**| **Baseline Model Training**                  | Mar 17 | Mar 18 | 2 days   | Building initial models                       |
| **3**| **Hyperparameter Tuning**                    | Mar 19 | Mar 21 | 3 days   | Optimizing model performance                  |
| **4**| **Evaluation & Cross-Validation**            | Mar 22 | Mar 24 | 3 days   | Evaluating model results                      |
| **4**| **Explainability & Final Presentation Prep** | Mar 25 | Mar 28 | 4 days   | Explaining results and preparing presentation |
| **later on** | **Model Dashboard Development** | Apr 20 | Apr 21 | 2 days   | Building and integrating the final visualization dashboard |



---










# IV. Problems, Terminology & Requirements

## 1. Core Research Problems
**Main Question:** How can we build an AI that predicts stroke risk accurately, especially when we only have a small number of stroke cases compared to healthy people?

**Sub-Questions:**
* How do we handle missing data (like BMI) through smart **Imputation** techniques, without ruining the dataset or deleting important patients?
* How can we pick the most important medical signs (Feature Selection) without squashing data together through complex math like **PCA (Dimensionality Reduction)**, which creates a "black box" that doctors can't understand?
* How do we deal with extreme **Class Imbalance** (healthy people outnumbering sick people), while ensuring our training data stays 100% real (without generating fake synthetic patients using **Oversampling or SMOTE**)?
* How can we fix overlapping or redundant features (**multicollinearity**) without breaking the model?
* What evaluation metrics (like the **F2-Score**) can we use to avoid the **"Accuracy Paradox"** and force the AI to care more about catching strokes than getting a high general accuracy score?



## 2. Key Terminology

### **The Confusion Matrix**
* **TP / TN:** Correct predictions (Positive/Negative).
* **FP (False Positive):** A false alarm (healthy patient flagged).
* **FN (False Negative):** A critical miss (sick patient sent home). **Our goal is to minimize this.**

### **Core Formulas**
* $Accuracy = \frac{TP + TN}{Total}$
* $Recall = \frac{TP}{TP + FN}$ (Our Priority)
* $F2\text{-}Score = 5 \times \frac{Precision \times Recall}{4 \times Precision + Recall}$ (Our Success Metric)

### **Statistical Concepts**
* **Correlation Matrix:** Used to find redundant data and simplify the model.
* **Chi-Square Test:** Used to find relationships between medical features and stroke risk.
* **SHAP:** Used to explain exactly *why* the AI flagged a patient (Transparency).




## 3. Technology Stack
* **Language:** Python 3.14
* **Data:** `pandas`, `numpy` 
* **ML & Metrics:** `scikit-learn`, `imblearn` (for class imbalance).
* **Visualization:** `matplotlib`, `seaborn`, `shap`.




## 4. Key Constraints & Decisions
* **Medical Logic:** Clipped outliers instead of deleting them to keep real clinical data.
* **Ethics:** Avoided PCA to keep the model explainable for doctors.
* **Teamwork:** Resolved conflicts through testing and shared knowledge; refactored 1,100+ lines of code into a modular, professional system.



---




# V. Methodology 

## 0. Project Workflow (8 Phases) (Workflow Summary)

### **Phase 1 — Data cleaning**
Fix and standardise the raw data: remove missing values, drop duplicates, and make sure all values are within valid medical ranges.

### **Phase 2 — Feature engineering**
Create new useful columns from the existing data to give the model better signals to learn from.

### **Phase 3 — Encoding and type fixing**
Convert text categories into numbers and fix column data types so the model can process them correctly.

### **Phase 4 — Feature selection**
Remove useless or redundant columns. Use correlation analysis, chi-square testing, and RFE to keep only the features that actually matter — without losing the ability to explain the model.

### **Phase 5 — Model training and tuning**
Train six models using cross-validation and tune each one with grid search to find the best settings. Models: Logistic Regression, KNN, Decision Tree, Random Forest, AdaBoost, and Bagging Classifier.

### **Phase 6 — Explainability**
Use SHAP to show exactly why the model made each prediction and which features drove it. (prevent blackbox)

### **Phase 7 — Scenario analysis**
Compare different approaches: how to handle class imbalance, which hyperparameters work best, and how stratification affects results.

### **Phase 8 — Dashboard**
Present all results in a visual, interactive dashboard.



## 1. Data Source Selection:
[Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data)
This dataset is a widely used clinical collection (hosted by fedesoriano on Kaggle). While the specific hospital source is confidential, it is modeled after real-world medical records. It contains 5,110 observations with 12 clinical features (gender, age, hypertension, heart disease, smoking status, etc.) used to predict stroke, which the WHO cites as the second leading cause of death globally.

To avoid bias, we selected datasets with balanced gender representation, since some datasets include only male or only female participants.

### 1.1 Why choosing this dataset:

This dataset was selected for several key reasons:
* **Real-world relevance:** It reflects realistic clinical data used in stroke risk assessment.

* **Balanced complexity:** It includes a moderate number of features, making it suitable for both feature engineering and model interpretation.

* **Room for preprocessing:** Unlike overly cleaned datasets, it contains missing values, inconsistencies, and outliers, allowing demonstration of full data engineering and preprocessing skills.

**Anonymized data:** Patient identity is fully protected, ensuring compliance with ethical standards and privacy requirements.

---

## 2. Data Exploration & Cleaning

Initially, we performed an exploratory analysis of the dataset to understand its structure, data types, column names, and the distribution of unique values across features.

The process followed a clear and systematic workflow:

* Data inspection using `.info()` and `.head()` to understand structure and detect inconsistencies.

---

### **2.1 Missing Value Handling**

We identified missing values in the **BMI** feature. To address this, three approaches were considered:

* **Dropping records (rejected):**
  In a medical AI context, data is highly valuable and often imbalanced. Since stroke cases are relatively rare, removing rows would lead to a significant loss of critical information.

* **Direct BMI calculation (not feasible):**
  The dataset does not include required attributes such as height and weight, making direct computation impossible.

* **Imputation (selected approach):**
  We applied **Conditional Median Imputation**, stratified by age groups and work categories.
  This method preserves the internal variability of the dataset and maintains clinically relevant relationships between lifestyle, profession, and body mass index.

---

Additional checks included:

* Verification of duplicate records (none detected)
* Assessment of temporal features (no datetime/date features found)

---

### **2.2 Membership Constraints & Data Standardization**

#### **2.2.1 Membership Constraints**

This step ensures that all values belong to valid and predefined categories. For example, a feature such as *Gender* must only contain acceptable values (e.g., Male/Female), and any invalid entry (e.g., “Dragon”) would violate this constraint.

#### **2.2.2 Standardization of Values**

This process ensures consistency in categorical representations. For instance, variations such as *“Female”*, *“female”*, *“F”*, or misspellings are unified into a single standardized label to avoid ambiguity during model training.

During this step, we observed an additional category **“Other”** in the Gender column. Since it represented a very small proportion of the dataset and was not suitable for binary encoding in this context, it was removed.

This decision is justified by a common preprocessing convention: when a category represents less than ~3% of the dataset and does not carry critical predictive value, it can be safely excluded to reduce noise.

Other features such as **stroke, smoking_status, ever_married, residence_type, work_type, heart_disease, and hypertension** were consistent and did not require correction. However, categorical variables like *work_type* and *ever_married* were later prepared for encoding.

---

### **2.3 Data Range Constraints & Outlier Handling**

We validated numerical features against medically plausible ranges:

* Age: within expected human lifespan range (valid)
* Average glucose level: detected anomalies (correct range: 50, 251)
* BMI: detected extreme values (correct range: 10, 61)

To handle outliers, we applied **medical-range clipping** instead of the traditional IQR method.
This choice was made to preserve clinically meaningful extreme values that might otherwise be incorrectly removed by purely statistical filtering.(like IQR method)

---

### **2.4 Final Preprocessing Step**

* The **ID column** was removed prior to model training, as it represents a non-informative identifier. Retaining it could introduce artificial patterns and negatively impact model generalization.

---

### **2.5 Scaling and Data Leakage Prevention**

Feature scaling will be applied within a machine learning pipeline to prevent data leakage.

Data leakage occurs when information from the test set is unintentionally used during training, leading to overly optimistic performance estimates and poor generalization to unseen data.

To avoid this, scaling parameters are learned only from the training set, and then applied to the test set through the pipeline. This ensures that the model does not “peek” at the test data during preprocessing, preserving the integrity of the evaluation process.


---


## 3. Feature engineering:

- **average glucose level**
We transformed the continuous average glucose level into clinically meaningful categories using categorical binning.

Since the dataset does not explicitly specify whether glucose measurements are fasting or random, we assumed a general random testing context, based on the distribution of values. We then applied thresholds inspired by WHO and American Diabetes Association (ADA) guidelines:

Normal: ≤ 140

Prediabetic: 140 – 200

Diabetic: > 200



- **Risk Factors Count**
We engineered a composite feature called risk_factors_count, which quantifies the cumulative health risk of a patient.

This feature aggregates several major stroke-related risk indicators:

 * Hypertension
 * Heart disease
 * High glucose level (≥ 140)
 * Smoking status (smoker or former smoker)
 * Overweight condition (BMI ≥ 25)

The resulting variable ranges from 0 to 5, representing the total number of risk conditions per patient.




- **Interaction Features**

Interaction Terms: We use multiplication x1 * x2 as the mathematical standard to represent variable interactions.
This allows the model to capture how the impact of a risk factor (like BMI) is not constant, but is instead amplified as another factor (like Age) increases.

 * Age × BMI interaction
 * Age × Hypertension interaction



 
- **Age Grouping**
We discretized age into meaningful life stages to capture differences in stroke risk across population groups.



---

## 4. Data Encoding

To prepare categorical variables for machine learning algorithms, appropriate encoding strategies were applied based on feature characteristics.

### **4.1 Binary Encoding**

Binary encoding was used for variables containing two categories:

* **ever_married** (Yes / No)
* **gender** (Male / Female)
* **Residence_type** (Urban / Rural)

Each category was transformed into numerical values *(0 or 1)* to allow model compatibility while preserving interpretability.

---

### **4.2 One-Hot Encoding**

For multi-category variables, **One-Hot Encoding** was applied to prevent artificial ordinal relationships between categories.

Encoded features:

* **work_type**
  *(Private, Self-employed, Govt_job, children, Never_worked)*

* **smoking_status**
  *(formerly smoked, never smoked, smokes, Unknown)*

Before encoding, category names were standardized:

```python
df["smoking_status"] = df["smoking_status"].str.replace(" ","_")
```

One-Hot Encoding was then performed:

```python
df = pd.get_dummies(
    df,
    columns=['work_type', 'smoking_status', "Glucose_Status", "age_group"],
    drop_first=True,
    dtype=int
)
```

* `drop_first=True` prevents multicollinearity (Dummy Variable Trap).
* `dtype=int` ensures numerical representation (0/1) instead of boolean values.

---

### **4.3 Code Book (Encoding Documentation)**

To guarantee transparency and reproducibility, a **code book** was created to document all encoding decisions and feature meanings.

The code book includes:

* Target variable interpretation
* Binary feature mappings
* One-Hot baseline categories
* Engineered feature definitions

This documentation ensures that model outputs remain interpretable and reproducible.

---

## 5. Memory Optimization

After preprocessing, memory usage was optimized to improve computational efficiency without sacrificing model performance.

### **5.1 Applied Optimizations**

* Binary columns → converted to **int32**
* Continuous numerical features → converted to **float32**
* Target variable (`stroke`) → converted to **float32**

Using `float32` allows storage of **probability predictions** instead of strict binary integers during model evaluation and deployment.

---

### **5.2 Results**

| Stage               | Memory Usage |
| ------------------- | ------------ |
| Before Optimization | 997.9 KB     |
| After Optimization  | 518.9 KB     |

This represents **≈48% memory reduction**.

---

### **5.3 Why Not `uint8`?**

Unsigned integers (`uint8`) were intentionally avoided because:

* Many ML frameworks expect **signed integer types** (`int32` / `int64`)
* Unsigned formats may create compatibility issues during training or evaluation pipelines

---






## 6. Feature Selection

Feature selection was performed to reduce noise, prevent overfitting, and improve model interpretability while preserving medically relevant information.

Rather than relying on a single technique, a **multi-step validation strategy** was adopted to ensure that removed variables were truly non-informative.

---

### **6.1 Feature Selection Strategy**

Four complementary approaches were combined:

1. **Variance Filtering**
2. **Correlation Analysis (Numerical Features)**
3. **Chi-Square Test of Independence (Categorical Features)**
4. **Recursive Feature Elimination with Cross-Validation (RFECV)**

Features were **not removed immediately** after one test.
A feature was discarded only when multiple methods confirmed low predictive contribution.



---




### **6.2 Variance Filtering**

Variance filtering identifies features that contain little or no information.

#### **6.2.1 Zero-Variance Features**

A feature where all observations share identical values cannot distinguish between classes.

Example:

```
0000000000
1111111111
```

Such features were automatically removed.

---

#### **6.2.2 Near-Zero Variance Features**

Features dominated by a single value (e.g., 95–99% identical observations) provide minimal predictive information.

Example:

```
0 0 0 0 0 0 0 0 1
```

Detection rule:

```python
top_freq = df[col].value_counts(normalize=True).iloc[0]
if top_freq > 0.95:
    near_zero_cols.append(col)
```

These variables were marked as potential noise candidates.

---






### **6.3 Correlation Analysis (Multicollinearity Check)**

For numerical variables, a correlation matrix was used to detect **multicollinearity**.

Highly correlated features introduce redundancy and may:

* inflate coefficients,
* destabilize models,
* reduce generalization performance.

#### **6.3.1 Removed Redundant Features**

| Feature                      | Reason                                    |
| ---------------------------- | ----------------------------------------- |
| age_hypertension_interaction | 0.97 correlation with hypertension        |
| age_bmi_interaction          | 0.91 correlation with age                 |
| age_group_child              | Highly correlated with work_type_children |

When two variables conveyed nearly identical information, only one representative feature was retained.







---

### **6.4 Chi-Square Test of Independence**

The **Chi-Square test** was applied to evaluate relationships between categorical features and the target variable (`stroke`).

#### **6.4.1 Hypotheses**

* **H₀ (Null Hypothesis):** Feature and stroke occurrence are independent.
* **H₁ (Alternative Hypothesis):** A dependency exists.

The test compares:

* **Observed frequencies**
* **Expected frequencies under independence**

A large deviation indicates a meaningful relationship.

---

#### **6.4.2 Decision Rule**

```
  A p-value is the probability that the connection you see between a feature and a stroke is random
      rather than a real relationship.
```

* **p-value < 0.05** → Feature retained (statistically significant)
* **p-value > 0.05** → Feature considered noise

Additional validity checks:

* Expected cell frequency ≥ 5 *(Rule of 5)*
* Nominal categorical variables
* Independent observations (each row represents one patient)



#### **6.4.3 Assumptions Verification**

Before applying the Chi-Square test, its statistical assumptions were verified:

* **Nominal Data Requirement**
  The variable *smoking_status* was treated as nominal rather than ordinal.
  No evidence exists to establish a meaningful ranking between categories (e.g., *formerly smoked* cannot be assumed less or greater than *smokes*) because detailed exposure information such as smoking duration or quantity was unavailable.

* **Rule of 5 (Expected Frequency Condition)**
  The expected frequency of each contingency table cell was verified to be ≥ 5, ensuring validity of the Chi-Square approximation.

* **Independence of Observations**
  Each dataset row represents a unique patient, therefore observations are statistically independent.

Since all assumptions were satisfied, the Chi-Square test was considered appropriate for evaluating relationships between categorical variables and stroke occurrence.



#### **6.4.4 Implementation logic:**
```python
chi2, p, dof, expected = chi2_contingency(contingency_table)

if (expected < 5).any():
    continue

if p <= 0.05:
    keep.append(col)
```

---

#### **6.4.5 Clinical Interpretation Example**

Although smoking is clinically associated with stroke risk, the variable **smoking_status_smokes** showed:

* Non-significant Chi-Square result (p > 0.05)
* Low importance ranking in RFECV

This outcome is likely caused by:

* High proportion of *Unknown* entries
* Predictive information already captured by stronger physiological variables such as:

  * hypertension
  * risk_factors_count

This highlights an important principle:

> Statistical relevance is dataset-dependent even when clinical relevance exists.







---

### **6.5 Recursive Feature Elimination with Cross-Validation (RFECV)**

To validate statistical findings, **RFECV** was applied.

#### Methodology

RFECV:

1. Starts with all features
2. Trains the model
3. Ranks feature importance
4. Iteratively removes the least informative features
5. Uses cross-validation to determine the optimal subset

---

#### Evaluation Metric: ROC-AUC

Accuracy was avoided due to class imbalance.

**ROC-AUC** was selected because it measures the model’s ability to distinguish between:

* Stroke
* No Stroke

independent of classification threshold.

This metric is more appropriate for medical risk prediction.

---

#### Why Random Forest as the RFE Estimator?

Random Forest was chosen because it:

* captures **non-linear relationships**
* models **feature interactions**
* requires minimal assumptions about data distribution

Stroke risk factors often interact in complex ways (e.g., Age × Glucose), making tree-based models suitable for feature ranking.

---





### **6.6 Final Feature Removal Decision**

Only features identified as weak across multiple methods were removed:

```python
noise_cols = [
    'gender',
    'Residence_type',
    'smoking_status_smokes',
    'work_type_Private'
]

```

| Status | Feature | Justification |
| --- | --- | --- |
| **KEEP** | `age`, `bmi`, `avg_glucose_level` | Core biological drivers of stroke risk. |
| **KEEP** | `risk_factors_count` | Engineered feature summarizing overall risk; high RFE ranking. |
| **KEEP** | `hypertension`, `heart_disease` | Statistically significant ($p < 0.05$) and essential clinical indicators. |
| **KEEP** | `Glucose_Status_Diabetic` | Captures the clinical threshold (>200 mg/dL) more effectively than continuous data, ensuring model stability for Logistic Regression. |
| **DROP** | `gender` | Failed Chi-Square ($p > 0.05$) despite RFE Rank #1; removed to ensure clinical validity (prioritizing medical signals over random patterns) and avoid bias. |
| **DROP** | `age_group_senior` | Initially kept, but later dropped due to severe multicollinearity with continuous `age` identified during SHAP analysis. |
| **DROP** | `Residence_type` | Failed Chi-Square ($p > 0.05$) and ranked low by RFECV. |
| **DROP** | `smoking_status_smokes` | Failed Chi-Square ($p > 0.05$) due to high "Unknown" data volume; its predictive influence is already captured by `hypertension` and `risk_factors_count`. |
| **DROP** | `work_type_Private` | High p-value indicating redundancy with other features. |



1. **Statistical Noise Reduction:** We utilized the **Chi-Square Test of Independence** to identify and remove features (like `Residence Type` and `Gender`) that failed to show a statistically significant relationship ($p > 0.05$) with the target variable, preventing the model from learning random noise.

2. **Multicollinearity & Interpretability:** We eliminated redundant interaction terms and categorical age groups that caused severe multicollinearity (detected via Correlation Matrix and **SHAP analysis**), ensuring the model follows logical clinical pathways.

3. **Recursive Optimization:** We applied **RFECV** (Recursive Feature Elimination) using a `balanced` class weight and `ROC-AUC` scoring to identify the optimal feature subset that best balances sensitivity (Recall) and model stability.




## 7. Model training & Scenario analysis

### 7.1 Training Strategy

The model training phase aimed to develop a reliable Artificial Intelligence system capable of predicting stroke risk while prioritizing **patient safety**.
Since the dataset is highly imbalanced (approximately 5% stroke cases), traditional accuracy-based optimization was considered inappropriate.

Instead, the training process was designed around the following principles:

* **Maximize Recall (Sensitivity)** to detect as many stroke cases as possible.
* **Prevent data leakage** during validation.
* **Ensure fair comparison** between multiple machine learning algorithms.
* **Maintain reproducibility** using fixed random seeds.

To achieve this, a structured experimental workflow was implemented.

---

### 7.2 Data Splitting Strategy

Two splitting strategies were compared:

* Standard random train-test split
* **Stratified train-test split**

Stratification preserves the original stroke ratio across training and testing datasets.
This step is critical in imbalanced medical datasets because random splitting may accidentally remove minority-class samples from validation sets.

The final approach used:

* **80% Training set**
* **20% Testing set**
* **Stratified sampling based on stroke outcome**

Additionally, model validation used **Stratified K-Fold Cross Validation (k = 5)** to ensure robust and unbiased performance estimation.

---

### 7.3 Pipeline-Based Training

All models were trained using Scikit-learn pipelines.

Pipelines guarantee that:

* feature scaling is learned **only from training data**,
* transformations are applied consistently across folds,
* validation data remains unseen during training.

For experiments involving class imbalance techniques, an **Imbalanced-Learn Pipeline** was used because sampling methods modify the number of observations and cannot be handled by standard pipelines.

---

### 7.4 Baseline Model Comparison

Before optimization, several machine learning algorithms were evaluated using their default configurations:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* AdaBoost
* Bagging Classifier

This baseline step served two purposes:

1. Establish a reference performance level.
2. Demonstrate the **Accuracy Paradox**, where models achieve high accuracy by predicting the majority class while failing to detect stroke patients.

Results confirmed that default models achieved high accuracy but extremely low recall for stroke detection, making them clinically unsafe.



baseline_all_models_no_tuning(X_train, X_test, y_train, y_test)

- Output:
```
Logistic Regression:
                precision    recall  f1-score   support

    No Stroke       0.95      1.00      0.97       972
    Stroke          0.00      0.00      0.00        50

    accuracy                            0.95      1022
    macro avg       0.48      0.50      0.49      1022
    weighted avg    0.90      0.95      0.93      1022
```

```
K-Nearest Neighbors:
                precision    recall  f1-score   support

    No Stroke       0.95      0.99      0.97       972
    Stroke          0.18      0.04      0.07        50

    accuracy                            0.94      1022
    macro avg       0.57      0.52      0.52      1022
    weighted avg    0.91      0.94      0.93      1022



Decision Tree:
                precision    recall  f1-score   support

    No Stroke       0.96      0.95      0.95       972
    Stroke          0.13      0.14      0.13        50

    accuracy                            0.91      1022
    macro avg       0.54      0.55      0.54      1022
    weighted avg    0.92      0.91      0.91      1022



Random Forest:
                precision    recall  f1-score   support

    No Stroke       0.95      0.99      0.97       972
    Stroke          0.25      0.04      0.07        50

    accuracy                            0.95      1022
    macro avg       0.60      0.52      0.52      1022
    weighted avg    0.92      0.95      0.93      1022



AdaBoost:
                precision    recall  f1-score   support

    No Stroke       0.95      1.00      0.97       972
    Stroke          0.00      0.00      0.00        50

    accuracy                            0.95      1022
    macro avg       0.48      0.50      0.49      1022
    weighted avg    0.90      0.95      0.93      1022



Bagging:
                precision    recall  f1-score   support

    No Stroke       0.95      0.99      0.97       972
    Stroke          0.21      0.06      0.09        50

    accuracy                            0.94      1022
    macro avg       0.58      0.52      0.53      1022
    weighted avg    0.92      0.94      0.93      1022

```

- Interpretation (the accuracy paradox):
```
Notice how all the default models boast a massive 95% Accuracy. However, look at the Recall for 'Stroke':
- KNN and AdaBoost literally caught 0% of Strokes.
- Logistic Regression and Random Forest caught 2% of Strokes.
- Decision Tree caught 10% of Strokes.
Because the dataset is heavily imbalanced (Strokes ~5%), the models are blindly predicting "No Stroke" for almost everyone. 
This is why Accuracy is a misleading ("liar") metric in medical datasets. We must optimize for Recall.
```











---

### 7.5 Handling Class Imbalance

Because stroke cases are rare, three imbalance-handling strategies were compared:

#### 7.5.1. Random UnderSampling

Randomly removes majority-class samples to balance classes.

* Advantage: improves minority detection.
* Limitation: loss of valuable patient information.

#### 7.5.2. SMOTE Oversampling

Synthetic Minority Oversampling Technique generates artificial stroke samples.

* Advantage: improves recall without deleting data.
* Limitation: introduces synthetic observations.

#### 7.5.3. Algorithm-Level Class Weighting

Assigns higher penalties to misclassified stroke cases within the loss function.

* Advantage: preserves original data integrity.
* No artificial data generation required.

Experimental results showed that **Class Weighting** achieved performance comparable to SMOTE while maintaining real clinical data.
Therefore, class weighting was selected as the preferred strategy.



compare_sampling_methods(X_train, X_test, y_train, y_test)

- Output:

1. UnderSampling (Randomly deleting healthy patients):

```
              precision    recall  f1-score   support

  No Stroke       0.99      0.67      0.80       972
  Stroke          0.12      0.84      0.20        50

  accuracy                            0.68      1022
  macro avg       0.55      0.76      0.50      1022
  weighted avg    0.95      0.68      0.77      1022
```



2. OverSampling (Creating synthetic stroke patients with SMOTE):
```
                precision    recall  f1-score   support

    No Stroke       0.99      0.70      0.82       972
    Stroke          0.13      0.84      0.22        50

    accuracy                            0.71      1022
    macro avg       0.56      0.77      0.52      1022
    weighted avg    0.95      0.71      0.79      1022
```



3. Class Weighting (Penalizing mistakes mathematically):
```
                precision    recall  f1-score   support

    No Stroke       0.99      0.70      0.82       972
    Stroke          0.13      0.84      0.22        50

    accuracy                            0.71      1022
    macro avg       0.56      0.77      0.52      1022
    weighted avg    0.95      0.71      0.79      1022

```

- Interpretation:
```
By utilizing sampling and class weighting, we force the models to learn what a stroke actually looks like.
- Recall skyrockets from 2% to ~82-84%. The models are successfully identifying the vast majority of real strokes.
- The cost is Precision: Precision drops to ~12-13%. Because our models are now highly sensitive to strokes, they misdiagnose many healthy people.

In healthcare, a false positive (low precision) means extra medical tests, which is inconvenient. A false negative (low recall) means sending a dying patient home. 
Therefore, this trade-off is often ideal.

Which is best?
Oversampling (SMOTE) performed marginally better (0.84 recall vs 0.82 recall).
However, **Class Weighting is the preferred approach for this project**. 

Why? Because Class Weighting achieves near-identical performance relying purely 
on mathematics and the algorithm's loss function. It doesn't require us to delete real,
valuable data about healthy patients, and it doesn't require us to fabricate imaginary
, synthetic stroke patients to artificially balance the ratio. In a sensitive medical context,
retaining the integrity of the original patient records is ideal.

Both SMOTE and Class Weighting are excellent choices, but Class Weighting is cleaner and more medically sound here.
```



---

### 7.6 Evaluation Metric Selection

Model optimization used the **F2-score** instead of accuracy or F1-score.

The F2-score emphasizes Recall twice as much as Precision:

[
F_2 = \frac{(1+2^2)\cdot Precision \cdot Recall}{4\cdot Precision + Recall}
]

This choice reflects medical priorities:

* False positives → additional medical tests.
* False negatives → missed stroke diagnosis.

Thus, maximizing recall provides safer clinical outcomes.

---

### 7.7 Hyperparameter Optimization

Hyperparameter tuning was conducted using **GridSearchCV** combined with stratified cross-validation.

The following models were optimized:

* Logistic Regression:
    - Definition: Estimates stroke probability by fitting a linear separator between classes.

    - classifier__C: Controls regularization; smaller values penalize large weights more (prevents overfitting).
      It's basically the alpha in Ridge and Lasso.

    - classifier__penalty: L1 (Lasso) uses absolute values; L2 (Ridge) uses squares.
      It's just how the loss function is written; adds a penalty to the loss function.

    - class_weight='balanced': Automatically weights the minority class (Stroke) higher to handle imbalance.



* KNN (K-Nearest Neighbors):
    - weights: 'uniform' (equal weight to all neighbors) vs 'distance' (closer neighbors have more influence).
    - metric: Calculation method for "closeness" (Euclidean = straight line, Manhattan = grid-like path).



* Decision Tree:
    - Definition: A binary tree structure that splits data into groups based on feature rules (e.g., feature <, > split_point).

    - max_depth: Limits how deep the tree grows to prevent it from "memorizing" (overfitting) the data.

    - min_samples_split: The minimum patients required before the tree is allowed to split further.

    - criterion: The mathematical metric (Gini vs Entropy) used to measure the "purity" of a split. 
      Used in the loss function formula to find the split_point that maximizes information gain.



* Bagging:
    - Definition: Short for Bootstrap Aggregating; it trains multiple models on random subsets of data 
      and averages them to reduce overall variance and prevent overfitting.



* Random Forest: (Ensemble learning method)
    - Relationship: Random Forest is a bagging-based ensemble method in which multiple Decision Trees are trained on different bootstrap samples of the data. Additionally, at each split, a random subset of features is selected without replacement, which increases model diversity and reduces overfitting.

    - Definition: An ensemble of many Decision Trees; it is more stable and accurate than a single tree.

    - n_estimators: The number of trees in the forest. More trees lead to better stability.

    - max_features: The subset of features each tree sees, ensuring the trees are diverse.



* AdaBoost (Boosting):
    - Definition: A sequential model where each new "weak" learner (e.g., Decision Tree with max depth 1) 
      focuses on correcting the mistakes of the previous one to form a strong learner.

    - learning_rate: Controls how aggressively the model corrects previous errors.


Each configuration was evaluated using the **F2-score**.




model = perform_hyperparameter_tuning(X_train, y_train, cv, X_test, y_test)


- Output:

The Hyperparameter Winner is: LogisticRegression
Best Parameters found:
```
{
    'classifier__C': 10, 
    'classifier__penalty': 'l1',
    'classifier__solver': 'liblinear',
    'classifier__class_weight': 'balanced'
}
```

Validating Performance (Threshold = 0.5):

```
                precision    recall  f1-score   support

    No Stroke       0.99      0.71      0.82       972
        Stroke       0.13      0.84      0.22        50

    accuracy                           0.71      1022
    macro avg       0.56      0.77      0.52      1022
weighted avg       0.95      0.71      0.79      1022
```


- Interpretation:
```
Tuning confirmed that for this medical dataset, a simple 'L1' (Lasso) penalty was ideal. 
Lasso effectively performs feature selection by zeroing out the weights of features 
that it deems unimportant, allowing the model to focus purely on the most critical indicators 
of stroke risk.
```

---


### 7.8 Final Model Selection

Hyperparameter search identified **Logistic Regression with L1 regularization and balanced class weights** as the optimal model.

Key advantages:

* Strong recall performance,
* Built-in feature selection through L1 regularization,
* High interpretability for medical decision support,
* Stable generalization across validation folds.

The final trained model therefore balances **clinical safety**, **model simplicity**, and **interpretability**, making it suitable for real-world healthcare assistance.







# VI. Explainability

## 1. SHAP Explainability

To ensure transparency and clinical interpretability, we applied **SHAP (SHapley Additive exPlanations)** to analyze how the final model makes predictions.

The SHAP summary plot provides **global feature importance**, explaining which variables most influence stroke prediction decisions.

### 1.1 Interpretation of the SHAP Summary Plot

**Y-Axis — Features**
Features are ranked according to their overall contribution to the model’s predictions.
The top features represent the strongest determinants of stroke risk.

As expected, the most influential variables were:

* **Age**
* **risk_factors_count** (engineered feature)
* Glucose-related indicators
* Hypertension and heart disease status

---

**X-Axis — SHAP Value**

* **Positive SHAP values (Right side):**
  Increase the predicted probability of stroke.

* **Negative SHAP values (Left side):**
  Reduce the predicted stroke risk.

This allows direct observation of how each feature pushes predictions toward *Stroke* or *No Stroke*.

---

**Color Encoding — Feature Value**

* **Red points:** High feature values
  (e.g., older age, high BMI, presence of risk factors)

* **Blue points:** Low feature values
  (e.g., younger age, low BMI, absence of conditions)

This visualization confirms whether medical relationships learned by the model align with established clinical knowledge.

---

## 2. Additional Exploratory Visualizations

Several complementary visualizations were produced to better understand the dataset and validate model behavior:

* Histograms of **Age**, **Average Glucose Level**, and **BMI**
* Visualization of **target class imbalance**
* Scatter plot of **Age vs. Glucose Level**, colored by stroke status
* **Pearson correlation heatmap** between numerical variables
* **Confusion Matrix** comparison between model configurations

These plots support both data understanding and model validation.

---









# VII. Results

## 1. Model Diagnosis Using Explainability

After the first execution of model training and SHAP analysis, an important issue was discovered.

Initially, categorical age groups were engineered:

* `age_group_child`
* `age_group_teen`
* `age_group_senior`

However, SHAP analysis revealed **severe multicollinearity**.

Because Logistic Regression models linear relationships, the algorithm attempted to compensate for overlapping age representations:

* Continuous **age** received strong negative coefficients for young individuals.
* The binary feature **age_group_child** received an artificially large positive weight to mathematically correct model bias.

This behavior contradicted medical reality by incorrectly suggesting children were at high stroke risk.

### 1.1 Decision

To maintain clinical validity and interpretability:

- All categorical age group features were removed
- Only the continuous **age** variable was retained

This step significantly improved model reliability.

---

## 2. Final Model Selection

After retraining and performing a new hyperparameter search, the winning model was:

**Final Model:** Logistic Regression

**Best Hyperparameters**

```
C = 0.1
Penalty = L2 (Ridge Regularization)
Solver = liblinear
Class Weight = balanced
```

The model primarily relies on biologically meaningful predictors such as **Age** and **risk_factors_count**.

---

## 3. Performance Evaluation — Default Threshold (0.5)

| Class     | Precision | Recall   | F1-score |
| --------- | --------- | -------- | -------- |
| No Stroke | 0.99      | 0.72     | 0.84     |
| Stroke    | 0.13      | **0.82** | 0.23     |


Although recall slightly decreased compared to the initial version, performance became **medically trustworthy**.

The previous higher recall was artificially inflated due to multicollinearity effects.

The corrected model now follows clinically logical behavior.

---

## 4. Threshold Optimization

Because stroke prediction prioritizes **Sensitivity (Recall)**, an alternative decision threshold was evaluated.

### 4.1 Threshold = 0.3 (High Sensitivity Mode)

| Class  | Precision | Recall   |
| ------ | --------- | -------- |
| Stroke | 0.09      | **0.84** |

Results:

* Detects ~2% more stroke cases
* Produces a large increase in false positives
* Significantly lowers overall accuracy (59%)

---

## 5. Recommended Operating Threshold

Despite slightly higher recall at 0.3, the **0.5 threshold** was selected as the final configuration.

**Reasoning:**

* Maintains strong recall (**82%**)
* Reduces unnecessary medical alarms
* Preserves overall model stability and reliability

---

## 6 Final Summary

### 6.1. Resolving Multicollinearity

Explainability analysis revealed that engineered age categories were misleading the model. Removing them restored medical coherence and improved interpretability.

---

### 6.2. Final Model

The selected model is **Logistic Regression with Ridge (L2) regularization (C = 0.1)**, balancing interpretability, robustness, and clinical realism.

---

### 6.3. Performance Recommendation

* **Recommended Configuration (Threshold = 0.5)**
  → Detects **82 out of 100** potential stroke cases while limiting false alarms.

* **High-Sensitivity Configuration (Threshold = 0.3)**
  → Optional deployment setting when maximum safety screening is required.








# VIII. Dashboard presentation
with streamlit library 
2 pages:
...







# IX. Discussion, problems, limitation and implications

## 1. Discussion of Results

This project aimed to build a stroke risk prediction system optimized for clinical sensitivity (Recall) rather than traditional accuracy. This design choice is justified by the medical context, where missing a stroke case (false negative) is significantly more critical than incorrectly flagging a healthy patient.

The experimental results clearly showed that accuracy alone is misleading in highly imbalanced datasets. Initial baseline models achieved high accuracy (~94–95%) while failing to detect most stroke cases, confirming the presence of the accuracy paradox in medical classification problems.

After addressing class imbalance using class weighting and sampling techniques, the models achieved a strong improvement in recall, increasing stroke detection from near-zero levels to approximately 82–84% recall. This confirms that imbalance handling successfully forced the models to learn minority-class patterns instead of defaulting to majority-class predictions.

However, it is important to clarify a key observation:

> The increase in false positives is not primarily caused by imbalance techniques, but rather by threshold adjustment decisions.

When using a lower decision threshold (0.3), the model becomes more sensitive to the positive class, resulting in:

- Higher recall (more stroke cases detected)
- But significantly more false positives
- Lower overall accuracy

In contrast, the 0.5 threshold provides a more balanced operating point, achieving:

- High recall (~82%)
- Controlled false positives
- More stable and clinically reliable predictions

Therefore, the trade-off between recall and precision is mainly controlled by decision threshold tuning, not by sampling or class weighting strategies.


## 2. Model Behavior Interpretation

Explainability analysis using SHAP confirmed that the model’s predictions are primarily driven by medically meaningful variables, especially:

- Age
- Risk factor accumulation (risk_factors_count)
- Hypertension
- Glucose-related indicators

This aligns with established medical literature, indicating that the model is not learning arbitrary patterns but rather clinically consistent relationships.

However, SHAP also revealed a critical modeling issue during early iterations: multicollinearity introduced by engineered age-group features. These categorical age bins conflicted with the continuous age variable, leading to unstable coefficient distributions in linear models. Removing these redundant features restored logical consistency and improved interpretability.


## 3. Challenges & Design Decisions

### 3.1 Technical Problems

* **Dataset selection:** The first two datasets were unsuitable; one contained only female patients and the other was already fully cleaned. A third dataset was selected because it contained realistic challenges worth solving.

* **Missing BMI:** Since height and weight were unavailable, BMI was estimated using a statistical approximation method.

* **Outliers:** Extreme values were clipped rather than removed, because abnormal values in medical datasets may represent real patient conditions.

* **Glucose categories:** Glucose measurements can be fasting or random, each with different thresholds. As the dataset did not specify the measurement type, a clear assumption was made and documented.

* **Feature engineering:** New features were created carefully to avoid noise and redundant information.

* **Multicollinearity:** Some engineered variables were highly correlated, negatively affecting certain models. This issue was detected and handled during feature selection.

* **Scaling:** Scaling was applied only to models that require it and was fitted exclusively on training data to avoid data leakage.

* **Feature selection vs dimensionality reduction:** PCA was intentionally avoided because it reduces interpretability. Explainable feature selection methods were preferred.

* **Class imbalance:** Stroke cases were rare, causing models to ignore the minority class. Multiple imbalance handling techniques were tested and compared.

---

### 3.2 Team Challenges

* **1. Missing value debate:** Team members proposed different strategies such as dropping rows, mean imputation, and median imputation. Decisions were made after comparing experimental results rather than personal preference.

* **2. Model choice debate:** During model training and hyperparameter tuning, Achraf and Mohammed implemented different modeling approaches and obtained different results. Instead of selecting one approach, the team analyzed both and merged their strongest components into a unified solution.

* **3. Collaborative decision process:**

  * Each member explained why their solution fit the project.
  * Others evaluated limitations and provided constructive criticism.
  * Useful ideas were kept regardless of their origin.
  * Discussions functioned as organized technical debates based on reasoning and experimentation.

* **4. Conflict resolution:** Technical disagreements were resolved through communication, testing, and evidence-based decisions.

* **5. Strict Deadline**: We have a hard deadline to finish the entire project the week before exams, so strict
time management is required


* **6. Different initial knowledge levels**
  Not all team members had the same background in AI, Python, or data science.
  **Solution:** The team organized internal training sessions (Phase 0 — Team Training) where Python, Pandas, and basic AI concepts were explained so every member could understand the project logic and actively participate.

* **7. Project organization and coordination**
  Managing a complex AI project within a limited time required strong planning.
  **Solution:** The project was structured into clear phases. During Phase 1 — Initiation, objectives were defined, the dataset was selected, and ethical AI principles such as fairness(avoid bias by using class imbalance solutions) and transparency (dimensionality reduction methods were avoided to preserve model interpretability and prevent black-box behavior) were established.

* **8. Task distribution and workload balance**
  To avoid confusion and ensure efficiency, responsibilities were assigned according to project phases.
  **Solution:**

  * Weeks 1–2 (Data Preparation): El Mehdi Terech and El Kabraoui Hamza handled EDA, data cleaning, and feature engineering.
  * Weeks 3–4 (Advanced Modeling): Achraf Salimi and Mohammed Hafiane worked on feature selection, modeling, hyperparameter tuning, evaluation, and explainability.
  * Achraf and Mohammed also supervised technical consistency across stages to maintain methodological correctness.

* **9. Knowledge sharing and communication**
  * Our team operates on a strict rule: everyone is present when
a task is being executed. If a team member is coding, the others observe, provide help, and offer domain
knowledge when needed. If someone uses a complex math trick, they have to teach the rest of the team how
it works on the spot.

---

### 3.3 Design Problems

* **Code design:** The project initially existed as a single 1,124-line file, making maintenance difficult. The codebase was refactored into modular components with documentation, improving readability, scalability, and future development speed.


---



## 4. Team Communication and Collaboration

* Several project decisions required continuous discussion and agreement between team members.
* Multiple solutions were proposed, tested, and validated collaboratively.
* Experimental comparison was used to support decisions instead of assumptions.
* Different model implementations were combined to benefit from complementary strengths.
* The dashboard idea was inspired by other teams, showing openness to feedback and improvement.
* Mehdi was responsible for developing and integrating the dashboard into the system.

---

## 5. Team Organization and Soft Skills

* Tasks were distributed clearly to ensure efficient progress.
* Achraf refactored the codebase into modular, documented components.
* Mehdi developed the dashboard interface.
* Hamza and Simo prepared the presentation.
* The report was written collaboratively by all members.
* The project strengthened teamwork, communication, coordination, and problem-solving skills.
* Technical disagreements became opportunities for improvement rather than obstacles.
* Shared responsibility and collaboration played a key role in the final project quality.







# X. Conclusion

This project successfully developed an end-to-end machine learning pipeline for stroke risk prediction, integrating data preprocessing, feature engineering, model training, evaluation, and explainability within a clinically oriented framework.

A key achievement of this work is the consistent prioritization of clinical safety over traditional machine learning metrics. By focusing on Recall and F2-score instead of accuracy, the model was able to significantly improve detection of stroke cases, which is essential in real-world healthcare applications.

The project also demonstrated the importance of careful data preprocessing decisions, particularly in handling missing values, multicollinearity, and feature engineering. These steps directly impacted model stability and interpretability, showing that preprocessing is as important as model selection in medical AI systems.

Furthermore, the integration of SHAP-based explainability ensured that model predictions remain transparent and aligned with medical knowledge. This strengthens trust in the system and makes it suitable as a decision-support tool rather than a black-box predictor.

From a systems perspective, the final model shows that a well-tuned Logistic Regression model with appropriate threshold selection (0.5) can achieve strong clinical performance while maintaining interpretability and stability. The analysis also confirmed that threshold tuning plays a central role in controlling the recall–precision trade-off.

Finally, beyond the technical contributions, this project highlights the importance of teamwork, structured experimentation, and evidence-based decision-making. The combination of technical rigor and collaborative problem-solving contributed significantly to the quality of the final system.

In conclusion, this work demonstrates how machine learning, when guided by medical reasoning and ethical design principles, can effectively contribute to early stroke risk detection and support healthcare professionals in improving patient outcomes.