

# **6. Feature Selection**

Feature selection was conducted to balance **predictive performance** and **interpretability**, which is essential in medical machine learning applications.

---

## **Variance Filtering**

The following features were removed:

* Zero-variance features
* Near-constant features

These variables do not contribute meaningful information to the model and may introduce unnecessary noise.

---

## **Correlation Analysis**

A correlation matrix was used to:

* Identify highly correlated variables
* Detect redundancy among features
* Reduce multicollinearity

Highly correlated features were carefully reviewed, and redundant variables were removed to simplify the model while preserving predictive information.

---

## **Chi-Square Test**

The Chi-Square test was applied to evaluate the relationship between categorical variables and stroke occurrence.

This step allowed:

* Identification of statistically significant features
* Elimination of weak categorical predictors

---

## **RFECV (Recursive Feature Elimination with Cross-Validation)**

An embedded feature selection approach was applied using:

* **Estimator:** Random Forest
* **Evaluation metric:** ROC-AUC

RFECV iteratively removes the least important features while validating performance at each step.

This ensures:

* Optimal subset selection
* Strong predictive performance
* Improved interpretability

---

# **7. Model Training**

Multiple machine learning models were trained to evaluate different learning paradigms and select the most suitable approach for stroke prediction.

---

## **Models Evaluated**

The following algorithms were tested:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* AdaBoost
* Bagging Classifier

---

## **Training Strategy**

To ensure robust evaluation:

* Stratified train-test split was applied to preserve class distribution
* Cross-validation was used to ensure stability and reduce variance
* A unified pipeline was implemented for preprocessing and model training

---

## **Class Imbalance Handling**

Given the strong imbalance in stroke occurrence, three strategies were considered:

* Random undersampling
* SMOTE (oversampling)
* Class weighting (**final choice**)

### **Rationale for Class Weighting**

Class weighting was selected because it:

* Avoids synthetic data generation
* Preserves the original medical distribution
* Improves recall without introducing artificial bias

---

## **Hyperparameter Optimization**

GridSearchCV was used to optimize model hyperparameters.

This ensured:

* Optimal model configuration
* Fair comparison across models
* Improved generalization performance

---

# **8. Evaluation Metrics**

Given the medical context, evaluation prioritized **minimizing false negatives**, as missing a stroke case can have critical consequences.

---

## **Primary Metric: F2-Score**

The F2-score was selected as the main evaluation metric because it:

* Places higher weight on Recall than Precision
* Is suitable for high-risk medical classification tasks

---

## **Additional Metrics**

The following metrics were also used:

* **Recall (Sensitivity):** Primary clinical priority
* **ROC-AUC:** Measures overall discrimination ability
* **Confusion Matrix:** Provides detailed classification breakdown

---

# **VI. Explainability**

## **SHAP Analysis**

To ensure model transparency and interpretability, SHAP (SHapley Additive Explanations) was used.

---

## **Global Explainability**

SHAP values were analyzed to identify globally important features.

Key findings:

* **Age**, **average glucose level**, and **hypertension** were the most influential predictors
* Feature importance aligned with established medical knowledge

This reinforces the reliability of the model.

---

## **Local Explainability**

For individual predictions, SHAP provides:

* Contribution of each feature to the prediction
* Direction of impact (risk increase or decrease)

This enables clinicians to understand **why a specific patient is classified as high risk**.

---

## **Medical Consistency Validation**

Model behavior was validated against clinical expectations.

Findings:

* Increased age → higher stroke risk
* Higher glucose levels → increased risk
* Hypertension → strong positive contribution

This confirms that the model captures **real clinical patterns rather than spurious correlations**.

---

## **Additional Visualizations**

To support interpretability, the following visual tools were used:

* Feature distribution plots
* Risk contribution visualizations
* Correlation heatmaps

These tools helped validate both preprocessing and model behavior.

---

# **VII. Results**

## **Model Evaluation**

All models were evaluated using cross-validation and tested on unseen data.

---

## **Comparative Performance**

Observations across models:

* **Logistic Regression:**
  High interpretability but limited recall on minority class

* **KNN:**
  Sensitive to scaling and affected by class imbalance

* **Decision Tree:**
  Interpretable but prone to overfitting

* **Random Forest:**
  Strong overall performance with good balance between recall and precision

* **AdaBoost & Bagging:**
  Improved stability and better handling of complex patterns

---

## **Final Model Selection**

The final model was selected based on:

* High Recall (primary criterion)
* Acceptable Precision
* Stable cross-validation performance

The selected model demonstrated strong generalization capability across validation folds.

---

## **Threshold Optimization**

The default classification threshold (0.5) was not optimal for medical objectives.

---

### **Observed Limitation**

* High overall accuracy
* Insufficient recall (missed positive cases)

---

### **Adjusted Threshold**

The decision threshold was reduced to:

**0.3**

---

### **Impact of Adjustment**

* Increased Recall significantly
* Improved detection of high-risk patients
* Slight reduction in Precision (acceptable trade-off in medical context)

---

## **Recommended Configuration**

A high-sensitivity configuration is recommended, prioritizing:

* Detection of all potential stroke cases
* Minimization of false negatives
* Alignment with clinical safety requirements

---

# **VIII. Dashboard Presentation**

To enhance usability, an interactive dashboard was developed.

---

## **Clinical Interface**

This interface allows:

* Input of patient data
* Real-time stroke risk prediction
* Probability score output

It simulates real-world clinical usage.

---

## **Explainability Interface**

This section provides:

* SHAP-based explanations
* Global feature importance
* Model performance metrics

---

## **Purpose**

The dashboard transforms the model into a **decision-support system**, enabling:

* Accessibility for non-technical users
* Transparent AI decision-making
* Increased trust in predictions

---

# **IX. Discussion, Challenges & Implications**

## **1. Discussion of Results**

The project demonstrates that **interpretable machine learning models** can effectively support medical decision-making when:

* Recall is prioritized
* Explainability is preserved

The system achieves a balance between performance and clinical usability.

---

## **2. Model Behavior**

The model successfully learned clinically meaningful relationships, including:

* Age-related risk increase
* Impact of chronic conditions
* Influence of metabolic indicators

This confirms that predictions are grounded in real-world medical patterns.

---

## **3. Technical Challenges**

Key challenges included:

* Severe class imbalance
* Missing data handling
* Feature redundancy
* Maintaining interpretability over complexity

Each challenge was addressed using structured preprocessing and modeling strategies.

---

## **4. Team Challenges**

Additional non-technical challenges included:

* Differences in technical expertise
* Difficulty in explaining complex ML concepts
* Time constraints under project deadlines

---

### **Solutions Implemented**

* Collaborative development approach
* Continuous knowledge sharing
* Structured communication workflow

---

## **5. Design Decisions**

Important design choices included:

* Avoiding PCA to preserve interpretability
* Avoiding SMOTE to maintain data realism
* Prioritizing ethical and transparent AI design

These decisions ensured clinical trustworthiness of the system.

---

## **6. Soft Skills Development**

The project contributed to development of:

* Leadership skills
* Communication skills
* Team coordination
* Conflict resolution
* Project management abilities

---

# **X. Conclusion**

This project implemented a complete machine learning pipeline for stroke risk prediction, integrating:

* Data preprocessing
* Feature engineering
* Machine learning modeling
* Explainable AI
* Interactive dashboard development

---

## **Key Achievements**

* Development of a high-recall predictive model
* Integration of SHAP-based explainability
* Creation of a functional decision-support dashboard
* Effective teamwork and project execution

---

## **Impact**

The system demonstrates how artificial intelligence can:

* Support clinical decision-making
* Improve early detection of stroke risk
* Enhance patient safety

while maintaining **transparency and human oversight**.

