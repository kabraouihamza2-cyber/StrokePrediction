
Stroke Prediction

A machine learning project to predict the risk of stroke in patients based on clinical and demographic features.


About the Project
Stroke is one of the leading causes of death and disability worldwide. Early detection of at-risk patients can significantly improve outcomes. This project builds a predictive model using patient health data to identify individuals at high risk of stroke.
The model is trained on the Healthcare Stroke Dataset (Kaggle), which contains 5,110 patient records with 12 features including age, BMI, glucose levels, and comorbidities.


Project Structure
PROJECT/
├── src/
│   ├── data_processing.py     # Data cleaning, encoding, and feature engineering
│   ├── modeling.py            # Model training, evaluation, and export
│   └── visualization.py       # Plots: ROC curve, confusion matrix, feature importance
├── .env                       # Environment variables (not committed)
├── code_book.json             # Feature descriptions and variable mappings
├── data.csv                   # Raw dataset (not committed)
├── final_stroke_model.pkl     # Trained model (serialized)
├── main.py                    # Entry point — runs the full pipeline
└── requirements.txt           # Python dependencies


Dataset
Feature                 Type               Description
 age                    Numeric            Age of the patient 
hypertension            Binary             1 if the patient has hypertension 
heart_disease           Binary             1 if the patient has heart disease 
avg_glucose_level       Numeric            Average blood glucose level
bmi                     Numeric            Body Mass Index 
gender                  Categorical        Male / Female / Other 
ever_married            Categorical        Yes / No 
work_type               Categorical        Type of employment
Residence_type          Categorical U      rban / Rural 
smoking_status          Categorical        Smoking habits 
stroke                  Binary             Target      — 1 if stroke occurred

Télécharge le dataset sur Kaggle et place-le en tant que data.csv à la racine.


Best Model & Results
Logistic Regression:
                precision    recall  f1-score   support

       No Stroke       0.99      0.71      0.82       972
          Stroke       0.13      0.84      0.22        50

        accuracy                           0.71      1022
       macro avg       0.56      0.77      0.52      1022
    weighted avg       0.95      0.71      0.79      1022
