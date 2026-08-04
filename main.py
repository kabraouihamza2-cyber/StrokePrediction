import pandas as pd
import pickle
import os
import sys
from sklearn.preprocessing import StandardScaler

# Set working directory to the project folder (where this file is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report

from src import data_processing, modeling, visualization, constants

def main():

    ## I. Preprocessing --------------------------------------------------------

    # 1. Loading and initial cleaning
    df = data_processing.load_and_clean_data(constants.DATA_PATH)
    
    # 1.1. saving first plots
    visualization.plot_feature_distributions(df, filename='feature_distributions.png')
    visualization.plot_stroke_distribution(df, filename='stroke_distribution.png')
    visualization.plot_age_vs_glucose(df, filename='age_vs_glucose.png')
    
    # 2. Feature Engineering
    df = data_processing.engineer_features(df)
    
    # 3. data encoding
    df = data_processing.encode_data(df)
    
    # 4. Memory Optimization
    df = data_processing.optimize_memory(df)
    
    # 5. Feature Selection
    # Internally runs Variance Filtering, Correlation Analysis, Chi-Square, and RFECV.
    # Note: Correlation matrix is saved internally within this step.
    df = data_processing.select_features(df)
    

    # 6. data preparation:
    X = df.drop(columns="stroke")
    y = df["stroke"]

    # 6.1 Splitting Strategy Comparison
    modeling.compare_splitting_strategies(X, y)

    # 6.2 Split data:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=constants.SEED, stratify=y)
    






    ## II. Model training & Shap analysis -------------------------------------- 
    
    # 7. Model Research (Baseline, Sampling, Tuning)
    model_winner = modeling.choose_best_model(X, y)
    

    # 9. SHAP Analysis 
    # Based on SHAP results from the best initial model, we identify multicollinearity
    visualization.explain_with_shap(X_train, model_winner, filename='shap_research_initial.png')

  
    """
    We need to drop age_group_child, age_group_teen, and age_group_senior.

    "In our initial feature engineering we created categorical age groups. However, SHAP analysis
    revealed severe multicollinearity. Because the Logistic Regression algorithm attempts 
    to fit a linear line to an exponential biological risk, it assigned a heavily negative weight
    to young continuous ages, and then confusingly assigned a massive positive weight to the 
    age_group_child feature to mathematically correct its own overshoot.

    To ensure our model's logic remains medically sound and interpretable,we must rely solely
    on the continuous age feature and drop the categorical interaction groups."
    """

    cols_to_drop_shap = ['age_group_child', 'age_group_teen', 'age_group_senior']
    df = df.drop(columns=[col for col in cols_to_drop_shap if col in df.columns])






    ## Retry data preparation, model training and Shap explanability ----------------

    # 11. data preparation (second time)
    X_f = df.drop(columns="stroke")
    y_f = df["stroke"]
    X_train_f, X_test_f, y_f_train, y_f_test = train_test_split(X_f, y_f, test_size=0.2, random_state=constants.SEED, stratify=y_f)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=constants.SEED)
    

    # 12. model training
    print("\n--- Model after the final multicollinarity resolved ")
    model_final = modeling.perform_hyperparameter_tuning(X_train_f, y_f_train, cv, X_test_f, y_f_test)


    ## Output (Threshold = 0.5):
    """
    Final Model Winner: LogisticRegression
    Best Parameters: {
        'classifier__C': 0.1, 
        'classifier__penalty': 'l2',
        'classifier__solver': 'liblinear',
        'classifier__class_weight': 'balanced'
    }
    
    
    Performance at Default Threshold (0.5):
                  precision    recall  f1-score   support

       No Stroke       0.99      0.72      0.84       972
          Stroke       0.13      0.82      0.23        50

        accuracy                           0.73      1022
       macro avg       0.56      0.77      0.53      1022
    weighted avg       0.95      0.73      0.81      1022

    Although the baseline recall is lower in this version, these results are significantly more 
    medically reliable. The high recall in the initial model was "artificial," driven by 
    multicollinearity that incorrectly labeled children as high-risk. By resolving these 
    conflicts, the model now respects established clinical logic. We can safely reclaim the 
    required sensitivity by adjusting the classification threshold to 0.3.
    """

    # 13. Apply Custom Threshold (0.3)
    probs = model_final.predict_proba(X_test_f)[:, 1]
    preds = (probs >= constants.CLASSIFICATION_THRESHOLD).astype(int)
    
    print("\n" + "="*50)
    print(f"Results with Threshold = {constants.CLASSIFICATION_THRESHOLD}")
    print("="*50)
    print(classification_report(y_f_test, preds, target_names=['No Stroke', 'Stroke']))
    """
    Performance using (threshold 0.3):
                    precision    recall  f1-score   support

        No Stroke       0.99      0.58      0.73       972
        Stroke          0.09      0.84      0.17        50

        accuracy                            0.59      1022
        macro avg       0.54      0.71      0.45      1022
        weighted avg    0.94      0.59      0.70      1022


    We recommend the 0.5 threshold because it is much more balanced. While the 0.3 threshold 
    catches 2% more strokes, it creates way too many false alarms and lowers the overall accuracy.
     Keeping it at 0.5 ensures the model stays reliable while still maintaining a high recall of 82%.
    """
    


    # 14. Final evaluation:
    visualization.plot_confusion_matrix(model_final, X_test_f, y_f_test, filename='final_confusion_matrix_thresh0_3.png', threshold=constants.CLASSIFICATION_THRESHOLD)
    visualization.plot_confusion_matrix(model_final, X_test_f, y_f_test, filename='final_confusion_matrix_thresh0_5.png')

    visualization.explain_with_shap(X_train_f, model_final, filename='final_shap_importance.png')
    

    # 15. Final Model Export for Dashboard
    with open('final_stroke_model.pkl', 'wb') as f:
        pickle.dump(model_final, f)


    """
    General Resume:

    1. Solving Multicollinearity:
    During the research phase, we discovered that categorical age groups were 
    confusing the model (multicollinearity). By deleting these groups and 
    keeping only the continuous 'age' feature, we made the model much more 
    reliable and medically sound.

    2. Final Model Selection: 
    The winning model is Logistic Regression using Ridge (L2) regularization 
    and C=0.1. It relies heavily on Age and our engineered "risk_factors_count".

    3. Performance & Recommendation:
    - Recommended (0.5 threshold): 82% Recall with significantly fewer false alarms.
    - High-Sensitivity (0.3 threshold): 84% Recall for maximum safety.
    
    We recommend the 0.5 setting because it keeps the model stable and reliable, 
    successfully identifying 82 out of 100 potential stroke cases.
    """


if __name__ == "__main__":
    main()
