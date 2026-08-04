import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, make_scorer, fbeta_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from src.constants import SEED




"""
Note:
1. Why use ImbPipeline? 
   The pipeline ensures that scaling and sampling are fit ONLY on the training folds of 
   our cross-validation, preventing the model from "peeking" at information in the test/validation sets.
   
   Why not use a normal Pipeline?
   Standard Scikit-learn pipelines do not support "samplers" that change the number of 
   observations. If you try to use SMOTE with a normal pipeline, it will crash because
   it expects the number of samples to remain constant during the transformation steps.

2. Why F2-score?
   In a medical context, we optimize for the F2-score because it weights Recall twice as heavily 
   as Precision. While we need high Precision to avoid unnecessary medical tests, a 100% Recall 
   model with 0% Precision would be useless. The F2-score finds the safe clinical sweet spot.
"""


def compare_splitting_strategies(X, y):
    """
    Compares Standard train_test_split vs. Stratified splitting.
    Demonstrates why stratification is critical for imbalanced datasets.
    """
    from sklearn.model_selection import train_test_split
    
    # Non-Stratified
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    # Stratified
    s_X_train, s_X_test, s_y_train, s_y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    
    print("\n\n" + "="*50)
    print(" Comparing splitting strategies ")
    print("="*50)
    print(f"Non-Stratified Stroke Count in Test: {sum(y_test)}")
    print(f"Stratified Stroke Count in Test:     {sum(s_y_test)}")
    print("Note: Stratification ensures the 5% stroke ratio is preserved across sets.")  # 5110 patient, 249 positive -> 249/5110 ~ 4.87% ~ 5%




def baseline_all_models_no_tuning(X_train, X_test, y_train, y_test):
    """
    Evaluates default models without tuning or sampling.
    Note: Accuracy is misleading in imbalanced medical data (The Accuracy Paradox).
    """
    print("\n\n" + "="*60)
    print(" Baseline Models (No Tuning, No Sampling, Default Params)")
    print("="*60)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(random_state=SEED),
        "AdaBoost": AdaBoostClassifier(random_state=SEED),
        "Bagging": BaggingClassifier(random_state=SEED)
    }
    
    for name, model in models.items():
        print(f"\nEvaluating: {name}")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model) 
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=['No Stroke', 'Stroke']))







def compare_sampling_methods(X_train, X_test, y_train, y_test):
    """
    Compares 3 approaches using Logistic Regression (Winner in the gridSearch solution):
        1. Undersampling (RandomUnderSampler)
        2. Oversampling (SMOTE)
        3. Algorithm-Level Weighting (class_weight='balanced')
    """

    print("\n\n" + "="*50)
    print(" Comparing Sampling Methods (Logistic Regression)")
    print("="*50)

    # -----------------------------------------------------------------

    
    # 1. Undersampling
    print("\n1. UnderSampling (Randomly deleting healthy patients):")
    under_pipe = ImbPipeline([
        ('scaler', StandardScaler()),
        ('sampler', RandomUnderSampler(random_state=SEED)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=SEED))
    ])
    under_pipe.fit(X_train, y_train)
    print(classification_report(y_test, under_pipe.predict(X_test), target_names=['No Stroke', 'Stroke']))
    

    # -----------------------------------------------------------------



    # 2. Oversampling (SMOTE)
    print("\n2. OverSampling (Creating synthetic stroke patients with SMOTE):")
    over_pipe = ImbPipeline([
        ('scaler', StandardScaler()),
        ('sampler', SMOTE(random_state=SEED)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=SEED))
    ])
    over_pipe.fit(X_train, y_train)
    print(classification_report(y_test, over_pipe.predict(X_test), target_names=['No Stroke', 'Stroke']))
    

    # -----------------------------------------------------------------


    # 3. Class Weighting
    print("\n3. Class Weighting (Penalizing mistakes mathematically):")
    weight_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=SEED, class_weight='balanced'))
    ])
    weight_pipe.fit(X_train, y_train)
    print(classification_report(y_test, weight_pipe.predict(X_test), target_names=['No Stroke', 'Stroke']))







def perform_hyperparameter_tuning(X_train, y_train, cv, X_test, y_test):
    """
    Uses GridSearchCV to find the optimal model and parameters.
    Optimizes for F2-score to prioritize Recall (Sensitivity) over Precision.


    Logistic Regression:
    - Definition: Estimates stroke probability by fitting a linear separator between classes.

    - classifier__C: Controls regularization; smaller values penalize large weights more (prevents overfitting).
      It's basically the alpha in Ridge and Lasso.

    - classifier__penalty: L1 (Lasso) uses absolute values; L2 (Ridge) uses squares.
      It's just how the loss function is written; adds a penalty to the loss function.

    - class_weight='balanced': Automatically weights the minority class (Stroke) higher to handle imbalance.



    KNN (K-Nearest Neighbors):
    - weights: 'uniform' (equal weight to all neighbors) vs 'distance' (closer neighbors have more influence).
    - metric: Calculation method for "closeness" (Euclidean = straight line, Manhattan = grid-like path).



    Decision Tree:
    - Definition: A binary tree structure that splits data into groups based on feature rules (e.g., feature <, > split_point).

    - max_depth: Limits how deep the tree grows to prevent it from "memorizing" (overfitting) the data.

    - min_samples_split: The minimum patients required before the tree is allowed to split further.

    - criterion: The mathematical metric (Gini vs Entropy) used to measure the "purity" of a split. 
      Used in the loss function formula to find the split_point that maximizes information gain.



    Bagging:
    - Definition: Short for Bootstrap Aggregating; it trains multiple models on random subsets of data 
      and averages them to reduce overall variance and prevent overfitting.



    Random Forest: (Ensemble learning method)
    - Relationship: Random Forest is Bagging where the base model is a Decision Tree.

    - Definition: An ensemble of many Decision Trees; it is more stable and accurate than a single tree.

    - n_estimators: The number of trees in the forest. More trees lead to better stability.

    - max_features: The subset of features each tree sees, ensuring the trees are diverse.



    AdaBoost (Boosting):
    - Definition: A sequential model where each new "weak" learner (e.g., Decision Tree with max depth 1) 
      focuses on correcting the mistakes of the previous one to form a strong learner.

    - learning_rate: Controls how aggressively the model corrects previous errors.


    """

    ## Scaling is the first step. The 'classifier' is a placeholder.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression()) 
    ])




    param_grid = [
        # 1. Logistic Regression: Testing different regularization strengths and types
        {
            'classifier': [LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced')],
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2']
        },
        # 2. KNN: Checking more neighbors and distance weighting
        {
            'classifier': [KNeighborsClassifier()],
            'classifier__n_neighbors': [3, 5, 7, 11, 15, 21],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan']
        },
        # 3. Decision Tree: Controlling growth to prevent overfitting
        {
            'classifier': [DecisionTreeClassifier(random_state=SEED, class_weight='balanced')],
            'classifier__max_depth': [None, 3, 5, 10, 20],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__criterion': ['gini', 'entropy']
        },
        # 4. Random Forest: Searching for the right "breadth" and "depth"
        {
            'classifier': [RandomForestClassifier(random_state=SEED, class_weight='balanced')],
            'classifier__n_estimators': [100, 200, 500],
            'classifier__max_features': ['sqrt', 'log2', None],
            'classifier__max_depth': [None, 10, 20]
        },
        # 5. AdaBoost (Boosting): Tuning the learning rate vs. number of estimators
        ## It balances by iteratively increasing weights of misclassified samples.
        {
            'classifier': [AdaBoostClassifier(random_state=SEED)],
            'classifier__n_estimators': [50, 100, 200, 500],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 1.0]
        },
        # 6. Bagging: Trying different base estimators (Trees vs KNN)
        {
            'classifier': [BaggingClassifier(random_state=SEED)],
            'classifier__estimator': [
                None, # Defaults to DecisionTree; it won't be balanced unless specified
                DecisionTreeClassifier(class_weight='balanced'), 
                KNeighborsClassifier()
            ],
            'classifier__n_estimators': [10, 50, 100],
            'classifier__max_samples': [0.5, 1.0],
            'classifier__max_features': [0.5, 1.0]
        }
    ]

    # Create a custom F2-scorer
    f2_scorer = make_scorer(fbeta_score, beta=2)

    # Execution:
    ## 'cv=kf' is where the K-Fold split is actually injected into the search process
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring=f2_scorer, n_jobs=-1)
    ## Using f2-score because it weights recall twice as important as precision
    ## f1-score prioritize precision and recall equally.
    grid.fit(X_train, y_train)
    
    print(f"\nThe winner is: {grid.best_estimator_.named_steps['classifier'].__class__.__name__}")
    print("Best Parameters found:")
    print(grid.best_params_)

    print("\n--- CLASSIFICATION REPORT (Threshold = 0.5) ---")
    y_pred = grid.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['No Stroke', 'Stroke']))

    return grid.best_estimator_




def choose_best_model(X, y):
    """
    Compare 3 methods:
        default (Baseline),
        when sampling is used, 
        hyperparameter tuning used, with a class-weight solution for class imbalance (best found in method 2)
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    # -----------------------------------------------------------------

    ## BaseLine Models, No-Tuning
    baseline_all_models_no_tuning(X_train, X_test, y_train, y_test)

    ## Output:
    """
    Logistic Regression:
                    precision    recall  f1-score   support

        No Stroke       0.95      1.00      0.97       972
        Stroke          0.00      0.00      0.00        50

        accuracy                            0.95      1022
        macro avg       0.48      0.50      0.49      1022
        weighted avg    0.90      0.95      0.93      1022



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

    """

    ## Interpretation (the accuracy paradox):
    """
    Notice how all the default models boast a massive 95% Accuracy. However, look at the Recall for 'Stroke':
    - KNN and AdaBoost literally caught 0% of Strokes.
    - Logistic Regression and Random Forest caught 2% of Strokes.
    - Decision Tree caught 10% of Strokes.
    Because the dataset is heavily imbalanced (Strokes ~5%), the models are blindly predicting "No Stroke" for almost everyone. 
    This is why Accuracy is a misleading ("liar") metric in medical datasets. We must optimize for Recall.
    """


    # -----------------------------------------------------------------

    ## Compare sampling methods:
    compare_sampling_methods(X_train, X_test, y_train, y_test)

    ## Output:
    """
    1. UnderSampling (Randomly deleting healthy patients):
                precision    recall  f1-score   support

    No Stroke       0.99      0.67      0.80       972
    Stroke          0.12      0.84      0.20        50

    accuracy                            0.68      1022
    macro avg       0.55      0.76      0.50      1022
    weighted avg    0.95      0.68      0.77      1022




    2. OverSampling (Creating synthetic stroke patients with SMOTE):
                    precision    recall  f1-score   support

        No Stroke       0.99      0.70      0.82       972
        Stroke          0.13      0.84      0.22        50

        accuracy                            0.71      1022
        macro avg       0.56      0.77      0.52      1022
        weighted avg    0.95      0.71      0.79      1022




    3. Class Weighting (Penalizing mistakes mathematically):
                    precision    recall  f1-score   support

        No Stroke       0.99      0.70      0.82       972
        Stroke          0.13      0.84      0.22        50

        accuracy                            0.71      1022
        macro avg       0.56      0.77      0.52      1022
        weighted avg    0.95      0.71      0.79      1022

    """

    ## Interpretation:
    """
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
    """

    # -----------------------------------------------------------------

    ## use hyperparameter tuning: with class-weight='balanced' best solution for class imbalance
    model = perform_hyperparameter_tuning(X_train, y_train, cv, X_test, y_test)


    ## Output:
    """
    The Hyperparameter Winner is: LogisticRegression
    Best Parameters found:
    {
        'classifier__C': 10, 
        'classifier__penalty': 'l1',
        'classifier__solver': 'liblinear',
        'classifier__class_weight': 'balanced'
    }

    Validating Performance (Threshold = 0.5):
                  precision    recall  f1-score   support

       No Stroke       0.99      0.71      0.82       972
          Stroke       0.13      0.84      0.22        50

        accuracy                           0.71      1022
       macro avg       0.56      0.77      0.52      1022
    weighted avg       0.95      0.71      0.79      1022
    """


    ## Interpretation:
    """
    Tuning confirmed that for this medical dataset, a simple 'L1' (Lasso) penalty was ideal. 
    Lasso effectively performs feature selection by zeroing out the weights of features 
    that it deems unimportant, allowing the model to focus purely on the most critical indicators 
    of stroke risk.
    """



    ## END: return the best model ---------------------------------------
    return model 



