import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

def _save_plot(filename):
    """
    save plots to the 'plots/' directory.
    """
    if filename:
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig(f'plots/{filename}')
        print(f"Plot saved to plots/{filename}")

def plot_feature_distributions(df, filename=None):
    """
    Plots histograms for Age, Average Glucose Level, and BMI.
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.histplot(df['age'], kde=True, color='skyblue')
    plt.title('Age Distribution')
    
    plt.subplot(1, 3, 2)
    sns.histplot(df['avg_glucose_level'], kde=True, color='salmon')
    plt.title('Avg Glucose Level Distribution')
    
    plt.subplot(1, 3, 3)
    sns.histplot(df['bmi'], kde=True, color='lightgreen')
    plt.title('BMI Distribution')
    
    plt.tight_layout()
    _save_plot(filename)
    plt.show()




def plot_stroke_distribution(df, filename=None):
    """
    Visualizes the target class imbalance.
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(x='stroke', data=df, hue='stroke', palette='Set1', legend=False)
    plt.title('Stroke Distribution')
    plt.xticks([0, 1], ['No Stroke', 'Stroke'])
    _save_plot(filename)
    plt.show()



def plot_age_vs_glucose(df, filename=None):
    """
    Scatter plot of Age vs Glucose colored by Stroke status.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='age', y='avg_glucose_level', hue='stroke', data=df, alpha=0.6, palette='coolwarm')
    plt.title('Age vs Avg Glucose Level (by Stroke Status)')
    plt.grid(True, linestyle='--', alpha=0.5)
    _save_plot(filename)
    plt.show()




def plot_correlation_matrix(df, filename=None):
    """
    Visualizes the Pearson correlation between numerical features.
    """
    plt.figure(figsize=(20, 15))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 8})
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    _save_plot(filename)
    plt.show()




def plot_confusion_matrix(model, X_test, y_test, filename=None, threshold=None):
    """
    Plots the confusion matrix .
    """
    if threshold is not None:
        probs = model.predict_proba(X_test)[:, 1]
        y_pred = (probs >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Stroke', 'Stroke'])
    disp.plot(cmap='Reds', values_format='d')
    
    if isinstance(model, Pipeline):
        classifier_name = model.named_steps['classifier'].__class__.__name__
    else:
        classifier_name = model.__class__.__name__

    plt.title(f"Confusion Matrix: {classifier_name}")
    _save_plot(filename)
    plt.show()



def explain_with_shap(X_train, model, filename=None):
    """
    SHAP Summary Plot for global feature importance.
    Uses the winner from our analysis


    The Y-Axis (Features): The features are ranked from top to bottom by their absolute predictive power. 
    The feature at the very top is the #1 reason the model decides if someone is going to have a stroke. 
    (This will likely be age and your engineered risk_factors_count).

    The X-Axis (SHAP Value):
    If a dot lands on the Right side (Positive SHAP): This feature increased the patient's risk of stroke.
    If a dot lands on the Left side (Negative SHAP): This feature lowered the patient's risk, protecting them from a stroke.

    The Colors (Feature Value):
    Red dots: The patient had a High amount of this feature (e.g., 1 Old Age, High BMI, or "Yes" for a binary feature).
    Blue dots: The patient had a Low amount of this feature (e.g., 0, Young Age, Low BMI).
    """


    print("\n\n" + "="*50)
    print(" GENERATING SHAP EXPLANATIONS")
    print("="*50)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    
    # If the model is a Pipeline, we need to extract the classifier for SHAP
    if isinstance(model, Pipeline):
        inner_model = model.named_steps['classifier']
    else:
        inner_model = model

    explainer = shap.Explainer(inner_model, X_train_scaled_df)
    shap_values = explainer.shap_values(X_train_scaled_df)
    
    plt.figure(figsize=(10, 8))
    plt.title("SHAP Summary Plot: Global Feature Importance")
    shap.summary_plot(shap_values, X_train_scaled_df, show=False)
    plt.tight_layout()
    _save_plot(filename)
    plt.show()
