import pandas as pd
import numpy as np
import json
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from src.constants import SEED, CODE_BOOK_PATH
from src import visualization

def load_and_clean_data(file_path):
    """
    Data exploration and Cleaning:
       Handel missing values, duplicated values, (membership constrains/standarizing values), data range constraints 
    """
    df = pd.read_csv(file_path)

    ## Checking data:
    #print(df.info())
    #print(df.head(20))

    # -----------------------------------------------------------------

    ## Missing values:
    #print(df.isna().any()) -> prints if the column has missing values or no

    # Instead of a simple mean imputation, I utilized Conditional Median Imputation based on age-cohorts and occupational categories. 
    # This preserves the variance within the dataset and respects the clinical correlation between work environment and body mass index.
    df['bmi'] = df['bmi'].fillna(df.groupby(['age', 'work_type'])['bmi'].transform('median'))

    #print(df.isna().any())


    # -----------------------------------------------------------------

    ## Duplicated values:
    #print(df.duplicated(subset="id").any()) -> there is no full/partial duplicated values.

    # -----------------------------------------------------------------

    ## Membership constraints / Standardizing Values

    """
    Membership Constraints
    This ensures data belongs to a valid, logical group. For example, if a "Gender" column contains "Dragon" or "Airplane,"
      it violates a membership constraint because those values aren't part of the allowed categories.

    Standardizing Values
    This fixes how valid data is written. For example, "Female," "female," "F," and "Famale" all refer to the same thing,
      so we standardize them into one uniform label so the computer doesn't treat them as different groups.
    """

    # Gender
    df.drop(df[df["gender"] == "Other"].index, inplace=True) # dropped bacause it's the only one with this category and to allow binary encoding
    #print(df["gender"].unique())

    # hypertension
    #print(df["hypertension"].unique())

    # heart_disease
    #print(df["heart_disease"].unique())

    # work_type (ecoded later)
    #print(df["work_type"].unique())

    # Residence_type
    #print(df["Residence_type"].unique())

    # ever_married (encoded because some models can't work with categorical data) 
    #print(df["ever_married"].unique())

    # smoking_status
    #print(df["smoking_status"].unique())

    # stroke
    #print(df["stroke"].unique())

    # ------------------------------------------------------------

    ## Data range constraints

    # Age (Age range verification) good
    #print(df[(df["age"] < 0) & df["age"] > 120])


    # avg glucose range verification (not good)
    #print(df["avg_glucose_level"].describe())
    ## solution for avg glucose outliers:
    df["avg_glucose_level"] = df["avg_glucose_level"].clip(lower=50, upper=251)


    # bmi ranges
    #print(df["bmi"].describe())
    ## solution for bmi outliers:
    df["bmi"] = df["bmi"].clip(lower=10.0, upper=61.0)


    """
    To handle outliers, we used clipping based on realistic medical ranges instead of the standard IQR method.
    This prevents the accidental deletion of extreme but clinically valid data points that a purely mathematical
    approach would have removed.
    """

    # -----------------------------------------------------------------

    ## Drop the id column:
    #Since it's a random integer, it should be dropped before training so the model doesn't try to find a "fake"
    #pattern in patient ID
    df.drop(columns=['id'], inplace=True, errors='ignore')
    
    # END ------------------------------------------------------------- 
    return df








def engineer_features(df):
    """
       feature engineer glucose_status, risk_factor_count, (age X bmi interactiom), (age x Hypertension interaction), age groups
    """

    ## Glucose Status (Categorical Bins)
    df["Glucose_Status"] = pd.cut(
        df["avg_glucose_level"],
        bins=[0, 140, 200, np.inf],
        labels=["Normal", "Prediabetic", "Diabetic"]
    )
    """
    We assumed random testing because of the distribution of avg_glucose_level,
    We applied the World Health Organization (WHO) and American Diabetes Association (ADA) criteria for Random testing.
    """

    # -----------------------------------------------------------------

    ## risk_factos_count
    # Count how many major risk factors the patient has
    # (Hypertension, Heart Disease, and High Glucose, ..)
    # values range: [0, 1, 2, 3, 4, 5]
    # Before encoding smoking_status this works 
    df['risk_factors_count'] = (
        df['hypertension'] + 
        df['heart_disease'] + 
        (df['avg_glucose_level'] >= 140).astype(int) +
        df['smoking_status'].isin(['smokes', 'formerly_smoked']).astype(int) +
        (df['bmi'] >= 25).astype(int)
    )

    # -----------------------------------------------------------------

    ## Interaction terms
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    df['age_hypertension_interaction'] = df['age'] * df['hypertension']

    """
    Interaction Terms: We use multiplication x1 * x2 as the mathematical standard to represent variable interactions.
    This allows the model to capture how the impact of a risk factor (like BMI) is not constant,
    but is instead amplified as another factor (like Age) increases.
    """

    # -----------------------------------------------------------------

    ## Age Groups
    conditions = [
        (df['age'] <= 12),
        (df['age'] > 12) & (df['age'] <= 19),
        (df['age'] > 19) & (df['age'] < 65),
        (df['age'] >= 65)
    ]
    choices = ['child', 'teen', 'adult', 'senior']
    df['age_group'] = np.select(conditions, choices, default='unknown')

    # END ------------------------------------------------------------- 
    return df










def encode_data(df):
    """
    Binary encoding, one-hot encoding
    """

    ## Binary Encoding (0 and 1)

    # ever_married: (Yes/No)
    df["ever_married"] = df["ever_married"].replace({"Yes":1, 'No':0})
    #print(df["ever_married"].unique())


    # gender: (Male/Female)
    df["gender"] = df["gender"].replace({"Male":1, "Female":0})


    # Residence_type: (Urban/Rural)
    df["Residence_type"] = df["Residence_type"].replace({"Urban":1, "Rural":0})


    # -----------------------------------------------------------------

    ## One-Hot Encoding (Dummies)

    # work_type: (Private, Self-employed, Govt_job, children, Never_worked)
    # smoking_status: (formerly smoked, never smoked, smokes, Unknown)

    ## fix categories names before encoding the work_type
    df["smoking_status"] = df["smoking_status"].str.replace(" ","_")

    df = pd.get_dummies(df, columns=['work_type', 'smoking_status',"Glucose_Status", "age_group"], drop_first=True, dtype=int)
    # dtype=int : because by default it's True False not 0/1


    # -----------------------------------------------------------------


    ## Code book for encoding:
    code_book = {
        "Target Variable": {"stroke": {1: "Stroke Diagnosed", 0: "No Stroke"}},
        "Binary Features": {
            "gender": {1: "Male", 0: "Female"},
            "ever_married": {1: "Yes", 0: "No"},
            "Residence_type": {1: "Urban", 0: "Rural"},
            "hypertension": {1: "Yes", 0: "No"},
            "heart_disease": {1: "Yes", 0: "No"}
        },
        "One-Hot Encoding (Baselines)": {
            "Note": "drop_first=True used. If all related columns are 0, the baseline is active.",
            "work_type": {"Baseline": "Govt_job"},
            "smoking_status": {"Baseline": "Unknown"},
            "Glucose_Status": {"Baseline": "Normal"},
            "age_group": {"Baseline": "child"}
        },
        "Engineered Features": {
            "risk_factors_count": "Sum of risks",
            "age_bmi_interaction": "age * bmi",
            "age_hypertension_interaction": "age * hypertension"
        }
    }
    with open(CODE_BOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(code_book, f, indent=4)
        

    # END ------------------------------------------------------------- 
    return df













def optimize_memory(df):
    """
    clean data types: (memory optimization)
    """

    ## Before optimization:
    #print(df.info()) # memory usage: 997.9+ KB

    #print(df.head())
    #print(df.dtypes)
    #print(df.columns)

    # -----------------------------------------------------------------

    # List of binary/dummy columns: downcast to int32
    binary_cols = [
        'gender', 'hypertension', 'heart_disease', 'ever_married', 
        'Residence_type', 'work_type_Never_worked', 
        'work_type_Private', 'work_type_Self-employed', 'work_type_children', 
        'smoking_status_formerly_smoked', 'smoking_status_never_smoked', 
        'smoking_status_smokes', "Glucose_Status_Prediabetic", "Glucose_Status_Diabetic",
        "age_group_child", "age_group_senior", "age_group_teen"
    ]

    # List of continuous columns:
    df["risk_factors_count"] = df["risk_factors_count"].astype("int32")
    float_cols = ['age', 'avg_glucose_level', 'bmi', "age_bmi_interaction", "age_hypertension_interaction"]

    # -----------------------------------------------------------------

    df["stroke"] = df["stroke"].astype('float32') # -> to store continuous probability scores instead of strict binary integers.

    # Convert binary cols to int32
    for col in binary_cols:
        df[col] = df[col].astype('int32')

    # Convert continuous features to float32 (7 decimal places)
    for col in float_cols:
        df[col] = df[col].astype('float32')

    # After optimization
    #print(df.info()) # memory usage: 518.9 KB


    """
    We didn’t choose uint8 because, most ML frameworks expect signed integers (int32/int64) for class labels,
    and using unsigned types can cause compatibility issues.
    """
        
    # END ------------------------------------------------------------- 
    return df









def select_features(df):
    """
    variance filtering,
    linear correlation analysis (numerical vars),
    chi square(hypothesis testing method aka test of independence) (categorical vars),
    RFE (Recursive Feature Elimination)
    """

    ## 1. variance filtering:
    df = variance_filter(df)

    # -----------------------------------------------------------------


    ## 2. linear correlation analysis:
    df = linear_corr_analysis(df)

    # -----------------------------------------------------------------

    ## 3. chi-square:
    ## the columns chi-square decided to drop:
    cols_chi_square_to_drop = perform_chi2(df)
    #print("cols_to_drop", cols_chi_square_to_drop) 
    # -> that's our fianl result that will help us take the decision of dropping or no.

    """
    Notice here we didn't drop them immediately because we need to make sure that's a good decision 
    so we need to support the decision with rfe results.
    """

    # -----------------------------------------------------------------

    ## 4. RFE (Recursive Feature Elimination with Cross-Validation)
    features, rfecv = run_rfecv(df)

    # df of the result:
    rfe_results = pd.DataFrame({
        'Feature': features,
        'Selected': rfecv.support_,
        'Ranking': rfecv.ranking_
    }).sort_values(by='Ranking')

    #print("\nFeature Rankings:")
    #print(rfe_results)


    # -----------------------------------------------------------------

    ## Final decision: based on Chi-Square, Variance, and RFE Results
    # Define the noise columns selected by using all 4 methods
    noise_cols = ['gender', 'Residence_type', 'smoking_status_smokes', 'work_type_Private']

    df = df.drop(columns=noise_cols, errors='ignore')

    """
    While smoking is a known clinical risk factor, in this specific dataset, 
    the smoking_status_smokes feature failed the Chi-Square test (p > 0.05) and was ranked low by RFECV. 

    This is likely due to the high volume of 'Unknown' entries in the original data
    and the fact that its predictive power was already captured by more direct physiological indicators like
    hypertension and risk_factors_count.
    """


    # END ------------------------------------------------------------- 
    return df







def linear_corr_analysis(df):
    """
    Corr matrix: for numerical values: (Redundancy(Multicollinearity) Check)
    """

    visualization.plot_correlation_matrix(df, filename='correlation_matrix.png')

    """
    Identify Multicollinearity. If two features are twins, drop one:

        age_hypertension_interaction: (0.97 correlation with hypertension). It's basically the same variable. Delete.

        age_bmi_interaction: (0.91 correlation with age). Delete.

        age_group_child: (0.91 correlation with work_type_children). Delete one of them.
    """

    # Removing interactions that are duplicates of existing features, It makes the model simpler and the RFE faster.
    df.drop(columns=["age_hypertension_interaction"],  inplace=True, errors='ignore')
    df.drop(columns=["age_bmi_interaction"],  inplace=True, errors='ignore')
    df.drop(columns=["work_type_children"],  inplace=True, errors='ignore')

    """
    Fixed multicollinearity because highly correlated features confuse the model, 
    leading it to unfairly penalize or inflate coefficients and produce unreliable predictions. 
    """ 

    # END ------------------------------------------------------------- 
    return df







def variance_filter(df):
    """
    Variance filtering
        drop the column only if it has near-zero variance OR zero predictive power (Constants). 
    """

    # Create a list of features excluding the target
    features_to_check = [col for col in df.columns if col != 'stroke']

    # -----------------------------------------------------------------

    # constant_cols: 0000000000 or 111111111 or 555555555555 etc, variance = 0
    constant_cols = [col for col in features_to_check if df[col].nunique() == 1]
    ##print("cols to drop: ", constant_cols)
    df.drop(columns = constant_cols, inplace = True)

    # -----------------------------------------------------------------

    # near zero cols: 0 0 0 0 0 0 0 0 1, 1 1 1 1 1 1 1 0
    # The feature technically varies, but 99% of values are the same, so it carries almost no information.
    near_zero_cols = []

    for col in features_to_check:
        top_freq = df[col].value_counts(normalize=True).iloc[0] # sorted by defaut DESC
        #print(df[col].value_counts(normalize=True))
        
        if top_freq > 0.95:
            near_zero_cols.append(col)

    ##print(near_zero_cols)
    df.drop(columns = near_zero_cols, inplace = True)

    """
    Zero Variance Features:
    A feature where all values are identical cannot help distinguish between classes. Since the feature never changes,
    the model cannot learn any useful pattern from it.

    Near-Zero Variance Features:
    If one value dominates the column (e.g., 98% of observations), the feature rarely changes 
    and therefore provides very little information for prediction.
    """

    # END ------------------------------------------------------------- 
    return df








def perform_chi2(df):
    """
    The Chi-Square test is a statistical hypothesis test used to determine if a significant relationship exists between two categorical 
    variables. It essentially functions as a "test of independence."
    https://www.youtube.com/watch?v=EjtXk-yEK6w&t=138s

    We begin by defining two competing views of the data:
        - Null Hypothesis (H0): There is no relationship between the two variables (they are independent).
        Any observed patterns are due to random chance (noise).

        - Alternative Hypothesis (Ha): There is a relationship between the variables (they are dependent).


        
    The test compares two sets of data:
        - Observed Frequencies (O): The actual counts you collected from your data.
        -Expected Frequencies (E): The counts you would expect to see if the two variables were perfectly independent.

        
    The chi^2 statistic measures the "gap" between these two. If the gap is large enough, 
    we reject H0 and conclude that the relationship is likely real, not just noise.



    Key Assumptions & Constraints:
    To ensure the test results are reliable, follow these rules:
        - The "Rule of 5": The expected frequency for each cell in your data table should be at least 5. 
    If the samples are too small, the test becomes inaccurate.

        - Data Type: It is designed for categorical data (nominal). While it works for unordered categories (like "City" or "Color"), 
    it does not account for the natural ranking in ordered data (like "Low," "Medium," "High").

        - Independence: Each observation must be independent (one person cannot be counted in two different categories).

        

    The Decision (P-Value)
    After calculating the Chi-Square value, we look at the P-value:
        - If P-value < 0.05: We reject the Null Hypothesis. The relationship is statistically significant.
        - If P-value > 0.05: We fail to reject the Null Hypothesis. The relationship is likely just noise.
    """


    
    """
    A p-value is the probability that the connection you see between a feature and a stroke is random
      rather than a real relationship.
    """


    # conditions:
    ## - nominal because we can't prove that smoking_status is ordinal because we can't say formerly_smokes < smokes
    ## we dont have enough info about years_smoking, packets smoked, ..

    ## - rule of 5 is checked
    ## - independece : each row = one patient, patients are independent

    ## all conditions are satisfied
    
    def filter_noise(df, cols):
        keep = []
        for col in cols:
            contingency_table = pd.crosstab(df[col], df['stroke'])
            
            # Run chi square test
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            
            # If any expected frequency is < 5, the p-value is unreliable
            if (expected < 5).any():
                # skip this col: fails the condition counts >= 5
                continue
                
            if p <= 0.05: 
                # keep this col: higher importance
                keep.append(col)
            else:
                # drop this col: noise col
                pass
                
        return keep


    ## final step: (cetegorical variables excluding target)
    cols_to_check = [
    'gender', 'hypertension', 'heart_disease', 'ever_married', 
    'Residence_type', 'work_type_Private', 'work_type_Self-employed',
    'smoking_status_smokes', "Glucose_Status_Prediabetic",'Glucose_Status_Diabetic',
    "age_group_child", "age_group_senior", "age_group_teen"
    ]

    cols_to_keep = filter_noise(df, cols_to_check)
    #print("cols_to_keep:", cols_to_keep)
    cols_to_drop = set(cols_to_check).difference(cols_to_keep)

    """
    The Chi-Square test was used to evaluate the independence between categorical features and the target variable.
    Features with p-values greater than 0.05 were considered statistically independent from the target 
    and were therefore removed as they likely represent noise
    """

    return cols_to_drop





def run_rfecv(df):
    """
    Recursive Feature Elimination with Cross-Validation (RFECV).
    
    RFE works by searching for a subset of features by starting with all features in the training dataset 
    and successfully removing features until only the most important ones remain. 
    It fits the model and ranks features by importance, discarding the least significant ones 
    to prevent overfitting and improve model interpretability.

    """
    
    X = df.drop(columns=['stroke'])
    y = df['stroke']
    
    # We use a Random Forest as the estimator for feature importance
    estimator = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight='balanced')

    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=StratifiedKFold(n_splits=5),
        scoring='roc_auc', # Focus on catching the positive cases (strokes)
        min_features_to_select=5,
        n_jobs=-1
    )

    rfecv.fit(X, y)
    #print(f"\nOptimal number of features: {rfecv.n_features_}") -> 8   



    """
    The Choice of "ROC-AUC" for RFE:
    Accuracy is misleading for imbalanced medical data. I used ROC-AUC because 
    it evaluates how well the model distinguishes between the two classes (Stroke vs. No Stroke)
     regardless of the threshold, which is critical for identifying high-risk patients."

    what's ROC_AUC:
    https://www.youtube.com/watch?v=4jRBRDbJemM

    Why Random Forest for RFE?
    Random Forest is non-parametric and captures non-linear interactions between features (like Age and Glucose) 
    that a linear model might miss. Since stroke risk factors are often complex and interactive,
    a Tree-based estimator provides a more realistic ranking of feature importance
    """
    
    return X.columns ,rfecv
