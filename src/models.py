import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_time_series_split(df_clean):
    # 1. Year filtering 
    years_of_interest = [1991, 1996, 2001, 2006, 2011, 2016, 2021]
    df_panel = df_clean[df_clean['Year'].isin(years_of_interest)].copy()

    # 2. Critical sorting to ensure shift(-1) targets the correct year
    df_panel = df_panel.sort_values(by=['Country', 'Year'])

    # 3. Growth formula (Lead the Target)
    df_panel['Next_FLFP'] = df_panel.groupby('Country')['FLFP_Rate'].shift(-1)
    df_panel['FLFP_Growth_Next_5Y'] = (df_panel['Next_FLFP'] - df_panel['FLFP_Rate']) / df_panel['FLFP_Rate']

    # 4. Handle Outliers and NAs
    df_panel['FLFP_Growth_Next_5Y'] = df_panel['FLFP_Growth_Next_5Y'].clip(lower=-0.5, upper=0.5)
    df_model = df_panel.dropna(subset=['FLFP_Growth_Next_5Y']).copy()

    # 5. Define Feature and Target Columns
    X_cols = ['Fem_Emp_Pop_Ratio', 'Fem_Unemp_Rate', 'Mean_Age_Mothers', 
              'GDP_Per_Capita', 'Urban_Pop_Rate', 'Years_Schooling', 'Fertility_Rate']
    y_col = 'FLFP_Growth_Next_5Y'

    # 6. Manual mask splitting (Temporal Validation as per proposal)
    # Training: 1991, 1996, 2001, 2006 
    # Testing: 2011, 2016
    train_mask = df_model['Year'] <= 2006
    test_mask = (df_model['Year'] >= 2011) & (df_model['Year'] <= 2016)

    X_train_raw = df_model.loc[train_mask, X_cols]
    y_train = df_model.loc[train_mask, y_col]
    X_test_raw = df_model.loc[test_mask, X_cols]
    y_test = df_model.loc[test_mask, y_col]

    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()

    
    print(f"Temporal Split: Train years <= 2006 | Test years 2011-2016")
    print(f"Train size: {X_train.shape} | Test size: {X_test.shape}")


    # To get the exact number of contires in 'train' and 'test'
    n_countries_train = df_model.loc[train_mask, 'Country'].nunique()
    n_countries_test = df_model.loc[test_mask, 'Country'].nunique()

    print(f"Total Unique Countries in Training: {n_countries_train}")
    print(f"Total Unique Countries in Testing: {n_countries_test}")

    return X_train, X_test, y_train, y_test, X_cols
    
# Model training
def train_and_evaluate(X_train, y_train, X_test, y_test):
    # List of models to try
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
    }

    # Cross validation setup
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results_list = []

    # Loop through models
    for name, m in models.items():
        # Setup pipeline
        p = Pipeline([
            ('impute', SimpleImputer(strategy='mean')), 
            ('scale', StandardScaler()), 
            ('model', m)
        ])
        
        # CV scores
        scores = cross_val_score(p, X_train, y_train, cv=cv, scoring='r2')
        avg_cv = scores.mean()
        
        # Training
        p.fit(X_train, y_train)
        
        # Predict and score
        preds = p.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        # Manual dictionary storage
        d = {
            "Model": name,
            "CV_R2": avg_cv,
            "Test_RMSE": rmse,
            "Test_R2": r2,
            "Pipeline": p
        }
        results_list.append(d)

    # Sort to find best
    res_df = pd.DataFrame(results_list).sort_values(by='Test_R2', ascending=False)
    best_p = res_df.iloc[0]['Pipeline']
    
    return best_p, res_df