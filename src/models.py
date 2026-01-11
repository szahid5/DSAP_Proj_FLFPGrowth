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

def prepare_time_series_split(df_clean):
    # 1. Filter for 5-Year Intervals
    years_of_interest = [1991, 1996, 2001, 2006, 2011, 2016, 2021]
    df_panel = df_clean[df_clean['Year'].isin(years_of_interest)].copy()

    # 2. Critical sorting for time-series operations
    df_panel = df_panel.sort_values(by=['Country', 'Year'])

    # 3. Feature Engineering: 5-Year Lag for WBL
    # This checks if legal rights from 5 years ago impact growth today
    if 'WBL_Legal_Score' in df_panel.columns:
        df_panel['WBL_Lagged'] = df_panel.groupby('Country')['WBL_Legal_Score'].shift(1)

    # 4. Growth formula (Target variable)
    df_panel['Next_FLFP'] = df_panel.groupby('Country')['FLFP_Rate'].shift(-1)
    df_panel['FLFP_Growth_Next_5Y'] = (df_panel['Next_FLFP'] - df_panel['FLFP_Rate']) / df_panel['FLFP_Rate']

    # 5. Handle Outliers and NAs
    df_panel['FLFP_Growth_Next_5Y'] = df_panel['FLFP_Growth_Next_5Y'].clip(lower=-0.5, upper=0.5)
    
    # Drop rows missing the target OR the lagged legal data
    cols_to_check = ['FLFP_Growth_Next_5Y']
    if 'WBL_Lagged' in df_panel.columns:
        cols_to_check.append('WBL_Lagged')
        
    df_model = df_panel.dropna(subset=cols_to_check).copy()

    # 6. Define Feature List
    X_cols = [
        'Fem_Emp_Pop_Ratio', 'Fem_Unemp_Rate', 'Mean_Age_Mothers', 
        'GDP_Per_Capita', 'Urban_Pop_Rate', 'Years_Schooling', 'Fertility_Rate'
    ]
    
    if 'WBL_Lagged' in df_model.columns:
        X_cols.append('WBL_Lagged')

    y_col = 'FLFP_Growth_Next_5Y'

    # 7. Temporal Split
    # Training: 1996 - 2006 (Uses 1991 as lag)
    # Testing: 2011 - 2016 
    train_mask = (df_model['Year'] >= 1996) & (df_model['Year'] <= 2006)
    test_mask = (df_model['Year'] >= 2011) & (df_model['Year'] <= 2016)

    X_train = df_model.loc[train_mask, X_cols].copy()
    y_train = df_model.loc[train_mask, y_col]
    X_test = df_model.loc[test_mask, X_cols].copy()
    y_test = df_model.loc[test_mask, y_col]
    
    print(f"Temporal Split: Train years 1996-2006 | Test years 2011-2016")
    print(f"Train size: {X_train.shape} | Test size: {X_test.shape}")
    print(f"Features used: {X_cols}")

    return X_train, X_test, y_train, y_test, X_cols

def train_and_evaluate(X_train, y_train, X_test, y_test):
    # 8. Define Model Suite
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results_list = []

    # 9. Pipeline Loop
    for name, m in models.items():
        p = Pipeline([
            ('impute', SimpleImputer(strategy='mean')), 
            ('scale', StandardScaler()), 
            ('model', m)
        ])
        
        # CV scores (Training performance)
        scores = cross_val_score(p, X_train, y_train, cv=cv, scoring='r2')
        
        # Training and Testing
        p.fit(X_train, y_train)
        preds = p.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        results_list.append({
            "Model": name,
            "CV_R2": scores.mean(),
            "Test_RMSE": rmse,
            "Test_R2": r2,
            "Pipeline": p
        })

    # 10. Find Best Pipeline
    res_df = pd.DataFrame(results_list).sort_values(by='Test_R2', ascending=False)
    best_p = res_df.iloc[0]['Pipeline']
    
    print(f"Best Model Found: {res_df.iloc[0]['Model']} (Test R2: {res_df.iloc[0]['Test_R2']:.4f})")
    
    return best_p, res_df