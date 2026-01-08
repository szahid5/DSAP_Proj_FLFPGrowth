import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

def save_results(best_pipeline, X_test, y_test, df_model, X_cols):
    # 0. Setup results folder
    if not os.path.exists('results'):
        os.makedirs('results')

    ridge_model = best_pipeline.named_steps['model']
    
    # 1. Visualize Ridge Coefficients
    plt.figure(figsize=(10, 6))
    coef_df = pd.DataFrame({
        'Feature': X_cols,
        'Coefficient': ridge_model.coef_ if hasattr(ridge_model, 'coef_') else 0
    }).sort_values(by='Coefficient', ascending=False)
    
    sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='viridis')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title('Impact of Features on FLFP Growth')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.savefig('results/ridge_coefficients.png')
    plt.close()

    # 2. Permutation Importance
    perm_importance = permutation_importance(
        best_pipeline, X_test, y_test, 
        n_repeats=30, random_state=42, scoring='r2'
    )
    sorted_idx = perm_importance.importances_mean.argsort()
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        perm_importance.importances[sorted_idx].T,
        vert=False,
        labels=np.array(X_cols)[sorted_idx]
    )
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title("Feature Importance (R2 Drop)")
    plt.savefig('results/permutation_importance.png')
    plt.close()

    # 3. Partial Dependence Plot
    # We include both the Baseline Employment and WBL to compare convergence vs. legal impact
    pdp_features = ['Fem_Emp_Pop_Ratio']
    if 'WBL_Lagged' in X_cols: pdp_features.append('WBL_Lagged')
    elif 'WBL_Legal_Score' in X_cols: pdp_features.append('WBL_Legal_Score')

    fig, ax = plt.subplots(len(pdp_features), 1, figsize=(10, 5 * len(pdp_features)))
    if len(pdp_features) == 1: ax = [ax]
    
    PartialDependenceDisplay.from_estimator(
        best_pipeline, 
        X_test, 
        features=pdp_features, 
        kind='average',
        ax=ax
    )
    plt.savefig('results/partial_dependence.png')
    plt.close()

    # 4. Predicted vs. Actual Growth
    y_pred = best_pipeline.predict(X_test)
    results_df = X_test.copy()
    results_df['Actual_Growth'] = y_test
    results_df['Predicted_Growth'] = y_pred
    results_df['Residual'] = results_df['Actual_Growth'] - results_df['Predicted_Growth']
    
    results_df['Country'] = df_model.loc[X_test.index, 'Country']
    results_df['Year'] = df_model.loc[X_test.index, 'Year']

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=results_df, x='Predicted_Growth', y='Actual_Growth', alpha=0.6, hue='Year', palette='deep')
    
    line_min = min(results_df['Actual_Growth'].min(), results_df['Predicted_Growth'].min())
    line_max = max(results_df['Actual_Growth'].max(), results_df['Predicted_Growth'].max())
    plt.plot([line_min, line_max], [line_min, line_max], color='red', linestyle='--')
    
    plt.title('Reality Check: Actual vs. Predicted Growth')
    plt.savefig('results/reality_check.png')
    plt.close()

    # 5. Save Miracle Countries
    overperformers = results_df.sort_values(by='Residual', ascending=False).head(10)
    overperformers.to_csv('results/miracle_countries.csv', index=False)