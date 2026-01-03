import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

def save_results(best_pipeline, X_test, y_test, df_model, X_cols):
    # Setup results folder
    if not os.path.exists('results'):
        os.makedirs('results')

    # Extract model for coefficients
    ridge_model = best_pipeline.named_steps['model']
    
    # 1. Visualize Ridge Coefficients
    plt.figure(figsize=(10, 6))
    coef_df = pd.DataFrame({
        'Feature': X_cols,
        'Coefficient': ridge_model.coef_ if hasattr(ridge_model, 'coef_') else 0
    }).sort_values(by='Coefficient', ascending=False)
    
    sns.barplot(data=coef_df, x='Coefficient', y='Feature')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title('What Drives FLFP Growth?')
    plt.savefig('results/ridge_coefficients.png')
    plt.close()

    # 2. Permutation Importance
    perm_importance = permutation_importance(
        best_pipeline, X_test, y_test, 
        n_repeats=10, random_state=42, scoring='r2'
    )
    sorted_idx = perm_importance.importances_mean.argsort()
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        perm_importance.importances[sorted_idx].T,
        vert=False,
        labels=np.array(X_cols)[sorted_idx]
    )
    plt.title("Permutation Importance (Test Set)")
    plt.savefig('results/permutation_importance.png')
    plt.close()

    # 3. Partial Dependence Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(
        best_pipeline, 
        X_test, 
        features=['Fem_Emp_Pop_Ratio'], 
        kind='average',
        ax=ax
    )
    plt.title("Convergence Effect: Baseline Employment vs. Predicted Growth")
    plt.savefig('results/partial_dependence.png')
    plt.close()

    # 4. Predicted vs. Actual Growth
    y_pred = best_pipeline.predict(X_test)
    results_df = X_test.copy()
    results_df['Actual_Growth'] = y_test
    results_df['Predicted_Growth'] = y_pred
    results_df['Residual'] = results_df['Actual_Growth'] - results_df['Predicted_Growth']
    
    # Map country names back
    results_df['Country'] = df_model.loc[X_test.index, 'Country']
    results_df['Year'] = df_model.loc[X_test.index, 'Year']

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=results_df, x='Predicted_Growth', y='Actual_Growth', alpha=0.6)
    
    # Perfect prediction line
    line_min = min(results_df['Actual_Growth'].min(), results_df['Predicted_Growth'].min())
    line_max = max(results_df['Actual_Growth'].max(), results_df['Predicted_Growth'].max())
    plt.plot([line_min, line_max], [line_min, line_max], color='red', linestyle='--')
    
    plt.title('Reality Check: Did the Model Predict Correctly?')
    plt.savefig('results/reality_check.png')
    plt.close()

    # Save miracle countries
    overperformers = results_df.sort_values(by='Residual', ascending=False).head(5)
    overperformers.to_csv('results/miracle_countries.csv', index=False)