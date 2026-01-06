# 1. Import custom modules at the top
from src.data_loader import load_and_clean_all, get_summary_statistics
from src.models import prepare_time_series_split, train_and_evaluate
from src.evaluation import save_results
import pandas as pd

def main():
    # 2. Load and clean the data
    print("Step 1: Loading and Merging Data...")
    df_clean = load_and_clean_all()

    # 3. Generate and save summary statistics
    print("Step 2: Generating Summary Statistics...")
    # Use df_clean
    df_stats = get_summary_statistics(df_clean)
    df_stats.to_csv('results/summary_statistics.csv')
    print("Summary statistics saved to results/summary_statistics.csv")

    # 4. Prepare the split and calculate growth
    print("Step 3: Calculating Growth and Splitting Data...")
    X_train, X_test, y_train, y_test, X_cols = prepare_time_series_split(df_clean)

    # 5. Train models and select best performance
    print("Step 4: Training Models and Finding the Best One...")
    best_pipeline, results_df = train_and_evaluate(X_train, y_train, X_test, y_test)
    
    # ADD THIS LINE TO SAVE THE PERFORMANCE METRICS
    results_df.to_csv('results/model_performance_metrics.csv', index=False)
    print("Performance metrics saved to results/model_performance_metrics.csv")

    # 6. Save evaluation visuals and miracle countries list
    print("Step 5: Generating Evaluation Plots in /results...")
    save_results(best_pipeline, X_test, y_test, df_clean, X_cols)

    print("\nProject successfully executed!")
    print("Final results can be found in the 'results/' folder.")

# Run the project
if __name__ == "__main__":
    main()