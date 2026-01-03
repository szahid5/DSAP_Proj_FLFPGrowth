# Import my custom modules
from src.data_loader import load_and_clean_all
from src.models import prepare_time_series_split, train_and_evaluate
from src.evaluation import save_results

def main():
    # 1. Load and clean the data
    print("Step 1: Loading and Merging Data...")
    df_clean = load_and_clean_all()

    # 2. Prepare the split and calculate growth
    print("Step 2: Calculating Growth and Splitting Data...")
    X_train, X_test, y_train, y_test, X_cols = prepare_time_series_split(df_clean)

    # 3. Train models and select best performance
    print("Step 3: Training Models and Finding the Best One...")
    best_pipeline, results_df = train_and_evaluate(X_train, y_train, X_test, y_test)

    # 4. Save evaluation visuals and miracle countries list
    print("Step 4: Generating Evaluation Plots in /results...")
    save_results(best_pipeline, X_test, y_test, df_clean, X_cols)

    print("\nProject successfully executed!")
    print("Final results can be found in the 'results/' folder.")

# Run the project
if __name__ == "__main__":
    main()