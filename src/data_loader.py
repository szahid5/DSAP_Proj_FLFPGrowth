import pandas as pd
import os
import gc

def load_and_clean_all():
    # Setup paths
    base_dir = os.path.join('data', 'raw') 
    
    # Define files
    files = {
        "Average Age of Mothers": 'average-age-of-mothers.csv',
        "Children per Woman": 'children-per-woman-un.csv',
        "Female Emp/Pop Ratio": 'modeled-female-employment-to-population-ratio.csv',
        "FLFP Rates": 'female-labor-force-participation-rates.csv',
        "GDP per Capita": 'gdp-per-capita-worldbank.csv',
        "Urban Population": 'urban-and-rural-population.csv',
        "Female Unemployment": 'unemployment-rate-women.csv',
        "Years of Schooling": 'years-of-schooling.csv'
    }

    # Initial cleaning
    def process_single_file(path, name):
        try:
            df = pd.read_csv(path)
            if name == "Average Age of Mothers":
                df.columns = df.columns.str.replace('period-', '', regex=False)
            if name == "Children per Woman":
                long_col = "Fertility rate - Sex: all - Age: all - Variant: estimates"
                if long_col in df.columns:
                    df = df.rename(columns={long_col: "Children per Woman"})
                if 'time' in df.columns:
                    df['Year'] = df['time']
            
            # Standardize country column
            if 'Entity' not in df.columns:
                if 'Country Name' in df.columns: df = df.rename(columns={'Country Name': 'Entity'})
                elif 'Country' in df.columns: df = df.rename(columns={'Country': 'Entity'})
            
            # Filter years
            if 'Year' in df.columns:
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                df = df.dropna(subset=['Year'])
                df['Year'] = df['Year'].astype(int)
                return df[(df['Year'] >= 1991) & (df['Year'] <= 2021)].copy()
            return pd.DataFrame()
        except Exception as e:
            return pd.DataFrame()

    # Load data
    data_frames = {}
    for name, filename in files.items():
        data_frames[name] = process_single_file(os.path.join(base_dir, filename), name)

    # Prepare for merge
    dfs_to_merge = [
        data_frames["Average Age of Mothers"], data_frames["Children per Woman"],
        data_frames["Female Emp/Pop Ratio"], data_frames["FLFP Rates"],
        data_frames["GDP per Capita"], data_frames["Urban Population"],
        data_frames["Female Unemployment"], data_frames["Years of Schooling"]
    ]

    # Clean keys
    def clean_and_prep(df, index):
        df = df.copy()
        df.columns = df.columns.str.strip()
        if 'Country Code' in df.columns: df = df.rename(columns={'Country Code': 'Code'})
        if 'Country Name' in df.columns: df = df.rename(columns={'Country Name': 'Entity'})
        elif 'Country' in df.columns: df = df.rename(columns={'Country': 'Entity'})
        
        if 'Code' not in df.columns or 'Year' not in df.columns: return None, None
        
        df = df.dropna(subset=['Code', 'Year'])
        df['Code'] = df['Code'].astype(str).str.strip()
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').dropna().astype(int)
        
        # Deduplicate
        df = df.drop_duplicates(subset=['Code', 'Year'])
        
        # Separate names
        code_map = {}
        if 'Entity' in df.columns:
            code_map = df[['Code', 'Entity']].drop_duplicates().set_index('Code')['Entity'].to_dict()
            df = df.drop(columns=['Entity']) 
        return df, code_map

    # Merging
    df_master, master_code_map = clean_and_prep(dfs_to_merge[0], 0)
    for i in range(1, len(dfs_to_merge)):
        current_df, current_map = clean_and_prep(dfs_to_merge[i], i)
        if current_df is not None:
            master_code_map.update(current_map)
            df_master = pd.merge(df_master, current_df, on=['Code', 'Year'], how='outer')
            gc.collect()

    # Re-attach country names
    df_master['Country'] = df_master['Code'].map(master_code_map)
    
    # Rename columns
    rename_map = {
        "age at childbearing": "Mean_Age_Mothers", "Fertility": "Fertility_Rate",
        "Employment to population": "Fem_Emp_Pop_Ratio", "Labor force participation": "FLFP_Rate",
        "GDP per capita": "GDP_Per_Capita", "Urban population": "Urban_Pop_Rate",
        "Unemployment, female": "Fem_Unemp_Rate", "years of schooling": "Years_Schooling"
    }
    new_cols = df_master.columns.tolist()
    for i, col in enumerate(new_cols):
        for keyword, short_name in rename_map.items():
            if keyword.lower() in col.lower():
                new_cols[i] = short_name
                break 
    df_master.columns = new_cols
    
    # Filtering
    entities_to_drop = ["World", "Arab World", "European Union", "OECD members", "High income"]
    df_clean = df_master[~df_master['Country'].isin(entities_to_drop)].copy()
    
    # Save results
    if not os.path.exists('results'): os.makedirs('results')
    df_clean.to_csv('results/master_dataset_clean.csv', index=False)
    
    return df_clean

# Entry point
if __name__ == "__main__":
    load_and_clean_all()

#summary stats (for paper)
def get_summary_statistics(df):
    """
    Generates summary statistics for the merged dataset 
    filtered to the 1990-2020 analysis period.
    """
    # 1. Filter for the analysis years
    df_filtered = df[(df['Year'] >= 1990) & (df['Year'] <= 2020)].copy()
    
    # 2. Select only the data from 1990 to 2020 for each varibla
    cols_to_include = [
        'FLFP_Rate', 'GDP_Per_Capita', 'Years_Schooling', 
        'Fertility_Rate', 'Mean_Age_Mothers', 'Fem_Emp_Pop_Ratio', 
        'Fem_Unemp_Rate', 'Urban_Pop_Rate'
    ]
    
    # Ensure only columns that actually exist in the df are used
    existing_cols = [c for c in cols_to_include if c in df_filtered.columns]
    
    stats = df_filtered[existing_cols].describe().transpose()
    stats.index.name = 'Variable'
    
    return stats