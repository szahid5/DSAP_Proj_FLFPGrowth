import pandas as pd
import os
import gc

def load_and_clean_all():
    # Set paths
    base_dir = os.path.join('data', 'raw') 
    
    # 1. Define files
    files = {
        "Average Age of Mothers": 'average-age-of-mothers.csv',
        "Children per Woman": 'children-per-woman-un.csv',
        "Female Emp/Pop Ratio": 'modeled-female-employment-to-population-ratio.csv',
        "FLFP Rates": 'female-labor-force-participation-rates.csv',
        "GDP per Capita": 'gdp-per-capita-worldbank.csv',
        "Urban Population": 'urban-and-rural-population.csv',
        "Female Unemployment": 'unemployment-rate-women.csv',
        "Years of Schooling": 'years-of-schooling.csv',
        "WBL Index": 'WBLHistorical.xlsx' # New Excel File
    }

    # 2. Robust File Processing
    def process_single_file(path, name):
        try:
            # Handle Excel vs CSV
            if path.endswith('.xlsx') or path.endswith('.xls'):
                # Target specific WBL sheet if applicable
                if "WBL" in name:
                    df = pd.read_excel(path, sheet_name="WBL Panel 2024")
                else:
                    df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)

            # Cleanup headers
            df.columns = df.columns.astype(str).str.strip()

            # Special Handling
            if "WBL" in name:
                if 'Report Year' in df.columns: df = df.rename(columns={'Report Year': 'Year'})
                if 'ISO Code' in df.columns: df = df.rename(columns={'ISO Code': 'Code'})
                elif 'Economy Code' in df.columns: df = df.rename(columns={'Economy Code': 'Code'})
                # Map the main score column
                score_col = [c for c in df.columns if 'WBL' in c.upper() and 'INDEX' in c.upper()]
                if score_col: df = df.rename(columns={score_col[0]: 'WBL_Score'})

            if name == "Average Age of Mothers":
                df.columns = df.columns.str.replace('period-', '', regex=False)
            if name == "Children per Woman":
                long_col = "Fertility rate - Sex: all - Age: all - Variant: estimates"
                if long_col in df.columns: df = df.rename(columns={long_col: "Children per Woman"})
                if 'time' in df.columns: df['Year'] = df['time']
            
            # Standardize country column
            if 'Entity' not in df.columns:
                if 'Country Name' in df.columns: df = df.rename(columns={'Country Name': 'Entity'})
                elif 'Country' in df.columns: df = df.rename(columns={'Country': 'Entity'})
                elif 'Economy' in df.columns: df = df.rename(columns={'Economy': 'Entity'})
            
            # Filter years
            if 'Year' in df.columns:
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                df = df.dropna(subset=['Year'])
                df['Year'] = df['Year'].astype(int)
                return df[(df['Year'] >= 1991) & (df['Year'] <= 2021)].copy()
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading {name}: {e}")
            return pd.DataFrame()

    # 3. Load Data
    data_frames = {}
    for name, filename in files.items():
        data_frames[name] = process_single_file(os.path.join(base_dir, filename), name)

    # 4. Prepare Merge List
    dfs_to_merge = [
        data_frames["Average Age of Mothers"], data_frames["Children per Woman"],
        data_frames["Female Emp/Pop Ratio"], data_frames["FLFP Rates"],
        data_frames["GDP per Capita"], data_frames["Urban Population"],
        data_frames["Female Unemployment"], data_frames["Years of Schooling"],
        data_frames["WBL Index"] # Added to merge list
    ]

    def clean_and_prep(df, index):
        if df.empty: return None, None
        df = df.copy()
        df.columns = df.columns.str.strip()
        if 'Country Code' in df.columns: df = df.rename(columns={'Country Code': 'Code'})
        if 'Country Name' in df.columns: df = df.rename(columns={'Country Name': 'Entity'})
        elif 'Country' in df.columns: df = df.rename(columns={'Country': 'Entity'})
        
        if 'Code' not in df.columns or 'Year' not in df.columns: return None, None
        
        df = df.dropna(subset=['Code', 'Year'])
        df['Code'] = df['Code'].astype(str).str.strip()
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype(int)
        df = df.drop_duplicates(subset=['Code', 'Year'])
        
        code_map = {}
        if 'Entity' in df.columns:
            code_map = df[['Code', 'Entity']].drop_duplicates().set_index('Code')['Entity'].to_dict()
            df = df.drop(columns=['Entity']) 
        return df, code_map

    # 5. Execute Merge
    df_master, master_code_map = clean_and_prep(dfs_to_merge[0], 0)
    for i in range(1, len(dfs_to_merge)):
        current_df, current_map = clean_and_prep(dfs_to_merge[i], i)
        if current_df is not None:
            master_code_map.update(current_map)
            df_master = pd.merge(df_master, current_df, on=['Code', 'Year'], how='outer')
            gc.collect()

    df_master['Country'] = df_master['Code'].map(master_code_map)
    
    # 6. Unified Renaming
    rename_map = {
        "age at childbearing": "Mean_Age_Mothers", "Fertility": "Fertility_Rate",
        "Employment to population": "Fem_Emp_Pop_Ratio", "Labor force participation": "FLFP_Rate",
        "GDP per capita": "GDP_Per_Capita", "Urban population": "Urban_Pop_Rate",
        "Unemployment, female": "Fem_Unemp_Rate", "years of schooling": "Years_Schooling",
        "WBL_Score": "WBL_Legal_Score" # Added WBL to final rename
    }
    
    new_cols = []
    for col in df_master.columns:
        renamed = False
        for keyword, short_name in rename_map.items():
            if keyword.lower() in str(col).lower():
                new_cols.append(short_name)
                renamed = True
                break
        if not renamed: new_cols.append(col)
    df_master.columns = new_cols
    
    # 7. Filtering and Final Cleanup
    entities_to_drop = ["World", "Arab World", "European Union", "OECD members", "High income"]
    df_clean = df_master[~df_master['Country'].isin(entities_to_drop)].copy()
    
    # Organize columns
    desired_order = ['Country', 'Code', 'Year', 'FLFP_Rate', 'WBL_Legal_Score', 'GDP_Per_Capita', 'Years_Schooling']
    final_cols = [c for c in desired_order if c in df_clean.columns] + [c for c in df_clean.columns if c not in desired_order]
    df_clean = df_clean[final_cols]

    if not os.path.exists('results'): os.makedirs('results')
    df_clean.to_csv('results/master_dataset_clean.csv', index=False)
    
    return df_clean

# Entry point
if __name__ == "__main__":
    load_and_clean_all()

# Summary stats update
def get_summary_statistics(df):
    df_filtered = df[(df['Year'] >= 1990) & (df['Year'] <= 2020)].copy()
    cols_to_include = [
        'FLFP_Rate', 'GDP_Per_Capita', 'Years_Schooling', 
        'Fertility_Rate', 'Mean_Age_Mothers', 'Fem_Emp_Pop_Ratio', 
        'Fem_Unemp_Rate', 'Urban_Pop_Rate', 'WBL_Legal_Score' # Added WBL to stats
    ]
    existing_cols = [c for c in cols_to_include if c in df_filtered.columns]
    stats = df_filtered[existing_cols].describe().transpose()
    stats.index.name = 'Variable'
    return stats