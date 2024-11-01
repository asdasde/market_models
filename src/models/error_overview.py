import pandas as pd
import os
from pathlib import Path

def update_error_overview(mae, data_name, target_variable, error_overview_path):
    version = data_name.split('_')[-1]
    
    df = pd.read_excel(error_overview_path, index_col=0)

    if version not in df.columns:
        df[version] = None

    if target_variable in df.index:
        df.at[target_variable, version] = mae
    else:
        print(f"Warning: Target variable '{target_variable}' not found in the file '{error_overview_path}'.")

    with pd.ExcelWriter(error_overview_path, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
        df.to_excel(writer)