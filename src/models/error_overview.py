import os
import pandas as pd
from openpyxl.utils import get_column_letter


def update_error_overview(mae, data_name, target_variable, model_config, error_overview_path):
    """
    Updates an error overview Excel file with a new error for a given target variable,
    model configuration, and data version (derived from data_name). Records are stored in a MultiIndex
    DataFrame with (target_variable, model_config) as index. Additionally, for each target variable and version,
    the best (i.e. minimum) error over all configurations is computed and stored under the 'best' config.

    Parameters:
      mae: Numeric error metric.
      data_name: String; its last underscore-separated token is used as the version.
      target_variable: String representing the dependent variable.
      model_config: String representing the model configuration.
      error_overview_path: Path to the Excel file.

    The Excel file is assumed to have a header row (with error data versions) and a two-level row index.
    If not, it will be converted.
    """
    # Extract version from data_name (assumes version is the last underscore-separated token)
    version = data_name.split('_')[-1]

    if not os.path.exists(error_overview_path):
        index = pd.MultiIndex.from_tuples([], names=["target_variable", "model_config"])
        df = pd.DataFrame(columns=[version], index=index)
    else:
        df = pd.read_excel(error_overview_path, index_col=[0, 1], engine='openpyxl')
    print(df)
    # Add new version column if it doesn't exist.
    if version not in df.columns:
        df[version] = pd.NA

    # Update the row for the given target_variable and model_config.
    print(df)
    row_idx = (target_variable, model_config)
    df.loc[row_idx, version] = mae

    # Compute the best (minimum) error for non-'best' configurations.
    target_mask = df.index.get_level_values("target_variable") == target_variable
    non_best_mask = df.index.get_level_values("model_config") != 'best'
    subset = df[target_mask & non_best_mask][version]

    best_error = subset.min() if not subset.dropna().empty else pd.NA

    # Update (or add) the best row.
    best_row = (target_variable, 'best')
    df.loc[best_row, version] = best_error

    # Sort the DataFrame by index.
    print(df)
    df = df.sort_index()
    print(df)

    # Write to Excel with merge_cells disabled
    with pd.ExcelWriter(error_overview_path, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, merge_cells=True)

        # Auto-adjust the width of each column based on its content
        worksheet = writer.sheets['Sheet1']
        for column_cells in worksheet.columns:
            max_length = 0
            column = get_column_letter(column_cells[0].column)
            for cell in column_cells:
                try:
                    cell_length = len(str(cell.value))
                    if cell_length > max_length:
                        max_length = cell_length
                except Exception:
                    pass
            worksheet.column_dimensions[column].width = max_length + 2  # extra padding for clarity
