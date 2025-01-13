import glob
import json
import pickle

import joblib
import logging
import xgboost

import pandas as pd
from typing import Tuple, Dict, List

from sklearn.base import is_outlier_detector
from xgboost import Booster

from utilities.path_utils import *
from utilities.files_utils import read_data_frame

# TODO:
# 1. Create a file (path_manager.py) to handle all object paths.
# 2. Create a file (loader.py) to handle object loading using paths from path_manager.
# 3. Create a file (exporter.py) to handle object exports.
# 4. In the main code, use simple naming for objects, abstracting path, loading, and export logic.



def load_lookups_table(target_variable: str) -> pd.DataFrame:
    lookups_table_path = get_mtpl_postal_categories_path(target_variable)
    lookups_table = pd.read_csv(lookups_table_path)
    lookups_table = lookups_table[lookups_table['title'].str.contains('territory', case=False)].drop_duplicates(
        subset=['value'])
    lookups_table['value'] = lookups_table['value'].astype(int)
    lookups_table = lookups_table[['key', 'value']]
    return lookups_table


def load_lookups_default_value(target_variable: str):
    try:
        default_values_path = get_mtpl_default_values_path(target_variable)
        default_value = pd.read_csv(default_values_path)
        return default_value[default_value['key'] == 'territory'].iloc[0]['value']
    except IndexError:
        return None


def to_interval(x):
    if str(x) == 'nan':
        return x
    mn, mx = str(x).replace('(', '').replace(']', '').split(',')
    mn = int(mn)
    mx = int(mx)

    return pd.Interval(mn, mx, closed='right')


def encode_categorical_columns(values: pd.Series, feature: str) -> pd.Series:
    feature_encoder = joblib.load(get_encoder_path(feature))

    if feature_encoder.__class__.__name__ == 'OrdinalEncoder':
        tmp = values
        if '_cut' in feature:
            tmp = tmp.apply(to_interval)
        tmp = tmp.values.reshape(-1, 1)
        try:
            transformed_values = feature_encoder.transform(tmp).ravel()
        except Exception as e:
            print(f"Error transforming values for feature '{feature}': {e}")
            raise
        return pd.Series(transformed_values, index=values.index)

    elif feature_encoder.__class__.__name__ == 'LabelEncoder':
        def transform(x):
            try:
                return feature_encoder.transform([x])[0]
            except ValueError:
                return 1  # Or any other default value indicating an unknown label

        try:
            transformed_values = values.apply(transform)
        except Exception as e:
            print(f"Error applying transform for feature '{feature}': {e}")
            raise
        return transformed_values

    else:
        raise ValueError("Unknown encoder type")


def choose_columns_specific_for_target_variable(data : pd.DataFrame, features : list, target_variable : str) -> tuple:
    cut_cols_to_remove = [col for col in data.columns if target_variable not in col and 'cut' in col]
    is_outlier_to_remove = [col for col in data.columns if target_variable.strip('log_') not in col and col.startswith('is_outlier_')]
    features = [col for col in features if col not in cut_cols_to_remove]
    features = [col for col in features if col not in is_outlier_to_remove]
    return data.drop(columns = cut_cols_to_remove), features


def load_data_name_reference() -> dict:
    data_name_reference_path = get_data_name_references_path()
    if os.path.exists(data_name_reference_path):
        with open(data_name_reference_path, 'r') as json_file:
            data_name_reference = json.load(json_file)
    else:
        data_name_reference = {}
    return data_name_reference

def load_names_file(names_file) -> list:
    names_file_path = get_names_file_path(names_file)
    with open(names_file_path, 'r') as file:
        names = names_file_path.read_text().strip().split('\n')
    return sorted(names)

def reconstruct_features_model(features_model_dict: Dict[str, Dict[str, str]]) -> List[str]:
    reconstructed_features = []

    for col, props in features_model_dict.items():
        if "dummy_values" in props:
            dummy_values = props["dummy_values"].split('#')
            for val in dummy_values:
                dummy_column = f"{col}__{val}__dummy"
                reconstructed_features.append(dummy_column)
        else:
            reconstructed_features.append(col)

    return reconstructed_features


def load_data(processed_data_name: str, target_variable: str = None,
              drop_target_na=True) -> Tuple[pd.DataFrame, list, list, list]:

    data_path = get_processed_data_path(processed_data_name)

    data = read_data_frame(data_path)
    logging.info("Imported data...")

    logging.info("Imported feature data...")

    data_name_reference = load_data_name_reference()
    if processed_data_name in data_name_reference.keys():
        data_info = data_name_reference[processed_data_name]
    else:
        raise Exception('Data is not in the data name reference, something went wrong, please regenerate the data')

    features_info = data_info['features_info']
    features_on_top = data_info['features_on_top']
    features_model = reconstruct_features_model(data_info['features_model'])

    cut_cols = [col for col in features_model if col.endswith('_cut')]
    data[cut_cols] = data[cut_cols].astype('category')

    if target_variable is not None:
        data, features_model = choose_columns_specific_for_target_variable(data, features_model, target_variable)
        data = data[features_info + features_on_top + features_model + [target_variable]]
        if drop_target_na:
            data = data.dropna(subset=[target_variable])
    return data, features_info, features_on_top, features_model


def check_model_existence(model_name : str):
    model_path = get_model_path(model_name)
    return os.path.exists(model_path)

def load_model(model_name : str) -> Booster | None:
    if model_name is None:
        model = None
    else:
        model_path = get_model_path(model_name)
        try:
            model = xgboost.Booster(model_file=str(model_path))
        except Exception as e:
            model = None
    return model


def load_out_of_sample_predictions(model_name : str, target_variable: str) -> pd.Series:
    out_of_sample_predictions_path = get_model_cv_out_of_sample_predictions_path(model_name)
    out_of_sample_predictions = pd.read_csv(out_of_sample_predictions_path)
    out_of_sample_predictions.columns = ['id_case', target_variable]
    out_of_sample_predictions = out_of_sample_predictions.set_index('id_case')
    out_of_sample_predictions = out_of_sample_predictions[target_variable]
    return out_of_sample_predictions

def load_hyperopt_trials(model_name : str):
    trials_path = get_model_trials_path(model_name)
    with open(trials_path, 'rb') as file:
        trials = pickle.load(file)
    return trials

def load_params(service: str, params_v: str) -> dict[str, pd.DataFrame]:
    params_path = get_params_path(service, params_v)
    params = {}
    for param_file in params_path.glob('*.csv'):
        param = param_file.stem
        params[param] = pd.read_csv(param_file)
    return params


def load_other(service: str) -> dict[str, pd.DataFrame]:
    others_path = get_others_path(service)
    others = {}
    for other_file in others_path.glob('*'):
        other = other_file.stem
        ext = other_file.suffix[1:]
        if ext == 'csv':
            others[other] = pd.read_csv(other_file, low_memory=False)
        elif ext == 'xlsx':
            others[other] = pd.read_excel(other_file)
        else:
            others[other] = pd.read_table(other_file, header=None, usecols=[1])
    return others

def load_transposed_config(service : str, date : str):

    transposed_config_path = get_transposed_config_path(service, date)
    with open(transposed_config_path, 'r') as json_file:
        data_loaded = json.load(json_file)
    return data_loaded


def reconstruct_categorical_variables(data : pd.DataFrame) -> pd.DataFrame:
    dummy_columns = [col for col in data.columns if col.endswith('__dummy')]
    original_vars = set([col.split('__')[0] for col in dummy_columns])
    dummies_by_var = {var : [col for col in dummy_columns if col.startswith(var + '__')] for var in original_vars }

    new_data = data.copy()
    for var, dummies in dummies_by_var.items():
        var_col = pd.from_dummies(data[dummies].rename(columns={col: col.replace('__dummy', '').replace(f'{var}__', '') for col in dummies}), default_category=None)
        new_data[var] = var_col
        new_data = new_data.drop(columns=dummies)
    return new_data

def load_distribution(service: str, params_v: str):
    return load_params(service, params_v), load_other(service)

def load_on_top_file(service : str) -> pd.DataFrame:
    on_top_path = get_on_top_factor_files(service)
    on_top = pd.read_csv(on_top_path, sep = ';')
    on_top['features'] = on_top['features'].apply(eval)
    on_top['feature_values'] = on_top['feature_values'].apply(eval)
    on_top['factor'] = on_top['factor'].fillna(1)
    return on_top


