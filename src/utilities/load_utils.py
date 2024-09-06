import glob
import joblib
import logging
import xgboost

import pandas as pd
from pathlib import Path
from typing import Tuple

from PIL.features import features

from utilities.path_utils import *
from utilities.files_utils import read_file
from utilities.constants import FEATURES_TO_IGNORE, BONUS_MALUS_CLASSES_DICT


def load_lookups_table(lookups_table_path: Path) -> pd.DataFrame:
    lookups_table = pd.read_csv(lookups_table_path)
    lookups_table = lookups_table[lookups_table['title'].str.contains('territory', case=False)].drop_duplicates(
        subset=['value'])
    lookups_table['value'] = lookups_table['value'].astype(int)
    lookups_table = lookups_table[['key', 'value']]
    return lookups_table


def load_lookups_default_value(default_values_path: Path):
    try:
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


def apply_features(data: pd.DataFrame, features: list, feature_dtypes: dict) -> pd.DataFrame:
    for feature in features:
        if feature_dtypes[feature] == 'index':
            data = data.set_index(feature)
            features.remove(feature)
            continue

        if feature == 'BonusMalus':
            data[feature] = data[feature].apply(lambda x: BONUS_MALUS_CLASSES_DICT.get(x, x))
        data[feature] = data[feature].astype(feature_dtypes[feature])

        if data[feature].dtype == 'category':
            data[feature] = encode_categorical_columns(data[feature], feature)
    return data


def choose_columns_specific_for_target_variable(data : pd.DataFrame, features : list, target_variable : str) -> tuple:
    cut_cols_to_remove = [col for col in data.columns if target_variable not in col and 'cut' in col]
    features = [col for col in features if col not in cut_cols_to_remove]
    return data.drop(columns = cut_cols_to_remove), features



def load_features(features_path : Path) -> tuple:
    with open(features_path) as file:
        features = file.readlines()
        features = [feature.replace('\n', '') for feature in features]
        feature_dtypes = {feature.split(' :: ')[0]: feature.split(' :: ')[1] for feature in features}
        features = [feature.split(' :: ')[0] for feature in features]
    FEATURES_TO_IGNORE = ['DateCrawled']
    features = [feature for feature in features if feature not in FEATURES_TO_IGNORE]
    return features, feature_dtypes

def load_data(data_path: Path, features_path: Path, target_variable: str = None, apply_feature_dtypes: bool = True,
              drop_target_na=True) -> Tuple[pd.DataFrame, list]:
    data = read_file(data_path)
    logging.info("Imported data...")

    features, feature_dtypes = load_features(features_path)
    logging.info("Imported feature data...")

    if apply_feature_dtypes:
        data = apply_features(data, features, feature_dtypes)

    if target_variable is not None:
        data, features = choose_columns_specific_for_target_variable(data, features, target_variable)

    columns = features

    if target_variable is None:
        data = data[columns]
    else:
        data = data[columns + [target_variable]]
        if drop_target_na:
            data = data.dropna(subset=[target_variable])

    return data, features


def load_model(model_path: Path) -> xgboost.Booster:
    try:
        return xgboost.Booster(model_file=str(model_path))
    except Exception:
        return None

def load_out_of_sample_predictions(out_of_sample_predictions_path : Path, target_variable: str) -> pd.DataFrame:
    out_of_sample_predictions = pd.read_csv(out_of_sample_predictions_path)
    out_of_sample_predictions.columns = ['id_case', target_variable]
    out_of_sample_predictions = out_of_sample_predictions.set_index('id_case')
    out_of_sample_predictions = out_of_sample_predictions[target_variable]
    return out_of_sample_predictions

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


def reconstruct_categorical_variables(data : pd.DataFrame) -> pd.DataFrame:
    dummy_columns = [col for col in data.columns if col.endswith('__dummy')]
    original_vars = set([col.split('__')[0] for col in dummy_columns])
    dummies_by_var = {var : [col for col in dummy_columns if col.startswith(var + '__')] for var in original_vars }

    new_data = data.copy()
    for var, dummies in dummies_by_var.items():
        var_col = pd.from_dummies(data[dummies].rename(columns={col: col.replace('__dummy', '').replace(f'{var}__', '') for col in dummies}))
        new_data[var] = var_col
        new_data = new_data.drop(columns=dummies)
    return new_data

def load_distribution(service: str, params_v: str):
    return load_params(service, params_v), load_other(service)
