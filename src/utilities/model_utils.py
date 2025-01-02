import numpy as np

from enum import Enum

import pandas as pd
import xgboost

from utilities.load_utils import *
import operator


class ModelType(Enum):
    MARKET_MODEL = '_market_model'
    ERROR_MODEL = '_error_model'
    PRESENCE_MODEL = '_presence_model'


def get_expected_features(model: xgboost.Booster):
    return getattr(model, "feature_names", [])


def is_compatible(model: xgboost.Booster, data: pd.DataFrame):
    expected_features = get_expected_features(model)
    print(set(expected_features).difference(data.columns))
    return set(expected_features).issubset(data.columns)

def make_d_matrix(data_features: pd.DataFrame,
                  data_target: pd.DataFrame,
                  is_classification: bool) -> xgboost.DMatrix:
    if is_classification:

        return xgboost.DMatrix(data_features, label=data_target, enable_categorical=True)
    else:
        return xgboost.DMatrix(data_features, data_target, enable_categorical=True)


def apply_threshold(arr: np.array, threshold: float):
    return (arr > threshold).astype(bool)




def merge_range(df, factors, col):
    merged = pd.merge(
        df,
        factors,
        on=['current_target'],
        suffixes=['', '_f'],
        how = 'left'
    )
    merged = merged[
        (merged[col] >= merged[f"{col}_f"].apply(lambda x: eval(x)[0] if x is not None else False)) &
        (merged[col] <= merged[f"{col}_f"].apply(lambda x: eval(x)[1] if x is not None else False))
    ]
    return merged.drop(columns=[f"{col}_f"])




def apply_on_top(data: pd.DataFrame,
                 target_variable: str,
                 target_variable_orig : str,
                 on_top: pd.DataFrame,
                 reverse : bool = False) -> pd.DataFrame:


    data['temp_id'] = range(len(data))

    corrected_target = f'corrected_{target_variable}'

    factor_types = ['absolute', 'relative']

    operations = {
        'absolute' : operator.add if reverse else operator.sub,
        'relative' : operator.mul if reverse else operator.truediv
    }

    data_c = data.copy()
    data_c['current_target'] = target_variable_orig
    on_top = on_top[on_top['current_target'] == target_variable_orig]
    data_c[corrected_target] = data_c[target_variable]

    for factor_type in factor_types:
        same_factor_type = on_top[on_top['factor_type'] == factor_type]

        for name, group in same_factor_type.groupby('features'):

            features = group.features.values[0]

            if isinstance(features, str):
                features = [features]

            if isinstance(features, tuple):
                features = list(features)

            named_factor = f'factor_{features}'
            rename_dict = {k: v for k, v in enumerate(features)}
            merge_type = group['merge_type'].iloc[0]

            unpacked = group['feature_values'].apply(pd.Series).rename(columns=rename_dict)
            unpacked = pd.concat([unpacked, group], axis=1)



            if merge_type == 'range':
                data_c = merge_range(data_c, unpacked, features[0])
            else:
                data_c = pd.merge(
                    data_c,
                    unpacked[features + ['current_target', 'factor']],
                    on = features + ['current_target'],
                    how='left'
                )

            data_c = data_c.rename(columns={'factor': named_factor})
            data_c[named_factor] = data_c[named_factor].fillna(1 if factor_type == 'relative' else 0)

            operation = operations[factor_type]
            data_c[corrected_target] = operation(data_c[corrected_target], data_c[named_factor])

    data = pd.merge(data, data_c[['temp_id', corrected_target]], on = 'temp_id')
    data[f'{target_variable}_orig'] = data[target_variable]
    data[target_variable] = data[f'corrected_{target_variable}']

    return data


def predict(model: xgboost.Booster, data: pd.DataFrame) -> np.ndarray:
    d_matrix = xgboost.DMatrix(data=data, enable_categorical=True)
    print(pd.DataFrame.sparse.from_spmatrix(d_matrix.get_data(), columns = data.columns))
    predictions = model.predict(d_matrix, output_margin=True)
    print(predictions)
    return predictions

def predict_on_top(
        model : xgboost.Booster,
        data : pd.DataFrame,
        on_top : pd.DataFrame,
        target_variable_orig : str) -> np.ndarray:

    data_c = data.copy()
    data_c['predictions'] = predict(model, data)
    data = apply_on_top(data_c, 'predictions', target_variable_orig, on_top, reverse = True)
    return data['corrected_predictions'].values


