import os
import xgboost

import numpy as np
import pandas as pd

from enum import Enum

from numpy.ma.core import absolute

from utilities.path_constants import MODELS_PATH
from utilities.load_utils import *


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


def get_all_models_trained_on(data_name: str, model_type: ModelType) -> list:
    model_names = []
    for model_name in os.listdir(MODELS_PATH):
        if model_type == ModelType.MARKET_MODEL:
            if ModelType.ERROR_MODEL.value not in model_name and ModelType.PRESENCE_MODEL.value not in model_name and data_name in model_name:
                model_names.append(model_name)
        else:
            if model_type.value in model_name and data_name in model_name:
                model_names.append(model_name)
    return model_names

import re
def make_d_matrix(data_features: pd.DataFrame,
                  data_target: pd.DataFrame,
                  is_classification: bool) -> xgboost.DMatrix:
    if is_classification:

        return xgboost.DMatrix(data_features, label=data_target, enable_categorical=True)
    else:
        return xgboost.DMatrix(data_features, data_target, enable_categorical=True)


def predict(model: xgboost.Booster, data: pd.DataFrame):
    d_matrix = xgboost.DMatrix(data=data, enable_categorical=True)
    predictions = model.predict(d_matrix, output_margin=True)
    return predictions


def apply_threshold(arr: np.array, threshold: float):
    return (arr > threshold).astype(bool)


def apply_on_top(data : pd.DataFrame, target_variable : str, on_top : pd.DataFrame) -> np.array:
    corrected_target = f'corrected_{target_variable}'
    factor_types = ['absolute', 'relative']

    data_c = data.copy()
    data_c['current_target'] = target_variable
    data_c['current_target'] = target_variable
    data_c[corrected_target] = data_c[target_variable]

    for ft in factor_types:

        same_ft = on_top[on_top['factor_type'] == ft]

        for name, group in same_ft.groupby('features'):

            features = group.features.values[0]

            if isinstance(features, str):
                features = [features]

            if isinstance(features, tuple):
                features = list(features)

            print(features)

            named_factor = f'factor_{features}'

            rename_dict = {k : v for k, v in enumerate(features)}

            unpacked = group['feature_values'].apply(pd.Series).rename(columns=rename_dict)
            unpacked = pd.concat([unpacked, group], axis=1)



            data_c = pd.merge(data_c, unpacked[features + ['current_target', 'factor']],
                            on= features + ['current_target'], how = 'left')

            data_c = data_c.rename(columns={'factor': named_factor})
            data_c[named_factor] = data_c[named_factor].fillna(1 if ft == 'relative' else 0)

            if ft == 'absolute':
                data_c[corrected_target] = data_c[corrected_target] - data_c[named_factor]
            else:
                data_c[corrected_target] = data_c[corrected_target] / data_c[named_factor]

    return data_c