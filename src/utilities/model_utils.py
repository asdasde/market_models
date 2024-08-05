import os
import xgboost

import numpy as np
import pandas as pd

from enum import Enum
from utilities.path_constants import MODELS_PATH


class ModelType(Enum):
    MARKET_MODEL = '_market_model'
    ERROR_MODEL = '_error_model'
    PRESENCE_MODEL = '_presence_model'


def get_expected_features(model: xgboost.Booster):
    return getattr(model, "feature_names", [])


def is_compatible(model: xgboost.Booster, data: pd.DataFrame):
    expected_features = get_expected_features(model)
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
