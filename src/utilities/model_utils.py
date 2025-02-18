import numpy as np
import pandas as pd
import xgboost
import os
import json
from enum import Enum
from typing import List, Dict
from pydantic import BaseModel
from typing import Optional
import operator
from pydantic import model_validator
from sqlalchemy.testing.plugin.plugin_base import warnings


class ModelType(Enum):
    MARKET_MODEL = '_market_model'
    ERROR_MODEL = '_error_model'
    PRESENCE_MODEL = '_presence_model'


class ModelConfig(BaseModel):
    model_config_name: str
    features_to_include: Optional[List[str]] = None
    features_to_exclude: Optional[List[str]] = None
    monotone_constraints: Optional[Dict[str, int]] = None


    def process_features(self, features : List[str]) -> List[str]:

        result = features.copy()

        if self.features_to_exclude:
            for feature in self.features_to_exclude:
                if feature in features:
                    result.remove(feature)
                else:
                    warnings.warn(f"Feature {feature} that should be excluded by the config, already not present, please check if this is intended!")
            return result
        else:
            for feature in self.features_to_include:
                if feature not in features:
                    raise Exception(f"Feature {feature} in feature_to_include not present!")
            return self.features_to_include


    @model_validator(mode="after")
    def check_features(cls, values: "ModelConfig") -> "ModelConfig":
        if values.features_to_include is not None and values.features_to_exclude is not None:
            raise ValueError("Both 'features_to_include' and 'features_to_exclude' cannot be provided at the same time.")
        return values

    @classmethod
    def load_from_json(cls, model_config_path: str) -> "ModelConfig":
        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"Configuration file not found: {model_config_path}")
        with open(model_config_path, "r") as f:
            model_config_data = json.load(f)
        return cls(**model_config_data)

    @classmethod
    def empty_config(cls) -> "ModelConfig":
        return cls(
            model_config_name="empty_config",
            features_to_include=None,
            features_to_exclude=[],
            monotone_constraints={},
        )

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


import ast
def preprocess_factors(factors, col):
    factors = factors.copy()
    def split_range(val):
        if pd.notnull(val):
            lower, upper = ast.literal_eval(str(val))
            return pd.Series({'lo': lower, 'hi': upper})
        else:
            return pd.Series({'lo': float('-inf'), 'hi': float('inf')})
    factors[[f"{col}_lo", f"{col}_hi"]] = factors[col].apply(split_range)
    return factors


def merge_range(df, factors, col):
    factors = preprocess_factors(factors, col)
    merged = pd.merge(
        df,
        factors,
        on=['current_target'],
        suffixes=['', '_f'],
        how='left'
    )
    lower_bound = merged[f"{col}_lo"]
    upper_bound = merged[f"{col}_hi"]

    mask = (merged[col] >= lower_bound) & (merged[col] <= upper_bound)
    merged = merged[mask]

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
    data_c[corrected_target] = data_c[target_variable]

    on_top = on_top[on_top['current_target'] == target_variable_orig]

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
    predictions = model.predict(d_matrix, output_margin=True)
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


