import os
import csv
import pandas as pd
import numpy as np
import logging

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import xgboost


def prepareDir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
    for file in os.listdir(dir):
        os.remove(dir + file)


def detect_csv_delimiter(file_path):
    with open(file_path, 'r', newline='') as file:
        dialect = csv.Sniffer().sniff(file.read(1024))
        return dialect.delimiter


NAME_MAPPING = {
    'Policy_Start_Bonus_Malus_Class': 'BonusMalus',
    'Vehicle_age': 'CarAge',
    'Vehicle_weight_empty': 'CarWeightMin',
    'Number_of_seats': 'NumberOfSeats',
    'Driver_Experience': 'DriverExperience',
    'Vehicle_weight_maximum': 'CarWeightMax',
    'Power_range_in_KW': 'kw',
    'Engine_size': 'engine_size',
    'DriverAge': 'driver_age',
    'PostalCode': 'PostalCode',
    'CarMake': 'CarMake',
    'Milage': 'Mileage'
}

BONUS_MALUS_CLASSES_GOOD = ['B10', 'B09', 'B08', 'B07', 'B06', 'B05', 'B04', 'B03', 'B02', 'B01', 'A00', 'M01', 'M02', 'M03', 'M04']
BONUS_MALUS_CLASSES_BAD = ['B10', 'B9', 'B8', 'B7', 'B6', 'B5', 'B4', 'B3', 'B2', 'B1', 'A0', 'M1', 'M2', 'M3', 'M4']


def read_file(file_path):
    file_extension = file_path.split('.')[-1].lower()
    if file_extension == 'csv':
        return pd.read_csv(file_path)
    elif file_extension in ['xls', 'xlsx']:
        return pd.read_excel(file_path)
    elif file_extension == 'json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def load_data(data_path, features_path, target_variable = None):
    data = read_file(data_path)
    logging.info("Imported data...")
    with open(features_path) as file:
        features = file.readlines()
        features = [feature.replace('\n', '') for feature in features]
        feature_dtypes = {feature.split(',')[0]: feature.split(',')[1] for feature in features}
        features = [feature.split(',')[0] for feature in features]


    for feature in features:
        if feature_dtypes[feature] == 'index':
            data = data.set_index(feature)
            features.remove(feature)
            continue

        data[feature] = data[feature].astype(feature_dtypes[feature])
        if data[feature].dtype == 'category' and feature == 'BonusMalus':

            ordinal_encoder = OrdinalEncoder(categories=[BONUS_MALUS_CLASSES_BAD])
            data[feature] = ordinal_encoder.fit_transform(data[[feature]])
        elif data[feature].dtype == 'category':
            label_encoder = LabelEncoder()
            data[feature] = label_encoder.fit_transform(data[feature])
    logging.info("Imported feature data...")

    if target_variable is None:
        data = data[features]
    else:
        data = data[features + [target_variable]]
        data = data.dropna(subset=[target_variable])

    return data, features

def makeDMatrix(data_features: pd.DataFrame,
                data_target: pd.DataFrame) -> xgboost.DMatrix:
    return xgboost.DMatrix(data_features, data_target, enable_categorical=True)

def load_model(model_path):
    # Load the XGBoost model
    return xgboost.Booster(model_file=model_path)

def predict(model, data):
    dmatrix = xgboost.DMatrix(data=data, enable_categorical=True)
    predictions = np.abs(model.predict(dmatrix))
    return predictions

