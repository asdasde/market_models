import os
import csv
import pandas as pd
import numpy as np
import logging
import joblib


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import xgboost


DISTRIBUTION_PATH = '../data/external/distributions/'
PROCESSED_DATA_PATH = '../data/processed/'
RAW_DATA_PATH = '../data/raw/'
MODELS_PATH = '../models/'
ENCODERS_PATH = '../models/encoders/'
ENCODERS_PATH = '../models/encoders/'


def prepareDir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
    for file in os.listdir(dir):
        os.remove(dir + file)


def detect_csv_delimiter(file_path):
    with open(file_path, 'r', newline='') as file:
        dialect = csv.Sniffer().sniff(file.read(1024))
        return dialect.delimiter


def get_processed_data_path(data_name):
    return f'{PROCESSED_DATA_PATH}{data_name}_processed.csv'

def get_raw_data_path(data_name):
    return f'{RAW_DATA_PATH}{data_name}.csv'

def get_features_path(data_name):
    return f'{PROCESSED_DATA_PATH}{data_name}_features.txt'


def get_model_name(data_name, target_variable):
    return f'{data_name}_{target_variable}_model'
def get_model_path(model_name):
    return f'{MODELS_PATH}{model_name}.json'

def get_error_model_name(data_name, target_variable):
    return f'{data_name}_{target_variable}_error_model'
def get_error_model_path(model_name):
    return f'{MODELS_PATH}{model_name}.json'

def get_params_path(service, params_v):
    return f'{DISTRIBUTION_PATH}{service}/params/{params_v}/'

def get_others_path(service):
    return f'{DISTRIBUTION_PATH}{service}/other/'
def get_encoder_path(feature):
    return f'{ENCODERS_PATH}{feature}_encoder.pkl'


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

BONUS_MALUS_CLASSES_DICT = dict(zip(BONUS_MALUS_CLASSES_BAD, BONUS_MALUS_CLASSES_GOOD))

FORINT_TO_EUR = 0.0026

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


def apply_features(data, features, feature_dtypes):
    for feature in features:
        if feature_dtypes[feature] == 'index':
            data = data.set_index(feature)
            features.remove(feature)
            continue

        if feature == 'BonusMalus':
            data[feature] = data[feature].apply(lambda x : BONUS_MALUS_CLASSES_DICT.get(x, x))
        data[feature] = data[feature].astype(feature_dtypes[feature])

        if data[feature].dtype == 'category':
            feature_encoder = joblib.load(get_encoder_path(feature))
            if feature_encoder.__class__.__name__ == 'OrdinalEncoder':
                data[feature] = feature_encoder.transform(data[feature].values.reshape(-1, 1)).ravel()
            else:
                data[feature] = feature_encoder.transform(data[feature])

    return data

def load_data(data_path, features_path, target_variable = None):
    data = read_file(data_path)
    logging.info("Imported data...")
    with open(features_path) as file:
        features = file.readlines()
        features = [feature.replace('\n', '') for feature in features]
        feature_dtypes = {feature.split(',')[0]: feature.split(',')[1] for feature in features}
        features = [feature.split(',')[0] for feature in features]

    data = apply_features(data, features, feature_dtypes)

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
    predictions = model.predict(dmatrix, output_margin = True)
    return predictions

