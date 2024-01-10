import json
import os
import csv
import pandas as pd
import numpy as np
import logging
import joblib
import zipfile

from datetime import datetime
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import xgboost


DISTRIBUTION_PATH = '../data/external/distributions/'
PROCESSED_DATA_PATH = '../data/processed/'
INTERIM_DATA_PATH = '../data/interim/'
RAW_DATA_PATH = '../data/raw/'
PREDICTIONS_PATH = '../data/predictions/'
MODELS_PATH = '../models/'
ENCODERS_PATH = '../models/encoders/'
REPORTS_PATH = '../reports/'
PRIVATE_KEY_PATH = "../../../ssh_key"
REFERENCES_PATH = "../references/"
BRACKETS_PATH = "../data/external/feature_brackets/"

REMOTE_HOST_NAME = "43mp.l.time4vps.cloud"
REMOTE_CRAWLER_DIRECTORY = "crawler-mocha/"
def prepareDir(dir_path: str):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    else:
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)


def dict_to_json(dictionary: dict, output_path: str):
    serializable_dict = dictionary.copy()
    for k, v in dictionary.items():
        if isinstance(v, np.int64):
            serializable_dict[k] = int(v)

    with open(output_path, 'w') as f:
        json.dump(serializable_dict, f, indent=4)

def detect_csv_delimiter(file_path : str) -> str:
    with open(file_path, 'r', newline='') as file:
        dialect = csv.Sniffer().sniff(file.read(1024))
        return dialect.delimiter


def get_raw_data_path(data_name : str) -> str:
    return f'{RAW_DATA_PATH}{data_name}.csv'
def get_processed_data_path(data_name : str) -> str:
    return f'{PROCESSED_DATA_PATH}{data_name}_processed.csv'

def get_interim_data_path(data_name : str) -> str:
    return f'{INTERIM_DATA_PATH}{data_name}_before_crawling.csv'

def get_predictions_all_path(data_name : str) -> str:
    return f'{PREDICTIONS_PATH}{data_name}_all_predictions.csv'

def get_predictions_path(data_name : str, model_name : str) -> str:
    return f'{PREDICTIONS_PATH}{data_name}__{model_name}.csv'

def get_features_path(data_name : str) -> str:
    return f'{PROCESSED_DATA_PATH}{data_name}_features.txt'

def get_model_name(data_name : str, target_variable : str) -> str:
    return f'{data_name}_{target_variable}_model'

def get_error_model_name(data_name : str, target_variable : str) -> str:
    return f'{data_name}_{target_variable}_error_model'

def get_model_directory(model_name : str) -> str:
    return f'{MODELS_PATH}{model_name}/'
def get_model_path(model_name : str) -> str:
    model_dir = get_model_directory(model_name)
    return f'{model_dir}{model_name}.json'

def get_model_hyperparameters_path(model_name : str) -> str:
    model_dir = get_model_directory(model_name)
    return f'{model_dir}{model_name}_hyperparameters.json'

def get_model_cv_out_of_sample_predictions_path(model_name : str) -> str:
    model_dir = get_model_directory(model_name)
    return f'{model_dir}{model_name}_cv_out_of_sample_predictions.csv'

def get_params_path(service : str, params_v : str) -> str:
    return f'{DISTRIBUTION_PATH}{service}/params/{params_v}/'

def get_others_path(service : str) -> str:
    return f'{DISTRIBUTION_PATH}{service}/other/'
def get_template_path(service : str, date : str) -> str:
    return f'{DISTRIBUTION_PATH}{service}/templates/{service}_template_{date}.xlsx'
def get_row_values_path(service : str, date : str) -> str:
    return f'{DISTRIBUTION_PATH}{service}/templates/{service}_row_values_{date}.txt'

def get_sampled_data_name(service : str, params_v : str) -> str:
    current_date = datetime.now().strftime("%Y_%m_%d")
    return f"{service}_sampled_data_{params_v}_{current_date}"

def get_incremental_data_name(service : str, base_profile_v : str, values_v : str) -> str:
    return f"{service}_incremental_data_base_profile_{base_profile_v}_values_{values_v}"
def get_incremental_base_profile_path(service : str, v : str) -> str:
    return f'{DISTRIBUTION_PATH}{service}/incremental_params/base_profiles/base_profile_{v}.csv'

def get_incremental_values_path(service : str, v : str) -> str:
    return f'{DISTRIBUTION_PATH}{service}/incremental_params/values/values_{v}.csv'

def get_private_key_file_path() -> str:
    return '../../../ssh_key'

def get_encoder_path(feature : str) -> str:
    return f'{ENCODERS_PATH}{feature}_encoder.pkl'

def get_report_path(model_name : str) -> str:
    return f'{REPORTS_PATH}{model_name}/'

def get_report_resource_path(model_name : str) -> str:
    return f'{REPORTS_PATH}{model_name}/resources/'

def get_profiles_for_crawling_dir(data_name : str) -> str:
    return f'{INTERIM_DATA_PATH}{data_name}/'
def get_profiles_for_crawling_transposed(data_name : str) -> str:
    dir = get_profiles_for_crawling_dir(data_name)
    return f'{dir}{data_name}.csv'

def get_profiles_for_crawling_zip_path(data_name) -> str:
    dir = get_profiles_for_crawling_dir(data_name)
    return f'{dir}{data_name}.zip'
def get_remote_profiles_after_crawling_zip_path(service: str) -> str:
    current_date = datetime.now().strftime("%Y_%m_%d")
    return f'{REMOTE_CRAWLER_DIRECTORY}{service}_{current_date}.zip'

def get_profiles_after_crawling_zip_path(service : str, data_name : str) -> str:
    dir = get_profiles_for_crawling_dir(data_name)
    current_date = datetime.now().strftime("%Y_%m_%d")
    return f'{dir}{service}_{current_date}.zip'

# Temoporary solution, because crawler needs to be run from its directory
def get_remote_crawler_path() -> str:
    return f'crawler.py'
def get_remote_queue_path() -> str:
    return f'{REMOTE_CRAWLER_DIRECTORY}queue/'

def get_remote_profiles_path() -> str:
    return f'{REMOTE_CRAWLER_DIRECTORY}profiles/'

def get_remote_profioles_path() -> str:
    return f'{REMOTE_CRAWLER_DIRECTORY}profiles'

def get_data_name_references_path():
    return f'{REFERENCES_PATH}data_name_references.json'

def get_feature_brackets_dir(target_variable):
    return f'{BRACKETS_PATH}{target_variable}_brackets/'

def get_brackets_path(target_variable, feature):
    dir = get_feature_brackets_dir(target_variable)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return f'{dir}{target_variable}_{feature}_brackets.csv'

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

NETRISK_CASCO_DTYPES = {'isRecent' : 'bool', 'CarMake' : 'category', 'CarAge' : 'int', 'ccm' : 'int', 'kw' : 'int', 'kg' : 'int',
                       'car_value' : 'float', 'CarMakerCategory' : 'float', 'PostalCode' : 'int', 'PostalCode2' : 'int', 'PostalCode3' : 'int',
                       'Category' : 'int', 'Longitude': 'float', 'Latitude': 'float', 'Age' : 'int', 'LicenseAge' : 'int', 'BonusMalus' : 'category',
                       'BonusMalusCode' : 'category'}

#FEATURES_TO_IGNORE = ['PostalCode2', 'PostalCode3']
FEATURES_TO_IGNORE = []


BONUS_MALUS_CLASSES_GOOD = ['B10', 'B09', 'B08', 'B07', 'B06', 'B05', 'B04', 'B03', 'B02', 'B01', 'A00', 'M01', 'M02', 'M03', 'M04']
BONUS_MALUS_CLASSES_BAD = ['B10', 'B9', 'B8', 'B7', 'B6', 'B5', 'B4', 'B3', 'B2', 'B1', 'A0', 'M1', 'M2', 'M3', 'M4']

BONUS_MALUS_CLASSES_DICT = dict(zip(BONUS_MALUS_CLASSES_BAD, BONUS_MALUS_CLASSES_GOOD))

FORINT_TO_EUR = 0.0026
ERROR_MODEL_CLASSIFICATION_THRESHOLD = 0.8
MAX_EVALS = 100

CURRENT_YEAR = datetime.today().year

QUANTILE_RANGE = [0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 1]

def get_all_models_trained_on(data_name : str) -> list:
    model_names = []
    for model_name in os.listdir(MODELS_PATH):
        if data_name in model_name and '_error_model' not in model_name:
            model_names.append(model_name)
    return model_names


def read_file(file_path : str) -> pd.DataFrame:
    file_extension = file_path.split('.')[-1].lower()
    if file_extension == 'csv':
        return pd.read_csv(file_path)
    elif file_extension in ['xls', 'xlsx']:
        return pd.read_excel(file_path)
    elif file_extension == 'json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def zip_list_of_files(file_paths: list, zip_file_path: str) -> None:
    with zipfile.ZipFile(zip_file_path, 'w') as zipMe:
        for file in file_paths:
            arcname = os.path.basename(file)
            zipMe.write(file, arcname = arcname, compress_type=zipfile.ZIP_DEFLATED)

def apply_features(data : pd.DataFrame, features : list, feature_dtypes : dict) -> pd.DataFrame:
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

                def transform(x):
                    try:
                        return feature_encoder.transform(x)
                    except ValueError:
                        return 1
                data[feature] = data[feature].apply(transform)
    return data

def choose_postal_categories(data : pd.DataFrame, target_variable : str = None) -> pd.DataFrame:

    if target_variable is None:
        target_variable = 'ALFA_price'

    postal_category_column_name = target_variable.replace("_price", "_postal_category")

    if postal_category_column_name in data.columns:
        data['Category'] = data[postal_category_column_name]

    return data

def load_data(data_path : str, features_path : str, target_variable : str = None, apply_feature_dtypes : bool = True) -> tuple:
    data = read_file(data_path)
    logging.info("Imported data...")
    with open(features_path) as file:
        features = file.readlines()
        features = [feature.replace('\n', '') for feature in features]
        feature_dtypes = {feature.split(',')[0]: feature.split(',')[1] for feature in features}
        features = [feature.split(',')[0] for feature in features]
    logging.info("Imported feature data...")

    if apply_feature_dtypes:
        data = apply_features(data, features, feature_dtypes)

    data = choose_postal_categories(data, target_variable)

    features = [feature for feature in features if '_postal_category' not in feature]
    features = [feature for feature in features if feature not in FEATURES_TO_IGNORE]

    columns = features
    columns = (['DateCrawled'] if 'DateCrawled' in data.columns else []) + columns

    if target_variable is None:
        data = data[columns]
    else:
        data = data[columns + [target_variable]]
        data = data.dropna(subset=[target_variable])

    return data, features

def makeDMatrix(data_features: pd.DataFrame,
                data_target: pd.DataFrame) -> xgboost.DMatrix:
    return xgboost.DMatrix(data_features, data_target, enable_categorical=True)

def load_model(model_path : str) -> xgboost.Booster:
    # Load the XGBoost model
    return xgboost.Booster(model_file=model_path)

def predict(model : xgboost.Booster, data : pd.DataFrame):
    dmatrix = xgboost.DMatrix(data=data, enable_categorical=True)
    predictions = model.predict(dmatrix, output_margin = True)
    return predictions


def apply_threshold(arr : np.array, threshold : float):
    return (arr > threshold).astype(bool)
