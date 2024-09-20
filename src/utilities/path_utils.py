from pathlib import Path
from utilities.path_constants import *
from utilities.constants import column_to_folder_mapping
from datetime import datetime

def get_raw_data_path(data_name: str, extension = '.csv') -> Path:
    return RAW_DATA_PATH / f'{data_name}{extension}'

def get_processed_data_path(data_name: str) -> Path:
    return PROCESSED_DATA_PATH / f'{data_name}_processed.parquet'

def get_interim_data_path(data_name: str) -> Path:
    return INTERIM_DATA_PATH / f'{data_name}_before_crawling.csv'

def get_predictions_all_path(data_name: str, train_data_name : str) -> Path:
    return PREDICTIONS_PATH / f'{data_name}_via_{train_data_name}_all_predictions.parquet'

def get_predictions_path(data_name: str, model_name: str) -> Path:
    return PREDICTIONS_PATH / f'{data_name}__{model_name}.csv'

def get_features_path(data_name: str) -> Path:
    return PROCESSED_DATA_PATH / f'{data_name}_features.txt'

def get_model_name(data_name: str, target_variable: str) -> str:
    return f'{data_name}_{target_variable}_model'

def get_error_model_name(data_name: str, target_variable: str) -> str:
    return f'{data_name}_{target_variable}_error_model'

def get_presence_model_name(data_name: str, target_variable: str) -> str:
    return f'{data_name}_{target_variable}_presence_model'

def get_model_directory(model_name: str) -> Path:
    return MODELS_PATH / model_name

def get_model_path(model_name: str) -> Path:
    return get_model_directory(model_name) / f'{model_name}.json'

def get_model_hyperparameters_path(model_name: str) -> Path:
    return get_model_directory(model_name) / f'{model_name}_hyperparameters.json'

def get_model_cv_out_of_sample_predictions_path(model_name: str) -> Path:
    return get_model_directory(model_name) / f'{model_name}_cv_out_of_sample_predictions.csv'

def get_model_trials_path(model_name : str) -> Path:
    return get_model_directory(model_name) / f'{model_name}_hyperopt_trials.pkl'

def get_params_path(service: str, params_v: str) -> Path:
    return DISTRIBUTION_PATH / service / 'params' / params_v

def get_others_path(service: str) -> Path:
    return DISTRIBUTION_PATH / service / 'other'

def get_template_path(service: str, date: str) -> Path:
    return DISTRIBUTION_PATH / service / 'templates' / f'{service}_template_{date}.xlsx'

def get_row_values_path(service: str, date: str) -> Path:
    return DISTRIBUTION_PATH / service / 'templates' / f'{service}_row_values_{date}.txt'

def get_transposed_config_path(service: str, date: str) -> Path:
    return DISTRIBUTION_PATH / service / 'templates' / f'{service}_transposed_config_{date}.json'

def get_sampled_data_name(service: str, params_v: str) -> str:
    current_date = datetime.now().strftime("%Y_%m_%d")
    return f"{service}_sampled_data_{params_v}_{current_date}"

def get_incremental_data_name(service: str, base_profile_v: str, values_v: str) -> str:
    return f"{service}_incremental_data_base_profile_{base_profile_v}_values_{values_v}"

def get_incremental_base_profile_path(service: str, v: str) -> Path:
    return DISTRIBUTION_PATH / service / 'incremental_params' / 'base_profiles' / f'base_profile_{v}.csv'

def get_incremental_values_path(service: str, v: str) -> Path:
    return DISTRIBUTION_PATH / service / 'incremental_params' / 'values' / f'values_{v}.csv'

def get_mtpl_postal_categories_path(target_variable: str) -> Path:
    tables_name = column_to_folder_mapping.get(target_variable)
    comp_name = tables_name.replace('_tables', '')
    return MTPL_POSTAL_CATEGORIES_PATH / f'{comp_name}_lookups_table.csv'


def get_mtpl_default_values_path(target_variable: str) -> Path:
    tables_name = column_to_folder_mapping.get(target_variable)
    comp_name = tables_name.replace('_tables', '')
    return MTPL_POSTAL_CATEGORIES_PATH / f'{comp_name}_factor_default_values_table.csv'

def get_encoder_path(feature: str) -> Path:
    return ENCODERS_PATH / f'{feature}_encoder.pkl'

def get_report_path(model_name: str) -> Path:
    return REPORTS_PATH / model_name

def get_report_resource_path(model_name: str) -> Path:
    return REPORTS_PATH / model_name / 'resources'

def get_data_name_references_path() -> Path:
    return REFERENCES_PATH / 'data_name_references.json'

def get_brackets_path(feature: str) -> Path:
    return BRACKETS_PATH / f'{feature}_brackets.json'


def get_profiles_for_crawling_dir(data_name: str) -> Path:
    return INTERIM_DATA_PATH / data_name

def get_profiles_for_crawling_transposed(data_name: str) -> Path:
    return get_profiles_for_crawling_dir(data_name) / f'{data_name}.csv'

def get_profiles_for_crawling_zip_path(data_name: str) -> Path:
    return get_profiles_for_crawling_dir(data_name) / f'{data_name}.zip'

def get_profiles_after_crawling_zip_path(data_name: str) -> Path:
    return get_profiles_for_crawling_dir(data_name) / f'{data_name}_crawled.zip'


def get_private_key_file_path() -> Path:
    return Path('../../../ssh_key')

def get_remote_crawl_sh_path() -> str:
    return (REMOTE_CRAWLER_DIRECTORY / 'crawl.sh').as_posix()


def get_remote_profiles_path(data_name: str) -> str:
    return f'{REMOTE_CRAWLER_DIRECTORY}{data_name}/'

def get_remote_profiles_zip_path(data_name : str) -> str:
    return f"{get_remote_profiles_path(data_name)}{data_name}.zip"

def get_remote_profiles_subdirectory_path(data_name: str) -> str:
    return f'{get_remote_profiles_path(data_name)}profiles/'

def get_remote_profiles_after_crawling_zip_path(data_name: str) -> str:
    return f'{get_remote_profiles_path(data_name)}{data_name}_crawled.zip'

