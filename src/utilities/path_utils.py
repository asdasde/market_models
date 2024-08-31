from utilities.path_constants import *
from utilities.constants import column_to_folder_mapping

from datetime import datetime


def get_raw_data_path(data_name: str) -> str:
    return f'{RAW_DATA_PATH}{data_name}.csv'


def get_processed_data_path(data_name: str) -> str:
    return f'{PROCESSED_DATA_PATH}{data_name}_processed.csv'


def get_interim_data_path(data_name: str) -> str:
    return f'{INTERIM_DATA_PATH}{data_name}_before_crawling.csv'


def get_predictions_all_path(data_name: str) -> str:
    return f'{PREDICTIONS_PATH}{data_name}_all_predictions.csv'


def get_predictions_path(data_name: str, model_name: str) -> str:
    return f'{PREDICTIONS_PATH}{data_name}__{model_name}.csv'


def get_features_path(data_name: str) -> str:
    return f'{PROCESSED_DATA_PATH}{data_name}_features.txt'


def get_model_name(data_name: str, target_variable: str) -> str:
    return f'{data_name}_{target_variable}_model'


def get_error_model_name(data_name: str, target_variable: str) -> str:
    return f'{data_name}_{target_variable}_error_model'


def get_presence_model_name(data_name: str, target_variable: str) -> str:
    return f'{data_name}_{target_variable}_presence_model'


def get_model_directory(model_name: str) -> str:
    return f'{MODELS_PATH}{model_name}/'


def get_model_path(model_name: str) -> str:
    model_dir = get_model_directory(model_name)
    return f'{model_dir}{model_name}.json'


def get_model_hyperparameters_path(model_name: str) -> str:
    model_dir = get_model_directory(model_name)
    return f'{model_dir}{model_name}_hyperparameters.json'


def get_model_cv_out_of_sample_predictions_path(model_name: str) -> str:
    model_dir = get_model_directory(model_name)
    return f'{model_dir}{model_name}_cv_out_of_sample_predictions.csv'


def get_params_path(service: str, params_v: str) -> str:
    return f'{DISTRIBUTION_PATH}{service}/params/{params_v}/'


def get_others_path(service: str) -> str:
    return f'{DISTRIBUTION_PATH}{service}/other/'


def get_template_path(service: str, date: str) -> str:
    return f'{DISTRIBUTION_PATH}{service}/templates/{service}_template_{date}.xlsx'


def get_row_values_path(service: str, date: str) -> str:
    return f'{DISTRIBUTION_PATH}{service}/templates/{service}_row_values_{date}.txt'


def get_sampled_data_name(service: str, params_v: str) -> str:
    current_date = datetime.now().strftime("%Y_%m_%d")
    return f"{service}_sampled_data_{params_v}_{current_date}"


def get_incremental_data_name(service: str, base_profile_v: str, values_v: str) -> str:
    return f"{service}_incremental_data_base_profile_{base_profile_v}_values_{values_v}"


def get_incremental_base_profile_path(service: str, v: str) -> str:
    return f'{DISTRIBUTION_PATH}{service}/incremental_params/base_profiles/base_profile_{v}.csv'


def get_incremental_values_path(service: str, v: str) -> str:
    return f'{DISTRIBUTION_PATH}{service}/incremental_params/values/values_{v}.csv'


def get_private_key_file_path() -> str:
    return '../../../ssh_key'


def get_encoder_path(feature: str) -> str:
    return f'{ENCODERS_PATH}{feature}_encoder.pkl'


def get_report_path(model_name: str) -> str:
    return f'{REPORTS_PATH}{model_name}/'


def get_report_resource_path(model_name: str) -> str:
    return f'{REPORTS_PATH}{model_name}/resources/'


def get_profiles_for_crawling_dir(data_name: str) -> str:
    return f'{INTERIM_DATA_PATH}{data_name}/'


def get_profiles_for_crawling_transposed(data_name: str) -> str:
    dir = get_profiles_for_crawling_dir(data_name)
    return f'{dir}{data_name}.csv'


def get_profiles_for_crawling_zip_path(data_name) -> str:
    dir = get_profiles_for_crawling_dir(data_name)
    return f'{dir}{data_name}.zip'


def get_remote_profiles_after_crawling_zip_path(data_name) -> str:
    return f'{REMOTE_CRAWLER_DIRECTORY}{data_name}/{data_name}_crawled.zip'


def get_profiles_after_crawling_zip_path(data_name: str) -> str:
    dir = get_profiles_for_crawling_dir(data_name)
    return f'{dir}{data_name}_crawled.zip'


def get_mtpl_postal_categories_path(target_variable):
    tables_name = column_to_folder_mapping.get(target_variable)
    comp_name = tables_name.replace('_tables', '')
    return f'{MTPL_POSTAL_CATEGORIES_PATH}{comp_name}_lookups_table.csv'


def get_mtpl_default_values_path(target_variable):
    tables_name = column_to_folder_mapping.get(target_variable)
    comp_name = tables_name.replace('_tables', '')
    return f'{MTPL_POSTAL_CATEGORIES_PATH}{comp_name}_factor_default_values_table.csv'


def get_remote_crawler_path() -> str:
    return f'crawler.py'

def get_remote_cralw_sh_path() -> str:
    return f'{REMOTE_CRAWLER_DIRECTORY}crawl.sh'

def get_remote_queue_path() -> str:
    return f'{REMOTE_CRAWLER_DIRECTORY}queue/'


def get_remote_profiles_path(data_name: str) -> str:
    return f'{REMOTE_CRAWLER_DIRECTORY}{data_name}/'


def get_remote_profiles_subdirectory_path(data_name: str) -> str:
    return f'{REMOTE_CRAWLER_DIRECTORY}{data_name}/profiles/'


def get_data_name_references_path():
    return f'{REFERENCES_PATH}data_name_references.json'


def get_brackets_path(feature):
    return f'{BRACKETS_PATH}{feature}_brackets.json'
