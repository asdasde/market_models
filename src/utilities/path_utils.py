from jupyterlab.utils import deprecated


from utilities.path_constants import *
from utilities.constants import column_to_folder_mapping
from datetime import datetime
from curses.ascii import isdigit


class PathManager:
    def __init__(self, service : str):
        self.service = service

    def get_raw_data_path(self, data_name: str, extension = '.csv') -> Path:
        return RAW_DATA_PATH / self.service / f'{data_name}{extension}'

    def get_processed_data_path(self, data_name: str) -> Path:
        return PROCESSED_DATA_PATH / self.service / f'{data_name}_processed.parquet'

    @staticmethod
    def get_predictions_all_path(data_name: str, train_data_name : str, apply_presence_models : str) -> Path:
        if apply_presence_models:
            return PREDICTIONS_PATH / f'{data_name}_via_{train_data_name}_with_presence_models_all_predictions.parquet'
        else:
            return PREDICTIONS_PATH / f'{data_name}_via_{train_data_name}_all_predictions.parquet'

    @staticmethod
    def get_predictions_path(data_name: str, model_name: str) -> Path:
        return PREDICTIONS_PATH / f'{data_name}_via_{model_name}.csv'


# MODEL PATHS

    @staticmethod
    def get_model_name(data_name: str, target_variable: str) -> str:
        return f'{data_name}_{target_variable}_model'

    @staticmethod
    def get_error_model_name(data_name: str, target_variable: str) -> str:
        return f'{data_name}_{target_variable}_error_model'

    @staticmethod
    def get_presence_model_name(data_name: str, target_variable: str) -> str:
        return f'{data_name}_{target_variable}_presence_model'

    @staticmethod
    def get_presence_model_name_from_model_name(model_name : str) -> str:
        return f'{model_name.replace('_model', '')}_presence_model'


    def get_models_for_train_data_directory(self, train_data_name : str) -> Path:
        return MODELS_PATH / self.service / train_data_name

    def get_model_directory(self, train_data_name : str, model_name: str) -> Path:
        return self.get_models_for_train_data_directory(train_data_name) /  model_name

    def get_all_models_on_train_data(self, train_data_name : str, is_presence_model : bool) -> list:
        model_names = []
        models_for_train_data_directory = self.get_models_for_train_data_directory(train_data_name)
        for model_directory in models_for_train_data_directory.glob('*'):
            model_name = model_directory.stem
            c2 = len(model_name) >= len(train_data_name) and not isdigit(model_name[len(train_data_name)])
            c3 = (is_presence_model == ('presence_model' in model_name))
            if c2 and c3:
                model_names.append(model_name)
        return model_names


    def get_model_path(self, train_data_name : str, model_name: str) -> Path:
        return self.get_model_directory(train_data_name, model_name) / f'{model_name}.json'

    def get_model_hyperparameters_path(self, train_data_name : str, model_name: str) -> Path:
        return self.get_model_directory(train_data_name, model_name) / f'{model_name}_hyperparameters.json'

    def get_model_cv_out_of_sample_predictions_path(self, train_data_name : str, model_name: str) -> Path:
        return self.get_model_directory(train_data_name, model_name) / f'{model_name}_cv_out_of_sample_predictions.csv'

    def get_model_trials_path(self, train_data_name : str, model_name: str) -> Path:
        return self.get_model_directory(train_data_name, model_name) / f'{model_name}_hyperopt_trials.pkl'


    def get_external_path(self):
        return EXTERNAL_DATA_PATH / 'services' / self.service

    def get_others_path(self) -> Path:
        return self.get_external_path() / 'other'

    def get_mtpl_postal_categories_path(self, target_variable: str) -> Path:
        tables_name = column_to_folder_mapping.get(target_variable)
        comp_name = tables_name.replace('_tables', '')
        return self.get_external_path() / 'mtpl_postal_categories' / f'{comp_name}_lookups_table.csv'


    def get_mtpl_default_values_path(self, target_variable: str) -> Path:
        tables_name = column_to_folder_mapping.get(target_variable)
        comp_name = tables_name.replace('_tables', '')
        return self.get_external_path() / 'mtpl_postal_categories' / f'{comp_name}_factor_default_values_table.csv'


    def get_on_top_factor_file_path(self) -> Path:
        return self.get_external_path() / 'on_top' / f'{self.service}_on_top.csv'

    @staticmethod
    def to_s3_key(path : Path) -> str:
        return str(path).replace(str(BASE_DIR), '').strip('/\\')

    @staticmethod
    def from_s3_key(s3_key : str) -> Path:
        return BASE_DIR / s3_key

    @staticmethod
    def get_api_configuration_path(api_configuration_name : str) -> Path:
        return API_CONFIGURATIONS_PATH / f'{api_configuration_name}_configuration.json'

    @staticmethod
    def get_encoder_path(feature: str) -> Path:
        return ENCODERS_PATH / f'{feature}_encoder.pkl'

    def get_report_path(self, train_data_name : str, target_variable : str) -> Path:
        model_name = self.get_model_name(train_data_name, target_variable)
        return REPORTS_PATH / self.service / train_data_name / model_name

    @staticmethod
    def get_report_resource_path(report_path : Path) -> Path:
        return report_path / 'resources'

    def get_error_overview_path(self) -> Path:
        return ERROR_OVERVIEW_PATH / f"error_overview_{self.service}" / "error_overview.xlsx"

    @staticmethod
    def get_report_data_overview_path(report_resources_path: Path, idx) -> Path:
        return report_resources_path / f"{idx:02d}_data_overview.jpg"

    @staticmethod
    def get_report_feature_distribution_path(report_resources_path: Path, feature : str, idx) -> Path:
        return report_resources_path / f"{idx:02d}_distribution_{feature}.jpg"

    @staticmethod
    def get_report_error_overview_path(report_resources_path: Path, idx) -> Path:
        return report_resources_path / f"{idx:02d}_error_overview.jpg"

    @staticmethod
    def get_report_error_quantiles_path(report_resources_path: Path, idx) -> Path:
        return report_resources_path / f"{idx:02d}_error_quantiles.jpg"

    @staticmethod
    def get_report_error_percentage_distribution_path(report_resources_path: Path, idx) -> Path:
        return report_resources_path / f"{idx:02d}_hist_error_percentage.jpg"

    @staticmethod
    def get_report_top_k_largest_errors_path(report_resources_path: Path, idx) -> Path:
        return report_resources_path / f"{idx:02d}_top_k_largest_errors.jpg"

    @staticmethod
    def get_report_feature_importance_path(report_resources_path: Path, kind: str, idx) -> Path:
        return report_resources_path / f"{idx:02d}_feature_importance_{kind}.jpg"

    @staticmethod
    def get_report_partial_dependence_plot_path(report_resources_path: Path, feature: str, idx) -> Path:
        return report_resources_path / f"{idx:02d}_pdp_{feature}.jpg"

    @staticmethod
    def get_report_real_vs_predicted_quantiles_path(report_resources_path: Path, idx) -> Path:
        return report_resources_path / f"{idx:02d}_real_vs_predicted_quantiles.jpg"

    @staticmethod
    def get_report_real_vs_predicted_by_feature_path(report_resources_path: Path, feature: str, idx) -> Path:
        return report_resources_path / f"{idx:02d}_a_real_vs_predicted_by_{feature}.jpg"

    @staticmethod
    def get_report_top_10_biggest_errors_by_feature_path(report_resources_path: Path, feature: str, idx) -> Path:
        return report_resources_path / f"{idx:02d}_b_top_10_biggest_errors_by_{feature}.jpg"

    @staticmethod
    def get_report_k_largest_errors_path(report_resources_path: Path, k: int, idx) -> Path:
        return report_resources_path / f"{idx:02d}_{k}_largest_errors.jpg"

    @staticmethod
    def get_report_learning_curve_path(report_resources_path: Path, idx) -> Path:
        return report_resources_path / f"{idx:02d}_learning_curve.jpg"

    @staticmethod
    def get_report_cover_image_path(report_resources_path: Path, idx) -> Path:
        return report_resources_path / f"{idx:02d}_report_cover_image.jpg"

    @staticmethod
    def get_data_name_references_path() -> Path:
        return REFERENCES_PATH / 'data_name_references.json'

    @staticmethod
    def get_names_file_path(names_file_name : str) -> Path:
        return REFERENCES_PATH / f'{names_file_name}.txt'

    @staticmethod
    def get_brackets_path(feature: str) -> Path:
        return BRACKETS_PATH / f'{feature}_brackets.json'



## DEPRECATED

@deprecated
def get_template_path(service: str, date: str) -> Path:
    return DISTRIBUTION_PATH / service / 'templates' / f'{service}_template_{date}.xlsx'

@deprecated
def get_row_values_path(service: str, date: str) -> Path:
    return DISTRIBUTION_PATH / service / 'templates' / f'{service}_row_values_{date}.txt'

@deprecated
def get_transposed_config_path(service: str, date: str) -> Path:
    return DISTRIBUTION_PATH / service / 'templates' / f'{service}_transposed_config_{date}.json'

@deprecated
def get_sampled_data_name(service: str, params_v: str) -> str:
    current_date = datetime.now().strftime("%Y_%m_%d")
    return f"{service}_sampled_data_{params_v}_{current_date}"

@deprecated
def get_incremental_data_name(service: str, base_profile_v: str, values_v: str) -> str:
    return f"{service}_incremental_data_base_profile_{base_profile_v}_values_{values_v}"

@deprecated
def get_incremental_base_profile_path(service: str, v: str) -> Path:
    return DISTRIBUTION_PATH / service / 'incremental_params' / 'base_profiles' / f'base_profile_{v}.csv'

@deprecated
def get_incremental_values_path(service: str, v: str) -> Path:
    return DISTRIBUTION_PATH / service / 'incremental_params' / 'values' / f'values_{v}.csv'

@deprecated
def get_profiles_for_crawling_dir(data_name: str) -> Path:
    return INTERIM_DATA_PATH / data_name

@deprecated
def get_profiles_for_crawling_transposed(data_name: str) -> Path:
    return get_profiles_for_crawling_dir(data_name) / f'{data_name}.csv'

@deprecated
def get_profiles_for_crawling_zip_path(data_name: str) -> Path:
    return get_profiles_for_crawling_dir(data_name) / f'{data_name}.zip'

@deprecated
def get_profiles_after_crawling_zip_path(data_name: str) -> Path:
    return get_profiles_for_crawling_dir(data_name) / f'{data_name}_crawled.zip'
@deprecated
def get_private_key_file_path() -> Path:
    return Path('../../../ssh_key')

@deprecated
def get_remote_crawl_sh_path() -> str:
    return (REMOTE_CRAWLER_DIRECTORY / 'crawl.sh').as_posix()

@deprecated
def get_remote_profiles_path(data_name: str) -> str:
    return f'{REMOTE_CRAWLER_DIRECTORY}{data_name}/'

@deprecated
def get_remote_profiles_zip_path(data_name : str) -> str:
    return f"{get_remote_profiles_path(data_name)}{data_name}.zip"

@deprecated
def get_remote_profiles_subdirectory_path(data_name: str) -> str:
    return f'{get_remote_profiles_path(data_name)}profiles/'

@deprecated
def get_remote_profiles_after_crawling_zip_path(data_name: str) -> str:
    return f'{get_remote_profiles_path(data_name)}{data_name}_crawled.zip'

@deprecated
def get_features_path(data_name: str) -> Path:
    return PROCESSED_DATA_PATH / f'{data_name}_features.txt'

@deprecated
def get_params_path(service: str, params_v: str) -> Path:
    return EXTERNAL_DATA_PATH / 'services' / service / 'params' / params_v


