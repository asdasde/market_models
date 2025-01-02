import os
import sys

from hyperopt import Trials
from utilities.files_utils import *
from utilities.load_utils import *
from utilities.path_utils import *


def export_sampled_data(sampled_data : pd.DataFrame, service : str, params_v : str, custom_name : str = None):
    sampled_data_name = get_sampled_data_name(service, params_v) + custom_name
    prepare_dir(get_profiles_for_crawling_dir(sampled_data_name))
    sampled_data_path = get_profiles_for_crawling_transposed(sampled_data_name)
    sampled_data.to_csv(Path(sampled_data_path))
    logging.info(f"Exported sampled data to {sampled_data_path}")


def export_data_name_reference(data_name_reference : dict):
    data_name_reference_path = get_data_name_references_path()
    with open(data_name_reference_path, 'w') as json_file:
        json.dump(data_name_reference, json_file, indent=2)


def export_trials(trials, model_name : str):
    trials_path = get_model_trials_path(model_name)
    with open(trials_path, 'wb') as f:
        pickle.dump(trials, f)

def export_model(model: xgboost.Booster, hyperparameters: dict, out_of_sample_predictions: pd.Series, trials : Trials, model_name : str) -> None:
    prepare_dir(get_model_directory(model_name))

    model_path = get_model_path(model_name)
    hyperparameters_path = get_model_hyperparameters_path(model_name)
    out_of_sample_predictions_path = get_model_cv_out_of_sample_predictions_path(model_name)

    model.save_model(model_path)
    dict_to_json(hyperparameters, hyperparameters_path)
    out_of_sample_predictions.to_csv(out_of_sample_predictions_path)
    export_trials(trials, model_name)