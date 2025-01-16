import os
import sys

import hyperopt
from hyperopt import Trials
from utilities.files_utils import *
from utilities.load_utils import *
from utilities.path_utils import *


class ExportManager:
    def __init__(self, path_manager : PathManager):
        self.path_manager = path_manager

    def export_data_name_reference(data_name_reference : dict):
        data_name_reference_path = PathManager.get_data_name_references_path()
        with open(data_name_reference_path, 'w') as json_file:
            json.dump(data_name_reference, json_file, indent=2)


    def export_trials(self, trials : Trials, train_data_name : str, model_name : str):
        trials_path = self.path_manager.get_model_trials_path(train_data_name, model_name)
        with open(trials_path, 'wb') as f:
            pickle.dump(trials, f)

    def export_model(self, model: xgboost.Booster, hyperparameters: dict, out_of_sample_predictions: pd.Series, trials : Trials, train_data_name : str, model_name : str) -> None:
        model_directory_path = self.path_manager.get_model_directory(train_data_name, model_name)
        print(model_directory_path)
        prepare_dir(model_directory_path)

        model_path = self.path_manager.get_model_path(train_data_name, model_name)

        print(model_path)

        hyperparameters_path = self.path_manager.get_model_hyperparameters_path(train_data_name, model_name)
        out_of_sample_predictions_path = self.path_manager.get_model_cv_out_of_sample_predictions_path(train_data_name, model_name)

        model.save_model(model_path)
        dict_to_json(hyperparameters, hyperparameters_path)
        out_of_sample_predictions.to_csv(out_of_sample_predictions_path)
        self.export_trials(trials, train_data_name, model_name)