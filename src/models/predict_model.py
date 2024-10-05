import os
import sys
import click
import pandas as pd
from warnings import simplefilter

from absl.logging import FATAL
from tensorboard.plugins.pr_curve.metadata import TRUE_NEGATIVES_INDEX

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from utilities.load_utils import *
from utilities.path_utils import *
from utilities.model_utils import *
from utilities.constants import DEFAULT_TARGET_VARIABLES

def get_variable_from_model_name(model_name: str) -> str:
    return '_'.join(model_name.split('_')[-3:-1])


def predict_all_models(data : pd.DataFrame, train_data_name : str, apply_presence_models : bool = False):

    predictions_all_models = {}

    for target_variable in DEFAULT_TARGET_VARIABLES:
        model_name = get_model_name(train_data_name, target_variable)
        presence_model_name = get_presence_model_name(train_data_name, target_variable)

        model = load_model(model_name)

        if (model is None) or (not is_compatible(model, data)):
            print('skipped ', target_variable, 'no model' if model is None else 'not compatible')
            continue

        expected_features = get_expected_features(model)
        predictions = predict(model, data[expected_features])

        presence_model = load_model(presence_model_name) if apply_presence_models else None
        if presence_model is not None:
            presence_predictions = apply_threshold(predict(presence_model, data[expected_features]), 0.5)
            predictions[~presence_predictions] = None
        predictions_all_models[model_name] = predictions

    return predictions_all_models

def validate_model_name(ctx, param, value):
    all_option = ctx.params.get('all_models')
    if not all_option and not value:
        raise click.BadParameter('--model_name is required when --all_models is False.')
    return value

def validate_train_data_name(ctx, param, value):
    all_option = ctx.params.get('all_models')
    if all_option and not value:
        raise click.BadParameter('--train_data_name is required when --all_models is True.')
    return value

@click.command(name="model_predict")
@click.option("--data_name", required=True, type=click.STRING,
              help="Name of the processed data file (without file extension).")
@click.option("--all_models", is_flag=True, help="Predict using all available compatible models.")
@click.option("--train_data_name", callback = validate_train_data_name, help = "Name of the train data for the models")
@click.option("--model_name", callback=validate_model_name, help="Name of the model to use for prediction.")
@click.option("--apply_presence_models", is_flag = True, default = False, help = 'Used to signal if presence_models should be used')
def model_predict(data_name: str, all_models: bool, train_data_name : str, model_name: str, apply_presence_models : bool):

    data, features_info, features_on_top, features_model = load_data(data_name)
    if all_models:
        predictions_all_models = predict_all_models(data, train_data_name, apply_presence_models = apply_presence_models)
        for model_name, predictions in predictions_all_models.items():
            data[model_name] = predictions

        predictions_path = get_predictions_all_path(data_name, train_data_name, apply_presence_models)
        logging.info(f"Exported predictions to {predictions_path}.")
        data.to_parquet(predictions_path)
    else:
        model = load_model(model_name)

        presence_model_name = get_presence_model_name_from_model_name(model_name)

        if model is None or not is_compatible(model, data):
            click.echo("Error: Model and data are not compatible.")
            return

        predictions = predict(model, data)

        presence_model = load_model(presence_model_name) if apply_presence_models else None

        if presence_model is not None:
            presence_predictions = apply_threshold(predict(presence_model, data), 0.5)
            predictions[~presence_predictions] = None

        logging.info(f"Predictions for {model_name}: {predictions}")

@click.group()
def cli():
    pass

cli.add_command(model_predict)

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    cli()
