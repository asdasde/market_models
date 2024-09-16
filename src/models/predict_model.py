import os
import sys
import click
import pandas as pd
from warnings import simplefilter
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

        model_path = get_model_path(model_name)
        presence_model_path = get_model_path(presence_model_name)
        model = load_model(model_path)
        print(model_path)
        if (model is None) or (not is_compatible(model, data)):
            print('skipped ', target_variable, 'no model' if model is None else 'not compatible')
            continue

        expected_features = get_expected_features(model)
        predictions = predict(model, data[expected_features])

        presence_model = load_model(presence_model_path) if (apply_presence_models and os.path.exists(presence_model_path)) else None
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
def model_predict(data_name: str, all_models: bool, train_data_name : str, model_name: str):

    data_path = get_processed_data_path(data_name)
    features_path = get_features_path(data_name)

    data, features_info, features_on_top, features_model = load_data(data_path)
    if all_models:
        predictions_all_models = predict_all_models(data, train_data_name, apply_presence_models=True)
        for model_name, predictions in predictions_all_models.items():
            data[model_name] = predictions

        print(data[predictions_all_models.keys()])
        print(data.index)

        predictions_path = get_predictions_all_path(data_name)
        logging.info(f"Exported predictions to {predictions_path}.")
        data.to_csv(predictions_path)
    else:
        model_path = get_model_path(model_name)

        if not model_path:
            click.echo("Error: Please specify a model path.")
            return

        model = load_model(model_path)
        if not is_compatible(model, data):
            click.echo("Error: Model and data are not compatible.")
            return

        predictions = predict(model, data)
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
