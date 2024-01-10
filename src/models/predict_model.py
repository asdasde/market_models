# predict_model.py
import sys
import os

import pandas as pd
import xgboost

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import click
import glob
import utils
import logging
from pathlib import Path

def is_compatible(model: xgboost.Booster, data: pd.DataFrame):
    expected_features = get_expected_features(model)
    return set(expected_features).issubset(data.columns)


def get_expected_features(model: xgboost.Booster):
    return getattr(model, "feature_names", [])


def predict_all_models(data: pd.DataFrame, train_data_name : str):
    compatible_models = {}

    all_model_names = utils.get_all_models_trained_on(train_data_name)

    for model_name in all_model_names:
        model_path = os.path.join(utils.MODELS_PATH, model_name)
        model_path = os.path.join(model_path, model_name + '.json')

        model = utils.load_model(model_path)

        if is_compatible(model, data):
            compatible_models[model_name] = model

    predictions_all_models = {}
    for model_name, model in compatible_models.items():
        predictions = utils.predict(model, data)
        predictions_all_models[model_name] = predictions

        # Added logging for predictions
        logging.info(f"Predictions for {model_name}: {predictions}")

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

    data_path = utils.get_processed_data_path(data_name)
    features_path = utils.get_features_path(data_name)
    model_path = utils.get_model_path(model_name)

    data, features = utils.load_data(data_path, features_path)

    if all_models:
        predictions_all_models = predict_all_models(data, train_data_name)
        for model_name, predictions in predictions_all_models.items():
            data[model_name] = predictions

        # Added logging for exporting predictions
        predictions_path = utils.get_predictions_all_path(data_name)
        logging.info(f"Exported predictions to {predictions_path}.")
        data.to_csv(predictions_path)
    else:
        if not model_path:
            click.echo("Error: Please specify a model path.")
            return
        model = utils.load_model(model_path)
        if not is_compatible(model, data):
            click.echo("Error: Model and data are not compatible.")
            return

        predictions = utils.predict(model, data)
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
