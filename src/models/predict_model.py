# predict_model.py
import sys
import os

import xgboost

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click
import glob
import utils
import logging
from pathlib import Path


INPUT_DATA = '../data/processed/'
INPUT_MODEL = '../models/'


def load_model(model_path):
    # Load the XGBoost model
    return xgboost.Booster(model_file=model_path)


def predict(model, data):
    dmatrix = xgboost.DMatrix(data=data, enable_categorical=True)
    predictions = model.predict(dmatrix)
    return predictions


def is_compatible(model, data):
    expected_features = get_expected_features(model)
    return set(expected_features).issubset(data.columns)


def get_expected_features(model):
    return getattr(model, "feature_names", [])


def predict_all_models(data):
    # Predict using all available compatible models in the specified directory
    compatible_models = {}

    for model_path in glob.glob(os.path.join(INPUT_MODEL, "*.json")):
        model = load_model(model_path)
        model_name = os.path.basename(model_path).replace(".json", "")
        if is_compatible(model, data):
            compatible_models[model_name] = model

    predictions_all_models = {}
    for model_name, model in compatible_models.items():
        predictions = predict(model, data)
        predictions_all_models[model_name] = predictions

    return predictions_all_models


def validate_model_name(ctx, param, value):
    # Custom callback to validate the --model_name option
    all_option = ctx.params.get('all')
    if not all_option and not value:
        raise click.BadParameter('--model_name is required when --all is False.')
    return value


@click.command()
@click.option("--data_name", required=True, type=click.STRING,
              help="Name of to the processed data file (without file extension).")
@click.option("--all", is_flag=True, help="Predict using all available compatible models.")
@click.option("--model_name", callback=validate_model_name, help="Name of the model to use for prediction.")
def main(data_name, all, model_name):
    # Load the data
    data_path = f'{INPUT_DATA}{data_name}_processed.csv'
    features_path = f'{INPUT_DATA}{data_name}_features.txt'
    model_path = f'{INPUT_MODEL}{model_name}.json'

    data, features = utils.load_data(data_path, features_path)

    if all:
        predictions_all_models = predict_all_models(data)
        for model_name, predictions in predictions_all_models.items():
            print(f"Predictions for {model_name}: {predictions}")
    else:
        if not model_path:
            click.echo("Error: Please specify a model path.")
            return
        model = load_model(model_path)
        if not is_compatible(model, data):
            click.echo("Error: Model and data are not compatible.")
            return

        predictions = predict(model, data)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
