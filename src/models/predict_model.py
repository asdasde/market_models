import os
import sys
import click
import pandas as pd
from warnings import simplefilter

from absl.logging import FATAL
from sqlalchemy.testing.plugin.plugin_base import logging
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

    for target_variable in DEFAULT_TARGET_VARIABLES + ['rank1_price']:
        model_name = get_model_name(train_data_name, target_variable)
        print(model_name)
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

def print_metrics(label, mae, mae_std, rmse, rmse_std, mape, mape_std, meanPrice, target_variable):
    print(
        f"{label} - Mean MAE is {round(mae, 2)} ± {round(mae_std / mae * 100, 3)}%, "
        f"which is {round(mae / meanPrice * 100, 3)} ± {round(mae_std / meanPrice * 100, 3)}% percent of mean {target_variable}."
    )
    print(f"Mean RMSE is {round(rmse, 2)} ± {round(rmse_std / rmse * 100, 2)}%.")
    print(f"Mean MAPE is {round(mape * 100, 2)} ± {round(mape_std / mape, 3)}%.")

@click.command("compare_rank1_model_with_ensemble")
@click.option("--service", type=click.STRING)
@click.option("--train_data_name", type=click.STRING)
def compare_rank1_model_with_ensemble(service: str, train_data_name: str):
    data, features_info, features_on_top, features_model = load_data(train_data_name)
    data = data.reset_index()
    predicted_prices = []
    for model_name in get_all_models_on_train_data(train_data_name, is_presence_model=False):
        target_variable = model_name.replace('_model', '')
        out_of_sample_predictions = load_out_of_sample_predictions(model_name, target_variable)
        data = pd.merge(data, out_of_sample_predictions, on = 'id_case', how = 'right')
        predicted_prices.append(target_variable)

    rank1_model = get_model_name(train_data_name, 'rank1_price').replace('_model', '')
    if rank1_model not in predicted_prices:
        logging.error("No rank1 model trained on this data, please train the model first")
        return

    predicted_prices.remove(rank1_model)
    data['rank1_price_ensemble'] = data[predicted_prices].min(axis=1)

    target = "rank1_price"
    mMae, sMae = calculate_mean_std_error(data[target], data[rank1_model], metric="mae")
    mRMse, sRMse = calculate_mean_std_error(data[target], data[rank1_model], metric="rmse")
    mMape, sMape = calculate_mean_std_error(data[target], data[rank1_model], metric="mape")

    emMae, esMae = calculate_mean_std_error(data[target], data['rank1_price_ensemble'], metric="mae")
    emRMse, esRMse = calculate_mean_std_error(data[target], data['rank1_price_ensemble'], metric="rmse")
    emMape, esMape = calculate_mean_std_error(data[target], data['rank1_price_ensemble'], metric="mape")

    print_metrics("Rank1 Model", mMae, sMae, mRMse, sRMse, mMape, sMape,
                      data['rank1_price'].mean(), 'rank1_price')
    print_metrics("Rank1 Ensemble", emMae, esMae, emRMse, esRMse, emMape, esMape,
                      data['rank1_price'].mean(), 'rank1_price')


def calculate_mean_std_error(actual, predicted, metric):
    if metric == "mae":
        errors = np.abs(actual - predicted)
    elif metric == "rmse":
        errors = np.sqrt((actual - predicted) ** 2)
    elif metric == "mape":
        errors = np.abs((actual - predicted) / actual)
    else:
        raise ValueError("Unsupported metric")

    mean_error = errors.mean()
    std_error = errors.std()
    return mean_error, std_error


@click.group()
def cli():
    pass

cli.add_command(model_predict)
cli.add_command(compare_rank1_model_with_ensemble)

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    cli()
