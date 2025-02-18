import os
import sys

import click
import pandas as pd
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilities.model_utils import *
from utilities.export_utils import *
from utilities.constants import DEFAULT_TARGET_VARIABLES


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


def print_metrics(label, mae, mae_std, rmse, rmse_std, mape, mape_std, meanPrice, target_variable):
    print(
        f"{label} - Mean MAE is {round(mae, 2)} ± {round(mae_std / mae * 100, 3)}%, "
        f"which is {round(mae / meanPrice * 100, 3)} ± {round(mae_std / meanPrice * 100, 3)}% of mean {target_variable}."
    )
    print(f"Mean RMSE is {round(rmse, 2)} ± {round(rmse_std / rmse * 100, 2)}%.")
    print(f"Mean MAPE is {round(mape * 100, 2)} ± {round(mape_std / mape, 3)}%.")


def predict_all_models(data: pd.DataFrame, train_data_name: str, target_variables : list, on_top : pd.DataFrame, apply_presence_models: bool,
                       path_manager: PathManager, load_manager: LoadManager):
    predictions_all_models = {}

    for target_variable in target_variables[:] + ['rank1_price']:
        model_name = path_manager.get_model_name(train_data_name, target_variable, 'remove_vehicle_value')
        click.echo(f"Loading model: {model_name}")
        presence_model_name = path_manager.get_presence_model_name(train_data_name, target_variable)
        model = load_manager.load_model(train_data_name, model_name)
        if (model is None) or (not is_compatible(model, data)):
            click.echo(f"skipped {target_variable} - {'no model' if model is None else 'not compatible'}")
            continue

        expected_features = get_expected_features(model)
        predictions = predict_on_top(model, data[expected_features], on_top, target_variable)

        presence_model = load_manager.load_model(train_data_name, presence_model_name) if apply_presence_models else None
        if presence_model is not None:
            presence_predictions = apply_threshold(predict(presence_model, data[expected_features]), 0.5)
            predictions[~presence_predictions] = None
        predictions_all_models[model_name] = predictions

    return predictions_all_models


# ------------------------------------------------------------------------------
# Option validation callbacks (unchanged except for ordering)
# ------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------
# CLI commands
# ------------------------------------------------------------------------------

@click.command(name="model_predict")
@click.option("--service", required=True, type=click.STRING,
              help="Service name for path management.")
@click.option("--data_name", required=True, type=click.STRING,
              help="Name of the processed data file (without file extension).")
@click.option("--all_models", is_flag=True,
              help="Predict using all available compatible models.")
@click.option("--train_data_name", callback=validate_train_data_name,
              help="Name of the train data for the models")
@click.option("--model_name", callback=validate_model_name,
              help="Name of the model to use for prediction.")
@click.option("--apply_presence_models", is_flag=True, default=False,
              help="Signal if presence models should be used")
def model_predict(service: str, data_name: str, all_models: bool, train_data_name: str,
                  model_name: str, apply_presence_models: bool):

    path_manager = PathManager(service)
    load_manager = LoadManager(path_manager)
    export_manager = ExportManager(path_manager)

    data, features_info, features_on_top, features_model = load_manager.load_data(data_name)

    on_top = load_manager.load_on_top_file()

    if all_models:
        predictions_all_models = predict_all_models(data,
                                                    train_data_name,
                                                    DEFAULT_TARGET_VARIABLES.get(service, []),
                                                    on_top,
                                                    apply_presence_models,
                                                    path_manager,
                                                    load_manager)

        for m_name, predictions in predictions_all_models.items():
            print(len(data), len(predictions))
            data[m_name] = predictions

        model_names = [x for x in predictions_all_models.keys()]
        data = data[~(data[model_names] < 0).any(axis = 1)]

        predictions_path = path_manager.get_predictions_all_path(data_name, train_data_name, apply_presence_models)
        logging.info(f"Exported predictions to {predictions_path}.")
        data.to_parquet(predictions_path)
    else:
        model = load_manager.load_model(train_data_name, model_name)
        presence_model_name = path_manager.get_presence_model_name_from_model_name(model_name)

        if model is None or not is_compatible(model, data):
            click.echo("Error: Model and data are not compatible.")
            return

        predictions = predict_on_top(model, data, on_top, model_name)
        presence_model = load_manager.load_model(train_data_name, presence_model_name) if apply_presence_models else None
        if presence_model is not None:
            presence_predictions = apply_threshold(predict(presence_model, data), 0.5)
            predictions[~presence_predictions] = None

        logging.info(f"Predictions for {model_name}: {predictions}")


@click.command("compare_rank1_model_with_ensemble")
@click.option("--service", required=True, type=click.STRING,
              help="Service name for path management.")
@click.option("--train_data_name", required=True, type=click.STRING,
              help="Name of the train data for the models")
def compare_rank1_model_with_ensemble(service: str, train_data_name: str):
    path_manager = PathManager(service)
    load_manager = LoadManager(path_manager)

    data, features_info, features_on_top, features_model = load_manager.load_data(train_data_name)
    data = data.reset_index()
    predicted_prices = []

    for model_name in load_manager.get_all_models_on_train_data(train_data_name, is_presence_model=False):
        target_variable = model_name.replace('_model', '')
        out_of_sample_predictions = load_manager.load_out_of_sample_predictions(train_data_name, model_name, target_variable)
        data = pd.merge(data, out_of_sample_predictions, on='id_case', how='right')
        predicted_prices.append(target_variable)

    rank1_model = path_manager.get_model_name(train_data_name, 'rank1_price').replace('_model', '')
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


@click.group()
def cli():
    pass


cli.add_command(model_predict)
cli.add_command(compare_rank1_model_with_ensemble)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    cli()
