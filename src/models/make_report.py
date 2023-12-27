import click
import dataframe_image as dfi
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost
from matplotlib.ticker import PercentFormatter
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import logging
from PIL import Image

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import os

os.environ["QT_QPA_PLATFORM"] = "wayland"
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils


def make_data_overview(data: pd.DataFrame, report_resources_path: str) -> None:
    unique = pd.DataFrame(data.nunique()).rename(columns={0: 'unique'}).T
    describe = pd.concat([data.describe(), unique])
    describeStyle = describe.T.style.format(precision=2)
    dfi.export(describeStyle, f'{report_resources_path}data_overview.jpg', dpi=200)


def plot_feature_distribution(feature: pd.Series, report_resources_path: str) -> None:
    plt.figure(figsize=(10, 6))

    if feature.dtype == 'bool':
        sns.countplot(x=feature.name)
        plt.title(f'Distribution of {feature.name}')
    else:
        sns.histplot(feature, bins=40, kde=False, stat='density', alpha=0.7)
        plt.title(f'Distribution of {feature.name}')
        plt.xlabel(feature.name)
        plt.ylabel('Percent of values')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.locator_params(axis='x', nbins=20)
        plt.xticks(rotation=45, ha='right')

    plt.savefig(f'{report_resources_path}distribution_{feature}.jpg', bbox_inches='tight')
    plt.close()


def make_error_overview(acutal: pd.Series, predicted: pd.Series, report_resources_path: str) -> None:
    mae = mean_absolute_error(acutal, predicted)
    maeP = mae / acutal.mean() * 100
    mse = mean_squared_error(acutal, predicted)
    mseP = mse / acutal.mean() * 100
    mape = mean_absolute_percentage_error(acutal, predicted)

    report = pd.DataFrame(columns=['Mean Absolute Error', 'Mean Absolute Error as % of average target',
                                   'Mean Squared Error', 'Mean absolute percentage error'],
                          index=[0])
    report['Mean Absolute Error'] = mae
    report['Mean Absolute Error as % of average target'] = maeP
    report['Mean Squared Error'] = mse
    report['Mean absolute percentage error'] = mape

    reportStyle = report.T.style.format(precision=2)
    dfi.export(reportStyle, f'{report_resources_path}error_overview.jpg', dpi=200)


def plot_hist_error_percentage(error: np.array, report_resources_path: str) -> None:
    plt.hist(error, range=[error.min() - 1, error.max() + 1], bins=40, weights=np.ones(len(error)) / len(error))
    plt.xlabel('Error percentage')
    plt.ylabel('Percent of errors')
    plt.savefig(f'{report_resources_path}hist_error_percentage.jpg')
    plt.close()


def make_error_quantiles(real: pd.Series, predicted: pd.Series, report_resources_path: str,
                         quantile_ranges: list = None) -> None:
    quantile_ranges = utils.QUANTILE_RANGE if quantile_ranges is None else quantile_ranges
    errors = real - predicted
    absolute_errors = errors.abs()
    relative_errors = errors / real
    errors_q = errors.quantile(quantile_ranges)
    abs_error_q = absolute_errors.quantile(quantile_ranges)
    relative_errors_q = relative_errors.quantile(quantile_ranges) * 100
    abs_relative_errors_q = np.abs(relative_errors).quantile(quantile_ranges) * 100
    quantile_df = pd.DataFrame(index=errors_q.index)
    quantile_df['error_q'] = errors_q
    quantile_df['abs_error_q'] = abs_error_q
    quantile_df['relative_error_q'] = relative_errors_q
    quantile_df['abs_relative_error_q'] = abs_relative_errors_q
    quantile_df = quantile_df.round(3).T
    quantile_df_style = quantile_df.T.style.format(precision=3)
    dfi.export(quantile_df_style, f'{report_resources_path}error_quantiles.jpg', dpi=200)


def partial_dependence_analysis(model: xgboost.Booster,

                                data: pd.DataFrame,
                                features: list,
                                target_variable: str,
                                grid_resolution: int = 100) -> tuple:
    importance_dict = {}
    pdp_dict = {}

    for feature in features:

        if data[feature].dtype == 'category':  # Check if the variable is categorical
            feature_range = data[feature].unique()
        else:
            feature_range = np.linspace(data[feature].min(), data[feature].max(), grid_resolution)

        partial_dependence_values = []
        for value in feature_range:
            data_copy = data.copy()
            data_copy[feature] = value
            predictions = utils.predict(model, data_copy[features])
            partial_dependence_values.append(np.mean(predictions))
            importance_dict[feature] = np.std(partial_dependence_values)
            pdp_dict[feature] = (feature_range, partial_dependence_values)

    return pdp_dict, importance_dict


def plot_pdp_importance(features: list, importances, title="Feature importance",
                        xlabel="Partial Dependence Feature Importance",
                        ylabel="Features", values_format="{:.2f}", save_path=None, **kwargs):
    sorted_indices = np.argsort(importances)
    features = [features[i] for i in sorted_indices]
    importances = [importances[i] for i in sorted_indices]

    # Plotting
    fig, ax = plt.subplots()
    ax.barh(features, importances, **kwargs)

    # Customize plot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if save_path is not None:
        plt.savefig(save_path)
    return ax


def plot_feature_importance(model: xgboost.Booster, importances: dict, report_resources_path: str):
    export_path = f'{report_resources_path}feature_importance'
    xgboost.plot_importance(model, importance_type='weight', xlabel='weight')
    plt.savefig(f'{export_path}_weight.jpg')
    plt.close()

    xgboost.plot_importance(model, importance_type='gain', xlabel='gain')
    plt.savefig(f'{export_path}_gain.jpg')
    plt.close()

    xgboost.plot_importance(model, importance_type='cover', xlabel='cover')
    plt.savefig(f'{export_path}_cover.jpg')
    plt.close()

    plot_pdp_importance(list(importances.keys()), list(importances.values()), save_path=f'{export_path}_pdp.jpg')
    plt.close()


def plot_pdps(features: list, pdp_dict: dict, report_resources_path: str):
    for feature in features:
        feature_range, pdp_values = pdp_dict[feature]
        plt.plot(feature_range, pdp_values, label=feature)
        plt.xlabel('Feature Values')
        plt.ylabel('Partial Dependence')
        plt.title('Partial Dependence Plots for Selected Features')
        plt.legend()
        plt.savefig(f'{report_resources_path}pdp_{feature}.jpg')
        plt.close()


def getQauntSplit(step=50):
    return [i / 1000 for i in range(0, 1001, step)]


def plot_real_vs_predicted_quantiles(real: pd.Series, predicted: pd.Series, report_resource_path: str) -> None:
    step = (100000 // len(real))

    quant = getQauntSplit(step)

    qr = np.quantile(real, quant)
    qp = np.quantile(predicted, quant)

    plt.scatter(qr, qp, alpha=0.8, s=40)

    x = np.linspace(qr.min(), qr.max())
    plt.plot(x, x, c='r')
    plt.xlabel('Real quantiles for training data')
    plt.ylabel('Predicted quantiles for training data')
    plt.savefig(f'{report_resource_path}' + 'real_vs_predicted_quantiles.jpg', bbox_inches='tight')
    plt.close()


def plot_real_vs_predicted_quantiles_by_feature(data: pd.DataFrame, predictions: pd.Series, feature: str,
                                                target_variable: str, report_resources_path: str,
                                                num_quantiles: int = 20):
    data['quantiles'] = pd.qcut(data[feature], num_quantiles, labels=False, duplicates='drop')
    data['predicted'] = predictions
    quantile_values = data.groupby('quantiles')[feature].mean().values

    mean_values = data.groupby('quantiles').agg({
        target_variable: 'mean',
        'predicted': 'mean'
    }).reset_index()

    plt.figure(figsize=(12, 8))
    sns.lineplot(x=quantile_values, y=mean_values[target_variable], label='Real Mean', marker='o')
    sns.lineplot(x=quantile_values, y=mean_values['predicted'], label='Predicted Mean', marker='x')

    plt.title(f'Mean Real vs Predicted Values for Different Feature Ranges of {feature}')
    plt.xlabel(feature)
    plt.ylabel(f'Mean {target_variable}')
    plt.savefig(f'{report_resources_path}real_vs_predicted_quantiles_by_{feature}.jpg')
    plt.close()


def make_pdf(reprot_resources_path: str, report_path: str):
    if not os.path.exists(reprot_resources_path):
        raise ValueError(f"Image directory does not exist: {reprot_resources_path}")

    image_paths = [
        os.path.join(reprot_resources_path, filename)
        for filename in os.listdir(reprot_resources_path)
        if filename.lower().endswith(".jpg")
    ]

    if not image_paths:
        raise ValueError(f"No JPG images found in directory: {reprot_resources_path}")

    image_paths = sorted(image_paths)
    images = [Image.open(path) for path in image_paths]

    with open(report_path, "wb") as pdf_file:
        images[0].save(pdf_file, "PDF", resolution=100.0, save_all=True, append_images=images[1:])


def generate_report_util(model: xgboost.Booster, data: pd.DataFrame, features: list, target_variable: str,
                         out_of_sample_predictions: pd.Series, report_path: str, report_resources_path: str):
    logging.info("Preparing directories for the report.")
    utils.prepareDir(report_path)
    utils.prepareDir(report_resources_path)

    logging.info("Creating data overview.")
    make_data_overview(data, report_resources_path)

    real = data[target_variable]
    # predictions = utils.predict(model, data[features])
    errors = real - out_of_sample_predictions
    errors_percentage = errors / data[target_variable] * 100
    logging.info("Making error overview")

    make_error_overview(real, out_of_sample_predictions, report_resources_path)

    logging.info("Making error quantiles.")
    make_error_quantiles(real, out_of_sample_predictions, report_resources_path)

    logging.info("Ploting error percentage distribution.")
    plot_hist_error_percentage(errors_percentage, report_resources_path)

    logging.info("Performing partial dependence analysis.")
    pdp_dict, importance = partial_dependence_analysis(model, data, features, target_variable, grid_resolution=20)

    logging.info("Plotting feature importance.")
    plot_feature_importance(model, importance, report_resources_path)

    logging.info("Plotting partial dependence plots.")
    plot_pdps(features, pdp_dict, report_resources_path)

    logging.info("Plotting real vs predicted quantiles.")
    plot_real_vs_predicted_quantiles(real, out_of_sample_predictions, report_resources_path)

    logging.info("Plotting real vs predicted quantiles by feature.")
    for feature in features:
        plot_real_vs_predicted_quantiles_by_feature(data, out_of_sample_predictions, feature, target_variable,
                                                    report_resources_path)

    make_pdf(report_resources_path, f'{report_path}report.pdf')

    logging.info("Report generation completed.")


@click.command("generate_report")
@click.option("--data_name", required=True, type=click.STRING)
@click.option("--target_variable", required=True, type=click.STRING)
def generate_report(data_name: str, target_variable: str):
    model_name = utils.get_model_name(data_name, target_variable)

    data_path = utils.get_processed_data_path(data_name)
    features_path = utils.get_features_path(data_name)
    model_path = utils.get_model_path(model_name)
    out_of_sample_predictions_path = utils.get_model_cv_out_of_sample_predictions_path(model_name)
    report_path = utils.get_report_path(model_name)
    report_resources_path = utils.get_report_resource_path(model_name)

    data, features = utils.load_data(data_path, features_path, target_variable)
    model = utils.load_model(model_path)
    out_of_sample_predictions = pd.read_csv(out_of_sample_predictions_path)[target_variable]

    generate_report_util(model, data, features, target_variable, out_of_sample_predictions, report_path,
                         report_resources_path)


@click.group()
def cli():
    pass


cli.add_command(generate_report)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    cli()
    # generate_report('netrisk_casco_2023_11_14__2023_11_20__2023_12_12', 'ALFA_price')
