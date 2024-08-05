import os
import sys
import click
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from matplotlib.ticker import PercentFormatter
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from matplotlib.colors import Normalize
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.constants import QUANTILE_RANGE
from utilities.model_utils import *
from utilities.files_utils import *
from utilities.load_utils import *

os.environ["QT_QPA_PLATFORM"] = "wayland"

def make_data_overview(data: pd.DataFrame, report_resources_path: str) -> None:
    unique = pd.DataFrame(data.nunique()).rename(columns={0: 'unique'}).T
    describe = pd.concat([data.describe().round(1), unique])
    describeStyle = describe.T.style.format(precision=2)

    fig, ax = plt.subplots(figsize=(10, len(describeStyle.data) * 0.5))
    ax.axis('off')

    table = ax.table(cellText=describeStyle.data.values,
                     colLabels=describeStyle.data.columns,
                     rowLabels=describeStyle.data.index,
                     cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(describeStyle.data.columns))))

    plt.savefig(f'{report_resources_path}01_data_overview.jpg', bbox_inches='tight', dpi=200)
    plt.close()


def plot_feature_distribution(feature: pd.Series, report_resources_path: str) -> None:
    plt.figure(figsize=(10, 6))

    if feature.dtype == 'bool':
        sns.countplot(x=feature)
        plt.title(f'Distribution of {feature.name}')
    else:
        if feature.dtype == 'object':
            try:
                feature = pd.to_numeric(feature, errors='coerce')
            except ValueError:
                pass

        sns.histplot(feature, bins=40, kde=False, stat='density', alpha=0.7)
        plt.title(f'Distribution of {feature.name}')
        plt.xlabel(str(feature.name))
        plt.ylabel('Percent of values')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.locator_params(axis='x', nbins=20)
        plt.xticks(rotation=45, ha='right')

    plt.savefig(f'{report_resources_path}07_distribution_{feature.name}.jpg', bbox_inches='tight')
    plt.close()


def make_error_overview(actual: pd.Series, predicted: pd.Series, report_resources_path: str) -> None:
    mae = mean_absolute_error(actual, predicted)
    maeP = mae / actual.mean() * 100
    mse = mean_squared_error(actual, predicted)
    mseP = mse / actual.mean() * 100
    mape = mean_absolute_percentage_error(actual, predicted)

    report = pd.DataFrame(columns=['Mean Absolute Error', 'Mean Absolute Error as % of average target',
                                   'Mean Squared Error', 'Root Mean Squared Error', 'Mean absolute percentage error'],
                          index=[0])
    report['Mean Absolute Error'] = round(mae, 2)
    report['Mean Squared Error'] = round(mse, 2)
    report['Mean Absolute Error as % of average target'] = round(maeP, 2)
    report['Root Mean Squared Error'] = round(np.sqrt(mse), 2)
    report['Mean absolute percentage error'] = round(mape, 2)

    reportStyle = report.T.style.format(precision=2)

    fig, ax = plt.subplots(figsize=(10, len(reportStyle.data) * 0.5))
    ax.axis('off')

    table = ax.table(cellText=reportStyle.data.values,
                     colLabels=reportStyle.data.columns,
                     rowLabels=reportStyle.data.index,
                     cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(reportStyle.data.columns))))

    plt.savefig(f'{report_resources_path}02_error_overview.jpg', bbox_inches='tight', dpi=200)
    plt.close()



def plot_hist_error_percentage(error: np.array, report_resources_path: str) -> None:

    delta = max(abs(error.quantile(0.1)), error.quantile(0.9))
    hist_range = (-delta, delta)
    plt.hist(error, range = hist_range, bins=40, weights=np.ones(len(error)) / len(error))
    plt.xlabel('Error percentage')
    plt.ylabel('Percent of errors')
    plt.savefig(f'{report_resources_path}04_hist_error_percentage.jpg')
    plt.close()


def make_error_quantiles(real: pd.Series, predicted: pd.Series, report_resources_path: str,
                         quantile_ranges: list = None) -> None:
    quantile_ranges = QUANTILE_RANGE if quantile_ranges is None else quantile_ranges
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

    fig, ax = plt.subplots(figsize=(10, len(quantile_df_style.data) * 0.5))
    ax.axis('off')

    table = ax.table(cellText=quantile_df_style.data.values,
                     colLabels=quantile_df_style.data.columns,
                     rowLabels=quantile_df_style.data.index,
                     cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(quantile_df_style.data.columns))))

    plt.savefig(f'{report_resources_path}03_error_quantiles.jpg', bbox_inches='tight', dpi=200)
    plt.close()


def partial_dependence_analysis(model: xgboost.Booster,
                                data: pd.DataFrame,
                                features: list,
                                target_variable: str,
                                grid_resolution: int = 100) -> tuple:
    importance_dict = {}
    pdp_dict = {}

    for feature in features:

        if feature in ['CarMake', 'CarModel']:
            continue


        if data[feature].dtype == 'category':  # Check if the variable is categorical
            feature_range = data[feature].unique()
        else:
            feature_range = np.linspace(data[feature].min(), data[feature].max(), grid_resolution)

        partial_dependence_values = []
        for value in feature_range:
            data_copy = data.copy()
            data_copy[feature] = value
            predictions = predict(model, data_copy[features])
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
    export_path = f'{report_resources_path}06_feature_importance'
    xgboost.plot_importance(model, importance_type='weight', xlabel='weight')
    plt.savefig(f'{export_path}_weight.jpg')
    plt.close()

    xgboost.plot_importance(model, importance_type='gain', xlabel='gain')
    plt.savefig(f'{export_path}_gain.jpg')
    plt.close()

    xgboost.plot_importance(model, importance_type='cover', xlabel='cover')
    plt.savefig(f'{export_path}_cover.jpg')
    plt.close()

    if importances is not None:
        plot_pdp_importance(list(importances.keys()), list(importances.values()), save_path=f'{export_path}_pdp.jpg')
    plt.close()


def plot_pdps(features: list, pdp_dict: dict, report_resources_path: str):
    for feature in features:
        if feature in ['CarMake', 'CarModel']:
            continue
        feature_range, pdp_values = pdp_dict[feature]
        plt.plot(feature_range, pdp_values, label=feature)
        plt.xlabel('Feature Values')
        plt.ylabel('Partial Dependence')
        plt.title('Partial Dependence Plots for Selected Features')
        plt.legend()
        plt.savefig(f'{report_resources_path}08_pdp_{feature}.jpg')
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
    plt.savefig(f'{report_resource_path}' + '09_real_vs_predicted_quantiles.jpg', bbox_inches='tight')
    plt.close()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error


def plot_real_vs_predicted_quantiles_by_feature(data_p: pd.DataFrame, predictions: pd.Series, feature: str,
                                                target_variable: str, report_resources_path: str,
                                                num_quantiles: int = 20):
    data = data_p.copy()
    data['predicted'] = predictions.values
    data = data.dropna(subset=[feature, target_variable])
    print(feature, data[feature].dtype)
    if data[feature].dtype in ['object', 'category'] or data[feature].nunique() < num_quantiles:
        data['quantiles'] = data[feature]
        quantile_values = sorted(data[feature].unique())
    else:
        data['quantiles'] = pd.qcut(data[feature], num_quantiles, labels=False, duplicates='drop')
        quantile_values = data.groupby('quantiles')[feature].mean().values

    data['error'] = abs(data[target_variable].values - data['predicted'].values) / data[target_variable].values * 100

    if all(isinstance(val, str) and len(val) == 10 and val[4] == '_' and val[7] == '_' for val in quantile_values):
        quantile_values = pd.to_datetime(quantile_values, format='%Y_%m_%d', errors='coerce')
    elif isinstance(quantile_values[0], (np.bool_, bool)):
        pass
    else:
        quantile_values = pd.to_numeric(quantile_values, errors='coerce').round(2)

    mean_values = data.groupby('quantiles').agg({
        target_variable: 'mean',
        'predicted': 'mean',
        'error': 'mean',
        feature : 'count'
    }).reset_index().dropna(subset=[target_variable])
    mean_values['error'] = mean_values['error'].round(decimals=2).sort_index()

    if feature == 'LicenseAge' or data[feature].nunique() < 2:
        return

    fig, axes = plt.subplots(nrows=3, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1, 1]})

    # Plot real vs predicted means
    sns.lineplot(x=quantile_values, y=mean_values[target_variable], label='Real Mean', marker='o', ax=axes[0])
    sns.lineplot(x=quantile_values, y=mean_values['predicted'], label='Predicted Mean', marker='x', ax=axes[0])
    try:
        axes[0].set_xticks(np.round(np.linspace(quantile_values.min(), quantile_values.max(), 15)))
    except Exception:
        pass

    # Plot histograms of feature value distributions
    sns.histplot(data=data, x=feature, bins=15, stat='percent', kde=False, color='blue', alpha=0.3, ax=axes[1])
    axes[1].set_title(f'Distribution of {feature}')
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel('Percentage of values')
    axes[1].set_xticklabels(quantile_values, rotation=45, ha='right')

    norm = Normalize(vmin=mean_values['error'].min(), vmax=mean_values['error'].max())
    color_map = plt.colormaps.get_cmap('magma_r')

    colors = color_map(norm(mean_values['error'])).tolist()  # Convert to list

    sns.barplot(x=quantile_values, y=mean_values['error'], palette=colors, ax=axes[2], hue=quantile_values, dodge=False,
                legend=False)

    axes[0].set_title(f'Mean Real vs Predicted Values for Different Feature Ranges of {feature}')
    axes[0].set_xlabel('')
    axes[0].set_ylabel(f'Mean {target_variable}')

    axes[2].set_xlabel(feature)
    axes[2].set_ylabel('Mean Absolute Percentage Error')

    # Set the ticks and labels manually
    axes[2].set_xticks(range(len(quantile_values)))
    axes[2].set_xticklabels(quantile_values, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'{report_resources_path}10_real_vs_predicted_quantiles_by_{feature}.jpg')
    plt.close()

    # Create a DataFrame for the 5 segments with the biggest errors


    if data[feature].dtype == 'category':
        encoder_path = get_encoder_path(feature)
        encoder = joblib.load(encoder_path)
        inv = encoder.inverse_transform(quantile_values)
    else:
        inv = quantile_values
    mean_values['values'] = inv
    top_errors = mean_values.nlargest(20, 'error')
    top_errors['feature_'] = feature
    top_errors['contrib'] = top_errors['error'] * top_errors[feature] / len(data)

    print(feature, top_errors['contrib'].sum())

    # Plot the DataFrame as a table
    fig, ax = plt.subplots(figsize=(10, 4))  # Set the size of the table
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=top_errors.values, colLabels=top_errors.columns, cellLoc='center', loc='center')



    plt.savefig(f'{report_resources_path}11_top_10_biggest_errors_by_{feature}.jpg')
    plt.close()
def k_largest_errors(data: pd.DataFrame, errors: pd.Series, k: int, report_resources_path: str):
    dc = data.copy()
    dc['abs_error'] = abs(errors)
    k_largest_errors = dc.sort_values(by = 'abs_error', ascending=False).iloc[:k]
    k_largest_errors_style = k_largest_errors.style.format(precision=2)

    fig, ax = plt.subplots(figsize=(10, len(k_largest_errors_style.data) * 0.5))
    ax.axis('off')

    table = ax.table(cellText=k_largest_errors_style.data.values,
                     colLabels=k_largest_errors_style.data.columns,
                     rowLabels=k_largest_errors_style.data.index,
                     cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(k_largest_errors_style.data.columns))))

    plt.savefig(f'{report_resources_path}05_{k}_largest_errors.jpg', bbox_inches='tight', dpi=200)
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



def generate_report_cover_image(report_resources_path: str, insurance_name: str, table_of_contents: list):
    image_width, image_height = 800, 600
    background_color = (255, 255, 255)
    text_color = (0, 0, 0)
    font_path = "arial.ttf"  # Make sure the font path is correct
    title_font_size = 40
    subtitle_font_size = 20
    toc_font_size = 16

    image = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.truetype(font_path, title_font_size)
        subtitle_font = ImageFont.truetype(font_path, subtitle_font_size)
        toc_font = ImageFont.truetype(font_path, toc_font_size)
    except IOError:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        toc_font = ImageFont.load_default()

    current_y = 50

    report_date = datetime.now().strftime("%Y-%m-%d")
    title_text = f"Report for {insurance_name}"
    subtitle_text = f"Date: {report_date}"

    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text((image_width // 2 - title_width // 2, current_y), title_text, fill=text_color, font=title_font)
    current_y += 60

    subtitle_bbox = draw.textbbox((0, 0), subtitle_text, font=subtitle_font)
    subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
    draw.text((image_width // 2 - subtitle_width // 2, current_y), subtitle_text, fill=text_color, font=subtitle_font)
    current_y += 40

    draw.text((50, current_y), "Table of Contents:", fill=text_color, font=subtitle_font)
    current_y += 40

    for item in table_of_contents:
        draw.text((70, current_y), f"- {item}", fill=text_color, font=toc_font)
        current_y += 30

    image_path = f"{report_resources_path}00_report_cover_image.jpg"
    image.save(image_path)
    return image_path


def generate_report_util(model: xgboost.Booster, data: pd.DataFrame, features: list, target_variable: str,
                         out_of_sample_predictions: pd.Series, report_path: str, report_resources_path: str, skip_pdp : bool = False):
    logging.info("Preparing directories for the report.")
    prepare_dir(report_path)
    prepare_dir(report_resources_path)

    table_of_contents = [
        "Data Overview",
        "Error Overview",
        "Error Quantiles",
        "Error Percentage Distribution",
        "Top k Largest Errors",
        "Feature Importance",
        "Feature Distribution",
        "Partial Dependence Plots",
        "Real vs Predicted Quantiles",
        "Real vs Predicted Quantiles by Feature",
    ]

    logging.info("Generating report cover image.")
    generate_report_cover_image(report_resources_path, target_variable, table_of_contents)

    logging.info("Creating data overview.")
    make_data_overview(data, report_resources_path)

    real = data[target_variable]
    errors = real - out_of_sample_predictions
    errors_percentage = errors / data[target_variable] * 100
    logging.info("Making error overview")

    make_error_overview(real, out_of_sample_predictions, report_resources_path)

    logging.info("Making error quantiles.")
    make_error_quantiles(real, out_of_sample_predictions, report_resources_path)

    logging.info("Plotting error percentage distribution.")
    plot_hist_error_percentage(errors_percentage, report_resources_path)

    logging.info("Finding k largest errors.")
    k_largest_errors(data, errors, 10, report_resources_path)

    pdp_dict = None
    importance = None
    if not skip_pdp:
        logging.info("Performing partial dependence analysis.")
        pdp_dict, importance = partial_dependence_analysis(model, data, features, target_variable, grid_resolution=20)

    logging.info("Plotting feature importance.")
    plot_feature_importance(model, importance, report_resources_path)


    logging.info("Plotting feature distributions.")
    for feature in features:
        plot_feature_distribution(data[feature], report_resources_path)

    if not skip_pdp:
        logging.info("Plotting partial dependence plots.")
        plot_pdps(features, pdp_dict, report_resources_path)

    logging.info("Plotting real vs predicted quantiles.")
    plot_real_vs_predicted_quantiles(real, out_of_sample_predictions, report_resources_path)

    logging.info("Plotting real vs predicted quantiles by feature.")
    plot_real_vs_predicted_quantiles_by_feature(data, out_of_sample_predictions, 'DateCrawled', target_variable, report_resources_path)
    for feature in ['DateCrawled'] + features:
        plot_real_vs_predicted_quantiles_by_feature(data, out_of_sample_predictions, feature, target_variable , report_resources_path)


    logging.info("Generating PDF report.")
    make_pdf(report_resources_path, f'{report_path}/report.pdf')

    logging.info("Report generation completed.")


@click.command("generate_report")
@click.option("--data_name", required=True, type=click.STRING)
@click.option("--target_variable", required=True, type=click.STRING)
@click.option('--skip_pdp', default = False, is_flag=True, help = 'Useful when testing, since pdp plots take a long time...')
def generate_report(data_name: str, target_variable: str, skip_pdp: bool):
    model_name = get_model_name(data_name, target_variable)


    data_path = get_processed_data_path(data_name)
    features_path = get_features_path(data_name)
    model_path = get_model_path(model_name)
    out_of_sample_predictions_path = get_model_cv_out_of_sample_predictions_path(model_name)
    report_path = get_report_path(model_name)
    report_resources_path = get_report_resource_path(model_name)

    data, features = load_data(data_path, features_path, target_variable)
    model = load_model(model_path)
    out_of_sample_predictions = pd.read_csv(out_of_sample_predictions_path)
    out_of_sample_predictions.columns = ['id_case', target_variable]
    out_of_sample_predictions = out_of_sample_predictions.set_index('id_case')
    out_of_sample_predictions = out_of_sample_predictions[target_variable]

    generate_report_util(model, data, features, target_variable, out_of_sample_predictions, report_path,
                         report_resources_path, skip_pdp)


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
