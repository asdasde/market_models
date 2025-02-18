import os
import random
import sys
import click
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from hyperopt import Trials

from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import CategoricalDtype
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.constants import QUANTILE_RANGE
from utilities.model_constants import DEFAULT_REPORT_TABLE_OF_CONTENTS, FEATURES_TO_SKIP_PDP
from utilities.model_utils import *
from utilities.files_utils import *
from utilities.load_utils import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.category")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.ticker")
os.environ["QT_QPA_PLATFORM"] = "wayland"


def make_data_overview(data: pd.DataFrame, features : List[str], report_resources_path: Path, idx) -> None:
    unique = pd.DataFrame(data[features].nunique()).rename(columns={0: 'unique'}).T
    describe = pd.concat([data[features].describe().round(1), unique])
    describeStyle = describe.T.style.format(precision=2)
    fig, ax = plt.subplots(figsize=(10, len(describeStyle.data) * 0.5))
    ax.axis('off')
    table = ax.table(cellText=describeStyle.data.values, colLabels=describeStyle.data.columns,
                     rowLabels=describeStyle.data.index, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(describeStyle.data.columns))))

    output_path = PathManager.get_report_data_overview_path(report_resources_path, idx)
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()




def plot_feature_distribution(feature: pd.Series, report_resources_path: Path, idx: int) -> None:
    print(feature.name, feature.dtype)
    if feature.dtype == 'object':
        try:
            feature = pd.to_numeric(feature, errors='raise')
            print(f'conversion to numeric succesfuly {feature.name}')
            print(feature)
        except ValueError:
            pass

    plt.figure(figsize=(10, 6))

    if feature.dtype == 'bool' or feature.dtype == 'object':

        if feature.nunique() > 60:
            return

        sns.countplot(x=feature, stat='percent', width=0.3, order=feature.value_counts().index)
        plt.xticks(rotation=90)
        plt.title(f'Distribution of {feature.name}')
    else:
        sns.histplot(feature, bins=40, kde=False, stat='density', alpha=0.7)
        plt.title(f'Distribution of {feature.name}')
        plt.xlabel(str(feature.name))
        plt.ylabel('Percent of values')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

        if pd.api.types.is_numeric_dtype(feature):
            plt.locator_params(axis='x', nbins=20)

        plt.xticks(rotation=45, ha='right')

    output_path = get_report_feature_distribution_path(report_resources_path, str(feature.name), idx)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def get_report_feature_distribution_path(report_resources_path: Path, feature_name: str, idx: int) -> Path:
    return report_resources_path / f"{idx:02d}_feature_distribution_{feature_name}.jpg"


def make_error_overview(actual: pd.Series, predicted: pd.Series, report_resources_path: Path, idx : int) -> None:
    mae = mean_absolute_error(actual, predicted)
    maeP = mae / actual.mean() * 100
    mse = mean_squared_error(actual, predicted)
    mseP = mse / actual.mean() * 100
    mape = mean_absolute_percentage_error(actual, predicted)
    report = pd.DataFrame(
        columns=['Mean Absolute Error', 'Mean Absolute Error as % of average target', 'Mean Squared Error',
                 'Root Mean Squared Error', 'Mean absolute percentage error'], index=[0])
    report['Mean Absolute Error'] = round(mae, 2)
    report['Mean Squared Error'] = round(mse, 2)
    report['Mean Absolute Error as % of average target'] = round(maeP, 2)
    report['Root Mean Squared Error'] = round(np.sqrt(mse), 2)
    report['Mean absolute percentage error'] = round(mape, 2)
    reportStyle = report.T.style.format(precision=2)
    fig, ax = plt.subplots(figsize=(10, len(reportStyle.data) * 0.5))
    ax.axis('off')
    table = ax.table(cellText=reportStyle.data.values, colLabels=reportStyle.data.columns,
                     rowLabels=reportStyle.data.index, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(reportStyle.data.columns))))

    output_path = PathManager.get_report_error_overview_path(report_resources_path, idx)
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()


def plot_hist_error_percentage(error: np.array, report_resources_path: Path, idx : int) -> None:
    delta = max(abs(error.quantile(0.1)), error.quantile(0.9))
    hist_range = (-delta, delta)
    plt.hist(error, range=hist_range, bins=40, weights=np.ones(len(error)) / len(error))
    plt.xlabel('Error percentage')
    plt.ylabel('Percent of errors')

    output_path = PathManager.get_report_error_percentage_distribution_path(report_resources_path, idx)
    plt.savefig(output_path)
    plt.close()


def make_error_quantiles(real: pd.Series, predicted: pd.Series, report_resources_path: Path,
                         quantile_ranges: list, idx : int) -> None:
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
    table = ax.table(cellText=quantile_df_style.data.values, colLabels=quantile_df_style.data.columns,
                     rowLabels=quantile_df_style.data.index, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(quantile_df_style.data.columns))))

    output_path = PathManager.get_report_error_quantiles_path(report_resources_path, idx)
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()


def partial_dependence_analysis(model: xgboost.Booster, data: pd.DataFrame, features_model: list,
                                grid_resolution: int = 100) -> tuple:
    importance_dict = {}
    pdp_dict = {}
    for feature in features_model:

        if data[feature].nunique() > 300:
            continue

        if feature in FEATURES_TO_SKIP_PDP:
            continue

        if isinstance(data[feature].dtype, CategoricalDtype) or data[feature].dtype == 'object':
            feature_range = data[feature].dropna().unique()
        else:
            feature_range = np.linspace(data[feature].min(), data[feature].max(), grid_resolution)

        partial_dependence_values = []

        for value in feature_range:
            data_copy = data.copy()

            if isinstance(data[feature].dtype, CategoricalDtype) or data[feature].dtype == 'object':
                data_copy[feature] = pd.Categorical([value] * len(data_copy), categories=feature_range)
            else:
                data_copy[feature] = value

            predictions = predict(model, data_copy[features_model])
            partial_dependence_values.append(np.mean(predictions))

        importance_dict[feature] = np.std(partial_dependence_values)
        pdp_dict[feature] = (feature_range, partial_dependence_values)

    return pdp_dict, importance_dict


def partial_dependence_analysis_new(model: xgboost.Booster, data: pd.DataFrame, features_model: list,
                                    grid_resolution: int = 10) -> tuple:
    """
    Compute partial dependence plots (PDP) and a feature importance proxy (std of PDP curve)
    for each feature in features_model (unless skipped).

    For numeric features, the predictions over the grid are computed in a vectorized fashion
    (one batched prediction call) to speed up the computation.

    Parameters:
      - model: an xgboost.Booster object.
      - data: a pandas DataFrame containing the data.
      - features_model: list of feature names used for prediction.
      - grid_resolution: number of grid points for continuous features.

    Returns:
      A tuple (pdp_dict, importance_dict) where:
        - pdp_dict maps each feature to a tuple (grid_values, partial_dependence_values)
        - importance_dict maps each feature to the standard deviation of its partial dependence values.
    """
    pdp_dict = {}
    importance_dict = {}

    def batch_predict(model, X):
        dmatrix = xgboost.DMatrix(X, enable_categorical=True)
        return model.predict(dmatrix)

    for feature in features_model:
        if feature in FEATURES_TO_SKIP_PDP:
            continue

        logging.info(f"Doing partial dependence for {feature} dtype is {data[feature].dtype}")

        if data[feature].dtype == 'bool':
            feature_values = data[feature].dropna().unique()
            pdp_values = []
            for val in feature_values:
                data_mod = data.copy()
                data_mod[feature] = val
                preds = batch_predict(model, data_mod[features_model])
                pdp_values.append(np.mean(preds))
            grid = feature_values
        elif isinstance(data[feature].dtype, CategoricalDtype):
            feature_values = data[feature].dropna().unique()
            pdp_values = []
            for val in feature_values:
                data_mod = data.copy()
                data_mod[feature] = val
                data_mod[feature] = pd.Categorical(data_mod[feature], categories=data[feature].cat.categories)
                preds = batch_predict(model, data_mod[features_model])
                pdp_values.append(np.mean(preds))
            grid = feature_values
        else:
            print(data[feature].min())
            print(data[feature].max())
            try:
                grid = np.linspace(data[feature].dropna().min(), data[feature].dropna().max(), grid_resolution).astype(int)
            except Exception as e:
                continue
            X = data[features_model].copy()
            n_rows = X.shape[0]
            # Replicate the data for each grid value
            X_rep = pd.concat([X] * len(grid), ignore_index=True)

            # For each block of rows, replace the column with the grid value.
            for i, val in enumerate(grid):
                start = i * n_rows
                end = (i + 1) * n_rows
                X_rep.loc[start:end, feature] = val

            # Batched prediction call
            preds_all = batch_predict(model, X_rep)
            preds_all = np.array(preds_all).reshape(len(grid), n_rows)
            # Average over rows for each grid point
            pdp_values = preds_all.mean(axis=1)

        # Save the PDP and compute the importance measure (std. deviation of the PDP curve)
        pdp_dict[feature] = (grid, pdp_values)
        importance_dict[feature] = np.std(pdp_values)

    return pdp_dict, importance_dict
def plot_pdp_importance(features: list, importances, title="Feature Importance",
                        xlabel="Partial Dependence Feature Importance", ylabel="Features", values_format="{:.2f}",
                        save_path: Path = None, **kwargs):
    sorted_indices = np.argsort(importances)[::-1]
    features = [features[i] for i in sorted_indices]
    importances = [importances[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.barplot(x=importances, y=features, hue = features, palette='viridis', ax=ax, legend=False, **kwargs)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    return ax


def plot_feature_importance_individual(importance_dict: dict, kind: str, report_resources_path: Path, idx : int) -> None:
    importance = pd.Series(importance_dict, name='feature_with_dummies').reset_index()
    importance['feature'] = importance['index'].apply(lambda x: x.split('__')[0])
    importance_summary = importance.groupby('feature')['feature_with_dummies'].sum()
    importance_summary = importance_summary.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.barplot(
        x=importance_summary.values,
        y=importance_summary.index,
        hue = importance_summary.index,
        palette='viridis',
        ax=ax
    )

    # Customize the plot
    ax.set_title(f"Feature Importance ({kind})", fontsize=14)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)

    # Save the plot
    plt.tight_layout()
    output_path = PathManager.get_report_feature_importance_path(report_resources_path, kind, idx)
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_feature_importance(model: xgboost.Booster, importances: dict, report_resources_path: Path, idx : int):
    export_path = report_resources_path

    for kind in ['weight', 'gain', 'cover']:
        plot_feature_importance_individual(model.get_score(importance_type=kind), kind, export_path, idx)

    if importances is not None:
        save_path = PathManager.get_report_feature_importance_path(report_resources_path, 'pdp', idx)
        plot_pdp_importance(list(importances.keys()), list(importances.values()), save_path = save_path)


def plot_pdps(features: list, pdp_dict: dict, report_resources_path: Path, idx: int):
    for feature in features:
        if feature in FEATURES_TO_SKIP_PDP:
            continue
        try:
            feature_range, pdp_values = pdp_dict[feature]
        except Exception as e:
            continue
        plt.figure(figsize=(8, 6))
        sns.lineplot(x=feature_range, y=pdp_values, label=feature)

        plt.xlabel('Feature Values')
        plt.xticks(rotation = 90)
        plt.ylabel('Partial Dependence')
        plt.title(f'Partial Dependence Plots for {feature}')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        output_path = PathManager.get_report_partial_dependence_plot_path(report_resources_path, feature, idx)
        plt.savefig(output_path)
        plt.close()

def getQauntSplit(step=50):
    return [i / 1000 for i in range(0, 1001, step)]


def plot_real_vs_predicted_quantiles(real: pd.Series, predicted: pd.Series, report_resources_path: Path, idx) -> None:
    step = (100000 // len(real))
    quant = getQauntSplit(step)
    qr = np.quantile(real, quant)
    qp = np.quantile(predicted, quant)
    plt.scatter(qr, qp, alpha=0.8, s=40)
    x = np.linspace(qr.min(), qr.max())
    plt.plot(x, x, c='r')
    plt.xlabel('Real quantiles for training data')
    plt.ylabel('Predicted quantiles for training data')
    output_path = PathManager.get_report_real_vs_predicted_quantiles_path(report_resources_path, idx)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def get_mean_values(data, feature, target_variable, is_numeric : bool) -> pd.DataFrame:
    mean_values = data.groupby('quantiles').agg(
        **{f'mean_{target_variable}': (target_variable, 'mean')},
        mean_predicted=('predicted', 'mean'),
        mean_error=('error', 'mean'),
        **{f'count_{feature}': (feature, 'count')},
        **{f'min_{feature}': (feature, 'min')},
        **{f'max_{feature}': (feature, 'max')}
    ).dropna(subset=[f'mean_{target_variable}'])


    if is_numeric:
        mean_values[f'{feature}_range'] = mean_values.apply(
            lambda row: f'({row[f"min_{feature}"]},{row[f"max_{feature}"]}]', axis=1
        )
    else:
        mean_values[f'{feature}_range'] = mean_values[f'min_{feature}']

    mean_values = mean_values.drop([f'min_{feature}', f'max_{feature}'], axis=1)

    return mean_values


def configure_mean_vs_predicted_axes(axes, feature, ticks, target_variable, is_numeric : bool = False):
    axes[0].set_xticks(ticks)
    axes[0].set_xticklabels(ticks, rotation=90)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Mean Real vs Predicted Values for Different Feature Ranges of {feature}')
    axes[0].set_xlabel('')
    axes[0].set_ylabel(f'Mean {target_variable}')
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel('Mean Absolute Percentage Error')
    axes[1].set_xticks(range(len(ticks)))
    axes[1].set_xticklabels(ticks, rotation=90, ha='right')
    axes[2].set_title(f'Distribution of {feature}')
    axes[2].set_xlabel(feature)
    axes[2].set_ylabel('Percentage of values')
    if not is_numeric:
        axes[2].set_xticks(range(len(ticks)))
        axes[2].set_xticklabels(ticks, rotation=90)
    plt.tight_layout()
    return axes

def plot_real_vs_predicted_by_feature_numeric(data: pd.DataFrame, feature: str,
                                                target_variable: str, num_quantiles : int = 20) -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]:


    if data[feature].nunique() == 1:
        return None, None, None

    print(feature)
    data['quantiles'] = pd.qcut(data[feature], num_quantiles, labels=False, duplicates='drop')
    quantile_values = data.groupby('quantiles')[feature].mean().dropna().values.round(decimals=2)
    mean_values = get_mean_values(data, feature, target_variable, is_numeric=True)
    if len(quantile_values) == 0:
        return None, None, None
    ticks = np.round(np.linspace(quantile_values.min(), quantile_values.max(), len(quantile_values)), 2)

    norm = plt.Normalize(vmin=mean_values['mean_error'].min(), vmax=mean_values['mean_error'].max())
    color_map = plt.get_cmap('magma_r')
    colors = color_map(norm(mean_values['mean_error'])).tolist()

    fig, axes = plt.subplots(nrows=3, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1, 1]})

    sns.lineplot(x=quantile_values, y=mean_values[f'mean_{target_variable}'], label='Real Mean', marker='o', ax=axes[0])
    sns.lineplot(x=quantile_values, y=mean_values['mean_predicted'], label='Predicted Mean', marker='x', ax=axes[0])
    axes[0].set_ylim(0)

    sns.barplot(x=quantile_values, y=mean_values['mean_error'], palette=colors, ax=axes[1], hue=quantile_values, dodge=False,
                legend=False)
    axes[1].set_ylim(0)

    sns.histplot(data=data, x=feature, bins = len(quantile_values), stat='percent', kde=False, color='blue', alpha=0.3, ax=axes[2])

    axes = configure_mean_vs_predicted_axes(axes, feature, ticks, target_variable, True)

    plt.tight_layout()
    return fig, axes, mean_values



def plot_real_vs_predicted_by_feature_other(data: pd.DataFrame, feature: str,
                                            target_variable: str) -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    data['quantiles'] = data[feature]
    quantile_values = sorted(data[feature].unique())
    mean_values = get_mean_values(data, feature, target_variable, is_numeric = False)

    norm = plt.Normalize(vmin=mean_values['mean_error'].min(), vmax=mean_values['mean_error'].max())
    color_map = plt.get_cmap('magma_r')
    colors = color_map(norm(mean_values['mean_error'])).tolist()

    fig, axes = plt.subplots(nrows=3, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1, 1]})

    sns.lineplot(x=quantile_values, y=mean_values[f'mean_{target_variable}'], label='Real Mean', marker='o', ax=axes[0])
    sns.lineplot(x=quantile_values, y=mean_values['mean_predicted'], label='Predicted Mean', marker='x', ax=axes[0])
    axes[0].set_ylim(0)

    sns.barplot(x=quantile_values, y=mean_values['mean_error'], palette=colors, ax=axes[1], hue=quantile_values, dodge=False,
                legend=False)
    axes[1].set_ylim(0)

    sns.countplot(data=data, x=feature, stat='percent', color='blue', alpha=0.3, order = quantile_values,
                 ax=axes[2])

    axes = configure_mean_vs_predicted_axes(axes, feature, quantile_values, target_variable)
    return fig, axes, mean_values




def plot_real_vs_predicted_by_feature(data_p: pd.DataFrame, predictions : pd.Series, feature: str,
                                      target_variable: str, report_resources_path: Path, idx : int):

    if feature != target_variable:

        if data_p[feature].dtype == 'object':
            return

        data = data_p[[feature, target_variable]].copy()
    else:
        data = data_p[[feature]].copy()


    data['predicted'] = predictions
    data['error'] = abs(data[target_variable].values - data['predicted'].values) / data[target_variable].values * 100

    data = data.dropna(subset=[feature, target_variable])
    fig, axes, mean_values = None, None, None

    if pd.api.types.is_numeric_dtype(data[feature]):
        fig, axes, mean_values = plot_real_vs_predicted_by_feature_numeric(data, feature, target_variable)

    if isinstance(data[feature].dtype, CategoricalDtype):
        return

    if pd.api.types.is_string_dtype(data[feature]):
        fig, axes, mean_values = plot_real_vs_predicted_by_feature_other(data, feature, target_variable)

    if fig is None:
        return

    output_path = PathManager.get_report_real_vs_predicted_by_feature_path(report_resources_path, feature, idx)
    fig.savefig(output_path)
    plt.close(fig)


    top_errors = mean_values.nlargest(10, 'mean_error')
    top_errors['error_contribution'] = top_errors['mean_error'] * top_errors[f'count_{feature}'] / len(data)
    top_errors = top_errors[[f'{feature}_range', f'count_{feature}', f'mean_{target_variable}', f'mean_predicted', 'mean_error', 'error_contribution']]
    top_errors = top_errors.round(decimals = 2)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    ax.set_title(f'Segments with top error for {feature}')
    table = ax.table(cellText=top_errors.values, colLabels=top_errors.columns, cellLoc='center', loc='center')

    output_path = PathManager.get_report_top_10_biggest_errors_by_feature_path(report_resources_path, feature, idx)
    plt.savefig(output_path, dpi = 150)
    plt.close()


def get_k_largest_errors(data: pd.DataFrame, features : List[str], out_of_sample_predictions: pd.Series, errors: pd.Series, k: int,
                         report_resources_path: Path, idx: int):
    dc = data[features].copy()
    dc['error'] = errors
    dc['abs_error'] = abs(errors)
    dc['model_prediction'] = out_of_sample_predictions

    k_largest_errors = dc.sort_values(by='abs_error', ascending=False).iloc[:k]

    k_largest_errors_transposed = k_largest_errors.transpose()

    k_largest_errors_style = k_largest_errors_transposed.style.format(precision=2)

    fig, ax = plt.subplots(figsize=(10, len(k_largest_errors_style.data) * 0.4))  # Adjust 0.4 as needed
    ax.axis('off')

    table = ax.table(cellText=k_largest_errors_style.data.values,
                     colLabels=k_largest_errors_style.data.columns,
                     rowLabels=k_largest_errors_style.data.index,
                     cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(k_largest_errors_style.data.columns))))

    output_path = PathManager.get_report_k_largest_errors_path(report_resources_path, k, idx)
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()

def plot_learning_curve(trials : Trials, report_resources_path: Path, idx : int) -> None:
    losses = [x['result']['loss'] for x in trials.trials]

    best_losses = [min(losses[:i + 1]) for i in range(len(losses))]

    # Plot the errors
    plt.figure(figsize=(10, 6))
    plt.plot(best_losses, label="Best Error", color='blue')
    plt.plot(losses, label="Current Error", color='orange', alpha=0.5)
    plt.xlabel('Trials')
    plt.ylabel('Error')
    plt.title('Training Error Over Time')
    plt.legend()
    plt.grid(True)

    output_path = PathManager.get_report_learning_curve_path(report_resources_path, idx)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def compute_shap_values(model, data: pd.DataFrame, features_model: list, target_variable : str):
    explainer = shap.Explainer(model)
    d_matrix = make_d_matrix(data[features_model], data[target_variable], is_classification=False)
    shap_values = explainer(d_matrix)
    return shap_values

def plot_shap_summary(shap_values, data, features_model : list, report_resources_path, idx: int):
    plt.figure()
    shap.summary_plot(shap_values, data[features_model])
    output_path = report_resources_path / f"{idx:02d}_shap_summary_plot.jpg"
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()

def plot_shap_dependence(shap_values, data, feature, report_resources_path, idx: int):
    plt.figure()
    shap.dependence_plot(feature, shap_values.values, data)
    output_path = report_resources_path / f"{idx:02d}_shap_dependence_plot_{feature}.jpg"
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()


def plot_shap_waterfall(shap_values, report_resources_path: Path, idx : int, k : int = 3):
    random_indices = random.sample(range(shap_values.values.shape[0]), k)

    for i, random_idx in enumerate(random_indices):
        plt.figure()
        shap.plots.waterfall(shap_values[random_idx])

        output_path = report_resources_path / f"{idx:02d}_shap_waterfall_{i}.jpg"
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        plt.close()



def make_pdf(report_resources_path: Path, report_path: Path):
    if not report_resources_path.exists():
        raise ValueError(f"Image directory does not exist: {report_resources_path}")

    image_paths = sorted([path for path in report_resources_path.glob("*.jpg")])
    if not image_paths:
        raise ValueError(f"No JPG images found in directory: {report_resources_path}")

    images = [Image.open(str(path)) for path in image_paths]
    with open(report_path, "wb") as pdf_file:
        images[0].save(pdf_file, "PDF", resolution=100.0, save_all=True, append_images=images[1:])


def generate_report_cover_image(report_resources_path: Path, insurance_name: str, table_of_contents: list):
    image_width, image_height = 800, 600
    background_color = (255, 255, 255)
    text_color = (0, 0, 0)
    font_path = "arial.ttf"
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

    output_path = report_resources_path / "00_report_cover_image.jpg"
    image.save(output_path)
    return output_path


def generate_report_util(
        model: xgboost.Booster,
        data: pd.DataFrame,
        features_info: list,
        features_model : list,
        target_variable: str,
        out_of_sample_predictions: pd.Series,
        trials: Trials,
        report_path: Path,
        report_resources_path: Path,
        use_pdp: bool = False,
        use_shap: bool = False,

):
    data = reconstruct_categorical_variables(data)
    features_all = features_info + features_model
    all_info  = features_all + [f'{target_variable}_orig',
                                f'corrected_{target_variable}_orig',
                                f'corrected_corrected_{target_variable}',
                                'diff', target_variable]

    if target_variable.startswith('log_'):
        data[target_variable] = np.exp(data[target_variable])

    real = data[target_variable]
    errors = real - out_of_sample_predictions
    errors_percentage = errors / data[target_variable] * 100

    logging.info("Preparing directories for the report.")
    prepare_dir(report_path)
    prepare_dir(report_resources_path)


    table_of_contents = DEFAULT_REPORT_TABLE_OF_CONTENTS

    logging.info("Generating report cover image.")
    generate_report_cover_image(report_resources_path, target_variable, table_of_contents)

    pdp_dict = None
    importance = None
    shap_values = None
    if use_pdp:
        pdp_dict, importance = partial_dependence_analysis_new(model, data, features_model)
    if use_shap:
        shap_values = compute_shap_values(model, data, features_model, target_variable)
        shap_values.feature_names = features_model

    section_functions = {
        "Data Overview": lambda idx: make_data_overview(data, all_info, report_resources_path, idx),
        "Error Overview": lambda idx: make_error_overview(real, out_of_sample_predictions, report_resources_path, idx),
        "Error Quantiles": lambda idx: make_error_quantiles(real, out_of_sample_predictions, report_resources_path,
                                                            None, idx),
        "Error Percentage Distribution": lambda idx: plot_hist_error_percentage(errors_percentage,
                                                                                report_resources_path, idx),
        "Top k Largest Errors": lambda idx: get_k_largest_errors(data, all_info, out_of_sample_predictions, errors, 10
                                                                 , report_resources_path, idx),
        "Feature Importance": lambda idx: plot_feature_importance(model, importance,
                                                                  report_resources_path, idx),
        "Feature Distribution": lambda idx: [plot_feature_distribution(data[feature], report_resources_path, idx) for
                                             feature in features_all],
        "Partial Dependence Plots": lambda idx: plot_pdps(features_model, pdp_dict, report_resources_path,
                                                          idx) if use_pdp else None,
        "Real vs Predicted Quantiles": lambda idx: plot_real_vs_predicted_quantiles(real, out_of_sample_predictions,
                                                                                    report_resources_path, idx),
        "Real vs Predicted Quantiles by Feature": lambda idx: [
            plot_real_vs_predicted_by_feature(data, out_of_sample_predictions, feature, target_variable,
                                              report_resources_path, idx) for feature in features_all + [target_variable]],
        "Learning Curve": lambda idx: plot_learning_curve(trials, report_resources_path, idx),
        "Shapley Summary": lambda idx: plot_shap_summary(shap_values, data, features_model, report_resources_path, idx)
                                        if use_shap else None,
        "Shapley Waterfall": lambda idx : plot_shap_waterfall(shap_values, report_resources_path, idx, k = 3) if use_shap else None,
    }

    for section_id, section in enumerate(table_of_contents, start=1):
        logging.info(f"Generating section: {section}")
        if section in section_functions:
            section_functions[section](section_id)

    logging.info("Generating PDF report.")
    make_pdf(report_resources_path, report_path / "report.pdf")
    logging.info("Report generation completed.")

@click.command("generate_report")
@click.option("--service", required=True, type=click.STRING)
@click.option("--data_name", required=True, type=click.STRING)
@click.option("--target_variable", required=True, type=click.STRING)
@click.option("--model_config_name", required=True, type=click.STRING)
@click.option('--use_pdp', default=False, is_flag=True, help='Useful when testing, since pdp plots take a long time...')
@click.option('--use_shap', default=False, is_flag=True, help='Useful when testing, since shap plots take a long time...')
def generate_report(service : str, data_name: str, target_variable: str, model_config_name : str, use_pdp: bool, use_shap : bool):

    path_manager = PathManager(service)
    load_manager = LoadManager(path_manager)

    model_name = path_manager.get_model_name(data_name, target_variable, model_config_name)
    report_path = path_manager.get_report_path(service, data_name, target_variable)
    report_resources_path = path_manager.get_report_resource_path(report_path)

    data, features_info, features_on_top, features_model = load_manager.load_data(data_name, target_variable)

    model_config = load_manager.load_model_config(data_name, model_config_name)
    for col in model_config['features_to_exclude']:
        features_model.remove(col)
    model = load_manager.load_model(data_name, model_name)
    out_of_sample_predictions = load_manager.load_out_of_sample_predictions(data_name, model_name, target_variable)
    model_trials = load_manager.load_hyperopt_trials(data_name, model_name)

    print(report_path)
    print(report_resources_path)
    generate_report_util(model, data, features_info, features_model, target_variable, out_of_sample_predictions, model_trials, report_path,
                         report_resources_path, use_pdp, use_shap)


@click.group()
def cli():
    pass


cli.add_command(generate_report)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    cli()