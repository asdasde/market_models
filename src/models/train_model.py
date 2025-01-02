import os
import sys
from copyreg import pickle

import click
import time
from pathlib import Path
from typing import Optional, List, Union, final

import hyperopt.tpe
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from mlxtend.classifier import OneRClassifier
from hyperopt import STATUS_OK, SparkTrials, Trials, fmin, tpe, rand
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
    accuracy_score,
    log_loss,
    f1_score
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models.make_report as make_report
import models.error_overview as error_overview

from utilities.model_utils import *
from utilities.model_constants import *
from utilities.export_utils import *



def model_train(train_data: pd.DataFrame,
                test_data: pd.DataFrame,
                features: list,
                target_variable: str,
                is_classification: bool,
                param: dict = None) -> Tuple[xgboost.Booster, np.ndarray, Dict]:

    param_ = None
    if param is None:
        if is_classification:
            param_ = DEFAULT_PARAMS_CLASSIFICATION.copy()
        else:
            param_ = DEFAULT_PARAMS_REGRESSION.copy()
    else:
        param_ = param.copy()

    dtrain = make_d_matrix(train_data[features], train_data[target_variable], is_classification=is_classification)
    dtest = make_d_matrix(test_data[features], test_data[target_variable], is_classification=is_classification)
    eval_list = [(dtrain, 'train'), (dtest, 'eval')]
    evals_result = {}

    num_rounds = param_['n_estimators']
    del param_['n_estimators']

    param_['eval_metric'] = ['logloss', 'aucpr'] if is_classification else ['mae', 'mape', 'rmse']

    bst =  xgboost.train(
        param_,
        dtrain,
        num_boost_round=num_rounds,
        early_stopping_rounds=300,
        evals=eval_list,
        verbose_eval=False,
        evals_result=evals_result
    )

    return bst, bst.predict(dtest), evals_result



def create_stratified_cv_splits(data: pd.DataFrame, target_variable: str, k: int = 3,
                                num_groups=1000, seed: int = 42) -> list:
    if num_groups:
        grp = pd.qcut(data[target_variable], min(len(data), num_groups), labels=False, duplicates='drop')
    else:
        grp = data[target_variable]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    cv_splits = [(train_idx, test_idx) for train_idx, test_idx in skf.split(X=data.index, y=grp)]
    return cv_splits


def _print_fold_metrics(fold_num: int, metrics: dict, is_classification: bool, mean_price: float,
                        target_variable: str) -> None:
    print(f"\nSummary for fold {fold_num}")
    if is_classification:
        print(f"Log Loss: {metrics['log_loss']:.3f}")
        print(f"F1 score: {metrics['f1_score']:.3f}")
    else:
        print(f"MAE: {metrics['mae']:.3f} ({metrics['mae'] / mean_price * 100:.3f}% of mean {target_variable})")
        print(f"RMSE: {metrics['rmse']:.3f}")
        print(f"MAPE: {metrics['mape'] * 100:.3f}%")
    print("-" * 60)

def _print_final_regression_metrics(k, mae, mae_std, mape, mape_std, rmse, rmse_std, target_variable, true_mean):
    print(f"\nMean MAE over {k} fold CV: {mae:.2f} ± {mae_std / mae * 100:.3f}% "
          f"({mae / true_mean * 100:.3f}% ± {mae_std / true_mean * 100:.3f}% of mean {target_variable})")
    print(f"Mean RMSE over {k} fold CV: {rmse:.2f} ± {rmse_std / rmse * 100:.2f}%")
    print(f"Mean MAPE over {k} fold CV: {mape * 100:.2f}% ± {mape_std / mape * 100:.3f}%")


def _calculate_classification_metrics(metrics: dict, k: int, debug: bool) -> Tuple[float, float]:
    mean_log_loss = np.mean(metrics['log_loss'])
    std_log_loss = np.std(metrics['log_loss'])
    mean_f1_score = np.mean(metrics['f1_score'])
    std_f1_score = np.std(metrics['f1_score'])

    if debug:
        print(f"\nMean Log Loss over {k} fold CV: {mean_log_loss:.3f} ± {std_log_loss:.3f}")
        print(f"Mean F1-score over {k} fold CV: {mean_f1_score:.3f} ± {std_f1_score:.3f}")

    return mean_log_loss, mean_f1_score


def _calculate_regression_metrics(metrics: dict, k: int, mean_price: float, target_variable: str, debug: bool) -> Tuple[
    float, float, float]:

    mae = np.mean(metrics['mae'])
    mae_std = np.std(metrics['mae'])
    rmse = np.mean(np.sqrt(metrics['mse']))
    rmse_std = np.std(np.sqrt(metrics['mse']))
    mape = np.mean(metrics['mape'])
    mape_std = np.std(metrics['mape'])

    if debug:
        _print_final_regression_metrics(k, mae, mae_std, mape, mape_std, rmse, rmse_std, target_variable, mean_price)

    return mae, rmse, mape



def kFoldCrossValidation(
    k: int,
    data: pd.DataFrame,
    features: List[str],
    target_variable: str,
    is_classification: bool,
    param: Optional[dict] = None,
    debug: bool = False
) -> Tuple[Dict[str, float], pd.Series]:

    metrics = {
        'classification': {'log_loss': [], 'f1_score': []},
        'regression': {'mae': [], 'rmse': [], 'mape': []}
    }

    predictions = pd.Series(index=data.index, dtype=float)
    is_log_transformed = target_variable.startswith('log_')
    true_mean = np.exp(data[target_variable]).mean() if is_log_transformed else data[target_variable].mean()
    kf = create_stratified_cv_splits(data, target_variable, k=k, num_groups=10)

    for fold_num, (train_ix, test_ix) in enumerate(kf, 1):
        train_data = data.iloc[train_ix]
        test_data = data.iloc[test_ix]

        model, test_predictions, eval_results = model_train(train_data, test_data, features, target_variable, is_classification, param)

        predictions.iloc[test_ix] = test_predictions

        if is_classification:
            fold_metrics = {
                'log_loss': eval_results['eval']['log_loss'][-1],
                'f1_score': eval_results['eval']['f1_score'][-1]
            }
            for metric, value in fold_metrics.items():
                metrics['classification'][metric].append(value)
        else:
            fold_metrics = {
                'mae': eval_results['eval']['mae'][-1],
                'rmse': eval_results['eval']['rmse'][-1],
                'mape': eval_results['eval']['mape'][-1]
            }
            for metric, value in fold_metrics.items():
                metrics['regression'][metric].append(value)

        if debug:
            _print_fold_metrics(fold_num, fold_metrics, is_classification, true_mean, target_variable)

    if is_classification:
        mean_log_loss = np.mean(metrics['classification']['log_loss'])
        mean_f1_score = np.mean(metrics['classification']['f1_score'])
        std_log_loss = np.std(metrics['classification']['log_loss'])
        std_f1_score = np.std(metrics['classification']['f1_score'])

        if debug:
            print(f"\nMean Log Loss over {k} fold CV: {mean_log_loss:.3f} ± {std_log_loss:.3f}")
            print(f"Mean F1-score over {k} fold CV: {mean_f1_score:.3f} ± {std_f1_score:.3f}")

        final_metrics = {
            'log_loss': mean_log_loss,
            'f1_score': mean_f1_score,
            'log_loss_std': std_log_loss,
            'f1_score_std': std_f1_score
        }

    else:
        mae = np.mean(metrics['regression']['mae'])
        mae_std = np.std(metrics['regression']['mae'])
        rmse =  np.sqrt(np.mean(np.square(metrics['regression']['rmse'])))
        rmse_std = np.std(metrics['regression']['rmse'])
        mape = np.mean(metrics['regression']['mape'])
        mape_std = np.std(metrics['regression']['mape'])

        if debug:
            _print_final_regression_metrics(k, mae, mae_std, mape, mape_std, rmse, rmse_std, target_variable, true_mean)

        final_metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'mae_std': mae_std,
            'rmse_std': rmse_std,
            'mape_std': mape_std,
            'mae_percent': mae / true_mean * 100
        }

    return final_metrics, predictions


def train_model_util(data: pd.DataFrame, features: list, target_variable: str, is_classification: bool,
                     previous_trials : Trials = None):

    train_data, validation_data = train_test_split(data, test_size = 0.1, random_state = 42)

    def objective(space):
        params = space.copy()
        metrics, predictions = kFoldCrossValidation(
            k=4,
            data=train_data,
            features=features,
            target_variable=target_variable,
            is_classification=is_classification,
            param=params,
            debug=False
        )
        return {"loss": metrics['mae'], 'status': STATUS_OK}

    logging.info("Starting hyper-parameter tuning...")
    if is_classification:
        space = SPACE_CLASSIFICATION
    else:
        space = SPACE_REGRESSION

    if previous_trials is not None:
        trials = previous_trials
        trials.refresh()
        evals = len(previous_trials) + 2
    else:
        trials = Trials()
        evals = MAX_EVALS

    best_hyperparams = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=evals,
        trials=trials,
        return_argmin=False
    )

    model, predictions, eval_results = model_train(train_data, train_data, features, target_variable, is_classification, best_hyperparams)
    predictions = predict(model, validation_data[features])
    validation_data['preds_validation'] = predictions
    print(validation_data[[target_variable, 'preds_validation']])
    print(mean_absolute_percentage_error(validation_data[target_variable], predictions))
    print(mean_absolute_error(validation_data[target_variable], predictions))
    logging.info("Finished hyper-parameter tuning...")

    metrics, predictions = kFoldCrossValidation(k=4, data=data, features=features, target_variable=target_variable,
                               is_classification=is_classification, param=best_hyperparams, debug=True)
    data['preds_out_of_sample'] = predictions
    data['preds_in_sample'] = predict(model, data[features])
    print(data[['contractor_personal_id', target_variable, 'preds_out_of_sample', 'preds_in_sample']])
    print(mean_absolute_percentage_error(data[target_variable],  data['preds_in_sample']))
    print(mean_absolute_percentage_error(data[target_variable], predictions))
    print(mean_absolute_error(data[target_variable], data['preds_in_sample']))
    print(mean_absolute_error(data[target_variable], predictions))

    return model, best_hyperparams, predictions, trials, metrics['mae_percent']

def find_best_previous_trials(data_name, target_variable):
    model_name = get_model_name(data_name, target_variable)
    if os.path.exists(get_model_trials_path(model_name)):
        return load_hyperopt_trials(model_name)
    return None


@click.command(name='train_model')
@click.option('--service', required=True, type=click.STRING)
@click.option('--data_name', required=True, type=click.STRING)
@click.option('--target_variable', required=True, type=click.STRING)
def train_model(service, data_name, target_variable):
    model_name = get_model_name(data_name, target_variable)

    data, features_info, features_on_top, features_model = load_data(data_name, target_variable)

    on_top = load_on_top_file(service)
    data = apply_on_top(data, target_variable, target_variable, on_top)
    data = apply_on_top(data, f'corrected_{target_variable}', target_variable, on_top, reverse=True)
    data['diff'] = data[f'{target_variable}_orig'] - data[f'corrected_corrected_{target_variable}']
    print(data[[f'{target_variable}_orig', f'corrected_{target_variable}_orig', f'corrected_corrected_{target_variable}', 'diff']])

    trials = find_best_previous_trials(data_name, target_variable)

    model, hyperparameters, out_of_sample_predictions, trials, percent_mMae \
        = train_model_util(data, features_model, target_variable, is_classification=False, previous_trials = trials)

    export_model(model, hyperparameters, out_of_sample_predictions, trials, model_name)

    report_path = get_report_path(model_name)
    report_resources_path = get_report_resource_path(model_name)
    make_report.generate_report_util(model, data, features_info, features_model, target_variable,
                                     out_of_sample_predictions, trials, report_path,
                                     report_resources_path, use_pdp=False, use_shap=False)
    
    error_overview_path = get_error_overview_path(service)
    error_overview.update_error_overview(round(percent_mMae, 2), data_name, target_variable, error_overview_path)



def get_feature_quartiles(data: pd.DataFrame):
    data_quartiles = data.copy()
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            if data[column].nunique() >= 4:
                data_quartiles[column] = pd.qcut(data[column], q=[0, 0.25, 0.5, 0.75, 1], labels=False,
                                                 duplicates='drop')

    return data_quartiles


def evaluate_baseline_error_model(data: pd.DataFrame, features: list, target_variable: str):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data[features] = get_feature_quartiles(train_data[features])
    test_data[features] = get_feature_quartiles(test_data[features])
    baseline_error_model = OneRClassifier()
    baseline_error_model.fit(train_data[features].values, train_data[target_variable].values)
    baseline_error_model_preds = baseline_error_model.predict(test_data[features].values)

    logging.info(
        f"Basline error models Log loss is {log_loss(test_data[target_variable], baseline_error_model_preds)}.")
    logging.info(
        f"Basline error models Accuracy score is {accuracy_score(test_data[target_variable], baseline_error_model_preds)}.")


@click.command(name='train_presence_model')
@click.option('--data_name', required=True, type=click.STRING)
@click.option('--target_variable', required=True, type=click.STRING)
def train_market_presence_model(data_name, target_variable):
    presence_model_name = get_presence_model_name(data_name, target_variable)
    presence_model_target_variable = f'{target_variable}_presence'

    data, features_info, features_on_top, features_model = load_data(data_name, target_variable, drop_target_na=False)

    data[presence_model_target_variable] = ~data[target_variable].isna()
    presence_model, presence_model_hyperparameters, presence_model_out_of_sample_predictions, presence_model_trials, percent_mMae = (
        train_model_util(data, features_model, presence_model_target_variable,True, None))

    export_model(presence_model, presence_model_hyperparameters, presence_model_out_of_sample_predictions,
                 presence_model_trials, presence_model_name)


@click.command(name='train_error_model')
@click.option('--data_name', required=True, type=click.STRING)
@click.option('--target_variable', required=True, type=click.STRING)
@click.option('--use_pretrained_model', required=False, type=click.BOOL, default=True, show_default=True)
def train_error_model(data_name, target_variable, use_pretrained_model):

    model_name = get_model_name(data_name, target_variable)
    error_model_name = get_error_model_name(data_name, target_variable)
    error_model_target_variable = f'{model_name}_error'

    data, features_info, features_on_top, features_model = load_data(data_name, target_variable)

    model_exists = check_model_existence(model_name)

    if (use_pretrained_model and not model_exists) or not use_pretrained_model:
        train_model(data_name, target_variable)

    predictions = load_out_of_sample_predictions(model_name, target_variable)

    errors = data[target_variable] - predictions
    errors[np.abs(errors) < 1000] = 0
    errors[np.abs(errors) > 1000] = 1
    errors = errors.astype(bool)


    data[error_model_target_variable] = errors

    data = data[features_model + [error_model_target_variable]]

    evaluate_baseline_error_model(data, features_model, error_model_target_variable)

    error_model, error_model_hyperparameters, error_model_out_of_sample_predictions, error_model_trials = train_model_util(data, features_model, error_model_target_variable, True, None)

    export_model(error_model, error_model_hyperparameters, error_model_out_of_sample_predictions, error_model_trials, error_model_name)



@click.group()
def cli():
    pass


cli.add_command(train_error_model)
cli.add_command(train_model)
cli.add_command(train_market_presence_model)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    cli()
