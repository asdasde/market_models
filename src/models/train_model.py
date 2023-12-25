import os
import sys
from pathlib import Path
from typing import Optional

import click
import logging

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    accuracy_score,
    log_loss,
)
from sklearn.model_selection import KFold, train_test_split

from mlxtend.classifier import OneRClassifier

import xgboost

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from dotenv import find_dotenv, load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import models.make_report as make_report

TEST_SIZE = 0.1
RANDOM_STATE = 42

DEFAULT_PARAMS_REGRESSION = {
    'objective': 'reg:squarederror',
    'booster': 'gbtree',
    'eval_metric': 'mae',
    'n_estimators': 100,
    'eta': 0.3,
    'max_depth': 6,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'num_boost_round': 100,
    'early_stopping_rounds': None,
    'seed': 42
}

DEFAULT_PARAMS_CLASSIFICATION = {
    'objective': 'binary:logistic',  # for binary classification
    'booster': 'gbtree',
    'eval_metric': 'logloss',  # use log loss for classification
    'n_estimators': 100,
    'eta': 0.3,
    'max_depth': 6,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'num_boost_round': 100,
    'early_stopping_rounds': None,
    'seed': 42
}

SPACE_REGRESSION = {
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'n_estimators': hp.choice('n_estimators', np.arange(100, 1300, 100, dtype=int)),
    'max_depth': hp.choice('max_depth', np.arange(2, 11, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(1, 11, dtype=int)),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'gamma': hp.uniform('gamma', 0, 0.2),
    'lambda': hp.uniform('lambda', 0, 1),
    'reg_alpha': hp.uniform('reg_alpha', 40, 180),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'seed': 0,
}

SPACE_CLASSIFICATION = {
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'n_estimators': hp.choice('n_estimators', np.arange(100, 1300, 100, dtype=int)),
    'max_depth': hp.choice('max_depth', np.arange(2, 11, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(1, 11, dtype=int)),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'gamma': hp.uniform('gamma', 0, 0.2),
    'lambda': hp.uniform('lambda', 0, 1),
    'reg_alpha': hp.uniform('reg_alpha', 40, 180),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'seed': 0,
}


def makeDMatrix(data_features: pd.DataFrame,
                data_target: pd.DataFrame,
                is_classification: bool) -> xgboost.DMatrix:
    if is_classification:
        return xgboost.DMatrix(data_features, label=data_target, enable_categorical=True)
    else:
        return xgboost.DMatrix(data_features, data_target, enable_categorical=True)


def model_train(train_data: pd.DataFrame,
                test_data: pd.DataFrame,
                features: list,
                target_variable: str,
                is_classification: bool,
                param: dict = None) -> xgboost.Booster:
    param_ = None
    if param is None:
        if is_classification:
            param_ = DEFAULT_PARAMS_CLASSIFICATION.copy()
        else:
            param_ = DEFAULT_PARAMS_REGRESSION.copy()
    else:
        param_ = param.copy()

    dtrain = makeDMatrix(train_data[features], train_data[target_variable], is_classification=is_classification)
    dtest = makeDMatrix(test_data[features], test_data[target_variable], is_classification=is_classification)

    num_rounds = param_['n_estimators']
    del param_['n_estimators']
    param_['max_depth'] = int(param_['max_depth'])
    param_['eval_metric'] = 'logloss' if is_classification else 'mae'

    eval_list = [(dtrain, 'train'), (dtest, 'eval')]

    return xgboost.train(param_, dtrain, num_boost_round=num_rounds, evals=eval_list, verbose_eval=False)


def merge_predictions(model: xgboost.Booster,
                      test_data: pd.DataFrame,
                      features: list,
                      target_variable: str,
                      is_classification: bool) -> pd.DataFrame:
    pred_target_variable = f'predicted_{target_variable}'
    output = test_data.copy()

    dtest = makeDMatrix(test_data[features], test_data[target_variable], is_classification=is_classification)
    output[pred_target_variable] = model.predict(dtest)

    output['error'] = output[target_variable] - output[pred_target_variable]
    output['percentageError'] = output['error'] / output[target_variable] * 100
    return output


def kFoldCrossValidation(k: int,
                         data: pd.DataFrame,
                         features: list,
                         target_variable: str,
                         is_classification: bool,
                         param: Optional[dict],
                         debug: bool) -> tuple:
    maes = []
    mses = []
    mapes = []
    log_losses = []
    accuracy_scores = []

    out_of_sample_predictions = data[target_variable].copy(deep=True)

    kf = KFold(n_splits=k)
    fold_num = 0
    for train_ix, test_ix in kf.split(data):
        fold_num += 1

        train_data, test_data = data.iloc[train_ix], data.iloc[test_ix]
        model = model_train(train_data, test_data, features, target_variable, is_classification, param)

        dtest = makeDMatrix(test_data[features], test_data[target_variable], is_classification=is_classification)
        test_preds = model.predict(dtest)
        out_of_sample_predictions.iloc[test_ix] = test_preds

        if is_classification:
            log_loss_value = log_loss(test_data[target_variable].values, test_preds)
            acc_score = accuracy_score(test_data[target_variable], test_preds.astype(bool))
            log_losses.append(log_loss_value)
            accuracy_scores.append(acc_score)
        else:
            mae = mean_absolute_error(test_data[target_variable].values, test_preds)
            mse = mean_squared_error(test_data[target_variable].values, test_preds)
            mape = mean_absolute_percentage_error(test_data[target_variable].values, test_preds)
            maes.append(mae)
            mses.append(mse)
            mapes.append(mape)

        if debug:
            if is_classification:
                print(f"Summary for fold {fold_num}")
                print("Log Loss is {}.".format(round(log_loss_value, 3)))
                print("Accuracy score is {}.".format(round(log_loss_value, 3)))
                print("-------------------------------------------------------------")

            else:
                print(f"Summary for fold {fold_num}")
                print("Mean absolute error is {}, which is {}% of mean {}.".format(round(mae, 3), round(
                    mae / data[target_variable].mean() * 100, 3), target_variable))
                print("Mean square error is {}.".format(round(mse, 3)))
                print("Mean absolute percentage error is {}%.".format(round(mape * 100, 3)))
                print("-------------------------------------------------------------")

    if is_classification:
        mean_log_loss = np.mean(log_losses)
        std_log_loss = np.std(log_losses)
        mean_acc_score = np.mean(accuracy_scores)
        std_acc_score = np.std(accuracy_scores)

        if debug:
            print(
                f"Mean Log Loss over {k} fold Cross-validation is {round(mean_log_loss, 3)} ± {round(std_log_loss, 3)}.")
            print(
                f"Mean Accuracy score over {k} fold Cross-validation is {round(mean_acc_score, 3)} ± {round(std_acc_score, 3)}.")

        return mean_log_loss, mean_acc_score, out_of_sample_predictions

    else:
        mMae, sMae = np.mean(maes), np.std(maes)
        mRMse, sRMse = np.mean(np.sqrt(mses)), np.std(np.sqrt(mses))
        mMape, sMape = np.mean(mapes), np.std(mapes)
        meanPrice = data[target_variable].mean()

        rmMae, rsMae = round(mMae, 2), round(sMae / mMae * 100, 3)
        rmRMse, rsRMse = round(mRMse, 2), round(sRMse / mRMse * 100, 2)
        rmMape, rsMape = round(mMape * 100, 2), round(sMape / mMape, 3)

        if debug:
            print(
                f"Mean MAE over {k} fold Cross-validation is {rmMae} ± {rsMae}%, which is {round(mMae / meanPrice * 100, 3)} ± {round(sMae / meanPrice * 100, 3)}% percent of mean {target_variable}.")
            print(f"Mean RMSE over {k} fold Cross-validation is {rmRMse} ± {rsRMse}%.")
            print(f"Mean MAPE over {k} fold Cross-validation is {rmMape} ± {rsMape}%.")

        return mMae, mRMse, mMape, out_of_sample_predictions


def train_model_util(data: pd.DataFrame, features: list, target_variable: str, is_classification: bool):
    logging.info("Removed columns that are not used and nan values on target variable...")
    trials = Trials()

    def objective(space):
        params = space.copy()
        loss = kFoldCrossValidation(k=3, data=data, features=features, target_variable=target_variable,
                                    is_classification=is_classification, param=params, debug=False)
        print(f'Loss is {loss[0]}')
        return {"loss": loss[0], 'status': STATUS_OK}

    logging.info("Starting hyper-parameter tuning...")

    if is_classification:
        space = SPACE_CLASSIFICATION
    else:
        space = SPACE_REGRESSION

    best_hyperparams = fmin(fn=objective,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=utils.MAX_EVALS,
                            trials=trials,
                            return_argmin=False)

    model = model_train(data, data, features, target_variable, is_classification, best_hyperparams)
    logging.info("Finished hyper-parameter tuning...")

    res = kFoldCrossValidation(k=3, data=data, features=features, target_variable=target_variable,
                               is_classification=is_classification, param=best_hyperparams, debug=True)

    return model, best_hyperparams, res[-1]


def export_model(model: xgboost.Booster, hyperparameters: dict, out_of_sample_predictions: pd.Series, model_path: str,
                 hyperparameters_path: str, out_of_sample_predictions_path: str) -> None:
    model.save_model(model_path)
    utils.dict_to_json(hyperparameters, hyperparameters_path)
    out_of_sample_predictions.to_csv(out_of_sample_predictions_path)


@click.command(name='train_model')
@click.option('--data_name', required=True, type=click.STRING)
@click.option('--target_variable', required=True, type=click.STRING)
def train_model(data_name, target_variable):
    model_name = utils.get_model_name(data_name, target_variable)

    data_path = utils.get_processed_data_path(data_name)
    features_path = utils.get_features_path(data_name)
    model_path = utils.get_model_path(model_name)
    hyperparameters_path = utils.get_model_hyperparameters_path(model_name)
    out_of_sample_predictions_path = utils.get_model_cv_out_of_sample_predictions_path(model_name)
    report_path = utils.get_report_path(model_name)
    report_resources_path = utils.get_report_resource_path(model_name)

    utils.prepareDir(utils.get_model_directory(model_name))

    data, features = utils.load_data(data_path, features_path, target_variable)
    model, hyperparameters, out_of_sample_predictions = train_model_util(data, features, target_variable, False)

    export_model(model, hyperparameters, out_of_sample_predictions, model_path, hyperparameters_path,
                 out_of_sample_predictions_path)

    make_report.generate_report_util(model, data, features, target_variable, out_of_sample_predictions, report_path,
                                     report_resources_path)


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


@click.command(name='train_error_model')
@click.option('--data_name', required=True, type=click.STRING)
@click.option('--target_variable', required=True, type=click.STRING)
@click.option('--use_pretrained_model', required=False, type=click.BOOL, default=True, show_default=True)
def train_error_model(data_name, target_variable, use_pretrained_model):
    model_name = utils.get_model_name(data_name, target_variable)
    error_model_name = utils.get_error_model_name(data_name, target_variable)

    data_path = utils.get_processed_data_path(data_name)
    features_path = utils.get_features_path(data_name)
    model_path = utils.get_model_path(model_name)
    hyperparameters_path = utils.get_model_hyperparameters_path(error_model_name)
    out_of_sample_predictions_path = utils.get_model_cv_out_of_sample_predictions_path(error_model_name)

    error_model_path = utils.get_model_path(error_model_name)
    error_model_hyperparameters_path = utils.get_model_hyperparameters_path(error_model_name)
    error_model_out_of_sample_predictions_path = utils.get_model_cv_out_of_sample_predictions_path(error_model_name)

    data, features = utils.load_data(data_path, features_path, target_variable)

    if (use_pretrained_model and not os.path.exists(model_path)) or not use_pretrained_model:
        model, hyperparameters, out_of_sample_predictions = train_model_util(data, features, target_variable, False)
        export_model(model, hyperparameters, out_of_sample_predictions, model_path, hyperparameters_path,
                     out_of_sample_predictions_path)

    model = utils.load_model(model_path)
    predictions = utils.predict(model, data[features])
    errors = data[target_variable] - predictions
    errors[np.abs(errors) < 1000] = 0
    errors[np.abs(errors) > 1000] = 1
    errors = errors.astype(bool)

    errors_feature = f'{model_name}_errors'
    data[errors_feature] = errors

    data = data[features + [errors_feature]]

    evaluate_baseline_error_model(data, features, errors_feature)

    error_model, error_model_hyperparameters, error_model_out_of_sample_predictions = train_model_util(data, features,
                                                                                                       errors_feature,
                                                                                                       True)

    utils.prepareDir(utils.get_model_directory(error_model_name))
    export_model(error_model, error_model_hyperparameters, error_model_out_of_sample_predictions, error_model_path,
                 error_model_hyperparameters_path,
                 error_model_out_of_sample_predictions_path)


@click.group()
def cli():
    pass


cli.add_command(train_error_model)
cli.add_command(train_model)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    cli()
