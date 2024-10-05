import os
import sys
from copyreg import pickle

import click

from pathlib import Path
from typing import Optional, List
from dotenv import find_dotenv, load_dotenv
from mlxtend.classifier import OneRClassifier
from hyperopt import STATUS_OK, Trials, fmin, tpe
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    accuracy_score,
    log_loss,
    f1_score as f_score,
)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models.make_report as make_report

from utilities.load_utils import *
from utilities.files_utils import *
from utilities.model_utils import *
from utilities.model_constants import *
from utilities.export_utils import *



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

    dtrain = make_d_matrix(train_data[features], train_data[target_variable], is_classification=is_classification)
    dtest = make_d_matrix(test_data[features], test_data[target_variable], is_classification=is_classification)
    num_rounds = param_['n_estimators']
    del param_['n_estimators']
    param_['max_depth'] = int(param_['max_depth'])
    param_['eval_metric'] = 'aucpr' if is_classification else 'mae'
    eval_list = [(dtrain, 'train'), (dtest, 'eval')]
    evals_result = {}
    bst =  xgboost.train(param_, dtrain, num_boost_round=num_rounds, early_stopping_rounds=300, evals=eval_list,
                         verbose_eval=False, evals_result=evals_result)
    return bst

def merge_predictions(model: xgboost.Booster,
                      test_data: pd.DataFrame,
                      features: list,
                      target_variable: str,
                      is_classification: bool) -> pd.DataFrame:
    pred_target_variable = f'predicted_{target_variable}'
    output = test_data.copy()

    dtest = make_d_matrix(test_data[features], test_data[target_variable], is_classification=is_classification)
    output[pred_target_variable] = model.predict(dtest)

    output['error'] = output[target_variable] - output[pred_target_variable]
    output['percentageError'] = output['error'] / output[target_variable] * 100
    return output


def create_stratified_cv_splits(data: pd.DataFrame, target_variable: str, k: int = 3, num_groups=1000,
                                seed: int = RANDOM_STATE) -> list:
    grp = pd.qcut(data[target_variable], num_groups, labels=False, duplicates='drop')

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    fold_nums = np.zeros(len(data))

    for fold_no, (train_index, test_index) in enumerate(skf.split(X=data, y=grp)):
        fold_nums[test_index] = fold_no

    cv_splits = [(np.where(fold_nums != i)[0], np.where(fold_nums == i)[0]) for i in range(k)]

    return cv_splits


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
    f1_scores = []

    out_of_sample_predictions = pd.Series(index=data.index, dtype=float)

    kf = create_stratified_cv_splits(data, target_variable, k=3, num_groups=10)
    fold_num = 0
    for train_ix, test_ix in kf:
        fold_num += 1

        train_data, test_data = data.iloc[train_ix], data.iloc[test_ix]
        model = model_train(train_data, test_data, features, target_variable, is_classification, param)
        dtest = make_d_matrix(test_data[features], test_data[target_variable], is_classification=is_classification)
        test_preds = model.predict(dtest)
        out_of_sample_predictions.iloc[test_ix] = test_preds

        if is_classification:

            log_loss_value = log_loss(test_data[target_variable].values, test_preds)
            f1_score = f_score(test_data[target_variable], test_preds > 0.5)

            log_losses.append(log_loss_value)
            f1_scores.append(f1_score)

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
                print("F1 score is {}.".format(round(f1_score, 3)))
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

        mean_f1_score = np.mean(f1_scores)
        std_f1_score = np.std(f1_scores)

        if debug:
            print(
                f"Mean Log Loss over {k} fold Cross-validation is {round(mean_log_loss, 3)} ± {round(std_log_loss, 3)}.")
            print(
                f"Mean F1-score score over {k} fold Cross-validation is {round(mean_f1_score, 3)} ± {round(std_f1_score, 3)}.")

        return mean_log_loss, mean_f1_score, out_of_sample_predictions

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
        return {"loss": loss[0 if is_classification else 0], 'status': STATUS_OK}

    logging.info("Starting hyper-parameter tuning...")

    if is_classification:
        space = SPACE_CLASSIFICATION
    else:
        space = SPACE_REGRESSION

    best_hyperparams = fmin(fn=objective,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=MAX_EVALS,
                            trials=trials,
                            return_argmin=False)


    model = model_train(data, data, features, target_variable, is_classification, best_hyperparams)
    logging.info("Finished hyper-parameter tuning...")

    res = kFoldCrossValidation(k=3, data=data, features=features, target_variable=target_variable,
                               is_classification=is_classification, param=best_hyperparams, debug=True)

    return model, best_hyperparams, res[-1], trials


@click.command(name='train_model')
@click.option('--data_name', required=True, type=click.STRING)
@click.option('--target_variable', required=True, type=click.STRING)
def train_model(data_name, target_variable):
    model_name = get_model_name(data_name, target_variable)

    data, features_info, features_on_top, features_model = load_data(data_name, target_variable)
    model, hyperparameters, out_of_sample_predictions, trials = train_model_util(data, features_model, target_variable, False)

    export_model(model, hyperparameters, out_of_sample_predictions, trials, model_name)

    report_path = get_report_path(model_name)
    report_resources_path = get_report_resource_path(model_name)
    make_report.generate_report_util(model, data, features_info, features_model, target_variable,
                                     out_of_sample_predictions, trials, report_path,
                                     report_resources_path, use_pdp = False, use_shap=False)


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
    presence_model, presence_model_hyperparameters, presence_model_out_of_sample_predictions, presence_model_trials = (
        train_model_util(data, features_model, presence_model_target_variable,True))

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

    error_model, error_model_hyperparameters, error_model_out_of_sample_predictions, error_model_trials = train_model_util(data, features_model, error_model_target_variable, True)

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
