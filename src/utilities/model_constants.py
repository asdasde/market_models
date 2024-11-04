import numpy as np

from hyperopt import hp

TEST_SIZE = 0.1
RANDOM_STATE = 42
ERROR_MODEL_CLASSIFICATION_THRESHOLD = 0.8
PRESENCE_MODEL_CLASSIFICATION_THRESHOLD = 0.5
MAX_EVALS =60

DEFAULT_PARAMS_REGRESSION = {
    'objective': 'reg:absoluteerror',
    'booster': 'gbtree',
    'eval_metric': 'mae',
    'eta': 0.3,
    'num_boost_round': 100,
    'early_stopping_rounds': None,
    "colsample_bytree": 0.9963829889516952,
    "gamma": 0.9,
    "lambda": 0.6687059388368572,
    "learning_rate": 0.14904051561388795,
    "max_depth": 7,
    "min_child_weight": 2,
    "n_estimators": 1000,
    "reg_alpha": 81.51686654321942,
    "reg_lambda": 0.1484520061561239,
    "seed": 0,
    "subsample": 0.7355645569069456,
}

DEFAULT_PARAMS_REGRESSION_DART = {
    'objective': 'reg:squarederror',
    'booster': 'dart',
    'eval_metric': 'mae',
    'eta': 0.3,
    'num_boost_round': 100,
    'early_stopping_rounds': None,
    "colsample_bytree": 0.9963829889516952,
    "gamma": 1.9729702249620764,
    "lambda": 0.6687059388368572,
    "learning_rate": 0.14904051561388795,
    "max_depth": 7,
    "min_child_weight": 2,
    "n_estimators": 1000,
    "reg_alpha": 81.51686654321942,
    "reg_lambda": 0.1484520061561239,
    "seed": 0,
    "subsample": 0.7355645569069456,
    'sample_type': 'uniform',
    'normalize_type': 'tree',
    'rate_drop': 0.1,
    'skip_drop': 0.5
}

DEFAULT_PARAMS_CLASSIFICATION = {
    'objective': 'binary:logistic',
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
    'objective': hp.choice('objective', ['reg:absoluteerror']),
    'booster': hp.choice('booster', ['gbtree']),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'n_estimators': hp.choice('n_estimators', np.arange(100, 1600, 50, dtype=int)),
    'max_depth': hp.choice('max_depth', np.arange(2, 11, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(1, 11, dtype=int)),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'gamma': hp.uniform('gamma', 0, 2),
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
    'gamma': hp.uniform('gamma', 0, 1),
    'lambda': hp.uniform('lambda', 0, 1),
    'reg_alpha': hp.uniform('reg_alpha', 40, 180),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'seed': 0,
}

DEFAULT_REPORT_TABLE_OF_CONTENTS = [
        "Data Overview",
        "Error Overview",
        "Error Quantiles",
        "Error Percentage Distribution",
        "Top k Largest Errors",
        "Learning Curve",
        "Feature Importance",
        "Feature Distribution",
        "Partial Dependence Plots",
        "Real vs Predicted Quantiles",
        "Real vs Predicted Quantiles by Feature",
        "Shapley Summary",
        "Shapley Waterfall"
    ]

FEATURES_TO_SKIP_PDP = ['CarMake', 'CarModel', 'vehicle_maker', 'vehicle_model', 'county', 'date_crawled']
