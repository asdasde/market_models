import numpy as np

from hyperopt import hp

TEST_SIZE = 0.1
RANDOM_STATE = 42

DEFAULT_PARAMS_REGRESSION = {
    'objective': 'reg:squarederror',
    'booster': 'gbtree',
    'eval_metric': 'mae',
    'eta': 0.3,
    'num_boost_round': 100,
    'early_stopping_rounds': None,
    "colsample_bytree": 0.603082145359716,
    "gamma": 1.1829268099644894,
    "lambda": 0.16917750724256597,
    "learning_rate": 0.07609398922995288,
    "max_depth": 7,
    "min_child_weight": 2,
    "n_estimators": 1000,
    "reg_alpha": 91.36426983962457,
    "reg_lambda": 0.7181972609115401,
    "seed": 0,
    "subsample": 0.8482744016970223
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
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'n_estimators': hp.choice('n_estimators', np.arange(100, 1300, 100, dtype=int)),
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
