.. module:: src


===========

Description
-----------

This module includes functionalities for training XGBoost models and performing hyperparameter tuning. It also includes utility functions for error model training and evaluation and creating datasets.

Usage
-----

Command Line Interface (CLI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    $ cd src
    $ python models/train_model.py train_model --data_name YOUR_DATA_NAME --target_variable YOUR_TARGET_VARIABLE
    $ python models/train_model.py train_error_model --data_name YOUR_DATA_NAME --target_variable YOUR_TARGET_VARIABLE [--use_pretrained_model]
    $ python models/predict_model.py model_predict --data_name YOUR_DATA_NAME [--all_models] [--model_name YOUR_MODEL_NAME]
    $ python data/make_dataset.py sample_crawling_data --error_model_name ERROR_MODEL_NAME --service SERVICE_NAME --params_v PARAMS_VERSION --policy_start_date POLICY_START_DATE --n NUM_SAMPLES


Requirements
------------

- Python 3.6 or later
- Required Python packages (specified in your script)
- Pre-trained XGBoost models in the `models` directory
- Running make_dataset.py requires already trained error models.

Installation
------------

No special installation is required. Ensure that the necessary Python packages are installed:

::

    $ cd src
    $ pip install -r requirements.txt

Configuration
-------------

The scripts use environment variables, and you can configure them in a `.env` file.

.. note::

    Make sure to set the required environment variables for the scripts to work properly.

Example
-------

Train a regression model:

::

    $ cd src
    $ python models/train_model.py train_model --data_name example_data --target_variable target_variable

Train an error model:

::

    $ cd src
    $ python models/train_model.py train_error_model --data_name example_data --target_variable target_variable

Predict using a specific model:

::

    $ cd src
    $ python models/predict_model.py model_predict --data_name processed_data --model_name example_model

Predict using all compatible models:

::

    $ cd src
    $ python models/predict_model.py model_predict --data_name processed_data --all_models


Creating a 1000 row synthetic dataset:

::

    $ cd src
    $ python data/make_dataset.py sample_crawling_data --error_model_name netrisk_casco_error_model --service netrisk_casco --params_v v1 --policy_start_date 2023_01_01 --n 1000



Options
-------

- `data_name` (str): The name of the dataset or processed data file (without file extension).
- `target_variable` (str): The target variable for the model.
- `use_pretrained_model` (bool, optional): Use a pre-trained model for error prediction. Default is True.
- `all_models` (flag): Predict using all available compatible models (for predict_model.py).
- `model_name` (str): Name of the specific model to use for prediction (for predict_model.py).
- `error_model_name` (str): Name of the specific model to use for filtering sampled data points (for make_dataset.py).
- `service` (str): The name of the insurance service for which to generate synthetic profiles (for make_dataset.py).
- `params_v` (str): The version of parameters to use for generating profiles (for make_dataset.py).
- `policy_start_date` (str): The policy start date in YYYY_MM_DD format (for make_dataset.py).
- `n` (int): Number of profiles to sample (for make_dataset.py).


Authors
-------

- Author Name (Luka Erceg)

License
-------

This project is licensed under the MIT License - see the `LICENSE` file for details.

