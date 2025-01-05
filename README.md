market_models
==============================

Modeling competitor quotes (prima.ry Casco) using xgboost and quantile regression.

All the scripts from src/ are meant to runned from it so current workting directory should be src.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data contaings brackets and on top stuff and distributions for PG (legacy)
    │   ├── interim        <- Intermediate data that has been transformed (not really used anymore).
    │   ├── processed      <- The final, canonical data sets for modeling. 
    │   └── raw            <- The original, immutable data dump. (ussualy crawling results)
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, model summaries and hyperparameters.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Contains a data name refernce (a registery of all processed datas and details about them) 
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── error_overview <- All model errors per service per version
    │         ├── pdf_reports    <- PDF reports for individual models
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_processed_data.py <- main script used for making processed datas for all services
    │   │   └── data_processors.py     <- data processing logic per service
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py (not used)
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py  <- uses trained models to make predictions
    │   │   └── train_model.py    <- trains a model
    │   │   └── make_report.py    <- generates a PDF report for a trained model
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations (not used)
    │       └── visualize.py
    │
    │   └── api  <- Directory containg api code
    │   │   └── main.py    <- api itself
    │   │   └── config.py  <- config script for Poland (will have to work more on it)
    │   │   └── test.py    <- local test script

    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
