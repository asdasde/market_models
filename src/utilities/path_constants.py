from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

REMOTE_SERVER_FOR_CRAWLING = os.getenv('SERVER_FOR_CRAWLING')
REMOTE_SERVER_FOR_SAVING_DATA = os.getenv('SERVER_FOR_SAVING_DATA')
REMOTE_CRAWLER_DIRECTORY = os.getenv('REMOTE_CRAWLER_DIRECTORY')

DISTRIBUTION_PATH = Path('../data/external/distributions/')
PROCESSED_DATA_PATH = Path('../data/processed/')
INTERIM_DATA_PATH = Path('../data/interim/')
RAW_DATA_PATH = Path('../data/raw/')
PREDICTIONS_PATH = Path('../data/predictions/')
MODELS_PATH = Path('../models/')
ENCODERS_PATH = Path('../models/encoders/')
REPORTS_PATH = Path('../reports/')
PRIVATE_KEY_PATH = Path('../../../ssh_key')
REFERENCES_PATH = Path('../references/')
BRACKETS_PATH = Path('../data/external/feature_brackets/')
MTPL_POSTAL_CATEGORIES_PATH = Path('../data/external/mtpl_postal_categories/')


