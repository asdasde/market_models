from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import os

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Environment variables
REMOTE_SERVER_FOR_CRAWLING = os.getenv('SERVER_FOR_CRAWLING')
REMOTE_SERVER_FOR_SAVING_DATA = os.getenv('SERVER_FOR_SAVING_DATA')
REMOTE_CRAWLER_DIRECTORY = os.getenv('REMOTE_CRAWLER_DIRECTORY')

# Base directory setup using relative paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Directory paths (relative to BASE_DIR or directly specified)

API_CONFIGURATIONS_PATH = BASE_DIR / 'src/api/configurations/'
DISTRIBUTION_PATH = BASE_DIR / 'data/external/distributions/'
ON_TOP_PATH = BASE_DIR / 'data/external/on_top/'
PROCESSED_DATA_PATH = BASE_DIR / 'data/processed/'
INTERIM_DATA_PATH = BASE_DIR / 'data/interim/'
RAW_DATA_PATH = BASE_DIR / 'data/raw/'
EXTERNAL_DATA_PATH = BASE_DIR / 'data/external'
PREDICTIONS_PATH = BASE_DIR / 'data/predictions/'
MODELS_PATH = BASE_DIR / 'models/'
ENCODERS_PATH = BASE_DIR / 'models/encoders/'
REPORTS_PATH = BASE_DIR / 'reports/pdf_reports/'
ERROR_OVERVIEW_PATH = BASE_DIR / 'reports/error_overview/'
PRIVATE_KEY_PATH = BASE_DIR / 'ssh_key'
REFERENCES_PATH = BASE_DIR / 'references/'
BRACKETS_PATH = BASE_DIR / 'data/external/feature_brackets/'
MTPL_POSTAL_CATEGORIES_PATH = BASE_DIR / 'data/external/mtpl_postal_categories/'
