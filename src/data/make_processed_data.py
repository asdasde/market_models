import os
import sys
import json
import click
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.load_utils import *
from utilities.constants import *


def cut_brackets_numeric(values: pd.Series, brackets: list) -> pd.Series:
    bins = [interval[0] - 1 for interval in brackets] + [brackets[-1][1]]
    return pd.cut(values, bins=bins)


def cut_brackets_categorical(values: pd.Series, brackets: list) -> pd.Series:
    return values


def make_postal_brackets_mtpl(data: pd.DataFrame, target_variables: list) -> pd.DataFrame:
    for price_col in target_variables:
        comp_name = column_to_folder_mapping.get(price_col).replace('_tables', '')
        lookups_table_path = get_mtpl_postal_categories_path(price_col)
        default_values_path = get_mtpl_default_values_path(price_col)
        lookups_table = load_lookups_table(lookups_table_path)
        default_value = load_lookups_default_value(default_values_path)
        if default_value is None:
            default_value = lookups_table.key.max()

        data = pd.merge(data, lookups_table, left_on='PostalCode', right_on='value', how='left')
        col_name = f'{price_col.replace("_price", "")}_postal_category'
        data = data.rename(columns={'key': col_name}).drop('value', axis=1)
        data[col_name] = data[col_name].fillna(default_value)

        if comp_name == 'allianz':
            data[col_name] = data[col_name].apply(lambda x: ord(x) - 97)

        if comp_name == 'magyar':
            data[col_name] = data[col_name].apply(
                lambda x: 7 + int(x.split('_')[-1]) if 'budapest_' in str(x) else x
            )

    return data


def add_bracket_features(data: pd.DataFrame, target_variables: list, feature: str) -> pd.DataFrame:
    brackets_path = get_brackets_path(feature)
    if not brackets_path.exists():
        return data

    with open(brackets_path, 'r') as json_file:
        brackets = json.load(json_file)

    if pd.api.types.is_numeric_dtype(data[feature]):
        cut_function = cut_brackets_numeric
    else:
        cut_function = cut_brackets_categorical

    for target_variable in target_variables:
        target_variable = 'WÁBERER_price' if target_variable == 'GRÁNIT_price' else target_variable
        if len(brackets[target_variable]) == 0:
            continue

        cut_name = f'{target_variable}_{feature}_cut'
        data[cut_name] = cut_function(data[feature], brackets[target_variable])

    return data


def make_processed_crawler_data(data: pd.DataFrame) -> pd.DataFrame:
    data['BonusMalus'] = data['BonusMalus'].astype('category')
    data['CarMake'] = data['CarMake'].astype('category')
    data['CarModel'] = data['CarModel'].astype('category')

    target_variables = data.filter(like='_price').columns.tolist()
    features = data.columns.difference(target_variables)

    for feature in ['Age']:
        data = add_bracket_features(data, target_variables, feature)

    data = data.drop(['id_case'], axis=1)
    if 'DateCrawled' not in data.columns:
        data['DateCrawled'] = datetime.now().strftime("%Y_%m_%d")
    data['isRecent'] = data['DateCrawled'].apply(lambda x: x.split('_')[0] == '2024')

    return data


def make_processed_singal_iduna_data(data: pd.DataFrame) -> pd.DataFrame:
    price_annotation = '_newprice'
    index_col = 'policyNr'
    data = data.set_index(index_col)
    data['BonusMalus'] = data['BonusMalus'].astype('category')
    data['CarMake'] = data['CarMake'].astype('category')

    data = add_bracket_features(data, USUAL_TARGET_VARIABLES, 'Age')
    return data


def make_processed_generator_data(data: pd.DataFrame) -> pd.DataFrame:
    data['BonusMalus'] = data['BonusMalus'].astype('category')
    data['CarMake'] = data['CarMake'].astype('category')
    data = data.drop(['CarModel'], axis=1, errors='ignore')

    for feature in ['Age']:
        data = add_bracket_features(data, DEFAULT_TARGET_VARIABLES, feature)

    data = make_postal_brackets_mtpl(data, DEFAULT_TARGET_VARIABLES)
    data['ALFA_postal_category'] = data['ALFA_postal_category'].astype(float)
    data['Category'] = data['ALFA_postal_category'].astype(float)

    return data


def export_features_file(data: pd.DataFrame, features_path: Path) -> None:
    with open(features_path, 'w') as file:
        feature_cols = [col for col in data.columns if not col.endswith('_price') and col != 'DateCrawled']
        for feature in feature_cols:
            file.write(f"{feature},{data[feature].dtype}\n")


def find_first_available_name(service: str, benchmark: bool) -> str:
    v_id = 1
    ext = 'benchmark_' if benchmark else ''
    while get_processed_data_path(f'{service}{ext}v{v_id}').exists():
        v_id += 1
    return f'{service}{ext}v{v_id}'


@click.command(name='make_processed_data')
@click.option('--service', required=False, default='netrisk_casco')
@click.option('--names', type=click.STRING)
@click.option('--data_source', type=click.STRING,
              help='Currently supported data sources are available are (crawler, signal_iduna, profile_generator)')
@click.option('--benchmark', default=False, type=click.BOOL,
              help='Signals that is purely for benchmarking purposes, and should not be used for training purposes.')
def make_processed_data(service: str, names: str, data_source: str, benchmark: bool) -> None:
    logger = logging.getLogger(__name__)
    names = names.split(',')
    service += '_'

    dates = [set(name.replace(service, '').split('__')) for name in names]
    all_dates = set()
    for date in dates:
        all_dates = all_dates.union(date)
    all_dates = sorted(list(all_dates))
    all_dates_data_name = f'{service}{"__".join(all_dates)}'

    logger.info('Started finding name load_daid for new file ...')
    processed_data_name = find_first_available_name(service, benchmark)
    logger.info(f'Found name {processed_data_name}')

    logger.info("Loading data name reference")
    data_name_reference_path = get_data_name_references_path()
    if data_name_reference_path.exists():
        with open(data_name_reference_path, 'r') as json_file:
            data_name_reference = json.load(json_file)
    else:
        data_name_reference = {}

    logger.info("Checking in data name reference for duplicate datasets...")
    if all_dates_data_name in [x['raw_name'] for x in data_name_reference.values()]:
        logger.info("Duplicate found")
        logger.info("Aborting ...")
        return
    logger.info("No duplicates found")

    logger.info("Loading raw data")
    paths = [get_raw_data_path(f'{service}{name}') for name in names]
    datas = [pd.read_csv(path) for path in paths]
    data = pd.concat(datas)
    data['id_case'] = range(len(data))
    data = data.drop_duplicates(subset=data.columns.difference(['id_case']))

    if data_source == "crawler":
        data = make_processed_crawler_data(data)
    elif data_source == "signal_iduna":
        data = make_processed_singal_iduna_data(data)
    elif data_source == "profile_generator":
        data = make_processed_generator_data(data)
    else:
        logger.info("Currently unsupported data source, aborting ...")
        return

    logger.info("Exporting processed data and feature file")
    processed_data_path = get_processed_data_path(processed_data_name)
    features_path = get_features_path(processed_data_name)
    data.to_csv(processed_data_path)
    export_features_file(data, features_path)

    logger.info("Adding it to the data name reference")
    data_name_reference[processed_data_name] = {'raw_name': all_dates_data_name, 'processed_name': processed_data_name}
    with open(data_name_reference_path, 'w') as json_file:
        json.dump(data_name_reference, json_file, indent=2)


def check_processed_data_name(ctx, param, value):
    if value is not None:
        path = get_processed_data_path(value)
        if path.exists():
            return value
        else:
            raise click.BadParameter(f"Processed data file '{path}' does not exist.")


@click.command(name='remove_processed_data')
@click.option('--processed_data_name', type=click.STRING, required=True, callback=check_processed_data_name)
def remove_processed_data(processed_data_name: str):
    processed_data_path = get_processed_data_path(processed_data_name)
    features_path = get_features_path(processed_data_name)
    data_name_reference_path = get_data_name_references_path()

    logging.info("Removal process started ...")
    processed_data_path.unlink()
    logging.info("Removed processed data ...")
    if features_path.exists():
        features_path.unlink()
        logging.info("Removed corresponding features file ...")

    if data_name_reference_path.exists():
        with open(data_name_reference_path, 'r+') as json_file:
            data_name_reference = json.load(json_file)
            if processed_data_name in data_name_reference:
                del data_name_reference[processed_data_name]
                logging.info(f"File '{processed_data_name}' removed from reference.")
                json_file.seek(0)
                json_file.truncate()
                json.dump(data_name_reference, json_file, indent=4)
            else:
                logging.info(f"File '{processed_data_name}' not found in reference.")
    else:
        logging.info(f"No data reference file found.")


@click.group()
def cli():
    pass


cli.add_command(make_processed_data)
cli.add_command(remove_processed_data)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    cli()