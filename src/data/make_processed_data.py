import os
import re
import sys
from pyexpat import features

import click

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.load_utils import *
from utilities.constants import *
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def cut_brackets_numeric(values: pd.Series, brackets: list) -> pd.Series:
    bins = [interval[0] - 1 for interval in brackets] + [brackets[-1][1]]
    return pd.cut(values, bins=bins)


def cut_brackets_categorical(values: pd.Series, brackets: list) -> pd.Series:
    return values


def make_postal_brackets_mtpl(data: pd.DataFrame, target_variables: list) -> Tuple[pd.DataFrame, List[str]]:
    bracket_cols = []
    for target_variable in target_variables:
        comp_name = column_to_folder_mapping.get(target_variable).replace('_tables', '')
        col_name = f'{target_variable}_PostalCode_cut'
        lookups_table_path = get_mtpl_postal_categories_path(target_variable)
        default_values_path = get_mtpl_default_values_path(target_variable)

        lookups_table = load_lookups_table(lookups_table_path)
        default_value = load_lookups_default_value(default_values_path)

        if default_value is None:
            default_value = lookups_table.key.max()

        if col_name in data.columns:
            data = data.drop(columns=[col_name])

        data = pd.merge(data, lookups_table, left_on='PostalCode', right_on='value', how='left')

        bracket_cols.append(col_name)

        data = data.rename(columns={'key': col_name}).drop('value', axis=1)


        data[col_name] = data[col_name].fillna(default_value)

        if comp_name == 'allianz':
            data[col_name] = data[col_name].apply(lambda x: ord(x) - 97)

        if comp_name == 'magyar':
            data[col_name] = data[col_name].apply(
                lambda x: 7 + int(x.split('_')[-1]) if 'budapest_' in str(x) else x
            )

        data[col_name] = data[col_name].astype(int)

    return data, bracket_cols




def get_target_variables(columns, suffix = '_price'):
    return [col for col in columns if col.endswith(suffix)]

def remove_special_chars_from_columns(data: pd.DataFrame, features_model: List[str], target_variables : List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    special_chars_mapping = {'[': '(', ']': ')', '<': '{'}

    def replace_special_chars(match):
        return special_chars_mapping.get(match.group(0), "?")

    pattern = re.compile("|".join(map(re.escape, special_chars_mapping.keys())))

    features_new = [pattern.sub(replace_special_chars, col) for col in features_model]
    target_variables_new = [pattern.sub(replace_special_chars, col) for col in target_variables]


    rename_dict = dict(zip(features_model + target_variables, features_new + target_variables_new))
    data = data.rename(columns=rename_dict)

    return data, features_new, target_variables_new


import pandas as pd
from datetime import datetime


def add_is_recent(data : pd.DataFrame) -> pd.DataFrame:
    if 'DateCrawled' not in data.columns:
        data['DateCrawled'] = datetime.now().strftime("%Y_%m_%d")
    data['DateCrawled'] = pd.to_datetime(data['DateCrawled'], format='%Y_%m_%d', errors='coerce')
    latest_date = data['DateCrawled'].max()

    data['isRecent'] = data['DateCrawled'].apply(
            lambda x: 'current_month' if x.month == latest_date.month and x.year == latest_date.year else
            'previous_month' if (latest_date - pd.DateOffset(months=1)).month == x.month and (
                    latest_date - pd.DateOffset(months=1)).year == x.year else
            '2_to_4_months_ago' if (latest_date - pd.DateOffset(months=4)) <= x < (
                    latest_date - pd.DateOffset(months=1)) else
            'older')
    return data


def add_bracket_feature(data: pd.DataFrame, target_variables: list, feature: str) -> Tuple[pd.DataFrame, List[str]]:
    if feature == 'PostalCode':
        return make_postal_brackets_mtpl(data, target_variables)

    brackets_path = get_brackets_path(feature)
    if not brackets_path.exists():
        return data, []

    with open(brackets_path, 'r') as json_file:
        brackets = json.load(json_file)

    if pd.api.types.is_numeric_dtype(data[feature]):
        cut_function = cut_brackets_numeric
    else:
        cut_function = cut_brackets_categorical

    cut_cols = []
    for target_variable in target_variables:
        target_variable = 'WÁBERER_price' if target_variable == 'GRÁNIT_price' else target_variable
        if len(brackets[target_variable]) == 0:
            continue
        cut_name = f'{target_variable}_{feature}_cut'
        cut_cols.append(cut_name)
        data[cut_name] = cut_function(data[feature], brackets[target_variable])
    return data, cut_cols

def add_bracket_features(data : pd.DataFrame, features_to_add_brackets : List[str]
                         , target_variables : List[str]) -> Tuple[pd.DataFrame, List[str]]:

    bracket_features = []
    for feature in features_to_add_brackets:
        data, brackets_feature = add_bracket_feature(data, target_variables, feature)
        bracket_features = bracket_features + brackets_feature
    return data, bracket_features


def generate_dummies(data : pd.DataFrame, categorical_columns : List[str]) -> pd.DataFrame:
    for col in categorical_columns:
        data[col] = data[col].astype('category')
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=False, prefix_sep='__', dummy_na=True)
    data.columns = [f'{col}__dummy' if any(col.startswith(f'{prefix}__') for prefix in categorical_columns) else col for
                    col in data.columns]
    return data


def make_processed_crawler_data(datas: List[pd.DataFrame]) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:

    processed_datas = []
    legacy_cols_to_drop = ['id_case', 'BonusMalusCode', 'CarMakerCategory', 'Category', 'WÁBERER_price_PostalCode_cut']
    legacy_rename = {'WÁBERER_price': 'GRÁNIT_price'}
    for data in datas:
        processed_data = data.drop(columns=legacy_cols_to_drop, errors='ignore')
        processed_data = processed_data.rename(columns=legacy_rename, errors='ignore')
        processed_datas.append(processed_data)

    data = pd.concat(processed_datas)
    data = data[data.columns.drop_duplicates()]

    target_variables = get_target_variables(data.columns)
    data = add_is_recent(data)

    features_to_add_bracket = ['Age', 'PostalCode']
    data, bracket_features = add_bracket_features(data, features_to_add_bracket, target_variables)

    categorical_columns = ['BonusMalus', 'CarMake', 'CarModel', 'isRecent'] + bracket_features
    data = generate_dummies(data, categorical_columns)

    data['DeductiblePercentage'] = data['DeductiblePercentage'].fillna(10)
    data['DeductibleAmount'] = data['DeductibleAmount'].fillna(100000)

    data[NETRISK_CASCO_EQUIPMENT_COLS] = data[NETRISK_CASCO_EQUIPMENT_COLS].fillna(False)

    features_info = NETRISK_CASCO_FEATURES_INFO
    features_on_top = NETRISK_CASCO_FEATURES_ON_TOP
    features_model = data.columns.difference(features_info + features_on_top + target_variables)


    data, features_model, target_variables = remove_special_chars_from_columns(data, features_model, target_variables)
    data = data[features_info + features_on_top + features_model + target_variables]

    return data, features_info, features_on_top, features_model



def make_processed_netrisk_like_data(datas : List[pd.DataFrame], data_name_reference : dict) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:

    processed_datas = []
    legacy_cols_to_drop = []
    legacy_rename = {'DriverAge' : 'Age'}
    legacy_cast = {'PostalCode' : int}

    for data in datas:
        processed_data = data.drop(columns=legacy_cols_to_drop, errors='ignore')
        processed_data = processed_data.rename(columns=legacy_rename, errors='ignore')
        processed_data = processed_data.dropna(subset = legacy_cast.keys()).astype(legacy_cast)
        processed_datas.append(processed_data)

    target_variables = DEFAULT_TARGET_VARIABLES

    data = pd.concat(processed_datas)
    data = data.set_index('unique_id')
    data = data[data['vehicle_make_year'] >= 2014]

    data = data[data.columns.drop_duplicates()]
    data = add_is_recent(data)

    if 'DeductibleAmount' not in data.columns:
        data['DeductibleAmount'] = 100000
    if 'DeductiblePercentage' not in data.columns:
        data['DeductiblePercentage'] = 10

    data['car_value'] = 15000
    data['LicenseAge'] = 18
    data['PostalCode2'] = data['PostalCode'].apply(lambda x: int(str(x)[:2]))
    data['PostalCode3'] = data['PostalCode'].apply(lambda x: int(   str(x)[:3]))

    data['BonusMalus'] = data['BonusMalus'].map(BONUS_MALUS_CLASSES_DICT_INV.get)

    others = load_other('netrisk_casco')
    geo_data_rename = {'latitude': 'Latitude', 'longitude': 'Longitude', 'postal_code': 'PostalCode'}
    others['hungary_postal_codes'] = (others['hungary_postal_codes'][['postal_code', 'latitude', 'longitude']]
                                     .drop_duplicates(subset=['postal_code']))
    others['hungary_postal_codes'] = others['hungary_postal_codes'].rename(columns=geo_data_rename)

    data = pd.merge(data, others['hungary_postal_codes'], on = 'PostalCode')


    data['DeductibleAmount'] = data['DeductibleAmount'].fillna(100000)
    data['DeductiblePercentage'] = data['DeductiblePercentage'].fillna(10)

    data[NETRISK_CASCO_EQUIPMENT_COLS] = False

    features_to_add_bracket = ['Age', 'PostalCode']
    data, bracket_features = add_bracket_features(data, features_to_add_bracket, target_variables)


    categorical_columns = ['BonusMalus', 'CarMake', 'CarModel', 'isRecent'] + bracket_features
    data = generate_dummies(data, categorical_columns)

    features_info = NETRISK_CASCO_FEATURES_INFO
    features_on_top = NETRISK_CASCO_FEATURES_ON_TOP

    latest_model_training_data = 'netrisk_casco_v1'
    features_model = reconstruct_features_model(data_name_reference[latest_model_training_data]['features_model'])

    data, features_model, target_variables = remove_special_chars_from_columns(data, features_model, target_variables)

    diff = sorted(list(set(features_model).difference(data.columns)))
    diff = [x for x in diff if x.endswith('__dummy')]
    data[diff] = False

    data = data[features_info + features_on_top + features_model]
    return data, features_info, features_on_top, features_model

def make_processed_punkta_data(datas : List[pd.DataFrame]) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:

    processed_datas = []
    legacy_cols_to_drop = []
    legacy_rename = {}
    legacy_cast = {}

    for data in datas:
        processed_data = data.drop(columns=legacy_cols_to_drop, errors='ignore')
        processed_data = processed_data.rename(columns=legacy_rename, errors='ignore')
        processed_data = processed_data.dropna(subset=legacy_cast.keys()).astype(legacy_cast)
        processed_datas.append(processed_data)

    data = pd.concat(processed_datas)
    data = data.set_index('unique_id')

    target_variables = get_target_variables(data, suffix = '-price')

    others = load_other('punkta')
    data = pd.merge(data, others['poland_postal_codes'][['postal_code', 'latitude', 'longitude', 'voivodeship', 'county']], on = 'postal_code')

    data['vehicle_maker'] = data['vehicle_maker'].apply(lambda x : x + '-' if any([c in x for c in ['[', ']']]) else x)
    data['contractor_age'] = CURRENT_YEAR - data['contractor_birth_year']
    data['licence_at_age'] = data['driver_licence_year'] - data['contractor_birth_year']
    data['driver_experience'] = CURRENT_YEAR - data['driver_licence_year']

    categorical_columns = ['vehicle_maker', 'vehicle_fuel_type', 'voivodeship', 'county', 'owner_driver_same', 'vehicle_parking_place']
    for categorical_col in categorical_columns:
        data[categorical_col] = data[categorical_col].astype('category')
    #data = generate_dummies(data, categorical_columns)

    features_info = ['calculation_time', 'contractor_birth_year', 'driver_licence_year']
    features_on_top = []
    features_model = (['vehicle_power', 'vehicle_engine_size', 'vehicle_weight_min', 'vehicle_weight_max',
                      'vehicle_make_year', 'number_of_damages_caused_in_last_5_years', 'mileage_domestic',
                      'contractor_age', 'licence_at_age', 'driver_experience', 'latitude', 'longitude']
                      + categorical_columns
                      + data.filter(like ='__dummy').columns.to_list())

    data, features_model, target_variables = remove_special_chars_from_columns(data, features_model, target_variables)
    data = data[features_info + features_on_top + features_model + target_variables]

    data = data.loc[:,~data.columns.duplicated()].copy()
    data = data[list(set(data.columns))]
    print(data)
    return data, features_info, features_on_top, features_model

def make_processed_generator_data(datas : List[pd.DataFrame], data_name_reference : dict) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    processed_datas = []
    legacy_cols_to_drop = ['id_case', 'BonusMalusCode', 'CarMakerCategory', 'Category', 'WÁBERER_price_PostalCode_cut', 'Longitude', 'Latitude']
    legacy_rename = {}
    legacy_cast = {}

    for data in datas:
        processed_data = data.drop(columns=legacy_cols_to_drop, errors='ignore')
        processed_data = processed_data.rename(columns=legacy_rename, errors='ignore')
        processed_data = processed_data.dropna(subset=legacy_cast.keys()).astype(legacy_cast)
        processed_datas.append(processed_data)

    data = pd.concat(processed_datas)
    data['id_case'] = range(len(data))

    data = data.set_index('id_case')

    target_variables = DEFAULT_TARGET_VARIABLES

    data = add_is_recent(data)

    if 'DeductibleAmount' not in data.columns:
        data['DeductibleAmount'] = 100000
    if 'DeductiblePercentage' not in data.columns:
        data['DeductiblePercentage'] = 10


    data['PostalCode2'] = data['PostalCode'].apply(lambda x: int(str(x)[:2]))
    data['PostalCode3'] = data['PostalCode'].apply(lambda x: int(str(x)[:3]))

    others = load_other('netrisk_casco')
    geo_data_rename = {'latitude': 'Latitude', 'longitude': 'Longitude', 'postal_code': 'PostalCode'}
    others['hungary_postal_codes'] = (others['hungary_postal_codes'][['postal_code', 'latitude', 'longitude']]
                                      .drop_duplicates(subset=['postal_code']))
    others['hungary_postal_codes'] = others['hungary_postal_codes'].rename(columns=geo_data_rename)
    data = pd.merge(data, others['hungary_postal_codes'], on='PostalCode')

    data['DeductibleAmount'] = data['DeductibleAmount'].fillna(100000)
    data['DeductiblePercentage'] = data['DeductiblePercentage'].fillna(10)
    data[NETRISK_CASCO_EQUIPMENT_COLS] = False

    features_to_add_bracket = ['Age', 'PostalCode']
    data, bracket_features = add_bracket_features(data, features_to_add_bracket, target_variables)

    categorical_columns = ['BonusMalus', 'CarMake', 'CarModel', 'isRecent'] + bracket_features
    data = generate_dummies(data, categorical_columns)

    features_info = NETRISK_CASCO_FEATURES_INFO
    features_on_top = NETRISK_CASCO_FEATURES_ON_TOP

    latest_model_training_data = 'netrisk_casco_v1'
    features_model = reconstruct_features_model(data_name_reference[latest_model_training_data]['features_model'])

    data, features_model, target_variables = remove_special_chars_from_columns(data, features_model, target_variables)
    diff = sorted(list(set(features_model).difference(data.columns)))
    diff = [x for x in diff if x.endswith('__dummy')]
    data[diff] = False


    data = data[features_info + features_on_top + features_model]
    return data, features_info, features_on_top, features_model


def make_processed_signal_iduna_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    target_variables = USUAL_TARGET_VARIABLES
    index_col = 'unique_id'
    index = data[index_col]

    data['Age'] = data['DriverAge']
    data['car_value'] = 15000
    data['PostalCode2'] = data['PostalCode'].apply(lambda x : int(str(x)[:2]))
    data['PostalCode3'] = data['PostalCode'].apply(lambda x : int(str(x)[:3]))

    data = add_is_recent(data)

    data, bracket_features = add_bracket_features(data, ['Age', 'PostalCode'], target_variables)

    categorical_columns = ['BonusMalus', 'CarMake', 'CarModel', 'isRecent'] + bracket_features

    data = generate_dummies(data, categorical_columns)
    features = data.columns.difference(target_variables)
    data, features = remove_special_chars_from_columns(data, features)
    dummies_d = set([col for col in data.columns if col.endswith('__dummy')])

    data_official, feature_official = load_data(get_processed_data_path('netrisk_casco_v36'), get_features_path('netrisk_casco_v36'))
    dummies_official = set([col for col in feature_official if col.endswith('__dummy')])

    for col in dummies_official:
        if col not in features:
            data[col] = False

    features = feature_official
    data = data[features]
    data[index_col] = index
    data = data.set_index(index_col)
    return data, features, [], []


def find_first_available_name(service: str, benchmark: bool) -> str:
    v_id = 1
    ext = 'benchmark_' if benchmark else ''
    while get_processed_data_path(f'{service}{ext}v{v_id}').exists():
        v_id += 1
    return f'{service}{ext}v{v_id}'


def extract_features_with_dtype(data: pd.DataFrame, features: List[str]) -> Dict[str, Dict[str, str]]:
    feature_dict = {}
    for column in data.columns:
        if column in features:
            if "__dummy" in column:
                parts = column.split('__')
                original_col = parts[0]
                dummy_val = '__'.join(parts[1:-1])
                if original_col not in feature_dict:
                    feature_dict[original_col] = {"dtype": 'category', "dummy_values": []}
                feature_dict[original_col]["dummy_values"].append(dummy_val)
            else:
                feature_dict[column] = {"dtype": str(data[column].dtype)}

    # Remove dummy_values key if it's empty
    for key in feature_dict:
        if "dummy_values" in feature_dict[key] and not feature_dict[key]["dummy_values"]:
            del feature_dict[key]["dummy_values"]
        elif "dummy_values" in feature_dict[key]:
            feature_dict[key]["dummy_values"] = '#'.join(feature_dict[key]["dummy_values"])

    return feature_dict



@click.command(name='make_processed_data')
@click.option('--service', required=False, default='netrisk_casco')
@click.option('--names', type=click.STRING, help='List of dates separated by ,')
@click.option('--data_source', type=click.STRING,
              help='Currently supported data sources are available are (crawler, signal_iduna)')
@click.option('--benchmark', default=False, type=click.BOOL,
              help='Signals that is purely for benchmarking purposes, and should not be used for training purposes.')
def make_processed_data(service: str, names: str, data_source: str, benchmark: bool) -> None:
    logger = logging.getLogger(__name__)
    names = names.split(',')
    service += '_'
    names = set(names)
    logger.info('Started finding name for new file ...')
    processed_data_name = find_first_available_name(service, benchmark)

    logger.info(f'Found name {processed_data_name}')
    logger.info("Loading data name reference")
    data_name_reference_path = get_data_name_references_path()

    if os.path.exists(data_name_reference_path):
        with open(data_name_reference_path, 'r') as json_file:
            data_name_reference = json.load(json_file)
    else:
        data_name_reference = {}

    logger.info("Checking in data name reference for duplicate datasets...")

    duplicate_found = any(
        names == set(entry['raw_data_used'])
        for entry in data_name_reference.values()
    )

    if duplicate_found:
        logger.info("Duplicate found, aborting ...")
        return

    logger.info("No duplicates found")
    logger.info("Loading raw data")

    extension = '.csv'
    if data_source in ['quotes_data', 'punkta_data']:
        extension = '.parquet'


    paths = [get_raw_data_path(f'{service}{name}', extension = extension) for name in names]
    datas = [read_data_frame(path) for path in paths]


    if data_source == "crawler":
        data, features_info, features_on_top, features_model = make_processed_crawler_data(datas)
    elif data_source == 'generator':
        data, features_info, features_on_top, features_model = make_processed_generator_data(datas, data_name_reference)
    elif data_source == "signal_iduna":
        data, features_info, features_on_top, features_model = make_processed_signal_iduna_data(datas)
    elif data_source == 'quotes_data':
        data, features_info, features_on_top, features_model = make_processed_netrisk_like_data(datas, data_name_reference)
    elif data_source == 'punkta_data':
        data, features_info, features_on_top, features_model = make_processed_punkta_data(datas)
    else:
        logger.info("Currently unsupported data source, aborting ...")
        return

    logger.info("Exporting processed data and feature file")
    processed_data_path = get_processed_data_path(processed_data_name)

    data.to_parquet(processed_data_path)

    logger.info("Adding it to the data name reference")
    file_size_mb = os.path.getsize(processed_data_path) / 1_048_576  # Convert bytes to MB
    file_size_mb = f"{file_size_mb:.1f}"  # Format to 1 decimal place

    features_model = extract_features_with_dtype(data, features_model)

    data_name_reference[processed_data_name] = {
        'raw_data_used': list(names),
        'processed_name': processed_data_name,
        'num_rows': len(data),
        'date_processed': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'data_source': data_source,
        'file_size_mb': file_size_mb,
        'is_benchmark': benchmark,
        'index_column_name': data.index.name,
        'features_info': features_info,
        'features_on_top': features_on_top,
        'features_model': features_model,
    }

    with open(data_name_reference_path, 'w') as json_file:
        json.dump(data_name_reference, json_file, indent=2)

    logger.info("Data name reference updated successfully.")

    logger.info('Checking if everyting went well, buy trying to load the data ...')
    data, features_info, features_on_top, features_model = load_data(processed_data_path)
    logger.info('All good')

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
    data_name_reference_path = get_data_name_references_path()

    logging.info("Removal process started ...")
    processed_data_path.unlink()
    logging.info("Removed processed data ...")

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