import os
import sys
import re

import numpy as np
from pyspark.sql.connect.functions import second
from sklearn.preprocessing import OrdinalEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.load_utils import *
from utilities.constants import *
from typing import Union

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def cut_brackets_numeric(values: pd.Series, brackets: list) -> pd.Series:
    bins = [interval[0] - 1 for interval in brackets] + [brackets[-1][1]]
    return pd.cut(values, bins=bins).astype(str)


def cut_brackets_categorical(values: pd.Series, brackets: list) -> pd.Series:
    return values


def sample_from_distribution(
        source_data: pd.DataFrame,
        columns: List[str],
        target_data: pd.DataFrame) -> pd.DataFrame:
    clean_source = source_data[columns].dropna()
    if len(clean_source) > 0:
        sample_rows = clean_source.sample(n = len(target_data), replace=True)
        for col in columns:
            target_data[col] = sample_rows[col].values

    return target_data


def make_postal_brackets_mtpl(data: pd.DataFrame, target_variables: list, load_manager : LoadManager) -> Tuple[pd.DataFrame, List[str]]:
    bracket_cols = []
    for target_variable in target_variables:

        comp_name = column_to_folder_mapping.get(target_variable, None)
        if comp_name is None:
            continue
        comp_name = comp_name.replace('_tables', '')

        col_name = f'{target_variable}_postal_code_cut'

        lookups_table = load_manager.load_lookups_table(target_variable)
        default_value = load_manager.load_lookups_default_value(target_variable)

        if default_value is None:
            default_value = lookups_table.key.max()

        if col_name in data.columns:
            data = data.drop(columns=[col_name])

        data = pd.merge(data, lookups_table, left_on='postal_code', right_on='value', how='left')

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


def get_target_variables(columns: List[str], suffixes: Union[List[str], str] = 'price') -> List[str]:
    if isinstance(suffixes, str):
        suffixes = [suffixes]
    return [col for col in columns if any(col.endswith(suffix) for suffix in suffixes)]


def remove_special_chars_from_columns(data: pd.DataFrame, features_model: List[str], target_variables : List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    special_chars_mapping = {
        '[': '(',
        ']': ')',
        '<': '{',
        '/' : '&'
    }

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
    if 'date_crawled' not in data.columns:
        if 'calculation_time' in data.columns:
            data['date_crawled'] = data['calculation_time']
        else:
            data['date_crawled'] = datetime.now().strftime("%Y_%m_%d")

    data['date_crawled'] = pd.to_datetime(data['date_crawled'], format='%Y_%m_%d', errors='coerce')

    latest_date = data['date_crawled'].max()
    data['is_recent'] = data['date_crawled'].apply(
            lambda x: 'current_month' if x.month == latest_date.month and x.year == latest_date.year else
            'previous_month' if (latest_date - pd.DateOffset(months=1)).month == x.month and (
                    latest_date - pd.DateOffset(months=1)).year == x.year else
            '2_to_4_months_ago' if (latest_date - pd.DateOffset(months=4)) <= x < (
                    latest_date - pd.DateOffset(months=1)) else
            'older')
    return data


def add_bracket_feature(data: pd.DataFrame, target_variables: list, feature: str, load_manager : LoadManager) -> Tuple[pd.DataFrame, List[str]]:
    if feature == 'postal_code':
        return make_postal_brackets_mtpl(data, target_variables, load_manager)

    brackets_path = load_manager.path_manager.get_brackets_path(feature)
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
        if target_variable not in brackets.keys() or len(brackets[target_variable]) == 0:
            continue
        cut_name = f'{target_variable}_{feature}_cut'
        cut_cols.append(cut_name)
        data[cut_name] = cut_function(data[feature], brackets[target_variable])
    return data, cut_cols

def add_bracket_features(data : pd.DataFrame, features_to_add_brackets : List[str]
                         , target_variables : List[str], load_manager : LoadManager) -> Tuple[pd.DataFrame, List[str]]:

    bracket_features = []
    for feature in features_to_add_brackets:
        print(feature)
        data, brackets_feature = add_bracket_feature(data, target_variables, feature, load_manager)
        bracket_features = bracket_features + brackets_feature
    return data, bracket_features


def generate_dummies(data : pd.DataFrame, categorical_columns : List[str]) -> pd.DataFrame:
    for col in categorical_columns:
        data[col] = data[col].astype('category')
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=False, prefix_sep='__', dummy_na=True)
    data.columns = [f'{col}__dummy' if any(col.startswith(f'{prefix}__') for prefix in categorical_columns) else col for
                    col in data.columns]
    return data

def add_rank1_price(data : pd.DataFrame, target_variables) -> Tuple[pd.DataFrame, List[str]]:
    data['rank1_price'] = data[target_variables].min(axis = 1)
    target_variables.append('rank1_price')
    return data, target_variables


def make_processed_crawler_data(datas: List[pd.DataFrame], data_name_reference : dict, encoding_type : str, path_manager : PathManager, load_manager : LoadManager) \
        -> Tuple[pd.DataFrame, List[str], List[str], List[str], List[str]]:

    processed_datas = []
    legacy_cols_to_drop = ['Unnamed: 0', 'id_case', 'BonusMalusCode', 'CarMakerCategory', 'Category', 'WÁBERER_price_PostalCode_cut']
    legacy_rename = {'WÁBERER_price': 'GRÁNIT_price',
                    "vehicle_equipment_air_conditioning": "vehicle_equipment_ac",
                    "vehicle_equipment_driving_support_system": "vehicle_equipment_driving_support",
                    "vehicle_equipment_led_headlights": "vehicle_equipment_led_lights",
                    "vehicle_equipment_navigation_systems": "vehicle_equipment_navigation",
                    "vehicle_equipment_polished_specialty": "vehicle_equipment_polished",
                    "vehicle_equipment_shock_absorber": "vehicle_equipment_shock_system",
                    "vehicle_equipment_ultra_sonic": "vehicle_equipment_parking_system",
                    "vehicle_equipment_xenon_headlights": "vehicle_equipment_xenon_lights"
     }
    for data in datas:
        processed_data = data.drop(columns=legacy_cols_to_drop, errors='ignore')
        processed_data = processed_data.rename(columns=legacy_rename, errors='ignore')
        processed_datas.append(processed_data)
    data = pd.concat(processed_datas)

    data = data[data.columns.drop_duplicates()]

    target_variables = get_target_variables(data.columns)
    data, target_variables = add_rank1_price(data, target_variables)

    data = add_is_recent(data)

    data['postal_code_2'] = data['postal_code'].apply(lambda x : int(str(x)[:2]))
    data['postal_code_3'] = data['postal_code'].apply(lambda x : int(str(x)[:3]))

    features_to_add_bracket = ['contractor_age', 'postal_code']
    data, bracket_features = add_bracket_features(data, features_to_add_bracket, target_variables, load_manager)


    if 'vehicle_eurotax_code' not in data.columns:
        data['vehicle_eurotax_code'] = 'unknown'

    if 'vehicle_trim' not in data.columns:
        data['vehicle_trim'] = 'unknown'

    if 'policy_start_date' not in data.columns:
        data['policy_start_date'] = None

    if 'payment_frequency' not in data.columns:
        data['payment_frequency'] = 'yearly'

    if 'payment_method' not in data.columns:
        data['payment_method'] = 'bank_transfer'

    data['vehicle_fuel_type'] = data['vehicle_fuel_type'].replace({
        'Benzin' : 'petrol',
        'benzin' : 'petrol'
    })

    data['payment_method'] = data['payment_method'].fillna('bank_transfer')
    data['payment_frequency'] = data['payment_frequency'].fillna('yearly')

    categorical_columns = NETRISK_CASCO_CATEGORICAL_COLUMNS + bracket_features
    data[categorical_columns] = data[categorical_columns].astype('category')
    #data = generate_dummies(data, categorical_columns)

    data['deductible_percentage'] = data['deductible_percentage'].fillna(10)
    data['deductible_amount'] = data['deductible_amount'].fillna(100000)

    for equipment in NETRISK_CASCO_EQUIPMENT_COLS:
        if equipment not in data.columns:
            data[equipment] = False
        data[equipment].fillna(False)

    data[NETRISK_CASCO_EQUIPMENT_COLS] = data[NETRISK_CASCO_EQUIPMENT_COLS].fillna(False)

    for rider in NETRISK_CASCO_RIDERS:
        if rider not in data.columns:
            data[rider] = False
        data[rider] = data[rider].fillna(False)

    # for col in data.columns:
    #     print(col, data[col].dtype)

    data[NETRISK_CASCO_RIDERS] = data[NETRISK_CASCO_RIDERS].astype(bool)

    features_info = NETRISK_CASCO_FEATURES_INFO
    features_on_top = NETRISK_CASCO_FEATURES_ON_TOP
    features_model = data.columns.difference(features_info + features_on_top + target_variables).tolist()


    data, features_model, target_variables = remove_special_chars_from_columns(data, features_model, target_variables)
    data = data[features_info + features_on_top + features_model + target_variables]
    data['id_case'] = range(len(data))
    data = data.set_index('id_case')
    return data, features_info, features_on_top, features_model, target_variables


def make_processed_netrisk_casco_like_data(datas : List[pd.DataFrame], data_name_reference : dict, encoding_type : str, path_manager : PathManager, load_manager : LoadManager) \
        -> Tuple[pd.DataFrame, List[str], List[str], List[str], List[str]]:


    processed_datas = []
    legacy_cols_to_drop = ['vehicle_value']
    legacy_rename = {
        'ALFA_a_price' : 'ALFA_price',
        'GRANIT_a_price' : 'GRÁNIT_price',
        'GROUPAMA_a_price' : 'GROUPAMA_price',
        'K&H_a_price' : 'K&AMP;H_price',
        'UNION_b_price' : 'UNION_price'
    }
    legacy_cast = {'postal_code' : int}

    for data in datas:
        processed_data = data.drop(columns=legacy_cols_to_drop, errors='ignore')
        processed_data = processed_data.rename(columns=legacy_rename, errors='ignore')
        processed_data = processed_data.dropna(subset = legacy_cast.keys()).astype(legacy_cast)
        processed_datas.append(processed_data)

    data = pd.concat(processed_datas)
    data = data[data['vehicle_type'] == 'passenger_car']
    data = data[data['contractor_gender'] != 'company']
    try:
        data['policy_start_date'] = pd.to_datetime(data['policy_start_date']).dt.strftime('%Y_%m_%d')
    except:
        pass


    target_variables = DEFAULT_TARGET_VARIABLES['netrisk_casco']
    data = add_is_recent(data)

    data['vehicle_trim'] = data['vehicle_model']
    data['vehicle_model'] = data['vehicle_model'].apply(lambda x : x.split('  ')[0] if x is not None else x)

    eurotax_prices = pd.read_csv(path_manager.get_others_path() / 'eurotax_car_db_prices.csv')
    eurotax_prices = eurotax_prices[['eurotax_code', 'new_price_1_gross']]
    eurotax_prices = eurotax_prices.rename(
        columns = {'eurotax_code': 'vehicle_eurotax_code', 'new_price_1_gross': 'vehicle_value'}
    )
    data = pd.merge(data, eurotax_prices, on = 'vehicle_eurotax_code', how = 'left')

    data['contractor_age'] = CURRENT_YEAR - data['contractor_birth_year']
    data['vehicle_age'] = CURRENT_YEAR - data['vehicle_make_year']
    data['licence_age'] = CURRENT_YEAR - data['driver_licence_year'].astype("Int64")
    data['postal_code_2'] = data['postal_code'].apply(lambda x: int(str(x)[:2]))
    data['postal_code_3'] = data['postal_code'].apply(lambda x: int(str(x)[:3]))

    hungary_postal_codes = pd.read_csv(path_manager.get_others_path() / 'hungary_postal_codes.csv')
    hungary_postal_codes = (hungary_postal_codes[['postal_code', 'latitude', 'longitude']]
                                     .drop_duplicates(subset=['postal_code']))

    data = pd.merge(data, hungary_postal_codes, on = 'postal_code', how = 'left')


    data['deductible_amount'] = data['deductible_amount'].fillna(100000)
    data['deductible_percentage'] = data['deductible_percentage'].fillna(10)

    for equipment in NETRISK_CASCO_EQUIPMENT_COLS:
        data[equipment] = data[equipment].map({'yes' : True, 'no' : False})
    data[NETRISK_CASCO_EQUIPMENT_COLS] = data[NETRISK_CASCO_EQUIPMENT_COLS].astype(bool)

    for rider in NETRISK_CASCO_RIDERS:
        data[rider] = data[rider].map({'yes' : True, 'no' : False})
    data[NETRISK_CASCO_RIDERS] = data[NETRISK_CASCO_RIDERS].astype(bool)

    data['vehicle_fuel_type'] = data['vehicle_fuel_type'].replace(
        {'benzin' : 'petrol'}
    )

    data['payment_frequency'] = data['payment_frequency'].replace({
        'half_year' : 'half_yearly'
    })

    features_to_add_bracket = ['contractor_age', 'postal_code']
    data, bracket_features = add_bracket_features(data, features_to_add_bracket, target_variables, load_manager)

    categorical_columns = NETRISK_CASCO_CATEGORICAL_COLUMNS + bracket_features
    data[categorical_columns] = data[categorical_columns].astype('category')

    features_info = NETRISK_CASCO_FEATURES_INFO
    features_on_top = NETRISK_CASCO_FEATURES_ON_TOP

    latest_model_training_data = 'netrisk_casco_v11'
    features_model = LoadManager.reconstruct_features_model(data_name_reference[latest_model_training_data]['features_model'])
    print(features_model, target_variables)
    data, features_model, target_variables = remove_special_chars_from_columns(data, features_model, target_variables)
    data = data.set_index('unique_id')
    if all([x in data.columns for x in target_variables]):
        data = data[features_info + features_on_top + features_model + target_variables]
    else:
        data = data[features_info + features_on_top + features_model]
    return data, features_info, features_on_top, features_model, target_variables


def make_processed_netrisk_like_data(
        datas: List[pd.DataFrame],
        data_name_reference: dict,
        encoding_type: str,
        path_manager : PathManager,
        load_manager : LoadManager
) -> Tuple[pd.DataFrame, List[str], List[str], List[str], List[str]]:

    processed_datas = []
    legacy_cols_to_drop = ['CarModel']
    legacy_rename = {'DriverAge': 'Age', 'vehicle_model': "CarModel"}
    legacy_cast = {'postal_code': int}

    for data in datas:
        processed_data = data.drop(columns=legacy_cols_to_drop, errors='ignore')
        processed_data = processed_data.rename(columns=legacy_rename, errors='ignore')
        processed_data = processed_data.dropna(subset=legacy_cast.keys()).astype(legacy_cast)
        processed_datas.append(processed_data)

    # Concatenate and prepare data
    target_variables = DEFAULT_TARGET_VARIABLES
    data = pd.concat(processed_datas)

    # Filter data for vehicles made from 2014 onwards
    data = data[data['vehicle_make_year'] >= 2014]

    # Remove duplicate columns
    data = data[data.columns.drop_duplicates()]

    # Add is_recent feature
    data = add_is_recent(data)

    # Add missing columns with default values
    default_columns = {
        'deductible_amount': 100000,
        'deductible_percentage': 10,
        'vehicle_value': 15000 / FORINT_TO_EUR,
        'bonus_malus_casco': 'CO1'
    }

    for col, default_val in default_columns.items():
        if col not in data.columns:
            data[col] = default_val

    data['contractor_age'] = CURRENT_YEAR - data['contractor_birth_year']
    data['vehicle_age'] = CURRENT_YEAR - data['vehicle_make_year']
    data['licence_age'] = CURRENT_YEAR - data['driver_licence_year']
    data['postal_code_2'] = data['postal_code'].apply(lambda x: int(str(x)[:2]))
    data['postal_code_3'] = data['postal_code'].apply(lambda x: int(str(x)[:3]))

    others = load_manager.load_other()
    others['hungary_postal_codes'] = (
        others['hungary_postal_codes'][['postal_code', 'latitude', 'longitude']]
        .drop_duplicates(subset=['postal_code'])
    )

    data = pd.merge(data, others['hungary_postal_codes'], on='postal_code', how='left')

    data[NETRISK_CASCO_RIDERS] = None
    data[NETRISK_CASCO_EQUIPMENT_COLS] = None

    netrisk_data, _, _, _ = load_manager.load_data('netrisk_casco_v10')

    sampling_columns = {
        'deductible': ['deductible_amount', 'deductible_percentage'],
        'bonus_malus': ['bonus_malus_casco'],
        'riders': NETRISK_CASCO_RIDERS,
        'equipment': NETRISK_CASCO_EQUIPMENT_COLS
    }

    for category, cols in sampling_columns.items():
        source_df = netrisk_data[cols].dropna()
        data = sample_from_distribution(source_df, cols, data)

    data['vehicle_model'] = ''
    data['vehicle_trim'] = ''
    data['vehicle_eurotax_code'] = ''

    features_to_add_bracket = ['contractor_age', 'postal_code']
    data, bracket_features = add_bracket_features(data, features_to_add_bracket, target_variables, load_manager)

    data[NETRISK_CASCO_CATEGORICAL_COLUMNS + bracket_features] = data[NETRISK_CASCO_CATEGORICAL_COLUMNS + bracket_features].astype('category')

    categorical_columns = (
            ['bonus_malus_current', 'bonus_malus_casco', 'vehicle_maker', 'vehicle_model', 'vehicle_fuel_type',
             'is_recent']
            + bracket_features
    )
    data[categorical_columns] = data[categorical_columns].astype('category')

    features_info = NETRISK_CASCO_FEATURES_INFO
    features_on_top = NETRISK_CASCO_FEATURES_ON_TOP

    latest_model_training_data = 'netrisk_casco_v11'
    features_model = LoadManager.reconstruct_features_model(data_name_reference[latest_model_training_data]['features_model'])

    data, features_model, target_variables = remove_special_chars_from_columns(
        data,
        features_model,
        target_variables
    )

    data = data.set_index('unique_id')
    data = data[features_info + features_on_top + features_model]

    return data, features_info, features_on_top, features_model, target_variables

import pandas as pd

def make_is_outlier_columns(data, target_variable):
    data['vehicle_power_cut'] = pd.cut(data['vehicle_power'].values, bins=20)
    data['contractor_age_cut'] = pd.cut(data['contractor_age'].values, bins=20)

    threshold = 0.85
    vehicle_power_threshold = (data.groupby('vehicle_power_cut', observed=False)[target_variable].
                               transform(lambda x: x.quantile(threshold)))
    contractor_age_threshold = (data.groupby('contractor_age_cut', observed=False)[target_variable].
                                transform(lambda x: x.quantile(threshold)))

    outlier_column_name = f'is_outlier_per_{target_variable}'
    data[outlier_column_name] = (
        (data[target_variable] >= vehicle_power_threshold) &
        (data[target_variable] >= contractor_age_threshold)
    )

    return data, outlier_column_name

def filter_punkta(data : pd.DataFrame) -> pd.DataFrame:
    original_len = len(data)

    data = data[data['contractor_age'] >= 18]
    data = data[data['licence_at_age'] >= 15]
    data = data[data['driver_experience'] >= 0]
    data = data[data['vehicle_age'] >= 0]
    data = data[data['vehicle_power'] >= 20]
    data = data[data['vehicle_engine_size'] >= 100]
    data = data[data['vehicle_weight_min'] >= 200]
    data = data[data['worth'] > 100]
    #data = data[data['date_crawled'].dt.month >= 9]
    data = data[data['vehicle_usage'] == 'private']
    data = data[data['vehicle_type'].isin(['Samochody osobowe', 'Osobowy'])]
    data = data[data['time_delta'] < 100]
    data = data[data['time_delta'] >= 0]

    filtered_len = len(data)
    print(f"Filtered out {original_len - filtered_len} rows, for having impossible values,"
          f" which is {round(1 - filtered_len / original_len, 2)}")
    return data

from pprint import pprint
def make_processed_punkta_data(datas : List[pd.DataFrame], data_name_reference : dict, encoding_type : str, path_manager : PathManager, load_manager : LoadManager) \
        -> Tuple[pd.DataFrame, List[str], List[str], List[str], List[str]]:

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


    others = load_manager.load_other()

    geo_data_columns = ['postal_code', 'latitude', 'longitude', 'voivodeship', 'county', 'postal_code_population', 'postal_code_area']
    data = pd.merge(data, others['poland_postal_codes'][geo_data_columns], on = 'postal_code')
    data['postal_code_population_density'] = round(data['postal_code_population'] / data['postal_code_area'], 3)


    data['date_crawled'] = pd.to_datetime(data['calculation_time'])
    data['vehicle_maker'] = data['vehicle_maker'].apply(lambda x : x + '-' if any([c in x for c in ['[', ']']]) else x)
    data['vehicle_age'] = CURRENT_YEAR - data['vehicle_make_year']
    data['contractor_age'] = CURRENT_YEAR - data['contractor_birth_year']
    data['licence_at_age'] = data['driver_licence_year'] - data['contractor_birth_year']
    data['driver_experience'] = CURRENT_YEAR - data['driver_licence_year']
    data['vehicle_weight_to_power_ratio'] = data['vehicle_weight_max'] / data['vehicle_power']


    data = pd.merge(data, others['mtu_contractor_age_factors'], how = 'left')
    data = pd.merge(data, others['mtu_vehicle_age_factors'], how = 'left')
    data['policy_start_month'] = data['policy_start_date'].dt.month

    data['time_delta'] = (data['policy_start_date'] - pd.to_datetime(data['calculation_time'])).dt.days
    pprint(data.head())

    target_variables = get_target_variables(data.columns, suffixes=['-price', '-isolated_price'])
    log_target_variables = []
    is_outlier_columns = []
    for i, target_variable in enumerate(target_variables):
        data, added_col = make_is_outlier_columns(data, target_variable)
        is_outlier_columns.append(added_col)
        if data[target_variable].min() > 1:
            data[f'log_{target_variable}'] = np.log(data[target_variable])
            log_target_variables.append(f'log_{target_variable}')

    target_variables += log_target_variables

    data[PUNKTA_CATEGORICAL_COLUMNS] = data[PUNKTA_CATEGORICAL_COLUMNS].astype('category')

    features_info = PUNKTA_FEATURES_INFO
    features_on_top = PUNKTA_FEATURES_ON_TOP
    features_model = (PUNKTA_FEATURES_MODEL + is_outlier_columns + data.filter(like ='__dummy').columns.to_list())

    data = filter_punkta(data)

    data, features_model, target_variables = remove_special_chars_from_columns(data, features_model, target_variables)
    data = data[features_info + features_on_top + features_model + target_variables]

    data = data.loc[:,~data.columns.duplicated()].copy()
    data = data[list(set(data.columns))]
    return data, features_info, features_on_top, features_model, target_variables

def make_processed_mubi_data(datas : List[pd.DataFrame], data_name_reference : dict, encoding_type : str, path_manager : PathManager, load_manager : LoadManager):


    MUBI_CATEGORICAL = ['vehicle_maker', 'vehicle_fuel_type', 'vehicle_model',
                        'vehicle_parking_place', 'vehicle_usage', 'vehicle_first_registration_country',
                        'vehicle_planned_annual_mileage',
                        'contractor_marital_status',
                        'voivodeship', 'county']

    MUBI_FEATURES_INFO = ['id_case', 'crawling_date', 'policy_start_date',
                          'contractor_birth_date',
                          'contractor_driver_licence_date', 'contractor_personal_id',
                          'contractor_postal_code',
                          'vehicle_make_year',
                          'vehicle_trim', 'vehicle_eurotax_version',
                          'vehicle_infoexpert_model', 'vehicle_infoexpert_version',
                          'vehicle_type', 'vehicle_licence_plate']

    MUBI_FEATURES_ON_TOP = []

    MUBI_VEHICLE_VALUE_FEATURES = ['allianz_vehicle_value', 'balcia_vehicle_value', 'beesafe_vehicle_value',
                                   'benefia_vehicle_value', 'ergohestia_vehicle_value',
                                   'generali_vehicle_value', 'link4_vehicle_value',
                                   'mtu24_vehicle_value', 'proama_vehicle_value',
                                   'trasti_vehicle_value', 'tuz_vehicle_value',
                                   'uniqa_vehicle_value', 'wefox_vehicle_value',
                                   'wiener_vehicle_value', 'ycd_vehicle_value']

    MUBI_MTPL_CLAIM_YEARS = [f'contractor_mtpl_{x}_claim' for x in ['first', 'second', 'third', 'fourth', 'fifth']]



    MUBI_FEATURES_MODEL = [
       # VEHICLE INFO
       'vehicle_engine_size', 'vehicle_power','vehicle_weight_to_power_ratio',
       'vehicle_net_weight', 'vehicle_gross_weight',
       'vehicle_number_of_seats', 'vehicle_number_of_doors',
       'vehicle_age',
       'vehicle_steering_wheel_right',
       'vehicle_imported', 'vehicle_imported_within_last_12_months',
       # CONTRACTOR INFO
       'contractor_age', 'licence_at_age', 'driver_experience',
       'latitude', 'longitude',
       'postal_code_population', 'postal_code_area', 'postal_code_population_density',
       # POLICY INFO
       'contractor_mtpl_policy_years',
       'contractor_mtpl_number_of_claims', 'contractor_mtpl_years_since_last_damage_caused',
       # DISCOUNTS
       'contractor_children_under_26',
       'additional_driver_under_26', 'additional_driver_under_26_license_obtained_year',
       'GENERALI_pesel_ab_test'
    ] + MUBI_VEHICLE_VALUE_FEATURES + MUBI_CATEGORICAL

    MUBI_NULLABLE_INT = ['additional_driver_under_26_license_obtained_year', 'contractor_mtpl_years_since_last_damage_caused']
    MUBI_NULLABLE_FLOAT = MUBI_VEHICLE_VALUE_FEATURES

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

    target_variables = get_target_variables(data.columns, suffixes=['-price', '-isolated_price'])

    if 'id_case' not in data.columns:
        data['id_case'] = range(len(data))
    if 'crawling_date' not in data.columns:
        data['crawling_date'] = '2000.01.01'

    data = data.dropna(subset=['contractor_birth_date'])

    print(data.filter(like = 'vehicle_value').columns)

    others = load_manager.load_other()
    geo_data_columns = ['postal_code', 'latitude', 'longitude', 'voivodeship', 'county', 'postal_code_population',
                        'postal_code_area']

    data = pd.merge(
        data,
        others['poland_postal_codes'][geo_data_columns],
        how='left',
        left_on='contractor_postal_code',
        right_on='postal_code',
    )

    data['postal_code_population_density'] = round(data['postal_code_population'] / data['postal_code_area'], 3)


    data['contractor_birth_year'] = data['contractor_birth_date'].apply(lambda x: int(str(x).split('.')[0].split('-')[0]))

    data['contractor_driver_licence_year'] = data['contractor_driver_licence_date'].apply(
        lambda x: int(str(x).split('.')[0].split('-')[0]))


    data['policy_start_date'] = data['policy_start_date'].fillna(datetime.today().strftime("%Y.%m.%d"))

    data['vehicle_age'] = CURRENT_YEAR - data['vehicle_make_year']
    data['contractor_age'] = (pd.to_datetime(data['policy_start_date'], format='%Y.%m.%d') -
                              pd.to_datetime(data['contractor_birth_date'], format='%Y.%m.%d'))
    data['contractor_age'] = data['contractor_age'].apply(lambda x: np.floor(x.days / 365.25))

    data['licence_at_age'] = (pd.to_datetime(data['contractor_driver_licence_date'], format='%Y.%m.%d') -
                              pd.to_datetime(data['contractor_birth_date'], format='%Y.%m.%d'))
    data['licence_at_age'] = data['licence_at_age'].apply(lambda x: np.floor(x.days / 365.25))

    data['driver_experience'] = (pd.to_datetime(data['policy_start_date'], format='%Y.%m.%d') -
                              pd.to_datetime(data['contractor_driver_licence_date'], format='%Y.%m.%d'))
    data['driver_experience'] = data['driver_experience'].apply(lambda x: np.floor(x.days / 365.25))

    data['pesel_last_digit'] = data['contractor_personal_id'].apply(lambda x: int(str(x)[-1]))
    data['generali_pesel_ab_test'] = data['pesel_last_digit'].apply(lambda x: 0 if (x == 0) or (x < 8 and x % 2) else 1)

    data['vehicle_weight_to_power_ratio'] = data['vehicle_gross_weight'] / data['vehicle_power']

    data['contractor_mtpl_policy_years'] = CURRENT_YEAR - data['mtpl_first_purchase_year']


    print(data.filter(like = 'vehicle_value').columns)


    def get_last_damage(row):
        num_claims = row['contractor_mtpl_number_of_claims']
        if num_claims == 0:
            return None

        claim_cols = MUBI_MTPL_CLAIM_YEARS[:num_claims]
        return CURRENT_YEAR - max(row[claim_cols])

    data['contractor_mtpl_years_since_last_damage_caused'] = (
        data.apply(lambda x : get_last_damage(x), axis = 1)
    )

    data['pesel_last_digit'] = data['contractor_personal_id'].apply(lambda x: int(str(x)[-1]))
    data['GENERALI_pesel_ab_test'] = data['pesel_last_digit'].apply(lambda x: 0 if (x == 0) or (x < 8 and x % 2) else 1)

    features_info = MUBI_FEATURES_INFO
    features_on_top = MUBI_FEATURES_ON_TOP
    features_model = MUBI_FEATURES_MODEL


    data[MUBI_CATEGORICAL] = data[MUBI_CATEGORICAL].astype('category')

    for vehicle_value_col in MUBI_VEHICLE_VALUE_FEATURES:
        if vehicle_value_col not in data.columns:
            data[vehicle_value_col] = None

    data[MUBI_NULLABLE_INT] = data[MUBI_NULLABLE_INT].astype('Int64')
    data[MUBI_NULLABLE_FLOAT] = data[MUBI_NULLABLE_FLOAT].astype('Float64')

    data, features_model, target_variables = remove_special_chars_from_columns(data, features_model, target_variables)
    data = data[features_info + features_on_top + features_model + target_variables]

    data = data.loc[:,~data.columns.duplicated()].copy()
    return data, features_info, features_on_top, features_model, target_variables

def make_processed_mubi_data_old(datas : List[pd.DataFrame], data_name_reference : dict, encoding_type : str, path_manager : PathManager, load_manager : LoadManager):
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
    print(data['contractor_birth_date'])
    data = data.dropna(subset = ['contractor_birth_date'])

    others = load_manager.load_other()
    geo_data_columns = ['postal_code', 'latitude', 'longitude', 'voivodeship', 'county', 'postal_code_population', 'postal_code_area']
    data = pd.merge(
        data,
        others['poland_postal_codes'][geo_data_columns],
        how = 'left',
        left_on = 'contractor_postal_code',
        right_on='postal_code',
    )
    data['postal_code_population_density'] = round(data['postal_code_population'] / data['postal_code_area'], 3)

    data['contractor_birth_year'] = data['contractor_birth_date'].apply(lambda x : int(x.split('_')[0]))
    data['contractor_driver_licence_year'] = data['contractor_driver_licence_date'].apply(lambda x : int(x.split('_')[0]))

    data['vehicle_age'] = CURRENT_YEAR - data['vehicle_make_year']
    data['contractor_age'] = (pd.to_datetime(data['policy_start_date'], format = '%Y_%m_%d') -
                              pd.to_datetime(data['contractor_birth_date'], format = '%Y_%m_%d'))
    data['contractor_age'] = data['contractor_age'].apply(lambda x : np.floor(x.days / 365.25))

    data['licence_at_age'] = (data['contractor_driver_licence_year'] - data['contractor_birth_year']).astype(int)
    data['driver_experience'] = (CURRENT_YEAR - data['contractor_driver_licence_year']).astype(int)


    data['vehicle_weight_to_power_ratio'] = data['vehicle_gross_weight'] / data['vehicle_power']

    target_variables = get_target_variables(data.columns, suffixes=['-price', '-isolated_price'])
    data[MUBI_CATEGORICAL] = data[MUBI_CATEGORICAL].astype('category')

    data = data[~(data['vehicle_model'] == 'Punto')]
    data = data[~(data['vehicle_model'] == 'Lupo')]

    features_info = MUBI_FEATURES_INFO
    features_on_top = MUBI_FEATURES_ON_TOP
    features_model = (
            MUBI_FEATURES_MODEL +
           data.filter(like ='__dummy').columns.to_list()
    )


    data, features_model, target_variables = remove_special_chars_from_columns(data, features_model, target_variables)
    data = data[features_info + features_on_top + features_model + target_variables]

    data = data.loc[:,~data.columns.duplicated()].copy()
    data = data[list(set(data.columns))]
    return data, features_info, features_on_top, features_model, target_variables


def make_processed_generator_data(
        datas: List[pd.DataFrame],
        data_name_reference: dict,
        encoding_type: str,
        path_manager : PathManager,
        load_manger : LoadManager
) -> Tuple[pd.DataFrame, List[str], List[str], List[str], List[str]]:

    processed_datas = []
    legacy_cols_to_drop = ['id_case', 'BonusMalusCode', 'CarMakerCategory', 'Category', 'WÁBERER_price_PostalCode_cut',
                           'Longitude', 'Latitude', 'latitude', 'longitude']
    legacy_rename = {}
    legacy_cast = {}

    for data in datas:
        processed_data = data.drop(columns=legacy_cols_to_drop, errors='ignore')
        processed_data = processed_data.rename(columns=legacy_rename, errors='ignore')
        processed_data = processed_data.dropna(subset=legacy_cast.keys()).astype(legacy_cast)
        processed_datas.append(processed_data)

    data = pd.concat(processed_datas)
    data['unique_id'] = range(len(data))  # Changed from id_case to unique_id to match Signal Iduna
    data = data.set_index('unique_id')

    target_variables = DEFAULT_TARGET_VARIABLES

    default_columns = {
        'deductible_amount': None,
        'deductible_percentage': None,
        'bonus_malus_casco': None
    }

    for col, default_val in default_columns.items():
        if col not in data.columns:
            data[col] = default_val

    data['postal_code_2'] = data['postal_code'].apply(lambda x: int(str(x)[:2]))
    data['postal_code_3'] = data['postal_code'].apply(lambda x: int(str(x)[:3]))

    others = load_manger.load_other()
    others['hungary_postal_codes'] = (
        others['hungary_postal_codes'][['postal_code', 'latitude', 'longitude']]
        .drop_duplicates(subset=['postal_code'])
    )
    data = pd.merge(data, others['hungary_postal_codes'], on='postal_code', how='left')
    pprint(data)
    data = add_is_recent(data)

    netrisk_data, _, _, _ = load_manger.load_data('netrisk_casco_v10')

    # Sample distributions for specific columns
    sampling_columns = {
        'deductible': ['deductible_amount', 'deductible_percentage'],
        'bonus_malus': ['bonus_malus_casco'],
        'riders': NETRISK_CASCO_RIDERS,
        'equipment': NETRISK_CASCO_EQUIPMENT_COLS
    }

    for category, cols in sampling_columns.items():
        # Create missing mask for the specific columns
        source_df = netrisk_data[cols].dropna()
        data = sample_from_distribution(source_df, cols, data)

    data['vehicle_trim'] = ''
    data['vehicle_eurotax_code'] = ''

    data, bracket_features = add_bracket_features(data, ['contractor_age', 'postal_code'], target_variables, load_manger)

    data[NETRISK_CASCO_CATEGORICAL_COLUMNS + bracket_features] = data[NETRISK_CASCO_CATEGORICAL_COLUMNS + bracket_features].astype('category')


    features_info = NETRISK_CASCO_FEATURES_INFO
    features_on_top = NETRISK_CASCO_FEATURES_ON_TOP

    latest_model_training_data = 'netrisk_casco_v11'
    features_model = LoadManager.reconstruct_features_model(data_name_reference[latest_model_training_data]['features_model'])

    # Remove special characters from columns
    data, features_model, target_variables = remove_special_chars_from_columns(
        data,
        features_model,
        target_variables
    )

    # Select final columns
    data = data[features_info + features_on_top + features_model]

    return data, features_info, features_on_top, features_model, target_variables


def make_processed_signal_iduna_data(datas: List[pd.DataFrame], data_name_reference : dict, encoding_type : str, path_manager : PathManager, load_manager : LoadManager) \
        -> Tuple[pd.DataFrame, List[str], List[str], List[str], List[str]]:

    processed_datas = []
    legacy_cols_to_drop = []
    legacy_rename = {}
    legacy_cast = {}

    for data in datas:
        processed_data = data.drop(columns=legacy_cols_to_drop, errors='ignore')
        processed_data = processed_data.rename(columns=legacy_rename, errors='ignore')
        processed_data = processed_data.dropna(subset=legacy_cast.keys()).astype(legacy_cast)
        processed_datas.append(processed_data)

    target_variables = DEFAULT_TARGET_VARIABLES
    data = pd.concat(processed_datas)

    if 'deductible_amount' not in data.columns:
        data['deductible_amount'] = None
    if 'deductible_percentage' not in data.columns:
        data['deductible_percentage'] = None
    if 'vehicle_value' not in data.columns:
        data['vehicle_value'] = 15000 / FORINT_TO_EUR


    data['bonus_malus_casco'] = None
    data['contractor_age'] = CURRENT_YEAR - data['contractor_birth_year']
    data['vehicle_age'] = CURRENT_YEAR - data['vehicle_make_year']
    data['licence_age'] = CURRENT_YEAR - data['driver_licence_year']
    data['postal_code_2'] = data['postal_code'].apply(lambda x : int(str(x)[:2]))
    data['postal_code_3'] = data['postal_code'].apply(lambda x : int(str(x)[:3]))

    others = load_manager.load_other()
    others['hungary_postal_codes'] = (others['hungary_postal_codes'][['postal_code', 'latitude', 'longitude']]
                                      .drop_duplicates(subset=['postal_code']))

    data = pd.merge(data, others['hungary_postal_codes'], on='postal_code', how='left')

    data = add_is_recent(data)

    netrisk_data, _, _, _ = load_manager.load_data('netrisk_casco_v10')

    sampling_columns = {
        'deductible': ['deductible_amount', 'deductible_percentage'],
        'bonus_malus': ['bonus_malus_casco'],
        'riders': NETRISK_CASCO_RIDERS,
        'equipment': NETRISK_CASCO_EQUIPMENT_COLS
    }

    for category, cols in sampling_columns.items():
        source_df = netrisk_data[cols].dropna()
        data = sample_from_distribution(source_df,cols,data)

    data['vehicle_trim'] = ''
    data['vehicle_eurotax_code'] = ''
    data[NETRISK_CASCO_CATEGORICAL_COLUMNS] = data[NETRISK_CASCO_CATEGORICAL_COLUMNS].astype('category')
    data, bracket_features = add_bracket_features(data, ['contractor_age', 'postal_code'], target_variables, load_manager)

    categorical_columns = ['bonus_malus_current', 'bonus_malus_casco', 'vehicle_maker', 'vehicle_model', 'vehicle_fuel_type', 'is_recent'] + bracket_features
    data[categorical_columns] = data[categorical_columns].astype('category')
    #data = generate_dummies(data, categorical_columns)

    features_info = NETRISK_CASCO_FEATURES_INFO
    features_on_top = NETRISK_CASCO_FEATURES_ON_TOP

    latest_model_training_data = 'netrisk_casco_v11'
    features_model = LoadManager.reconstruct_features_model(data_name_reference[latest_model_training_data]['features_model'])
    data = data.set_index('unique_id')
    data, features_model, target_variables = remove_special_chars_from_columns(data, features_model, target_variables)
    data = data[features_info + features_on_top + features_model]
    return data, features_info, features_on_top, features_model, target_variables

def make_processed_zmarta_data(datas: pd.DataFrame, data_name_reference : dict, encoding_type : str, path_manager : PathManager, load_manger : PathManager) -> Tuple[pd.DataFrame, List[str], List[str], List[str], List[str]]:
    processed_datas = []
    legacy_cols_to_drop = []
    legacy_rename = {}
    legacy_cast = {}

    for data in datas:
        processed_data = data.drop(columns=legacy_cols_to_drop, errors="ignore")
        processed_data = processed_data.rename(columns=legacy_rename, errors="ignore")
        processed_data = processed_data.dropna(subset=legacy_cast.keys()).astype(legacy_cast)
        processed_data = processed_data.reset_index(drop=True)
        processed_datas.append(processed_data)

    data = pd.concat(processed_datas, ignore_index=True)

    data = add_is_recent(data)

    target_variables = get_target_variables(data.columns, suffixes=["_price"])

    data["contractor_age"] = CURRENT_YEAR - data["contractor_birth_year"]
    data["vehicle_age"] = CURRENT_YEAR - data["vehicle_make_year"]

    categorical_columns = [
        "isRecent",
        "vehicle_maker",
        "vehicle_model",
        "vehicle_body_type",
        "vehicle_fuel_type",
        "contractor_expected_mileage",
        "contractor_living_place",
        "contractor_marital_status",
        "driver_under25",
        "deductible_level",
        "postal_code",
    ]
    for categorical_col in categorical_columns:
        data[categorical_col] = data[categorical_col].astype("category")
    data["postal_code"] = pd.to_numeric(data["postal_code"], errors="coerce")
    data["vehicle_number_of_seats"] = pd.to_numeric(data["vehicle_number_of_seats"], errors="coerce")

    features_info = ["contractor_birth_year", "vehicle_make_year"]
    features_on_top = []
    features_model = (
        [
            "vehicle_age",
            "contractor_age",
            "vehicle_engine_size",
            "vehicle_power",
            "vehicle_weight_min",
            "vehicle_weight_load",
            "vehicle_kg_per_kw",
            "vehicle_value",
            "vehicle_number_of_seats",
            "vehicle_ownership_duration",
            "vehicle_number_of_owners",
            "postal_code",
            "annualy_commuting_km",
            # "contractor_owned_vehicles",
            # "contractor_residence_living_duration",
            "postal_code_density",
            # "postal_code_median_age",
            "latitude",
            "longitude",
            # "postal_code_average_income",
            # "postal_code_payment_complaints_percent",
        ]
        + categorical_columns
        + data.filter(like="__dummy").columns.to_list()
    )

    data, features_model, target_variables = remove_special_chars_from_columns(
        data, features_model, target_variables
    )
    data = data[features_info + features_on_top + features_model + target_variables]

    data = data.loc[:, ~data.columns.duplicated()].copy()
    data = data[list(set(data.columns))]
    return data, features_info, features_on_top, features_model, target_variables

def find_first_available_name(path_manager : PathManager, benchmark: bool) -> str:
    v_id = 1
    ext = 'benchmark_' if benchmark else ''
    while path_manager.get_processed_data_path(f'{path_manager.service}{ext}_v{v_id}').exists():
        v_id += 1
    return f'{path_manager.service}{ext}_v{v_id}'


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

    for key in feature_dict:
        if "dummy_values" in feature_dict[key] and not feature_dict[key]["dummy_values"]:
            del feature_dict[key]["dummy_values"]
        elif "dummy_values" in feature_dict[key]:
            feature_dict[key]["dummy_values"] = '#'.join(feature_dict[key]["dummy_values"])

    return feature_dict