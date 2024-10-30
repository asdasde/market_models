import os
import sys
import re

from tensorboard import errors

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


def make_postal_brackets_mtpl(data: pd.DataFrame, target_variables: list) -> Tuple[pd.DataFrame, List[str]]:
    bracket_cols = []
    for target_variable in target_variables:

        comp_name = column_to_folder_mapping.get(target_variable).replace('_tables', '')
        col_name = f'{target_variable}_postal_code_cut'

        lookups_table = load_lookups_table(target_variable)
        default_value = load_lookups_default_value(target_variable)

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


def add_bracket_feature(data: pd.DataFrame, target_variables: list, feature: str) -> Tuple[pd.DataFrame, List[str]]:
    if feature == 'postal_code':
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



def make_processed_crawler_data(datas: List[pd.DataFrame], data_name_reference : dict, encoding_type : str) \
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

    data = add_is_recent(data)

    data['postal_code_2'] = data['postal_code'].apply(lambda x : int(str(x)[:2]))
    data['postal_code_3'] = data['postal_code'].apply(lambda x : int(str(x)[:3]))

    features_to_add_bracket = ['contractor_age', 'postal_code']
    data, bracket_features = add_bracket_features(data, features_to_add_bracket, target_variables)


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


    categorical_columns = NETRISK_CASCO_CATEGORICAL_COLUMNS + bracket_features
    data[categorical_columns] = data[categorical_columns].astype('category')
    #data = generate_dummies(data, categorical_columns)

    data['deductible_percentage'] = data['deductible_percentage'].fillna(10)
    data['deductible_amount'] = data['deductible_amount'].fillna(100000)

    for equipment in NETRISK_CASCO_EQUIPMENT_COLS:
        if equipment not in data.columns:
            data[equipment] = False

    data[NETRISK_CASCO_EQUIPMENT_COLS] = data[NETRISK_CASCO_EQUIPMENT_COLS].fillna(False)

    for rider in NETRISK_CASCO_RIDERS:
        if rider not in data.columns:
            data[rider] = False

    features_info = NETRISK_CASCO_FEATURES_INFO
    features_on_top = NETRISK_CASCO_FEATURES_ON_TOP
    features_model = data.columns.difference(features_info + features_on_top + target_variables).tolist()


    data, features_model, target_variables = remove_special_chars_from_columns(data, features_model, target_variables)
    data = data[features_info + features_on_top + features_model + target_variables]
    data['id_case'] = range(len(data))
    data = data.set_index('id_case')
    print(data.dropna(subset = ['policy_start_date']))
    return data, features_info, features_on_top, features_model, target_variables


def make_processed_netrisk_casco_like_data(datas : List[pd.DataFrame], data_name_reference : dict, encoding_type : str) \
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

    target_variables = DEFAULT_TARGET_VARIABLES

    data = pd.concat(processed_datas)
    data = data[data['vehicle_type'] == 'passenger_car']
    data = data[data['contractor_gender'] != 'company']
    data['policy_start_date'] = pd.to_datetime(data['policy_start_date']).dt.strftime('%Y_%m_%d')
    data = add_is_recent(data)

    data['vehicle_trim'] = data['vehicle_model']
    data['vehicle_model'] = data['vehicle_model'].apply(lambda x : x.split('  ')[0] if x is not None else x)

    eurotax_prices = pd.read_csv(get_others_path('netrisk_casco') / 'eurotax_car_db_prices.csv')
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

    hungary_postal_codes = pd.read_csv(get_others_path('netrisk_casco') / 'hungary_postal_codes.csv')
    hungary_postal_codes = (hungary_postal_codes[['postal_code', 'latitude', 'longitude']]
                                     .drop_duplicates(subset=['postal_code']))

    data = pd.merge(data, hungary_postal_codes, on = 'postal_code', how = 'left')


    data['deductible_amount'] = data['deductible_amount'].fillna(100000)
    data['deductible_percentage'] = data['deductible_percentage'].fillna(10)

    for equipment in NETRISK_CASCO_EQUIPMENT_COLS:
        data[equipment] = data[equipment].map({'yes' : True, 'no' : False})


    for rider in NETRISK_CASCO_RIDERS:
        data[rider] = data[rider].map({'yes' : True, 'no' : False})


    features_to_add_bracket = ['contractor_age', 'postal_code']
    data, bracket_features = add_bracket_features(data, features_to_add_bracket, target_variables)

    categorical_columns = NETRISK_CASCO_CATEGORICAL_COLUMNS + bracket_features
    data[categorical_columns] = data[categorical_columns].astype('category')

    features_info = NETRISK_CASCO_FEATURES_INFO
    features_on_top = NETRISK_CASCO_FEATURES_ON_TOP

    latest_model_training_data = 'netrisk_casco_v1'
    features_model = reconstruct_features_model(data_name_reference[latest_model_training_data]['features_model'])

    # %%
    data = data[~data[NETRISK_CASCO_EQUIPMENT_COLS + NETRISK_CASCO_RIDERS].any(axis=1)]
    # %%
    data = data[data['payment_frequency'] == 'yearly']
    # %%
    data = data[data['payment_method'] == 'bank_transfer']
    # %%
    data = data[data['deductible_percentage'] == 10]
    data = data[data['deductible_amount'] == 100000]

    data, features_model, target_variables = remove_special_chars_from_columns(data, features_model, target_variables)
    data = data.set_index('unique_id')
    data = data[features_info + features_on_top + features_model + target_variables]
    return data, features_info, features_on_top, features_model, target_variables


def make_processed_netrisk_like_data(datas : List[pd.DataFrame], data_name_reference : dict, encoding_type : str) \
        -> Tuple[pd.DataFrame, List[str], List[str], List[str], List[str]]:

    processed_datas = []
    legacy_cols_to_drop = ['CarModel']
    legacy_rename = {'DriverAge' : 'Age', 'vehicle_model' : "CarModel"}
    legacy_cast = {'postal_code' : int}

    for data in datas:
        processed_data = data.drop(columns=legacy_cols_to_drop, errors='ignore')
        processed_data = processed_data.rename(columns=legacy_rename, errors='ignore')
        processed_data = processed_data.dropna(subset = legacy_cast.keys()).astype(legacy_cast)
        processed_datas.append(processed_data)

    target_variables = DEFAULT_TARGET_VARIABLES

    data = pd.concat(processed_datas)
    data = data[data['vehicle_make_year'] >= 2014]

    data = data[data.columns.drop_duplicates()]
    data = add_is_recent(data)

    if 'deductible_amount' not in data.columns:
        data['deductible_amount'] = 100000
    if 'deductible_percentage' not in data.columns:
        data['deductible_percentage'] = 10

    data['vehicle_value'] = 15000 / FORINT_TO_EUR
    data['contractor_age'] = CURRENT_YEAR - data['contractor_birth_year']
    data['vehicle_age'] = CURRENT_YEAR - data['vehicle_make_year']
    data['licence_age'] = CURRENT_YEAR - data['driver_licence_year']
    data['postal_code_2'] = data['postal_code'].apply(lambda x: int(str(x)[:2]))
    data['postal_code_3'] = data['postal_code'].apply(lambda x: int(str(x)[:3]))

    data['bonus_malus_casco'] = 'CO1'

    others = load_other('netrisk_casco')
    others['hungary_postal_codes'] = (others['hungary_postal_codes'][['postal_code', 'latitude', 'longitude']]
                                     .drop_duplicates(subset=['postal_code']))

    data = pd.merge(data, others['hungary_postal_codes'], on = 'postal_code', how = 'left')


    data['deductible_amount'] = data['deductible_amount'].fillna(100000)
    data['deductible_percentage'] = data['deductible_percentage'].fillna(10)

    data[NETRISK_CASCO_EQUIPMENT_COLS] = False

    features_to_add_bracket = ['contractor_age', 'postal_code']
    data, bracket_features = add_bracket_features(data, features_to_add_bracket, target_variables)

    data['vehicle_model'] = ''

    categorical_columns = ['bonus_malus_current', 'bonus_malus_casco', 'vehicle_maker', 'vehicle_model', 'vehicle_fuel_type', 'is_recent'] + bracket_features
    data[categorical_columns] = data[categorical_columns].astype('category')
    #data = generate_dummies(data, categorical_columns)

    features_info = NETRISK_CASCO_FEATURES_INFO
    features_on_top = NETRISK_CASCO_FEATURES_ON_TOP

    latest_model_training_data = 'netrisk_casco_v1'
    features_model = reconstruct_features_model(data_name_reference[latest_model_training_data]['features_model'])

    data, features_model, target_variables = remove_special_chars_from_columns(data, features_model, target_variables)
    data = data.set_index('unique_id')
    data = data[features_info + features_on_top + features_model]
    return data, features_info, features_on_top, features_model, target_variables

def make_processed_punkta_data(datas : List[pd.DataFrame], encoding_type : str) -> Tuple[pd.DataFrame, List[str], List[str], List[str], List[str]]:

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

    target_variables = get_target_variables(data.columns, suffixes = ['-price', '-isolated_price'])

    others = load_other('punkta')
    data = pd.merge(data, others['poland_postal_codes'][['postal_code', 'latitude', 'longitude', 'voivodeship', 'county']], on = 'postal_code')

    data['date_crawled'] = pd.to_datetime(data['date_crawled'])
    data['vehicle_maker'] = data['vehicle_maker'].apply(lambda x : x + '-' if any([c in x for c in ['[', ']']]) else x)
    data['contractor_age'] = CURRENT_YEAR - data['contractor_birth_year']
    data['licence_at_age'] = data['driver_licence_year'] - data['contractor_birth_year']
    data['driver_experience'] = CURRENT_YEAR - data['driver_licence_year']

    categorical_columns = ['vehicle_maker', 'vehicle_fuel_type', 'voivodeship', 'county', 'owner_driver_same', 'vehicle_parking_place']
    for categorical_col in categorical_columns:
        data[categorical_col] = data[categorical_col].astype('category')
    #data = generate_dummies(data, categorical_columns)

    features_info = ['date_crawled', 'contractor_birth_year', 'driver_licence_year']
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
    return data, features_info, features_on_top, features_model, target_variables

def make_processed_generator_data(datas : List[pd.DataFrame], data_name_reference : dict, encoding_type : str) \
        -> Tuple[pd.DataFrame, List[str], List[str], List[str], List[str]]:
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

    if 'deductible_amount' not in data.columns:
        data['deductible_amount'] = 100000
    if 'deductible_percentage' not in data.columns:
        data['deductible_percentage'] = 10


    data['PostalCode2'] = data['PostalCode'].apply(lambda x: int(str(x)[:2]))
    data['PostalCode3'] = data['PostalCode'].apply(lambda x: int(str(x)[:3]))

    others = load_other('netrisk_casco')
    geo_data_rename = {'latitude': 'Latitude', 'longitude': 'Longitude', 'postal_code': 'PostalCode'}
    others['hungary_postal_codes'] = (others['hungary_postal_codes'][['postal_code', 'latitude', 'longitude']]
                                      .drop_duplicates(subset=['postal_code']))
    others['hungary_postal_codes'] = others['hungary_postal_codes'].rename(columns=geo_data_rename)
    data = pd.merge(data, others['hungary_postal_codes'], on='PostalCode')

    data['deductible_amount'] = data['deductible_amount'].fillna(100000)
    data['deductible_percentage'] = data['deductible_percentage'].fillna(10)
    data[NETRISK_CASCO_EQUIPMENT_COLS] = False

    features_to_add_bracket = ['Age', 'PostalCode']
    data, bracket_features = add_bracket_features(data, features_to_add_bracket, target_variables)

    categorical_columns = ['BonusMalus', 'CarMake', 'CarModel', 'isRecent'] + bracket_features
    #data = generate_dummies(data, categorical_columns)
    data[categorical_columns] = data[categorical_columns].astype('category')

    features_info = NETRISK_CASCO_FEATURES_INFO
    features_on_top = NETRISK_CASCO_FEATURES_ON_TOP

    latest_model_training_data = 'netrisk_casco_v1'
    features_model = reconstruct_features_model(data_name_reference[latest_model_training_data]['features_model'])

    data, features_model, target_variables = remove_special_chars_from_columns(data, features_model, target_variables)
    #diff = sorted(list(set(features_model).difference(data.columns)))
    #diff = [x for x in diff if x.endswith('__dummy')]
    #data[diff] = False


    data = data[features_info + features_on_top + features_model]
    return data, features_info, features_on_top, features_model, target_variables


def make_processed_signal_iduna_data(datas: List[pd.DataFrame], data_name_reference : dict, encoding_type : str) \
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
        data['deductible_amount'] = 100000
    if 'deductible_percentage' not in data.columns:
        data['deductible_percentage'] = 10
    if 'vehicle_value' not in data.columns:
        data['vehicle_value'] = 15000 / FORINT_TO_EUR


    data['bonus_malus_casco'] = 'C01'
    data['contractor_age'] = CURRENT_YEAR - data['contractor_birth_year']
    data['vehicle_age'] = CURRENT_YEAR - data['vehicle_make_year']
    data['licence_age'] = CURRENT_YEAR - data['driver_licence_year']
    data['postal_code_2'] = data['postal_code'].apply(lambda x : int(str(x)[:2]))
    data['postal_code_3'] = data['postal_code'].apply(lambda x : int(str(x)[:3]))

    others = load_other('netrisk_casco')
    others['hungary_postal_codes'] = (others['hungary_postal_codes'][['postal_code', 'latitude', 'longitude']]
                                      .drop_duplicates(subset=['postal_code']))

    data = pd.merge(data, others['hungary_postal_codes'], on='postal_code', how='left')

    data = add_is_recent(data)

    data['deductible_amount'] = data['deductible_amount'].fillna(100000)
    data['deductible_percentage'] = data['deductible_percentage'].fillna(10)
    data[NETRISK_CASCO_EQUIPMENT_COLS] = False

    data, bracket_features = add_bracket_features(data, ['contractor_age', 'postal_code'], target_variables)

    categorical_columns = ['bonus_malus_current', 'bonus_malus_casco', 'vehicle_maker', 'vehicle_model', 'vehicle_fuel_type', 'is_recent'] + bracket_features
    data[categorical_columns] = data[categorical_columns].astype('category')
    #data = generate_dummies(data, categorical_columns)

    features_info = NETRISK_CASCO_FEATURES_INFO
    features_on_top = NETRISK_CASCO_FEATURES_ON_TOP

    latest_model_training_data = 'netrisk_casco_v1'
    features_model = reconstruct_features_model(data_name_reference[latest_model_training_data]['features_model'])
    data = data.set_index('unique_id')
    data, features_model, target_variables = remove_special_chars_from_columns(data, features_model, target_variables)
    data = data[features_info + features_on_top + features_model + target_variables]
    return data, features_info, features_on_top, features_model, target_variables

def make_processed_zmarta_data(datas: pd.DataFrame, data_name_reference : dict, encoding_type : str) -> Tuple[pd.DataFrame, List[str], List[str], List[str], List[str]]:
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
            "contractor_owned_vehicles",
            "contractor_residence_living_duration",
            "postal_code_density",
            # "postal_code_median_age",
            "latitude",
            "longitude",
            "postal_code_average_income",
            "postal_code_payment_complaints_percent",
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

    for key in feature_dict:
        if "dummy_values" in feature_dict[key] and not feature_dict[key]["dummy_values"]:
            del feature_dict[key]["dummy_values"]
        elif "dummy_values" in feature_dict[key]:
            feature_dict[key]["dummy_values"] = '#'.join(feature_dict[key]["dummy_values"])

    return feature_dict