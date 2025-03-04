from datetime import datetime

NAME_MAPPING = {
    'Policy_Start_Bonus_Malus_Class': 'BonusMalus',
    'Vehicle_age': 'CarAge',
    'Vehicle_weight_empty': 'CarWeightMin',
    'Number_of_seats': 'NumberOfSeats',
    'Driver_Experience': 'DriverExperience',
    'Vehicle_weight_maximum': 'CarWeightMax',
    'Power_range_in_KW': 'kw',
    'Engine_size': 'engine_size',
    'DriverAge': 'driver_age',
    'PostalCode': 'PostalCode',
    'CarMake': 'CarMake',
    'Milage': 'Mileage'
}

NETRISK_CASCO_DTYPES = {'isRecent': 'bool', 'CarMake': 'category', 'CarModel': 'category', 'CarAge': 'int',
                        'ccm': 'int', 'kw': 'int', 'kg': 'int',
                        'car_value': 'float', 'CarMakerCategory': 'float', 'PostalCode': 'int', 'PostalCode2': 'int',
                        'PostalCode3': 'int',
                        'Category': 'int', 'Longitude': 'float', 'Latitude': 'float', 'Age': 'int', 'LicenseAge': 'int',
                        'BonusMalus': 'category',
                        'BonusMalusCode': 'category'}
NETRISK_CASCO_EQUIPMENT_COLS = [
    "vehicle_equipment_polished",
    "vehicle_equipment_ac",
    "vehicle_equipment_alloy_wheels",
    "vehicle_equipment_automatic_transmission",
    "vehicle_equipment_leather_seats",
    "vehicle_equipment_navigation",
    "vehicle_equipment_xenon_lights",
    "vehicle_equipment_camera",
    "vehicle_equipment_sunroof",
    "vehicle_equipment_led_lights",
    "vehicle_equipment_shock_system",
    "vehicle_equipment_parking_system",
    "vehicle_equipment_driving_support",
]

NETRISK_CASCO_RIDERS = [
    "glass_damage_deductible",
    "additional_insurance",
    "accident_insurance",
    "luggage_insurance",
    "legal_protection",
    "rental",
    "assistance"
]

# NETRISK_CASCO_FEATURES_ON_TOP = (['bonus_malus_casco', 'deductible_percentage','deductible_amount', 'payment_frequency',
#                                 'payment_method'] + NETRISK_CASCO_EQUIPMENT_COLS + NETRISK_CASCO_RIDERS)
NETRISK_CASCO_FEATURES_ON_TOP = []
NETRISK_CASCO_FEATURES_INFO = ['date_crawled', 'policy_start_date', 'vehicle_trim']

NETRISK_CASCO_CATEGORICAL_COLUMNS = ['vehicle_eurotax_code', 'bonus_malus_current', 'bonus_malus_casco',
                                     'vehicle_maker',
                                     'vehicle_model', 'vehicle_fuel_type', 'is_recent', 'payment_frequency',
                                     'payment_method']
FEATURES_TO_IGNORE = []
DEFAULT_TARGET_VARIABLES = {
    'netrisk_casco': ['ALFA_price',
                      'ALLIANZ_price',
                      'GENERALI_price',
                      'GENERTEL_price',
                      'GROUPAMA_price', 'K&AMP;H_price', 'KÖBE_price', 'MAGYAR_price',
                      'SIGNAL_price', 'UNION_price', 'UNIQA_price', 'GRÁNIT_price', 'OMINIMO_price'],
    'mubi': [
        "BEESAFE-(OC)-price",
        "TUZ-(OC),(NNW),(Assistance=100 km PL)-price",
        "BENEFIA-(OC),(NNW),(Assistance=150 km EU,After breakdown)-price",
        "BALCIA-(OC)-price",
        "WEFOX-(OC),(Assistance=150 km PL)-price",
        "ERGOHESTIA-(OC)-price",
        "UNIQA-(OC),(NNW),(Assistance=75 km PL,After breakdown,Replacement vehicle)-price",
        "TRASTI-(OC)-price",
        "GENERALI-(OC)-price",
        "MTU24-(OC)-price",
        "PROAMA-(OC)-price",
        "LINK4-(OC),(Assistance=100 km PL,Replacement vehicle)-price"
    ]
}

BONUS_MALUS_CLASSES_GOOD = ['B10', 'B09', 'B08', 'B07', 'B06', 'B05', 'B04', 'B03', 'B02', 'B01', 'A00', 'M01', 'M02',
                            'M03', 'M04']
BONUS_MALUS_CLASSES_BAD = ['B10', 'B9', 'B8', 'B7', 'B6', 'B5', 'B4', 'B3', 'B2', 'B1', 'A0', 'M1', 'M2', 'M3', 'M4']

BONUS_MALUS_CLASSES_DICT = dict(zip(BONUS_MALUS_CLASSES_BAD, BONUS_MALUS_CLASSES_GOOD))
BONUS_MALUS_CLASSES_DICT_INV = dict(zip(BONUS_MALUS_CLASSES_GOOD, BONUS_MALUS_CLASSES_BAD))

FORINT_TO_EUR = 0.0026

USUAL_TARGET_VARIABLES = ['K&AMP;H_price', 'ALFA_price', 'SIGNAL_price', 'KÖBE_price', 'GROUPAMA_price', 'UNIQA_price',
                          'GENERALI_price', 'ALLIANZ_price', 'MAGYAR_price', 'UNION_price', 'GRÁNIT_price',
                          'GENERTEL_price']

CURRENT_YEAR = datetime.today().year

QUANTILE_RANGE = [0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 1]

column_to_folder_mapping = {
    'ALFA_price': 'alfa_tables',
    'ALLIANZ_price': 'allianz_tables',
    'GENERALI_price': 'generali_tables',
    'GENERTEL_price': 'genertel_tables',
    'GROUPAMA_price': 'groupama_tables',
    'K&AMP;H_price': 'kh_tables',
    'KÖBE_price': 'alfa_tables',
    'MAGYAR_price': 'magyar_tables',
    'SIGNAL_price': 'si_tables',
    'UNION_price': 'union_tables',
    'UNIQA_price': 'uniqa_tables',
    'WÁBERER_price': 'granit_tables',
    'GRÁNIT_price': 'granit_tables',
}


MUBI_CATEGORICAL = ['vehicle_maker', 'vehicle_fuel_type', 'vehicle_parking_place',
                    'voivodeship', 'county',
                    'contractor_marital_status']

MUBI_FEATURES_INFO = ['id_case', 'policy_start_date', 'crawling_date',
                      'contractor_birth_date', 'contractor_personal_id',
                      'contractor_driver_licence_date',
                      'vehicle_make_year', 'vehicle_licence_plate',
                      'vehicle_trim','vehicle_eurotax_version',
                      'vehicle_infoexpert_model', 'vehicle_infoexpert_version']

MUBI_FEATURES_ON_TOP = []

MUBI_VEHICLE_VALUE_FEATURES = ['balcia_vehicle_value', 'beesafe_vehicle_value',
                               'benefia_vehicle_value', 'ergohestia_vehicle_value',
                               'generali_vehicle_value', 'link4_vehicle_value',
                               'mtu24_vehicle_value', 'proama_vehicle_value',
                               'trasti_vehicle_value', 'tuz_vehicle_value',
                               'uniqa_vehicle_value', 'wefox_vehicle_value',
                               'wiener_vehicle_value', 'ycd_vehicle_value']

MUBI_FEATURES_MODEL = ['vehicle_power', 'vehicle_engine_size', 'vehicle_weight_to_power_ratio',
                       'vehicle_net_weight', 'vehicle_gross_weight',
                       'vehicle_age'
                       'vehicle_number_of_seats', 'vehicle_number_of_doors',
                       'vehicle_steering_wheel_right',
                       'vehicle_imported', 'vehicle_imported_within_last_12_months'
                       'contractor_age', 'licence_at_age', 'driver_experience',
                       'contractor_mtpl_number_of_claims',
                       'postal_code_population', 'postal_code_area', 'postal_code_population_density',
                       'latitude', 'longitude',
                       ] + MUBI_VEHICLE_VALUE_FEATURES + MUBI_CATEGORICAL
