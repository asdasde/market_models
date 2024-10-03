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
NETRISK_CASCO_EQUIPMENT_COLS = ['polished_specialty','AirConditioning','AlloyWheels','AutomaticTransmission','LeatherSeats','NavigationSystems','XenonHeadlights','Camera','Sunroof','LedHeadlights','ShockAbsorber','UltraSonic','DrivingSupportSystem']
NETRISK_CASCO_FEATURES_ON_TOP = ['DeductiblePercentage','DeductibleAmount'] + NETRISK_CASCO_EQUIPMENT_COLS
NETRISK_CASCO_FEATURES_INFO = ['DateCrawled']


FEATURES_TO_IGNORE = []
DEFAULT_TARGET_VARIABLES = ['ALFA_price', 'ALLIANZ_price', 'GENERALI_price', 'GENERTEL_price',
                            'GROUPAMA_price', 'K&AMP;H_price', 'KÖBE_price', 'MAGYAR_price',
                            'SIGNAL_price', 'UNION_price', 'UNIQA_price', 'GRÁNIT_price']

BONUS_MALUS_CLASSES_GOOD = ['B10', 'B09', 'B08', 'B07', 'B06', 'B05', 'B04', 'B03', 'B02', 'B01', 'A00', 'M01', 'M02',
                            'M03', 'M04']
BONUS_MALUS_CLASSES_BAD = ['B10', 'B9', 'B8', 'B7', 'B6', 'B5', 'B4', 'B3', 'B2', 'B1', 'A0', 'M1', 'M2', 'M3', 'M4']

BONUS_MALUS_CLASSES_DICT = dict(zip(BONUS_MALUS_CLASSES_BAD, BONUS_MALUS_CLASSES_GOOD))
BONUS_MALUS_CLASSES_DICT_INV = dict(zip(BONUS_MALUS_CLASSES_GOOD, BONUS_MALUS_CLASSES_BAD))

FORINT_TO_EUR = 0.0026


USUAL_TARGET_VARIABLES = ['K&AMP;H_price', 'ALFA_price', 'SIGNAL_price', 'KÖBE_price', 'GROUPAMA_price', 'UNIQA_price',
                          'GENERALI_price', 'ALLIANZ_price', 'MAGYAR_price', 'UNION_price', 'GRÁNIT_price', 'GENERTEL_price']

CURRENT_YEAR = datetime.today().year

QUANTILE_RANGE = [0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 1]

column_to_folder_mapping = {
    'ALFA_price': 'aegon_tables',
    'ALLIANZ_price': 'allianz_tables',
    'GENERALI_price': 'generali_tables',
    'GENERTEL_price': 'genertel_tables',
    'GROUPAMA_price': 'groupama_tables',
    'K&AMP;H_price': 'kh_tables',
    'KÖBE_price': 'aegon_tables',
    'MAGYAR_price': 'magyar_tables',
    'SIGNAL_price': 'si_tables',
    'UNION_price': 'union_tables',
    'UNIQA_price': 'uniqa_tables',
    'WÁBERER_price': 'waberer_tables',
    'GRÁNIT_price' : 'waberer_tables',
}
