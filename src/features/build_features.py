# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import csv
import os

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

BONUS_MALUS_CLASSES = ['B10', 'B09', 'B08', 'B07', 'B06', 'B05', 'B04', 'B03', 'B02', 'B01', 'A00', 'M01', 'M02', 'M03', 'M04']


def prepareDir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
    for file in os.listdir(dir):
        os.remove(dir + file)


def detect_csv_delimiter(file_path):
    with open(file_path, 'r', newline='') as file:
        dialect = csv.Sniffer().sniff(file.read(1024))
        return dialect.delimiter


def read_file(file_path):
    file_extension = file_path.split('.')[-1].lower()
    if file_extension == 'csv':
        return pd.read_csv(file_path, sep=detect_csv_delimiter(file_path))
    elif file_extension in ['xls', 'xlsx']:
        return pd.read_excel(file_path)
    elif file_extension == 'json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def add_bracket_features(data, features, brackets_path):
    brackets = pd.read_json( brackets_path, orient='index').T

    for feature in brackets.columns:
        data[f'{feature}_cut'] = pd.cut(data[feature], brackets[feature].dropna(), labels = False).astype('category')
        data.drop(columns=[feature])
        features[features.index(feature)] = f'{feature}_cut'
    return data, features


def main(input_filepath, output_filepath, filename, only_convert_to_csv, index_col, price_annotation):
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Load the new data
    data = read_file(input_filepath)

    if only_convert_to_csv:
        data['BonusMalus'] = data['BonusMalus'].astype('category')
        data['CarMake'] = data['CarMake'].astype('category')
        processed_data_path = f'{output_filepath}{filename.split(".")[0]}_processed.csv'
        data.to_csv(processed_data_path)
        logger.info("Exported dataset to {}".format(processed_data_path))
        features_file_path = f'{output_filepath}{filename.split(".")[0]}_features.txt'
        with open(features_file_path, 'w') as file:

            feature_cols = [col for col in data.columns if '_price' not in col and col not in ['id_case', 'DateCrawled']]

            for feature in feature_cols:
                file.write(f"{feature},{str(data[feature].dtype)}\n")
        logger.info("Exported features to {}".format(features_file_path))

        return

    # Process the data
    price_columns = data.filter(regex=price_annotation, axis=1).columns.tolist()
    price_columns_mapping = {x: x.replace(price_annotation, '_price') for x in price_columns}

    NAME_MAPPING.update(price_columns_mapping)

    data = data.rename(NAME_MAPPING, axis=1)
    data = data.set_index(index_col)

    cols_to_use = []
    for value in NAME_MAPPING.values():
        if value in data.columns:
            cols_to_use.append(value)
    data = data[cols_to_use]
    data['BonusMalus'] = data['BonusMalus'].astype('category')
    data['CarMake'] = data['CarMake'].astype('category')

    # Save processed data to CSV
    processed_data_path = f'{output_filepath}{filename.split(".")[0]}_processed.csv'
    data.to_csv(processed_data_path)
    logger.info("Exported dataset to {}".format(processed_data_path))

    # Save features to a text file
    features_file_path = f'{output_filepath}{filename.split(".")[0]}_features.txt'
    with open(features_file_path, 'w') as file:
        feature_cols = [col for col in data.columns if '_price' not in col]
        for feature in feature_cols:
            file.write(f"{feature},{str(data[feature].dtype)}\n")

    logger.info("Exported features to {}".format(features_file_path))

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('filename')
@click.option('--only_convert_to_csv', is_flag=True, default=False)
@click.option('--index_col', default='id_case', help = 'annotation of index column (id_case, policyNr)')
@click.option('--price_annotation', default = '_price', help = 'annotation of price columns (_price, _newprice)')
def cli(input_filepath, output_filepath, filename, only_convert_to_csv, index_col, price_annotation):
    main(input_filepath, output_filepath, filename, only_convert_to_csv, index_col, price_annotation)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    cli()
