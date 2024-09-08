import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv, find_dotenv

from sqlalchemy.testing.plugin.plugin_base import logging


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.load_utils import *
from utilities.constants import *

import time
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import click
import zipfile
import shutil

def process_file(file, value_names : list, read_value_names : list, use_names : list) -> pd.Series:
    try:
        prof = pd.read_excel(file)
        prof = prof.set_index('name')

        value = prof.loc[value_names]['value']
        read_value = prof.loc[read_value_names]['ReadValue']
        use_value = prof.loc[use_names]['Use']

        return pd.concat([use_value, value, read_value], axis = 0)
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None


def load_crawled_profiles(files: list, transposed_config: dict) -> pd.DataFrame:
    logging.info(f"Starting to load profiles from {len(files)} files")

    vals = []
    idx = 0
    prev_time = time.time()

    value_names = transposed_config['relevant_names_value']
    read_value_names = transposed_config['relevant_names_read_value']
    use_names = transposed_config['relevant_names_use']

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, file, value_names, read_value_names, use_names): file for file in
                   files}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                vals.append(result)
                idx += 1
                if idx % 100 == 0:
                    logging.info(f"Processed {idx} files so far")
                    prev_time = time.time()

    transposed = pd.concat(vals, axis=1).T
    transposed.index.rename('id_case', inplace=True)
    transposed = transposed.rename(columns=transposed_config['renaming_dict'])
    transposed = transposed.loc[~transposed.apply(lambda row: row.astype(str).str.contains('skipped').any(), axis=1)]

    logging.info(f"Finished loading profiles. Processed {idx} valid profiles.")

    return transposed





def make_transposed_file_netrisk_casco(transposed : pd.DataFrame, other : dict) -> pd.DataFrame:


    other['full_trim_list'] = (other['full_trim_list']
                               [['eurotax_code', 'full_name']]
                               .drop_duplicates(subset=['full_name']))
    other['price'] = other['price'][['eurotax_code', 'new_price_1_gross']]
    other['price'] = other['price'].rename(columns={'new_price_1_gross': 'car_value'})
    other['price']['car_value'] = FORINT_TO_EUR * other['price']['car_value']

    geo_data_rename = {'latitude': 'Latitude', 'longitude': 'Longitude', 'postal_code': 'PostalCode'}
    other['hungary_postal_codes'] = other['hungary_postal_codes'][['postal_code', 'latitude', 'longitude']]
    other['hungary_postal_codes'] = other['hungary_postal_codes'].rename(columns=geo_data_rename)

    transposed['LicenseAge'] = transposed['LicenseAge'].astype(int) - transposed['Age'].astype(int)
    transposed['Age'] = CURRENT_YEAR - transposed['Age'].astype(int)
    transposed['CarAge'] = CURRENT_YEAR - transposed['CarAge'].astype(int)
    transposed['PostalCode'] = transposed['PostalCode'].astype(int)
    transposed['CarMake'] = transposed['CarMake'].apply(lambda x: x if x != 'VW' else 'VOLKSWAGEN')

    transposed['SpecificHtml'] = transposed['SpecificHtml'].apply(lambda x: x.replace('\n', '').replace('\t', ''))

    def extract_car_trim_from_html(row) -> list:

        soup = BeautifulSoup(row['SpecificHtml'], 'html.parser')

        lst = []
        cm_kw_kg = []
        for tag in soup.find_all('div', class_='col-xs-6 col-sm-3 j-r-leiras'):
            tag_text = tag.text.split(' ')
            if 'cm³' in tag_text:
                cm_kw_kg.append(tag_text[6])
            elif 'kW' in tag_text:
                cm_kw_kg.append(tag_text[6])
            elif 'kg' in tag_text and cm_kw_kg:
                cm_kw_kg.append(tag_text[6])
                lst.append(cm_kw_kg)
                cm_kw_kg = []
        full_model_name = None
        for i, tag in enumerate(soup.find_all('span', class_='reszlet1')):
            if i % 2 == 0 and (i // 2) + 1 == int(row['SpecificModelId']):
                full_model_name = tag.text
                break
        try:
            return lst[int(row['SpecificModelId']) - 1] + [full_model_name]
        except (IndexError, ValueError):
            return [-1, -1, -1, 1]

    transposed['cm_kw_kg_fn'] = transposed.apply(lambda row: extract_car_trim_from_html(row), axis=1)

    transposed['ccm'] = transposed['cm_kw_kg_fn'].apply(lambda x: int(x[0]))
    transposed['kw'] = transposed['cm_kw_kg_fn'].apply(lambda x: int(x[1]))
    transposed['kg'] = transposed['cm_kw_kg_fn'].apply(lambda x: int(x[2]))
    transposed['trim_name'] = transposed['cm_kw_kg_fn'].apply(lambda x: str(x[3]))

    transposed = transposed.drop(columns = ['cm_kw_kg_fn'])

    transposed = transposed[~(transposed['ccm'] == -1)]

    transposed['full_name'] = (transposed['CarMake'] + '_' +
                               transposed['trim_name'].apply(lambda x: '_'.join(x.split('  ')[ : 2])))

    transposed = pd.merge(pd.merge(transposed, other['full_trim_list'].drop_duplicates('full_name'), how='left', on='full_name'),
                          other['price'], how='left', on='eurotax_code')

    print(other['hungary_postal_codes'])
    transposed = pd.merge(transposed, other['hungary_postal_codes'], on='PostalCode', how='left')

    transposed['DeductiblePercentage'] = '10'
    transposed['DeductibleAmount'] = '100e'

    transposed['DeductiblePercentage'] = transposed['DeductiblePercentage'].apply(
        lambda x: (x.replace(' ', '').replace('%', ''))).astype(int)
    transposed['DeductibleAmount'] = transposed['DeductibleAmount'].apply(lambda x: x.replace('e', '000'))

    transposed['PostalCode2'] = transposed['PostalCode'].apply(lambda x: str(x)[: 2])
    transposed['PostalCode3'] = transposed['PostalCode'].apply(lambda x: str(x)[: 3])
    transposed['PostalCode'] = transposed['PostalCode'].astype(int)
    transposed['Latitude'] = transposed['Latitude'].astype(float)
    transposed['Longitude'] = transposed['Longitude'].astype(float)

    name_cols = [col for col in transposed.columns if col.startswith('Name')]

    def fix_prices(val):
        try:
            return float(val.replace(' ', '').replace('Ft', '').replace('.', '').split('<')[0])
        except Exception as e:
            return val

    def fix_names(val):
        try:
            return val.split(' ')[0]
        except AttributeError:
            return val

    for col in transposed.columns:
        if 'Price' in col:
            transposed[col] = transposed[col].apply(lambda x: fix_prices(x))
        if 'Name' in col:
            transposed[col] = transposed[col].apply(lambda x: fix_names(x))

    insurers = np.unique([str(val) for val in transposed[name_cols].values.flatten() if str(val) != 'nan'])
    price_cols = [col.upper() + "_price" for col in insurers]
    transposed[price_cols] = None

    def fill(row):
        for i in range(1, len(insurers) + 1):
            row[str(row['Name' + str(i)]).upper() + "_price"] = row['Price' + str(i)]
        return row

    transposed = transposed.apply(lambda row: fill(row), axis=1)

    transposed = transposed.drop(['Name' + str(i) for i in range(1, 13)] + ['Price' + str(i) for i in range(1, 13)],
                                   axis=1)
    return transposed

@click.command("make_transposed_file")
@click.option('--service', type=click.STRING)
@click.option('--profiles_name', type=click.STRING)
@click.option('--template_date', type=click.STRING, help='Template date in format YYYY_MM_DD')
def make_transposed_file(service: str, profiles_name: str, template_date: str):
    logging.info(f"Starting transposed file creation for service: {service}, profiles: {profiles_name}")

    profiles_after_crawling_zip_path = get_profiles_after_crawling_zip_path(profiles_name)
    data_raw_path = get_raw_data_path(profiles_name)
    template_config = load_transposed_config(service, template_date)

    unzip_dir = profiles_after_crawling_zip_path.parent / "unzipped_profiles"
    with zipfile.ZipFile(profiles_after_crawling_zip_path, 'r') as zip_ref:
        logging.info(f"Unzipping profiles from {profiles_after_crawling_zip_path}")
        zip_ref.extractall(unzip_dir)

    files = [file for file in unzip_dir.glob('*.xlsx')]
    transposed = load_crawled_profiles(files, template_config)

    logging.info("Cleaning up unzipped profiles directory")
    shutil.rmtree(unzip_dir)

    others = load_other(service)

    if service == 'netrisk_casco':
        transposed = make_transposed_file_netrisk_casco(transposed, others)
        transposed['DateCrawled'] = '_'.join(profiles_name.split('_')[-3:])
        transposed = transposed[template_config['final_columns']]

    logging.info(f"Saving transposed file to {data_raw_path}")
    transposed.to_csv(data_raw_path, index=False)

@click.group()
def cli():
    pass


cli.add_command(make_transposed_file)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    cli()