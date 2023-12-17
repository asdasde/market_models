# -*- coding: utf-8 -*-
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import glob
import pandas as pd
import numpy as np
import re
import random
import datetime
import utils
from bs4 import BeautifulSoup


CURRENT_YEAR = datetime.datetime.now().year


def range_to_list(x):
    n1, n2 = re.split(r'-|–', x)
    n1, n2 = int(n1), int(n2) + 1

    return list(range(n1, + n2))


def get_from_range(x):
    l = []
    if '+' in x:
        l1, l2 = x.split('+')
        l1, l2 = range_to_list(l1), range_to_list(l2)
        l = l1 + l2
    else:
        l = range_to_list(x)
    return random.choice(l)


get_from_range_v = np.vectorize(get_from_range)


def sample_univariate(cat, prob, num_samples=10000, is_car_make=False, range=True):
    if is_car_make:
        ind = np.arange(len(prob))
        res = np.column_stack(np.unravel_index(np.random.choice(ind, p=prob, size=num_samples), prob.shape))
        return cat[res].reshape(num_samples)

    ind = np.arange(len(prob))
    res = np.column_stack(np.unravel_index(np.random.choice(ind, p=prob, size=num_samples), prob.shape))
    if not range:
        return cat[res].reshape(num_samples)
    ret = get_from_range_v(cat[res])
    return ret.reshape(num_samples)


def getCarAgeAlternative(samples):
    ages = []
    for i in range(samples):
        youngOrOld = random.random() < 0.8
        cAge = -1
        if youngOrOld:
            cAge = random.choice([1, 2, 3, 4, 5])
        else:
            cAge = random.choice([6, 7, 8, 9, 10])
        ages.append(cAge)
    return ages


def get_car_model(prof, cars):
    n1, n2 = map(int, re.split(r'-|–', prof['kw']))
    n1 = int(n1)
    n2 = int(n2)
    kw_min, kw_max = min(n1, n2), max(n1, n2)

    subset = cars[(cars['car_make_year'] == CURRENT_YEAR - prof['CarAge']) &
                  (cars['car_make'] == prof['CarMake'].upper()) &
                  (cars['kw'].astype(int).between(kw_min, kw_max))]

    if not subset.empty:
        return subset[['car_model', 'car_trim_id', 'kw', 'ccm', 'kg', 'car_value']].sample(n=1).iloc[0].values
    else:
        return np.array([-1, -1, -1, -1, -1, -1])


def bm(val):
    if val[0] == 'B':
        return 11 - int(val[1:])
    if val[0] == 'A':
        return 11
    else:
        return 11 + int(val[1:])



def sample_profiles(samples, params, others, policy_start_date):
    profiles = pd.DataFrame(index=range(samples),
                            columns=['Age', 'PostalCode', 'BonusMalus', 'kw', 'CarMake', 'CarAge'])
    profiles = profiles.dropna()

    while len(profiles) < samples:
        sample = pd.DataFrame(index=range(samples),
                              columns=['Age', 'PostalCode', 'BonusMalus', 'kw', 'CarMake', 'CarAge', 'car_value'])
        sample['Age'] = sample_univariate(params['age_params']['cat'].values, params['age_params']['prob'].values,
                                          num_samples=samples)
        sample['PostalCode'] = sample_univariate(params['postal_code_params']['cat'].values,
                                                 params['postal_code_params']['prob'].values, num_samples=samples)
        sample['BonusMalus'] = sample_univariate(params['bonus_malus_params']['cat'].values,
                                                 params['bonus_malus_params']['prob'].values, num_samples=samples,
                                                 range=False)
        sample['kw'] = sample_univariate(params['power_params']['cat'].values, params['power_params']['prob'].values,
                                         num_samples=samples, range=False)


        sample['CarMake'] = sample_univariate(params['car_make_params']['cat'].values,
                                               params['car_make_params']['prob'].values, num_samples=samples,
                                               is_car_make=True, range=False)
        sample['CarAge'] = getCarAgeAlternative(samples)
        cars = others['netrisk_cars'][others['netrisk_cars']['car_make_year'] >= CURRENT_YEAR - sample['CarAge'].max()]
        car_model_data = np.array([get_car_model(x, cars) for _, x in sample.iterrows()])

        sample[['CarModel', 'CarModelSpecific', 'kw', 'ccm', 'kg', 'car_value']] = car_model_data

        sample['DateCrawled'] = policy_start_date

        sample = sample.dropna(subset=['CarModel'])
        sample = sample[~(sample['kw'] == -1)]
        sample = sample[sample['PostalCode'].isin(others['hungary_postal_codes']['postal_code'].tolist())]
        sample = pd.merge(sample, others['hungary_postal_codes'][['postal_code', 'latitude', 'longitude']],
                          left_on='PostalCode', right_on='postal_code', how='left').rename(
            columns={'latitude': 'Latitude', 'longitude': 'Longitude'})
        sample = pd.merge(sample, others['aegon_postal_categories'])
        sample['CarMake'] = sample['CarMake'].str.upper()
        sample = sample[
            ~((sample['BonusMalus'] > 'B5') & (sample['BonusMalus'] < 'M1') & (sample['Age'] < 29))]
        sample.index = range(len(sample))
        profiles = pd.concat([profiles, sample], axis=0)

    profiles.index = range(len(profiles))
    profiles = profiles[profiles.index < samples]
    profiles['Age'] = profiles['Age'].astype(int)
    profiles['LicenseAge'] = 18
    profiles['BonusMalus'] = profiles['BonusMalus'].astype('category')
    profiles['CarAge'] = profiles['CarAge'].astype(int)
    profiles['CarMake'] = profiles['CarMake'].astype('category').replace({'VOLKSWAGEN' : 'VW'})
    profiles['ccm'] = profiles['ccm'].astype(int)
    profiles['kw'] = profiles['kw'].astype(int)
    profiles['kg'] = profiles['kg'].astype(int)
    profiles['car_value'] = profiles['car_value'].astype(float) * utils.FORINT_TO_EUR
    profiles['PostalCode2'] = profiles['PostalCode'].apply(lambda x: str(x)[: 2]).astype(float)
    profiles['PostalCode3'] = profiles['PostalCode'].apply(lambda x: str(x)[: 3]).astype(float)
    profiles['PostalCode'] = profiles['PostalCode'].astype(int)
    profiles['Latitude'] = profiles['Latitude'].astype(float)
    profiles['Longitude'] = profiles['Longitude'].astype(float)
    profiles['CarMakerCategory'] = 1
    profiles['DateCrawled'] = profiles['DateCrawled'].astype('category')
    profiles['isRecent'] = profiles['DateCrawled'].apply(lambda x: x.split('_')[1] > '04')
    profiles['BonusMalusCode'] = profiles['BonusMalus'].apply(lambda x: bm(x))

    profiles = profiles[['isRecent', 'CarMake', 'CarAge', 'ccm', 'kw', 'kg', 'car_value', 'CarMakerCategory', 'PostalCode',
              'PostalCode2', 'PostalCode3', 'Category', 'Longitude', 'Latitude', 'Age', 'LicenseAge', 'BonusMalus',
              'BonusMalusCode']]

    return profiles

def load_distribution(service, params_v):


    params_path = utils.get_params_path(service, params_v)
    other_path = utils.get_others_path(service)

    params = {}
    for param_file in glob.glob(f'{params_path}*.csv'):
        param = param_file.split('/')[-1].split('.')[0]
        params[param] = pd.read_csv(param_file)

    others = {}
    for other_file in glob.glob(f'{other_path}*'):
        other = other_file.split('/')[-1].split('.')[0]
        ext = other_file.split('.')[-1]
        if ext == 'csv':
            others[other] = pd.read_csv(other_file)
        elif ext == 'xlsx':
            others[other] = pd.read_excel(other_file)
        else:
            others[other] = pd.read_table(other_file, header=None, usecols=[1])

    return params, others


@click.command(name='sample_crawling_data')
@click.option('--service', default='netrisk_casco', type=click.STRING)
@click.option('--params_v', default='v1', type=click.STRING)
@click.option('--policy_start_date', default=None, type=click.STRING, help='Should be in YYYY_MM_DD format.')
@click.option('--n', default=1000, type=click.INT, help='Number of profiles to sample.')
def sample_crawling_data(service, params_v, policy_start_date, n):
    params, others = load_distribution(service, params_v)

    sampled_data = sample_profiles(n, params, others, policy_start_date)

    print(sampled_data.head())


    data_name = 'netrisk_casco_2023_11_14__2023_11_20__2023_12_12'
    target_variable = 'ALFA_price'

    model_name = utils.get_error_model_name(data_name, target_variable)
    model_path = utils.get_error_model_path(model_name)

    model = utils.load_model(model_path)

    predictions = utils.predict(model, sampled_data)
    predictions[predictions < 0.8] = 0
    predictions[predictions > 0.8] = 1
    print(predictions)

    print(sampled_data.loc[predictions])


@click.group()
def cli():
    pass


cli.add_command(sample_crawling_data)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    cli()
