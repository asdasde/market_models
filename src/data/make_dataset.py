# -*- coding: utf-8 -*-
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

INPUT_DISTRIBUTION_PATH = '../data/external/distributions/'

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


def sample_univariate(cat, prob, num_samples = 10000, is_car_make = False, range = True):

    if is_car_make:
        ind = np.arange(len(prob))
        res = np.column_stack(np.unravel_index(np.random.choice(ind, p=prob, size= num_samples), prob.shape))
        return cat[res].reshape(num_samples)

    ind = np.arange(len(prob))
    res = np.column_stack(np.unravel_index(np.random.choice(ind, p=prob, size= num_samples), prob.shape))
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


def get_car_model(prof, manu):
    manu_set = manu.loc[(manu['Manufacturer'] == prof['CarMaker'].upper()) & (manu['MakeYear'] == CURRENT_YEAR - prof['CarAge'])]
    search = set(range_to_list(prof['kW']))
    try:
        manu_set = manu_set[[bool(search.intersection(eval(l))) for l in manu_set['kW']]]
        manu_set = manu_set.iloc[0]
    except Exception as e:
        return pd.Series([np.nan, np.nan])
    kws = eval(manu_set['kW'])
    for i in range(len(kws)):
        if kws[i] in range_to_list(prof['kW']):
            return pd.Series([manu_set['CarModel'], i + 1, kws[i]])


def sample_profiles(num_samples, params, others, policy_start_date):
    profiles = pd.DataFrame(index=range(num_samples),
                            columns=['Age', 'PostalCode', 'BonusMalus', 'kW', 'CarMaker', 'CarAge'])
    profiles = profiles.dropna()

    while len(profiles) < num_samples:

        sample = pd.DataFrame(index=range(num_samples),
                                columns=['Age', 'PostalCode', 'BonusMalus', 'kW', 'CarMaker', 'CarAge'])
        sample['Age'] = sample_univariate(params['age_params']['cat'].values, params['age_params']['prob'].values,
                                     num_samples=num_samples)
        sample['PostalCode'] = sample_univariate(params['postal_code_params']['cat'].values,
                                            params['postal_code_params']['prob'].values, num_samples=num_samples)
        sample['BonusMalus'] = sample_univariate(params['bonus_malus_params']['cat'].values,
                                            params['bonus_malus_params']['prob'].values, num_samples=num_samples, range=False)
        sample['kW'] = sample_univariate(params['power_params']['cat'].values, params['power_params']['prob'].values,
                                    num_samples=num_samples, range=False)
        sample['CarMaker'] = sample_univariate(params['car_make_params']['cat'].values, params['car_make_params']['prob'].values,
                                          num_samples=num_samples, is_car_make=True, range=False)
        sample['CarAge'] = getCarAgeAlternative(num_samples)
        sample['CarModel'] = np.nan
        sample['CarModelSpecific'] = np.nan
        sample[['CarModel', 'CarModelSpecific', 'kW']] = sample.apply(lambda x: get_car_model(x, others['ModelsSpecific']), axis=1)
        sample['PolicyStartDate'] = policy_start_date

        sample = sample.dropna(subset=['CarModel'])
        sample = sample[sample['PostalCode'].isin(list(others['HUplz'].iloc[:, 0].values))]
        sample['CarMaker'] = sample['CarMaker'].str.upper()
        sample = sample[~((sample['BonusMalus'] > 'B5') & (sample['BonusMalus'] < 'M1') & (sample['Age'] < 29))]


        profiles = pd.concat([profiles, sample], axis = 0)

    profiles.index = range(len(profiles))
    profiles = profiles[profiles.index < num_samples]
    return profiles


def load_distribution(service, params_v):
    params_path = f'{INPUT_DISTRIBUTION_PATH}{service}/params/{params_v}/'
    other_path = f'{INPUT_DISTRIBUTION_PATH}{service}/other/'

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
            print(other_file)
            others[other] = pd.read_table(other_file, header=None, usecols=[1])

    return params, others

@click.command(name = 'sample_crawling_data')
@click.option('--service', default = 'netrisk_casco', type = click.STRING)
@click.option('--params_v', default = 'v1', type = click.STRING)
@click.option('--policy_start_date', default = None, type = click.STRING, help = 'Should be in YYYY_MM_DD format.')
@click.option('--n', default = 1000, type = click.INT, help = 'Number of profiles to sample.')
def sample_crawling_data(service, params_v, policy_start_date, n):

    params, others = load_distribution(service, params_v)

    sampled_data = sample_profiles(n, params, others, policy_start_date)

    print(sampled_data)

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
