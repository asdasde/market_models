import sys
import os
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
import paramiko

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

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
                  (cars['kw'].between(kw_min, kw_max))]

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


def sample_profiles(samples, params, others, policy_start_date, error_model):
    profile_columns_dtypes = utils.NETRISK_CASCO_DTYPES
    profile_columns = list(profile_columns_dtypes.keys())

    logging.info("Started generating samples")

    samples_list = []
    tot = 0
    cnt = 0

    samples *= 3
    while tot < samples / 3:

        sample = pd.DataFrame()
        sample['Age'] = sample_univariate(params['age_params']['cat'].values, params['age_params']['prob'].values,
                                          num_samples=samples)

        sample['isRecent'] = True

        sample['LicenseAge'] = 18

        sample['PostalCode'] = sample_univariate(params['postal_code_params']['cat'].values,
                                                 params['postal_code_params']['prob'].values, num_samples=samples)

        sample['BonusMalus'] = sample_univariate(params['bonus_malus_params']['cat'].values,
                                                 params['bonus_malus_params']['prob'].values, num_samples=samples,
                                                 range=False)

        sample['BonusMalusCode'] = sample['BonusMalus'].apply(lambda x: bm(x))

        sample['kw'] = sample_univariate(params['power_params']['cat'].values, params['power_params']['prob'].values,
                                         num_samples=samples, range=False)

        sample['CarMake'] = sample_univariate(params['car_make_params']['cat'].values,
                                              params['car_make_params']['prob'].values, num_samples=samples,
                                              is_car_make=True, range=False)

        sample['CarAge'] = getCarAgeAlternative(samples)

        cars = others['netrisk_cars'][others['netrisk_cars']['car_make_year'] >= CURRENT_YEAR - sample['CarAge'].max()]
        cars = cars[cars['car_make'].isin(sample['CarMake'].apply(lambda x: x.upper()).unique().tolist())]
        car_model_data = np.array([get_car_model(x, cars) for _, x in sample.iterrows()])
        sample[['CarModel', 'CarModelSpecific', 'kw', 'ccm', 'kg', 'car_value']] = car_model_data

        sample = sample.dropna(subset=['CarModel'])
        sample['CarMake'] = sample['CarMake'].replace({'VOLKSWAGEN': 'VW'})

        sample = sample[~(sample['kw'] == -1)]

        sample['car_value'] = sample['car_value'] * utils.FORINT_TO_EUR

        sample = sample[sample['PostalCode'].isin(others['hungary_postal_codes']['postal_code'].tolist())]
        sample = pd.merge(sample, others['hungary_postal_codes'][['postal_code', 'latitude', 'longitude']],
                          left_on='PostalCode', right_on='postal_code', how='left').rename(
            columns={'latitude': 'Latitude', 'longitude': 'Longitude'})

        sample = pd.merge(sample, others['aegon_postal_categories'], left_on='PostalCode', right_on='PostalCode',
                          how='left')
        sample['Category'] = sample['Category'].fillna(8)

        sample['PostalCode2'] = sample['PostalCode'].apply(lambda x: str(x)[: 2])
        sample['PostalCode3'] = sample['PostalCode'].apply(lambda x: str(x)[: 3])

        sample['CarMake'] = sample['CarMake'].str.upper()
        sample = sample[~((sample['BonusMalus'] > 'B05') & (sample['BonusMalus'] < 'M01') & (sample['Age'] < 29))]
        sample['CarMakerCategory'] = 1

        for feature, dtype in profile_columns_dtypes.items():
            sample[feature] = sample[feature].astype(dtype)

        predictions = utils.predict(error_model, sample[profile_columns])
        predictions = utils.apply_threshold(predictions, np.percentile(predictions, 85))

        sample = sample.loc[predictions]
        samples_list.append(sample)

        tot += len(sample)
        cnt += 1

        logging.info(f"Generated {tot}/{samples // 3} profiles, in {cnt} tries")

    profiles = pd.concat(samples_list).iloc[: samples // 3]
    profiles['PolicyStartDate'] = policy_start_date
    profiles.index = range(len(profiles))
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
@click.option('--error_model_name', type=click.STRING)
@click.option('--service', default='netrisk_casco', type=click.STRING, help='Currently only supports netrisk casco')
@click.option('--params_v', default='v1', type=click.STRING, help='Use it to switch between versions of distributions.')
@click.option('--policy_start_date', default=None, type=click.STRING, help='Should be in YYYY_MM_DD format.')
@click.option('--n', default=1000, type=click.INT, help='Number of profiles to sample.')
def sample_crawling_data(error_model_name, service, params_v, policy_start_date, n):
    error_model_path = utils.get_model_path(error_model_name)
    print(error_model_path.split('/'))
    error_model = utils.load_model(error_model_path)
    logging.info("Loaded the error model.")

    params, others = load_distribution(service, params_v)
    logging.info("Loaded the distributions.")

    sampled_data = sample_profiles(n, params, others, policy_start_date, error_model)

    sampled_data_name = "sampled_data"
    utils.prepareDir(utils.get_profiles_for_crawling_dir("netrisk_casco", sampled_data_name))
    sampled_data_path = utils.get_profiles_for_crawling_transposed("netrisk_casco", sampled_data_name)
    sampled_data.to_csv(sampled_data_path)
    logging.info(f"Exported sampled data to {sampled_data_path}")


def replace_iii(row):
    val = row['value']
    try:
        val = int(float(val))
        row['value'] = val
    except Exception as e:
        pass
    val = str(val)
    row['tag'] = row['tag'].replace('iii', val)
    return row


def export_profile(profile: pd.DataFrame, template: pd.DataFrame, indicies: dict, row_values: list,
                   rows_to_not_use: list, profiles_export_path: str = None):
    prof = template
    for row, prof_col, expr in row_values:
        prof.at[indicies[row], 'value'] = expr(profile[prof_col]) if prof_col is not None else expr(1)
    for row in rows_to_not_use:
        prof.at[indicies[row], 'Use'] = False
    if profiles_export_path is not None:
        prof = prof.apply(lambda row: replace_iii(row), axis=1)
        prof['id_case'] = profile.name
        prof = prof.drop('Unnamed: 0', axis=1, errors='ignore')
        prof.to_csv(profiles_export_path + str(profile.name) + '.csv', index=False)
    return prof


@click.command(name='export_data_for_crawling')
@click.option("--service", required=True, type=click.STRING, help='Service name (example netrisk_casco).')
@click.option('--template_date', required=True, type=click.STRING, help='Date in the file name of the template')
def export_data_for_crawling(service: str, template_date: str):
    profiles_export_path = utils.get_profiles_for_crawling_dir(service, "sampled_data")  # promeniti u ID fajla
    data_path = utils.get_profiles_for_crawling_transposed(service, "sampled_data")  # promeniti u ID fajla
    zip_path = utils.get_profiles_for_crawling_zip_path(service, "sampled_data")
    template_path = utils.get_template_path(service, template_date)
    row_values_path = utils.get_row_values_path(service, template_date)

    if not os.path.exists(data_path):
        logging.info("No sampled data, please generate it first ...")
        logging.info("Aborting ...")
        return

    data = utils.read_file(data_path)
    template = utils.read_file(template_path)
    with open(row_values_path, 'r') as row_values:
        row_values = eval(' '.join(row_values.readlines()))

    indices = dict(zip(template['name'], template['id']))

    files_list = []
    for i in range(len(data)):
        export_profile(data.iloc[i], template, indices, row_values, [], profiles_export_path)
        files_list.append(f'{profiles_export_path}{i}.csv')

    utils.zip_list_of_files(files_list, zip_path)
    for file in files_list:
        os.remove(file)

    send_profiles_to_the_server(zip_path, "crawler-mocha/netirks_casco_sampled_data.zip", utils.get_remote_queue_path(),
                                utils.get_remote_crawler_path())


def send_profiles_to_the_server(local_zip_path, remote_zip_path, remote_unzip_path, remote_script_path):
    try:
        ssh = paramiko.SSHClient()
        private_key = paramiko.RSAKey(filename=utils.PRIVATE_KEY_PATH)
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=utils.REMOTE_HOST_NAME, username='root', pkey=private_key, allow_agent=True,
                    look_for_keys=True)

        sftp = ssh.open_sftp()
        sftp.put(local_zip_path, remote_zip_path)
        sftp.close()

        unzip_command = f"unzip -o {remote_zip_path} -d {remote_unzip_path}"
        stdin, stdout, stderr = ssh.exec_command(unzip_command)
        print(stdout.read().decode("utf-8"))

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        ssh.close()


def zip_and_fetch_profiles_from_the_server(remote_profiles_path, remote_zip_path, local_zip_path):
    try:
        ssh = paramiko.SSHClient()
        private_key = paramiko.RSAKey(filename=utils.PRIVATE_KEY_PATH)
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=utils.REMOTE_HOST_NAME, username='root', pkey=private_key, allow_agent=True,
                    look_for_keys=True)

        zip_command = f"zip -jr {remote_zip_path} {remote_profiles_path}/*"
        stdin, stdout, stderr = ssh.exec_command(zip_command)
        print(stdout.read().decode("utf-8"))

        sftp = ssh.open_sftp()
        sftp.get(remote_zip_path, local_zip_path)
        sftp.close()

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        ssh.close()


@click.command(name="fetch_profiles")
def fetch_profiles_from_the_server():
    zip_and_fetch_profiles_from_the_server(utils.get_remote_profiles_path(),
                                           utils.get_remote_profiles_after_crawling_zip_path("netrisk_casco"),
                                           utils.get_profiles_after_crawling_zip_path("netrisk_casco", "sampled_data"))


@click.group()
def cli():
    pass


cli.add_command(sample_crawling_data)
cli.add_command(export_data_for_crawling)
cli.add_command(fetch_profiles_from_the_server)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    cli()
