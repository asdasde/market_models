import os
import csv
import json
import zipfile

import numpy as np
import pandas as pd

from datetime import datetime


def prepare_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    else:
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)


def detect_csv_delimiter(file_path: str) -> property:
    with open(file_path, 'r', newline='') as file:
        dialect = csv.Sniffer().sniff(file.read(1024))
        return dialect.delimiter


def read_file(file_path: str) -> pd.DataFrame:
    file_extension = file_path.split('.')[-1].lower()
    if file_extension == 'csv':
        return pd.read_csv(file_path)
    elif file_extension in ['xls', 'xlsx']:
        return pd.read_excel(file_path)
    elif file_extension == 'json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def zip_list_of_files(file_paths: list, zip_file_path: str) -> None:
    with zipfile.ZipFile(zip_file_path, 'w') as zipMe:
        for file in file_paths:
            arcname = os.path.basename(file)
            zipMe.write(file, arcname=arcname, compress_type=zipfile.ZIP_DEFLATED)


def get_latest(path):
    date_format = "%Y_%B_%d"
    max_date = '9999_December_31'
    max_date = datetime.strptime(max_date, date_format)
    return max([datetime.strptime(x, date_format) for x in os.listdir(path) if
                datetime.strptime(x, date_format) <= max_date]).strftime(date_format)


def dict_to_json(dictionary: dict, output_path: str):
    serializable_dict = dictionary.copy()
    for k, v in dictionary.items():
        if isinstance(v, np.int64):
            serializable_dict[k] = int(v)

    with open(output_path, 'w') as f:
        json.dump(serializable_dict, f, indent=4)
