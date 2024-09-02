from pathlib import Path
import csv
import json
import zipfile

import numpy as np
import pandas as pd

from datetime import datetime


def prepare_dir(dir_path: Path):
    dir_path = Path(dir_path)  # Ensure dir_path is a Path object
    if not dir_path.exists():
        dir_path.mkdir(parents=True)
    else:
        for file in dir_path.iterdir():
            if file.is_file():
                file.unlink()


def detect_csv_delimiter(file_path: Path) -> str:
    with file_path.open('r', newline='') as file:
        dialect = csv.Sniffer().sniff(file.read(1024))
        return dialect.delimiter


def read_file(file_path: Path) -> pd.DataFrame:
    file_path = Path(file_path)  # Ensure file_path is a Path object
    file_extension = file_path.suffix.lower()
    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    elif file_extension == '.json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def zip_list_of_files(file_paths: list, zip_file_path: Path) -> None:
    zip_file_path = Path(zip_file_path)  # Ensure zip_file_path is a Path object
    with zipfile.ZipFile(zip_file_path, 'w') as zipMe:
        for file in file_paths:
            file = Path(file)  # Ensure each file in file_paths is a Path object
            arcname = file.name
            zipMe.write(file, arcname=arcname, compress_type=zipfile.ZIP_DEFLATED)


def get_latest(path: Path) -> str:
    path = Path(path)  # Ensure path is a Path object
    date_format = "%Y_%B_%d"
    max_date = '9999_December_31'
    max_date = datetime.strptime(max_date, date_format)
    return max([datetime.strptime(x.name, date_format) for x in path.iterdir() if
                datetime.strptime(x.name, date_format) <= max_date]).strftime(date_format)


def dict_to_json(dictionary: dict, output_path: Path):
    serializable_dict = dictionary.copy()
    for k, v in dictionary.items():
        if isinstance(v, np.int64):
            serializable_dict[k] = int(v)

    output_path = Path(output_path)  # Ensure output_path is a Path object
    with output_path.open('w') as f:
        json.dump(serializable_dict, f, indent=4)
