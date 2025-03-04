import click
from data_processors import *
from utilities.export_utils import *

def handle_names(names : str, names_file_name : str, logger : logging.Logger)->List:
    if bool(names) == bool(names_file_name):
        logger.error('You must provide either --names or --names_file_name, but not both.')
        raise ValueError('You must provide either --names or --names_file_name, but not both.')
    if names_file_name:
        logger.info(f'Loading names from file: {names_file_name}')
        names = LoadManager.load_names_file(names_file_name)
    else:
        names = names.split(',')
    return names

DATA_SOURCE_HANDLER = {
    "crawler": make_processed_crawler_data,
    "generator": make_processed_generator_data,
    "signal_iduna": make_processed_signal_iduna_data,
    "quotes_data": make_processed_netrisk_like_data,
    "netrisk_bought_data":  make_processed_netrisk_casco_like_data,
    "punkta_data":  make_processed_punkta_data,
    "zmarta_data":  make_processed_zmarta_data,
    'mubi': make_processed_mubi_data,
}

def check_duplicates(data_name_reference : dict, new_entry : dict):
    for entry in data_name_reference.values():
        c1 = set(new_entry['raw_data_used']) == set(entry['raw_data_used'])
        c2 = new_entry['service'] == entry['service']
        c3 = 'encoding_type' in entry.keys() and new_entry['encoding_type'] == entry["encoding_type"]
        c4 = new_entry['num_rows'] == entry['num_rows']
        c5 = new_entry['features_model'] == entry['features_model']
        if c1 and c2 and c3 and c4 and c5:
            print(entry['processed_name'])
            return True
    return False


@click.command(name='make_processed_data')
@click.option('--service', required=False, default='netrisk_casco')
@click.option('--names', type=click.STRING, help='List of dates separated by ",".')
@click.option('--names_file_name', type=click.STRING, help='Relative file path to a file containing names.')
@click.option('--data_source', type=click.STRING,
              help='Currently supported data sources are (crawler, signal_iduna)')
@click.option('--benchmark', default=False, type=click.BOOL,
              help='Signals that it is purely for benchmarking purposes, and should not be used for training purposes.')
@click.option('--encoding_type', type=click.STRING, default = 'xgb_categorical', help='Currently only xgboost category')
@click.option('--short_description', type=click.STRING, default = None)
def make_processed_data(service: str, names: str, names_file_name: str, data_source: str, benchmark: bool
                        , encoding_type : str, short_description : str) -> None:


    path_manager = PathManager(service)
    load_manager = LoadManager(path_manager)
    export_manager = ExportManager(path_manager)

    logger = logging.getLogger(__name__)

    names = handle_names(names, names_file_name, logger)

    service += '_'
    names = set(names)

    logger.info('Started finding name for new file ...')
    processed_data_name = find_first_available_name(path_manager, benchmark)

    logger.info(f'Found name {processed_data_name}')
    logger.info("Loading data name reference")
    data_name_reference = LoadManager.load_data_name_reference()

    logger.info("Loading raw data")

    extension = '.csv'
    if data_source in ['quotes_data', 'punkta_data', 'netrisk_bought_data', 'mubi']:
        extension = '.parquet'

    paths = [path_manager.get_raw_data_path(f'{service}{name}', extension=extension) for name in names]
    datas = [read_data_frame(path) for path in paths]

    processing_function = DATA_SOURCE_HANDLER.get(data_source, None)
    if processing_function is None:
        logger.info("Currently unsupported data source, aborting ...")
        return

    data, features_info, features_on_top, features_model, target_variables = processing_function(datas,
                                                                                                 data_name_reference,
                                                                                                 encoding_type,
                                                                                                 path_manager,
                                                                                                 load_manager)

    features_model = extract_features_with_dtype(data, features_model)

    new_entry = {
        'service': service,
        'data_source' : data_source,
        'short_description' : short_description,
        'raw_data_used': list(names),
        'processed_data_used': [],
        'processed_name': processed_data_name,
        'num_rows': len(data),
        'date_processed': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'file_size_mb': None,
        'is_benchmark': benchmark,
        'index_column_name': data.index.name,
        'encoding_type' : encoding_type,
        'features_info': features_info,
        'features_on_top': features_on_top,
        'features_model': features_model,
        'target_variables': target_variables,
    }

    logger.info("Checking in data name reference for duplicate datasets...")

    duplicate_found = check_duplicates(data_name_reference, new_entry)

    if duplicate_found:
        logger.info("Duplicate found, aborting ...")
        return

    logger.info("No duplicates found")

    logger.info("Exporting processed data and feature file")

    processed_data_path = path_manager.get_processed_data_path(processed_data_name)
    data.to_parquet(processed_data_path)

    file_size_mb = processed_data_path.stat().st_size / 1_048_576  # Convert bytes to MB
    file_size_mb = f"{file_size_mb:.1f}"  # Format to 1 decimal place
    new_entry['file_size_mb'] = file_size_mb

    logger.info("Adding it to the data name reference")

    data_name_reference[processed_data_name] = new_entry
    ExportManager.export_data_name_reference(data_name_reference)
    logger.info("Data name reference updated successfully.")
    logger.info('Checking if everything went well by trying to load the data ...')
    data, features_info, features_on_top, features_model = load_manager.load_data(processed_data_name)
    logger.info('All good')


@click.command(name='merge_processed_data')
@click.option('--service', required=False, default='netrisk_casco')
@click.option('--names', type=click.STRING, help='List of dates separated by ",".')
@click.option('--names_file_name', type=click.STRING, help='Relative file path to a file containing names.')
def merge_processed_datas(service: str, names: str, names_file_name: str):

    path_manager = PathManager(service)
    load_manager = LoadManager(path_manager)
    export_manager = ExportManager(path_manager)

    logger = logging.getLogger(__name__)
    names = handle_names(names, names_file_name, logger)

    service += '_'
    names = set(names)

    logger.info('Started finding name for new file ...')
    processed_data_name = find_first_available_name(path_manager, False)

    logger.info(f'Found name {processed_data_name}')
    logger.info("Loading data name reference")
    data_name_reference = LoadManager.load_data_name_reference()

    logger.info("Checking in data name reference for duplicate datasets...")

    duplicate_found = any(
        names == set(entry['processed_data_used'])
        for entry in data_name_reference.values()
    )

    if duplicate_found:
        logger.info("Duplicate found, aborting ...")
        return

    logger.info("No duplicates found")
    logger.info("Loading raw data")

    datas = []
    features_info_set = None
    features_on_top_set = None
    features_model_set = None

    for name in names:
        data_full_name = f'{service}{name}'
        data, _, _, _ = load_manager.load_data(data_full_name)

        features_model = data_name_reference[data_full_name]['features_model']
        features_on_top = data_name_reference[data_full_name]['features_on_top']
        features_info = data_name_reference[data_full_name]['features_info']

        if features_info_set is None:
            features_info_set = features_info
            features_model_set = features_model
            features_on_top_set = features_on_top
            print(features_info_set, features_on_top)
        else:
            if features_info != features_info_set:
                logger.error(f"Features info mismatch for {name}")
                raise ValueError("Inconsistent features_info across datasets")
            if features_on_top != features_on_top_set:
                logger.error(f"Features on top mismatch for {name}")
                raise ValueError("Inconsistent features_on_top across datasets")
            if sorted(features_model) != sorted(features_model_set):
                logger.error(f"Features model mismatch for {name}")
                logger.error(f"{set(features_model).difference(set(features_model_set))}")
                logger.error(f"{set(features_model_set).difference(set(features_model))}")
                raise ValueError("Inconsistent features_model across datasets")

        datas.append(data)

    categorical_columns = datas[0].select_dtypes(include=['category']).columns
    data = pd.concat(datas)
    data[categorical_columns] = data[categorical_columns].astype('category')

    logger.info("Exporting processed data and feature file")
    processed_data_path = path_manager.get_processed_data_path(processed_data_name)
    data.to_parquet(processed_data_path)

    target_variables = get_target_variables(data.columns)

    logger.info("Adding it to the data name reference")
    file_size_mb = processed_data_path.stat().st_size / 1_048_576  # Convert bytes to MB
    file_size_mb = f"{file_size_mb:.1f}"  # Format to 1 decimal place

    print(features_info_set)
    print(features_on_top_set)

    data_name_reference[processed_data_name] = {
        'service': service,
        'raw_data_used': [],
        'processed_data_used': list(names),
        'processed_name': processed_data_name,
        'num_rows': len(data),
        'date_processed': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'data_source': 'processed_data',
        'file_size_mb': file_size_mb,
        'is_benchmark': False,
        'index_column_name': data.index.name,
        'features_info': features_info_set,
        'features_on_top': features_on_top_set,
        'features_model': features_model_set,
        'target_variables': target_variables
    }

    ExportManager.export_data_name_reference(data_name_reference)
    logger.info("Data name reference updated successfully.")
    logger.info('Checking if everything went well by trying to load the data ...')
    data, features_info, features_on_top, features_model = load_manager.load_data(processed_data_name)
    logger.info('All good')


def check_processed_data_name(ctx, param, value):
    if value is not None:
        service = ctx.params.get('service')
        if service is None:
            return value
        path_manager = PathManager(service)
        path = path_manager.get_processed_data_path(value)
        if path.exists():
            return value
        else:
            raise click.BadParameter(f"Processed data file '{path}' does not exist.")


@click.command(name='remove_processed_data')
@click.option('--service', type=click.STRING, required=True,
              is_eager=True)
@click.option('--processed_data_name', type=click.STRING, required=True, callback=check_processed_data_name)
def remove_processed_data(service: str, processed_data_name: str):
    path_manager = PathManager(service)
    processed_data_path = path_manager.get_processed_data_path(processed_data_name)

    logging.info("Removal process started ...")
    print(processed_data_path)
    processed_data_path.unlink()
    logging.info("Removed processed data ...")

    data_name_reference = LoadManager.load_data_name_reference()
    if processed_data_name in data_name_reference:
        del data_name_reference[processed_data_name]
        logging.info(f"File '{processed_data_name}' removed from reference.")
        ExportManager.export_data_name_reference(data_name_reference)
    else:
        logging.info(f"File '{processed_data_name}' not found in reference.")
@click.group()
def cli():
    pass


cli.add_command(make_processed_data)
cli.add_command(merge_processed_datas)
cli.add_command(remove_processed_data)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    cli()