import click
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.load_utils import *
from api.config import ApiConfig


@click.command(name='upload_to_s3')
@click.option('--service', required=True)
@click.option('--api_configuration_name', required=True)
@click.option('--bucket_name', required=True, type=click.STRING, help='Name of the S3 bucket')
@click.option('--dry_run', is_flag=True, help='Print what would be uploaded without actually uploading')
def upload_to_s3(service: str, api_configuration_name: str, bucket_name: str, dry_run: bool) -> None:

    path_manager = PathManager(service)
    load_manager = LoadManager(path_manager)
    logger = logging.getLogger(__name__)
    s3_client = boto3.client('s3')
    api_config = ApiConfig.load_from_json(api_configuration_name)

    logger.info(f'{"DRY RUN - " if dry_run else ""}Starting upload to S3 bucket: {bucket_name}')

    response = s3_client.list_objects_v2(Bucket=bucket_name)

    s3_keys = [obj['Key'] for obj in response.get('Contents', [])]

    def upload_file(local_path: Path, file_type: str):
        try:
            if not local_path.exists():
                logger.warning(f'{file_type} file not found: {local_path}')
                return

            s3_key = PathManager.to_s3_key(local_path)

            if s3_key in s3_keys:
                s3_obj = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                local_size = local_path.stat().st_size
                if s3_obj['ContentLength'] == local_size:
                    logger.info(f'Skipping updating {s3_key} - bucket is up to date')
                    return

            logger.info(
                f'{"Would upload" if dry_run else "Uploading"} {file_type}: {local_path.stem} to s3://{bucket_name}/{s3_key}')

            if not dry_run:
                s3_client.upload_file(
                    str(local_path),
                    bucket_name,
                    s3_key
                )

        except ClientError as e:
            logger.error(f'Error uploading {file_type} {local_path}: {str(e)}')
        except Exception as e:
            logger.error(f'Unexpected error uploading {file_type} {local_path}: {str(e)}')

    logger.info("Syncing processed data files...")
    processed_data_name = api_config.train_data_name
    processed_data_path = path_manager.get_processed_data_path(processed_data_name)
    upload_file(processed_data_path, "processed data")

    logger.info("Syncing external files...")
    external_path = path_manager.get_external_path()
    if external_path.exists():
        for file_path in external_path.rglob('*'):
            if file_path.is_file():  # Skip directories
                upload_file(file_path, "external file")

    else:
        logger.warning(f"External directory not found: {external_path}")

    logger.info("Syncing model files...")
    try:
        model_names = path_manager.get_all_models_on_train_data(api_config.train_data_name, is_presence_model=False)
        for model_name in model_names:
            model_directory = path_manager.get_model_directory(api_config.train_data_name, model_name)
            for file_path in model_directory.rglob('*'):
                upload_file(file_path, "model file")
    except Exception as e:
        logger.error(f"Error accessing models for data version {api_config.train_data_name}: {str(e)}")\

    logger.info("Syncing data_name_reference")
    try:
        upload_file(PathManager.get_data_name_references_path(), 'data_name_reference')
    except Exception as e:
        logger.error(f"Error uploading data name reference: {str(e)}")

    logger.info(f'{"DRY RUN - " if dry_run else ""}Upload to S3 completed successfully')

@click.command(name='download_from_s3')
@click.option('--service', required=True)
@click.option('--api_configuration_name', required=True)
@click.option('--bucket_name', required=True, type=click.STRING, help='Name of the S3 bucket')
@click.option('--dry_run', is_flag=True, help='Print what would be downloaded without actually downloading')
def download_from_s3(service: str, api_configuration_name: str, bucket_name: str, dry_run: bool) -> None:

    path_manager = PathManager(service)
    load_manager = LoadManager(path_manager)
    logger = logging.getLogger(__name__)
    s3_client = boto3.client('s3')
    api_config = ApiConfig.load_from_json(api_configuration_name)

    logger.info(f'{"DRY RUN - " if dry_run else ""}Starting download from S3 bucket: {bucket_name}')

    response = s3_client.list_objects_v2(Bucket=bucket_name)
    s3_objects = {obj['Key']: obj for obj in response.get('Contents', [])}

    def download_file(local_path: Path, s3_key: str, file_type: str):
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if local_path.exists():
                local_size = local_path.stat().st_size
                if s3_key in s3_objects and s3_objects[s3_key]['Size'] == local_size:
                    logger.info(f'Skipping downloading {s3_key} - local file is up to date')
                    return

            if s3_key not in s3_objects:
                logger.warning(f'{file_type} not found in S3: {s3_key}')
                return

            logger.info(
                f'{"Would download" if dry_run else "Downloading"} {file_type}: s3://{bucket_name}/{s3_key} to {local_path}')

            if not dry_run:
                s3_client.download_file(
                    bucket_name,
                    s3_key,
                    str(local_path)
                )

        except ClientError as e:
            logger.error(f'Error downloading {file_type} {s3_key}: {str(e)}')
        except Exception as e:
            logger.error(f'Unexpected error downloading {file_type} {s3_key}: {str(e)}')

    logger.info("Syncing processed data files...")
    for processed_data_name in api_config.model_versions:
        processed_data_path = path_manager.get_processed_data_path(processed_data_name)
        s3_key = PathManager.to_s3_key(processed_data_path)
        download_file(processed_data_path, s3_key, "processed data")

    logger.info("Syncing external files...")
    external_path = path_manager.get_external_path()
    external_path_key = PathManager.to_s3_key(external_path)
    for s3_key in s3_objects:
        if s3_key.startswith(external_path_key):
            local_path = PathManager.from_s3_key(s3_key)
            download_file(local_path, s3_key, "external file")

    logger.info("Syncing model files...")

    for data_version in api_config.model_versions:
        model_directories_for_train_data_path = path_manager.get_models_for_train_data_directory(data_version)
        model_directories_for_train_data_key = PathManager.to_s3_key(model_directories_for_train_data_path)
        for s3_key in s3_objects:
            if s3_key.startswith(model_directories_for_train_data_key):
                local_path = PathManager.from_s3_key(s3_key)
                download_file(local_path, s3_key, "model file")

    logger.info("Syncing data_name_reference")
    try:
        data_ref_path = PathManager.get_data_name_references_path()
        s3_key = PathManager.to_s3_key(data_ref_path)
        download_file(data_ref_path, s3_key, 'data_name_reference')
    except Exception as e:
        logger.error(f"Error downloading data name reference: {str(e)}")

    logger.info(f'{"DRY RUN - " if dry_run else ""}Download from S3 completed successfully')

@click.group()
def cli():
    pass


cli.add_command(upload_to_s3)
cli.add_command(download_from_s3)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    load_dotenv(find_dotenv())
    cli()
