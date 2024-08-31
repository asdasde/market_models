.. module:: src
   :synopsis: This module includes functionalities for sampling, exporting, and crawling data.

===========
Description
===========

This module includes functionalities for sampling profiles, exporting data for crawling, fetching profiles, and executing the entire crawling process. It leverages XGBoost models for error prediction and uses various utility functions for data preparation and export.

Usage
=====

Command Line Interface (CLI)
----------------------------

To use the CLI commands, navigate to the `src` directory and run the respective Python scripts with the desired options.

.. code-block:: bash

    Replace `YOUR_COMMAND` with one of the supported CLI commands:

    $ cd src
    $ python data/make_dataset.py YOUR_COMMAND --OPTIONS

Supported CLI Commands
=======================

1. **generate_incremental_data**

   .. code-block:: bash

      $ python data/make_dataset.py generate_incremental_data --service SERVICE --base_profile_v BASE_PROFILE_VERSION --values_v VALUES_VERSION

   Generate incremental data based on specified profile and value versions.

   **Options:**

   - `--service` (str): The service name (currently only supports 'netrisk_casco').
   - `--base_profile_v` (str): The base profile version.
   - `--values_v` (str): The values version.

2. **sample_crawling_data**

   .. code-block:: bash

      $ python data/make_dataset.py sample_crawling_data --error_model_name ERROR_MODEL_NAME --service SERVICE --params_v PARAMS_VERSION --policy_start_date POLICY_START_DATE --n NUM_SAMPLES

   Sample profiles for crawling.

   **Options:**

   - `--error_model_name` (str): The error model name.
   - `--service` (str): The service name (currently only supports 'netrisk_casco').
   - `--params_v` (str): The version of parameter distributions.
   - `--policy_start_date` (str): The policy start date in `YYYY_MM_DD` format.
   - `--custom_name` (str): Custom name to append to the generated file name.
   - `--n` (int): The number of profiles to sample. Default is `1000`.

3. **export_data_for_crawling**

   .. code-block:: bash

      $ python data/make_dataset.py export_data_for_crawling --service SERVICE --data_name DATA_NAME --template_date TEMPLATE_DATE

   Export data for crawling.

   **Options:**

   - `--service` (str): The service name (example: 'netrisk_casco').
   - `--data_name` (str): The name of the data (example: 'incremental_data_base_profile_v1_values_v2').
   - `--template_date` (str): The date in the template file name.

4. **fetch_profiles**

   .. code-block:: bash

      $ python data/make_dataset.py fetch_profiles --service SERVICE --data_name DATA_NAME

   Fetch profiles from the server.

   **Options:**

   - `--service` (str): The service name (example: 'netrisk_casco').
   - `--data_name` (str): The data name (example: 'incremental_data_base_profile_v1_values_v2').

5. **run_crawler**

   .. code-block:: bash

      $ python data/make_dataset.py run_crawler --num_processes NUM_PROCESSES --remote_profiles_path REMOTE_PROFILES_PATH

   Run the crawler script on the server.

   **Options:**

   - `--num_processes` (int): The number of processes to start.
   - `--remote_profiles_path` (str): The location of the profiles on the server.

6. **execute_all_crawling**

   .. code-block:: bash

      $ python data/make_dataset.py execute_all_crawling --profile_type PROFILE_TYPE --service SERVICE --params_v PARAMS_VERSION --policy_start_date POLICY_START_DATE --custom_name CUSTOM_NAME --n NUM_SAMPLES --template_date TEMPLATE_DATE --num_processes NUM_PROCESSES

   Execute the complete crawling process.

   **Options:**

   - `--profile_type` (str): Type of profiles to use, either 'sampled' or 'incremental'.
   - `--service` (str): The service name (example: 'netrisk_casco').
   - `--params_v` (str): Version of distribution parameters (for sampled profiles).
   - `--policy_start_date` (str): Policy start date in `YYYY_MM_DD` format.
   - `--custom_name` (str): Custom name for the data.
   - `--n` (int): Number of profiles to sample (for sampled profiles). Default is `1000`.
   - `--template_date` (str): Date in the template file name.
   - `--num_processes` (int): Number of crawler processes to start.

Requirements
============

- Python 3.6 or later
- Required Python packages (specified in your script)
- Pre-trained XGBoost models in the `models` directory
- Downloaded and configured environment variables in the `.env` file.

Installation
============

No special installation is required. Ensure that the necessary Python packages are installed:

.. code-block:: bash

    $ cd src
    $ pip install -r requirements.txt

Configuration
=============

The scripts use environment variables, and you can configure them in a `.env` file.

.. note::

    Make sure to set the required environment variables for the scripts to work properly.

Example
=======

Generating incremental data:

.. code-block:: bash

    $ cd src
    $ python data/make_dataset.py generate_incremental_data --service netrisk_casco --base_profile_v v1 --values_v v1

Sampling crawling data:

.. code-block:: bash

    $ cd src
    $ python data/make_dataset.py sample_crawling_data --error_model_name netrisk_casco_error_model --service netrisk_casco --params_v v1 --policy_start_date 2023_01_01 --n 1000

Exporting data for crawlers:

.. code-block:: bash

    $ cd src
    $ python data/make_dataset.py export_data_for_crawling --service netrisk_casco --data_name incremental_data_base_profile_v1_values_v2 --template_date 2024_07_31

Fetching profiles:

.. code-block:: bash

    $ cd src
    $ python data/make_dataset.py fetch_profiles --service netrisk_casco --data_name incremental_data_base_profile_v1_values_v2

Running crawler script:

.. code-block:: bash

    $ cd src
    $ python data/make_dataset.py run_crawler --num_processes 2 --remote_profiles_path /path/to/profiles

Executing all crawling steps together:

.. code-block:: bash

    $ cd src
    $ python data/make_dataset.py execute_all_crawling --profile_type incremental --service netrisk_casco --base_profile_v v2 --values_v v15 --policy_start_date 2024_09_15 --custom_name _test --n 1000 --template_date 2024_07_31 --num_processes 2
