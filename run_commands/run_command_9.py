"""
Module for running daily reports.
"""

import os
import sys
import shutil

# Set the working directory to two levels up
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

from reporting_service.bulk_data_maker import generate_bulk_data, get_bulk_data_abs_path, get_output_data_abs_path
from reporting_service.bulk_data_cleaner import delete_old_files
from reporting_service.xlsx_report_maker import proces_bulk_data, prepare_output_report
from Const import SHARE_DRIVE_DIR

# Specify the type of run (development or production)
type_of_run = 'prod'
asset_type = 'crypto'

# Deleting old files (>20 days)
delete_old_files()
# Prepare and store bulk data
generate_bulk_data(type_of_run, asset_type)
# Get bulk data abs path
abs_path = get_bulk_data_abs_path(type_of_run)
# Get bulk data tables in a dict
stored_data = proces_bulk_data(abs_path)
# Prepare additional info and store the report
output_path = get_output_data_abs_path(type_of_run)
prepare_output_report(abs_path, stored_data, output_path)

# Copy the file to the Google Drive folder
destination_file_path = os.path.join(SHARE_DRIVE_DIR, os.path.basename(output_path))
shutil.copy(output_path, destination_file_path)
