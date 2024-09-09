import os
import re
from datetime import datetime, timedelta


def delete_old_files(age_threshold: int = 20) -> None:
    """
    Deletes files in the 'bulk_data' folder that are older than the specified age threshold.

    The function extracts dates from the filenames (expected format: YYYY-MM-DD) and deletes
    files that are older than the given number of days.

    Args:
        age_threshold (int): The age threshold in days. Files older than this number of days will be deleted.
                             Defaults to 20 days.

    Returns:
        None
    """
    # Define the path to the bulk_data folder
    folder_path = os.path.join(os.getcwd(), 'reporting_service', 'bulk_data')
    current_date = datetime.now()
    threshold_date = current_date - timedelta(days=age_threshold)

    # Define the date pattern in the filenames (e.g., YYYY-MM-DD)
    date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if it's a file
        if os.path.isfile(file_path):
            # Extract the date from the filename
            match = date_pattern.search(filename)
            if match:
                file_date_str = match.group()
                try:
                    file_date = datetime.strptime(file_date_str, '%Y-%m-%d')
                    # Check if the file is older than the threshold
                    if file_date < threshold_date:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                except ValueError:
                    print(f"Invalid date format in filename: {filename}")
    print('Function "delete_old_files" has been successfully executed.')
