import platform
import os
import time
from distutils.dir_util import copy_tree


def _slash_conversion():
    if platform.system() == "Linux":
        slash = '/'
    else:
        slash = '\\'
    return slash


def _discord_channel():
    return "1051536696434491442"


def calculate_patience(lr: float) -> int:
    """
    Calculate the patience value based on the learning rate (lr).

    Patience determines how many epochs to wait before early stopping
    is triggered when there is no improvement.

    Args:
        lr (float): The learning rate used for training.

    Returns:
        int: The patience value based on the input learning rate.

    Rules:
        - lr == 0.0001: patience = 20
        - 0.0001 < lr < 0.0005: patience = 15
        - lr == 0.0005: patience = 10
        - For any other lr, the default patience = 5
    """
    patience = 10
    if lr == 0.0001:
        patience = 40
    elif 0.0001 < lr < 0.0005:
        patience = 30
    elif lr == 0.0005:
        patience = 20
    print(f'For given learning rate {lr}, patience is {patience}')
    return patience


def clean_and_copy_pictures(source_folder, destination_folder, max_age_seconds=86400):
    """
    Clean up files older than a certain age in the source folder and copy the remaining files to the destination folder.

    :param source_folder: Path to the source folder.
    :param destination_folder: Path to the destination folder.
    :param max_age_seconds: Maximum age of files to keep (default is 1 day = 86400 seconds).
    """
    current_time = time.time()

    # Step 1: Clean up files older than max_age_seconds
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        try:
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:  # Check if the file is older than the threshold
                    os.unlink(file_path)  # Delete the file
                    print(f"Deleted old file: {file_path}")
        except Exception as e:
            print(f"Error while deleting file {file_path}: {e}")

    # Step 2: Copy remaining files to the destination folder
    copy_tree(source_folder, destination_folder)
    print(f"Copied files from {source_folder} to {destination_folder}")
