from fuzzywuzzy import fuzz

def check_name_in_dict(name: str, names_dict: dict) -> str:
    """
    Checks if a name (or a partial match of the name) is present in the keys of the dictionary, and returns the full
    name that matches (or a partial match of) the given name.

    Parameters:
    name (str): The name to check in the dictionary keys.
    names_dict (dict): A dictionary mapping names to values.

    Returns:
    str: The full name that matches (or a partial match of) the given name, or an empty string if no match is found.
    """
    # Check if the exact name is present in the keys of the dictionary
    if name in names_dict.keys():
        return name
    # Check if a partial match of the name is present in the keys of the dictionary
    else:
        max_score = 0
        matched_name = ''
        for key in names_dict.keys():
            key_words = key.split('_')
            for word in key_words:
                score = fuzz.partial_ratio(name.lower(), word.lower())
                if score > max_score:
                    max_score = score
                    matched_name = key
        if max_score >= 70:  # Minimum score for a fuzzy match
            return matched_name
        else:
            return None

def check_file_name_in_list(file_name: str, file_list: list) -> str:
    """
    Checks if a file name (or a partial match of the file name) is present in a list of file names, and returns the full
    file name that matches (or a partial match of) the given name.

    Parameters:
    file_name (str): The file name to check in the list.
    file_list (List[str]): A list of file names.

    Returns:
    str: The full file name that matches (or a partial match of) the given name, or an empty string if no match is found.
    """
    # Check if the exact file name is present in the list
    if file_name in file_list:
        return file_name
    # Check if a partial match of the file name is present in the list
    else:
        max_score = 0
        matched_file_name = ''
        for file in file_list:
            file_words = file.split('_')
            for word in file_words:
                score = fuzz.partial_ratio(file_name.lower(), word.lower())
                if score > max_score:
                    max_score = score
                    matched_file_name = file
        if max_score >= 70:  # Minimum score for a fuzzy match
            return matched_file_name
        else:
            return ''