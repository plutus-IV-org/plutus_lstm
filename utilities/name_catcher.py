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