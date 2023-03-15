import platform

def _slash_conversion():
    if platform.system() == "Linux":
        slash = '/'
    else:
        slash = '\\'
    return slash

def _discord_channel():
    return "1051536696434491442"