import platform


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
