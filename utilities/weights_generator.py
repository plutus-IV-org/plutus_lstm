"""
This module provides a function to generate a weight matrix. The matrix has the following properties:
- A specified number of columns, `N`.
- A specified number of rows, defaulting to 110.
- A specific distribution pattern: For the majority of the rows, each column takes turns receiving a
  maximum weight while the rest are distributed randomly such that they sum up to 1.
  For the last `N` rows, weights are generated around a mean value.
"""

import numpy as np


def generate_weight_matrix(N: int, rows: int = 110, max_weight: float = 0.65) -> np.ndarray:
    """
    Generates a weight matrix with specific distribution properties.

    Parameters:
    - N (int): Number of columns in the matrix.
    - rows (int): Total number of rows in the matrix. Default is 110.
    - max_weight (float): The maximum weight for a specific position in the distribution. Default is 0.65.

    Returns:
    - np.ndarray: A matrix with the specified weight distribution.
    """

    matrix = np.zeros((rows, N))
    rows_per_column = (rows - N) // N  # Deducting the last N rows for average distribution

    # For the first rows-N rows
    for i in range(rows - N):
        target_column = i // rows_per_column  # Determines which column gets the max weight

        # Assign the max_weight to the target column
        matrix[i][target_column] = max_weight

        # Distribute the remaining 1-max_weight among other columns
        remaining_weight = 1.0 - max_weight
        remaining_cols = [col for col in range(N) if col != target_column]  # Excludes the target column
        random_weights = np.random.rand(N - 1)  # Generate random weights for the non-target columns
        random_weights /= random_weights.sum()  # Normalize to sum up to 1
        random_weights *= remaining_weight  # Adjust the weights according to the remaining weight

        for j, col in enumerate(remaining_cols):
            matrix[i][col] = random_weights[j]

    # For the last N rows, generate random weights around 1/N
    for i in range(rows - N, rows):
        random_weights = np.random.rand(N)  # Generating random weights for all columns
        centered_weights = (random_weights + (1 / N)) / 2  # Adjusting the weights to be centered around 1/N
        matrix[i] = centered_weights / centered_weights.sum()  # Normalize to sum up to 1

    return matrix
