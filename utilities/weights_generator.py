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
    matrix = np.zeros((rows, N))

    # Compute the basic number of rows per column and the remainder
    base_rows_per_column = (rows - N) // N
    extra_rows = (rows - N) % N

    current_row = 0

    for col in range(N):
        # Assign max_weight to base_rows_per_column + (1 if this column is to get an extra row)
        num_rows_with_max_weight = base_rows_per_column + (1 if col < extra_rows else 0)

        for _ in range(num_rows_with_max_weight):
            matrix[current_row, col] = max_weight

            # Distribute the remaining weight among other columns
            remaining_weight = 1.0 - max_weight
            remaining_cols = [c for c in range(N) if c != col]
            random_weights = np.random.rand(N - 1)
            random_weights /= random_weights.sum()
            random_weights *= remaining_weight

            for j, other_col in enumerate(remaining_cols):
                matrix[current_row, other_col] = random_weights[j]

            current_row += 1

    # For the last N rows, generate random weights around 1/N
    for i in range(rows - N, rows):
        random_weights = np.random.rand(N)
        centered_weights = (random_weights + (1 / N)) / 2
        matrix[i] = centered_weights / centered_weights.sum()

    return matrix
