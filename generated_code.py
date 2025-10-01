from typing import List, Tuple


def _validate_matrix(matrix: List[List[int]], name: str) -> Tuple[int, int]:
    """
    Validate that a matrix is a non-empty, rectangular list of lists of ints.

    Parameters
    ----------
    matrix:
        The matrix to validate.
    name:
        The name used in error messages to identify the matrix.

    Returns
    -------
    (rows, cols)
        The number of rows and columns of the matrix.

    Raises
    ------
    ValueError
        If the matrix is empty, not rectangular, or contains non-int values.
    """
    if not isinstance(matrix, list) or not matrix:
        raise ValueError(f"{name} must be a non-empty list of lists")
    first_row = matrix[0]
    if not isinstance(first_row, list) or not first_row:
        raise ValueError(f"{name} must contain at least one column")
    cols = len(first_row)
    rows = len(matrix)
    for i, row in enumerate(matrix):
        if not isinstance(row, list):
            raise ValueError(f"{name} row {i} is not a list")
        if len(row) != cols:
            raise ValueError(f"{name} rows have inconsistent lengths")
        for j, value in enumerate(row):
            if not isinstance(value, int):
                raise ValueError(
                    f"{name}[{i}][{j}] must be an int, got {type(value)}"
                )
    return rows, cols


def multiply_matrices(
    a: List[List[int]], b: List[List[int]]
) -> List[List[int]]:
    """
    Multiply two matrices represented as lists of lists of ints.

    Parameters
    ----------
    a:
        Left-hand matrix with shape (m, n).
    b:
        Right-hand matrix with shape (n, p).

    Returns
    -------
    product:
        The resulting matrix with shape (m, p).

    Raises
    ------
    ValueError
        If either matrix is invalid or their dimensions are incompatible.
    """
    a_rows, a_cols = _validate_matrix(a, "A")
    b_rows, b_cols = _validate_matrix(b, "B")
    if a_cols != b_rows:
        raise ValueError("A columns must equal B rows for multiplication")
    # Transpose B for cache-friendly column access.
    b_t: List[List[int]] = [
        [b[r][c] for r in range(b_rows)] for c in range(b_cols)
    ]
    result: List[List[int]] = [[0] * b_cols for _ in range(a_rows)]
    for i in range(a_rows):
        a_row = a[i]
        res_row = result[i]
        for j in range(b_cols):
            b_col = b_t[j]
            s = 0
            for k in range(a_cols):
                s += a_row[k] * b_col[k]
            res_row[j] = s
    return result