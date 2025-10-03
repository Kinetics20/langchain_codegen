from typing import List, Tuple


def _validate_matrix(matrix: List[List[int]], name: str) -> Tuple[int, int]:
    """
    Validate that a matrix is a non-empty, rectangular list of lists
    of integers.

    Returns a tuple (rows, cols) describing the matrix dimensions.

    Raises ValueError on invalid input.
    """
    if not isinstance(matrix, list) or not matrix:
        raise ValueError(f"{name} must be a non-empty list of lists")
    row_count = len(matrix)
    first_row = matrix[0]
    if not isinstance(first_row, list) or not first_row:
        raise ValueError(f"{name} must contain non-empty rows")
    col_count = len(first_row)
    for i, row in enumerate(matrix):
        if not isinstance(row, list):
            raise ValueError(f"{name} row {i} is not a list")
        if len(row) != col_count:
            raise ValueError(f"{name} rows must all have the same length")
        for j, item in enumerate(row):
            if not isinstance(item, int):
                raise ValueError(
                    f"{name}[{i}][{j}] must be an integer"
                )
    return row_count, col_count


def multiply_matrices(a: List[List[int]],
                      b: List[List[int]]) -> List[List[int]]:
    """
    Multiply two matrices represented as lists of lists of integers.

    The function validates that both matrices are rectangular and that
    their dimensions are compatible for multiplication. It returns a
    new matrix containing integer products.

    Raises ValueError if inputs are invalid or dimensions mismatch.
    """
    a_rows, a_cols = _validate_matrix(a, "First matrix")
    b_rows, b_cols = _validate_matrix(b, "Second matrix")
    if a_cols != b_rows:
        raise ValueError(
            "Incompatible dimensions: "
            f"{a_rows}x{a_cols} cannot be multiplied with "
            f"{b_rows}x{b_cols}"
        )
    # Initialize result matrix with zeros.
    result: List[List[int]] = [
        [0 for _ in range(b_cols)] for _ in range(a_rows)
    ]
    # Compute multiplication.
    for i in range(a_rows):
        a_row = a[i]
        for j in range(b_cols):
            total = 0
            for k in range(a_cols):
                total += a_row[k] * b[k][j]
            result[i][j] = total
    return result