from typing import List, Tuple

__all__ = ["_validate_matrix", "multiply_matrices"]


def _validate_matrix(matrix: List[List[int]], name: str) -> Tuple[int, int]:
    """
    Validate that the provided matrix is a non-empty, rectangular matrix
    of integers.

    Parameters
    ----------
    matrix:
        Matrix to validate, represented as a list of integer lists.
    name:
        Descriptive name used in error messages.

    Returns
    -------
    (rows, cols)
        Number of rows and columns in the matrix.

    Raises
    ------
    ValueError
        If the matrix is empty, non-rectangular, or contains non-integers.
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise ValueError(f"{name} must be a non-empty list of rows.")
    if not all(isinstance(row, list) for row in matrix):
        raise ValueError(f"All rows of {name} must be lists.")
    row_count = len(matrix)
    col_count = len(matrix[0])
    if col_count == 0:
        raise ValueError(f"Rows of {name} must be non-empty.")
    for i, row in enumerate(matrix):
        if len(row) != col_count:
            raise ValueError(
                f"All rows of {name} must have the same length; "
                f"row 0 has length {col_count}, row {i} has length "
                f"{len(row)}."
            )
        if not all(isinstance(el, int) for el in row):
            raise ValueError(f"All elements of {name} must be integers.")
    return row_count, col_count


def multiply_matrices(a: List[List[int]],
                      b: List[List[int]]) -> List[List[int]]:
    """
    Multiply two matrices of integers.

    Parameters
    ----------
    a:
        Left matrix as a list of integer lists with dimensions m x n.
    b:
        Right matrix as a list of integer lists with dimensions n x p.

    Returns
    -------
    result:
        The product matrix with dimensions m x p.

    Raises
    ------
    ValueError
        If the matrices are invalid or their dimensions are incompatible.
    """
    a_rows, a_cols = _validate_matrix(a, "A")
    b_rows, b_cols = _validate_matrix(b, "B")
    if a_cols != b_rows:
        raise ValueError(
            "Incompatible dimensions: A has %d columns but B has %d rows."
            % (a_cols, b_rows)
        )
    # Transpose B to iterate over its columns efficiently.
    transposed_b = [tuple(col) for col in zip(*b)]
    result: List[List[int]] = []
    for row in a:
        result_row: List[int] = []
        for col in transposed_b:
            dot = sum(x * y for x, y in zip(row, col))
            result_row.append(dot)
        result.append(result_row)
    return result