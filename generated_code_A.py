from typing import List, Tuple


def _validate_matrix(matrix: List[List[int]], name: str) -> None:
    """
    Validate that the given matrix is a non-empty rectangular matrix
    of integers.

    Parameters
    ----------
    matrix:
        The matrix to validate as a list of lists of integers.
    name:
        The name of the matrix (used in error messages).

    Raises
    ------
    TypeError
        If the matrix or its elements are not of the expected types.
    ValueError
        If the matrix is empty or not rectangular.
    """
    if not isinstance(matrix, list):
        raise TypeError(f"{name} must be a list of lists of ints")

    if len(matrix) == 0:
        raise ValueError(f"{name} must be a non-empty matrix")

    row_length = None
    for row_index, row in enumerate(matrix):
        if not isinstance(row, list):
            raise TypeError(f"{name}[{row_index}] must be a list of ints")
        if row_length is None:
            row_length = len(row)
            if row_length == 0:
                raise ValueError(f"{name} rows must be non-empty")
        elif len(row) != row_length:
            raise ValueError(f"{name} must be rectangular; inconsistent row "
                             "lengths found")
        for col_index, item in enumerate(row):
            if not isinstance(item, int):
                raise TypeError(
                    f"{name}[{row_index}][{col_index}] must be an int"
                )


def _dimensions(matrix: List[List[int]]) -> Tuple[int, int]:
    """
    Return the dimensions of the matrix as (rows, cols).

    Parameters
    ----------
    matrix:
        A validated matrix.

    Returns
    -------
    Tuple[int, int]
        Number of rows and number of columns.
    """
    return len(matrix), len(matrix[0])


def multiply_matrices(a: List[List[int]], b: List[List[int]]
                      ) -> List[List[int]]:
    """
    Multiply two matrices of integers and return the product matrix.

    Both matrices must be non-empty rectangular lists of lists of ints.
    The number of columns in `a` must equal the number of rows in `b`.

    Parameters
    ----------
    a:
        Left matrix as a list of lists of ints.
    b:
        Right matrix as a list of lists of ints.

    Returns
    -------
    List[List[int]]
        The product matrix as a new list of lists of ints.

    Raises
    ------
    TypeError
        If inputs are not lists of lists of ints.
    ValueError
        If matrices are empty, non-rectangular, or have incompatible
        dimensions for multiplication.
    """
    _validate_matrix(a, "a")
    _validate_matrix(b, "b")

    a_rows, a_cols = _dimensions(a)
    b_rows, b_cols = _dimensions(b)

    if a_cols != b_rows:
        raise ValueError(
            "Incompatible dimensions: a columns must equal b rows for "
            "multiplication"
        )

    # Precompute columns of b to improve locality and avoid repeated
    # indexing of inner lists.
    b_columns: List[List[int]] = [
        [b[row][col] for row in range(b_rows)] for col in range(b_cols)
    ]

    result: List[List[int]] = [
        [0 for _ in range(b_cols)] for _ in range(a_rows)
    ]

    for i in range(a_rows):
        a_row = a[i]
        res_row = result[i]
        for j in range(b_cols):
            col = b_columns[j]
            # Compute dot product of a_row and col
            total = 0
            for k in range(a_cols):
                total += a_row[k] * col[k]
            res_row[j] = total

    return result