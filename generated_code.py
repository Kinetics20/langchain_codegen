from typing import List, Tuple

Matrix = List[List[int]]


def _validate_matrix(matrix: Matrix, name: str) -> Tuple[int, int]:
    """
    Validate that `matrix` is a non-empty, rectangular matrix of ints.

    Returns a tuple (rows, cols) describing the matrix dimensions.

    Raises:
        ValueError: If the matrix is empty, non-rectangular, has empty rows,
                    or contains non-integer elements.
    """
    if not matrix:
        raise ValueError(f"matrix '{name}' must not be empty")
    row_count = len(matrix)
    first_row_len = len(matrix[0])
    if first_row_len == 0:
        raise ValueError(f"matrix '{name}' must have non-empty rows")
    for r_index, row in enumerate(matrix):
        if len(row) != first_row_len:
            raise ValueError(
                f"matrix '{name}' must be rectangular; row {r_index} has "
                "different length"
            )
        for c_index, item in enumerate(row):
            if not isinstance(item, int):
                raise ValueError(
                    "matrix elements must be integers; found non-int at "
                    f"{name}[{r_index}][{c_index}]"
                )
    return row_count, first_row_len


def multiply_matrices(a: Matrix, b: Matrix) -> Matrix:
    """
    Multiply two matrices `a` and `b` and return the product.

    Both matrices must be lists of lists of integers. The number of columns
    in `a` must equal the number of rows in `b`. The result is a new
    matrix of size (rows of a) x (cols of b).

    Raises:
        ValueError: If either matrix is invalid or the shapes are incompatible.
    """
    a_rows, a_cols = _validate_matrix(a, "a")
    b_rows, b_cols = _validate_matrix(b, "b")
    if a_cols != b_rows:
        raise ValueError(
            "incompatible dimensions: a has %d cols but b has %d rows"
            % (a_cols, b_rows)
        )
    # Transpose b for better locality when computing dot products.
    b_t: Matrix = [list(col) for col in zip(*b)]
    result: Matrix = []
    for i in range(a_rows):
        row = a[i]
        result_row: List[int] = []
        for col in b_t:
            # Compute dot product of row and column.
            acc = 0
            for x, y in zip(row, col):
                acc += x * y
            result_row.append(acc)
        result.append(result_row)
    return result