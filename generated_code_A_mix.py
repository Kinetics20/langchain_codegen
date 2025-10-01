def _validate_matrix(matrix, name):
    """
    Validate that 'matrix' is a well-formed matrix represented as a list of lists of integers.

    Raises:
      TypeError: if the top-level object is not a list, or any row is not a list,
                 or any element is not an integer.
      ValueError: if the matrix is empty, any row is empty, or rows have inconsistent lengths.

    Returns:
      tuple (num_rows, num_cols)
    """
    # Top-level must be a list
    if not isinstance(matrix, list):
        raise TypeError(f"{name} must be a list of lists")
    # Must not be empty
    if len(matrix) == 0:
        raise ValueError(f"{name} cannot be empty")
    # Each row must be a list and non-empty; record row length consistency
    row_length = None
    for i, row in enumerate(matrix):
        if not isinstance(row, list):
            raise TypeError(f"{name} must be a list of lists (row {i} is not a list)")
        if len(row) == 0:
            raise ValueError(f"rows of {name} cannot be empty (row {i} is empty)")
        if row_length is None:
            row_length = len(row)
        elif len(row) != row_length:
            raise ValueError(f"each row of {name} must have the same length")
        # Check each element is an integer (exclude booleans)
        for j, val in enumerate(row):
            if not isinstance(val, int) or isinstance(val, bool):
                raise TypeError(f"{name}[{i}][{j}] must be an integer")
    return len(matrix), row_length


def multiply_matrices(a, b):
    """
    Multiply two matrices represented as lists of lists of integers.

    Parameters:
      a (list of list of int): Left matrix with dimensions m x n.
      b (list of list of int): Right matrix with dimensions n x p.

    Returns:
      list of list of int: Resulting matrix with dimensions m x p.

    Raises:
      TypeError: if input types are incorrect (not lists of lists, or non-integer elements).
      ValueError: if input values are invalid (empty matrices, inconsistent row sizes,
                  or dimensions that do not allow multiplication).
    """
    # Validate input matrices and obtain dimensions
    m, n = _validate_matrix(a, "A")
    n_b, p = _validate_matrix(b, "B")

    # Check that inner dimensions match for multiplication (n == n_b)
    if n != n_b:
        raise ValueError("matrices A and B cannot be multiplied: number of columns in A "
                         "must equal number of rows in B")

    # Perform matrix multiplication: result is m x p
    result = []
    # Precompute columns of B to improve locality (optional)
    # Build list of columns of B as lists for faster access in pure Python
    b_columns = [[b[row][col] for row in range(n)] for col in range(p)]

    for i in range(m):
        row_result = []
        for j in range(p):
            # Compute dot product of A's row i and B's column j
            dot = 0
            a_row = a[i]
            b_col = b_columns[j]
            for k in range(n):
                dot += a_row[k] * b_col[k]
            row_result.append(dot)
        result.append(row_result)

    return result