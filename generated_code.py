from typing import List

def merge_sort(arr: List[int]) -> List[int]:
    """
    Sorts a list of integers using merge sort algorithm.

    Args:
    arr: A list of integers to be sorted.

    Returns:
    A new sorted list of integers.
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]

    left_half = merge_sort(left_half)
    right_half = merge_sort(right_half)

    return merge(left_half, right_half)

def merge(left: List[int], right: List[int]) -> List[int]:
    """
    Merges two sorted lists into a single sorted list.

    Args:
    left: A sorted list of integers.
    right: A sorted list of integers.

    Returns:
    A new sorted list containing all elements from both input lists.
    """
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j])

    return result