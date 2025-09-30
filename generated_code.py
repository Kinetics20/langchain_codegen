from typing import List

def merge_sort(arr: List[int]) -> List[int]:
    """Sorts a list of integers using the merge sort algorithm.

    Args:
        arr (List[int]): The list of integers to be sorted.

    Returns:
        List[int]: A new list containing the sorted integers.
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left_half = merge_sort(arr[:mid])
    right_half = merge_sort(arr[mid:])

    return merge(left_half, right_half)

def merge(left: List[int], right: List[int]) -> List[int]:
    """Merges two sorted lists into a single sorted list.

    Args:
        left (List[int]): The first sorted list.
        right (List[int]): The second sorted list.

    Returns:
        List[int]: A new list containing the merged sorted integers.
    """
    merged = []
    left_index, right_index = 0, 0

    while left_index < len(left) and right_index < len(right):
        if left[left_index] < right[right_index]:
            merged.append(left[left_index])
            left_index += 1
        else:
            merged.append(right[right_index])
            right_index += 1

    merged.extend(left[left_index:])
    merged.extend(right[right_index:])

    return merged