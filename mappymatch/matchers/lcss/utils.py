import functools as ft
from itertools import groupby
from operator import itemgetter
from typing import Any, Callable, Generator, List


def forward_merge(merge_list: List, condition: Callable[[Any], bool]) -> List:
    """
    Merge items in a list by combining matching items with the next eligible element (left-to-right).

    This function scans through the list from left to right. When it encounters items that
    satisfy the condition, it combines them with the next item that doesn't satisfy the
    condition. This is useful for merging small trajectory segments with their neighbors.

    Args:
        merge_list: The list of items to merge. Items should support addition (+) for combining.
        condition: A function that returns True for items that should be merged forward. Items satisfying this condition will be combined with the next non-matching item.

    Returns:
        A new list with matching items merged into subsequent items
    """
    items = []

    def _flatten(ml):
        return ft.reduce(lambda acc, x: acc + x, ml)

    merge_items = []
    for i, item in enumerate(merge_list):
        if condition(item):
            merge_items.append(item)
        elif merge_items:
            # we found a large item and have short items to merge
            merge_items.append(item)
            items.append(_flatten(merge_items))
            merge_items = []
        else:
            items.append(item)

    if merge_items:
        # we got to the end but still have merge items;
        items.append(_flatten(merge_items))

    return items


def reverse_merge(merge_list: List, condition: Callable[[Any], bool]) -> List:
    """
    Merge items in a list by combining matching items with the previous eligible element (right-to-left).

    This function scans through the list from right to left. When it encounters items that
    satisfy the condition, it combines them with the previous item that doesn't satisfy the
    condition. This is the reverse of forward_merge.

    Args:
        merge_list: The list of items to merge. Items should support addition (+) for combining.
        condition: A function that returns True for items that should be merged backward. Items satisfying this condition will be combined with the previous non-matching item.

    Returns:
        A new list with matching items merged into preceding items
    """
    items = []

    def _flatten(ml):
        return ft.reduce(lambda acc, x: x + acc, ml)

    merge_items = []
    for i in reversed(range(len(merge_list))):
        item = merge_list[i]
        if condition(item):
            merge_items.append(item)
        elif merge_items:
            # we found a large item and have short items to merge
            merge_items.append(item)
            items.append(_flatten(merge_items))
            merge_items = []
        else:
            items.append(item)

    if merge_items:
        # we got to the end but still have merge items;
        items.append(_flatten(merge_items))

    return list(reversed(items))


def merge(merge_list: List, condition: Callable[[Any], bool]) -> List:
    """
    Merge items in a list using both forward and reverse merging to handle edge cases.

    This function first performs a forward merge, then checks if any items still satisfy
    the condition. If so, it performs a reverse merge to handle items at the end of the
    list that couldn't be merged forward.

    This two-pass approach ensures that items at both ends of the list can be successfully
    merged with their neighbors.

    Args:
        merge_list: The list of items to merge. Items should support addition (+) for combining.
        condition: A function that returns True for items that should be merged with neighbors.

    Returns:
        A new list with all matching items merged into neighbors
    """
    f_merge = forward_merge(merge_list, condition)

    if any(map(condition, f_merge)):
        return reverse_merge(f_merge, condition)
    else:
        return f_merge


def compress(cutting_points: List) -> Generator:
    """
    Compress adjacent cutting points by keeping only the middle point of each group.

    When multiple cutting points are directly adjacent (differ by 1 index), this function
    collapses them into a single representative cutting point. For each group, the middle
    point is selected as the representative.

    This prevents the LCSS algorithm from creating too many tiny segments when several
    adjacent points all have poor matches.

    Args:
        cutting_points: A list of CuttingPoint objects to compress

    Yields:
        CuttingPoint objects: One representative cutting point for each group of
        adjacent cutting points. The middle point of each group is selected.
    """
    sorted_cuts = sorted(cutting_points, key=lambda c: c.trace_index)
    for k, g in groupby(enumerate(sorted_cuts), lambda x: x[0] - x[1].trace_index):
        all_cps = list(map(itemgetter(1), g))
        yield all_cps[int(len(all_cps) / 2)]
