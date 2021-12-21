from copy import deepcopy
from collections import namedtuple
import sys

Rectangle = namedtuple('Rectangle', ['x', 'y', 'w', 'h'])


def phspprg(width, rectangles, sorting="height"):
    """
    The PH heuristic for the Strip Packing Problem. This is the RG variant, which means that rotations by
    90 degrees are allowed and that there is a guillotine constraint.

    Parameters
    ----------
    width
        The width of the strip.

    rectangles
        List of list containing width and height of every rectangle, [[w_1, h_1], ..., [w_n,h_h]].
        It is assumed that all rectangles can fit into the strip.

    sorting : string, {'width', 'height'}, default='width'
        The heuristic uses sorting to determine which rectangles to place first.
        By default sorting happens on the width but can be changed to height.

    Returns
    -------
    height
        The height of the strip needed to pack all the items.
    rectangles : list of namedtuple('Rectangle', ['x', 'y', 'w', 'h'])
        A list of rectangles, in the same order as the input list. This contains bottom left x and y coordinate and
        the width and height (which can be flipped compared to input).

    """
    if sorting not in ["width", "height" ]:
        raise ValueError("The algorithm only supports sorting by width or height but {} was given.".format(sorting))
    if sorting == "width":
        wh = 0
    else:
        wh = 1
    # logger.debug('The original array: {}'.format(rectangles))
    result = [None] * len(rectangles)
    remaining = deepcopy(rectangles)
    for idx, r in enumerate(remaining):
        if r[0] > r[1]:
            remaining[idx][0], remaining[idx][1] = remaining[idx][1], remaining[idx][0]
    # logger.debug('Swapped some widths and height with the following result: {}'.format(remaining))
    sorted_indices = sorted(range(len(remaining)), key=lambda x: -remaining[x][wh])
    # logger.debug('The sorted indices: {}'.format(sorted_indices))
    sorted_rect = [remaining[idx] for idx in sorted_indices]
    # logger.debug('The sorted array is: {}'.format(sorted_rect))
    x, y, w, h, H = 0, 0, 0, 0, 0
    while sorted_indices:
        idx = sorted_indices.pop(0)
        r = remaining[idx]
        if r[1] > width:
            result[idx] = Rectangle(x, y, r[0], r[1])
            x, y, w, h, H = r[0], H, width - r[0], r[1], H + r[1]
        else:
            result[idx] = Rectangle(x, y, r[1], r[0])
            x, y, w, h, H = r[1], H, width - r[1], r[0], H + r[0]
        recursive_packing(x, y, w, h, 1, remaining, sorted_indices, result)
        x, y = 0, H
    # logger.debug('The resulting rectangles are: {}'.format(result))

    return H, result


def phsppog(width, rectangles, sorting="width"):
    """
    The PH heuristic for the Strip Packing Problem. This is the OG variant, which means that rotations are
    NOT allowed and that there is a guillotine contraint.

    Parameters
    ----------
    width
        The width of the strip.

    rectangles
        List of list containing width and height of every rectangle, [[w_1, h_1], ..., [w_n,h_h]].
        It is assumed that all rectangles can fit into the strip.

    sorting : string, {'width', 'height'}, default='width'
        The heuristic uses sorting to determine which rectangles to place first.
        By default sorting happens on the width but can be changed to height.

    Returns
    -------
    height
        The height of the strip needed to pack all the items.
    rectangles : list of namedtuple('Rectangle', ['x', 'y', 'w', 'h'])
        A list of rectangles, in the same order as the input list. This contains bottom left x and y coordinate and
        the width and height (which can be flipped compared to input).

    """
    if sorting not in ["width", "height" ]:
        raise ValueError("The algorithm only supports sorting by width or height but {} was given.".format(sorting))
    if sorting == "width":
        wh = 0
    else:
        wh = 1
    # logger.debug('The original array: {}'.format(rectangles))
    result = [None] * len(rectangles)
    remaining = deepcopy(rectangles)
    # logger.debug('Swapped some widths and height with the following result: {}'.format(remaining))
    sorted_indices = sorted(range(len(remaining)), key=lambda x: -remaining[x][wh])
    # logger.debug('The sorted indices: {}'.format(sorted_indices))
    sorted_rect = [remaining[idx] for idx in sorted_indices]
    # logger.debug('The sorted array is: {}'.format(sorted_rect))
    x, y, w, h, H = 0, 0, 0, 0, 0
    while sorted_indices:
        idx = sorted_indices.pop(0)
        r = remaining[idx]
        result[idx] = Rectangle(x, y, r[0], r[1])
        x, y, w, h, H = r[0], H, width - r[0], r[1], H + r[1]
        recursive_packing(x, y, w, h, 0, remaining, sorted_indices, result)
        x, y = 0, H

    return H, result


def recursive_packing(x, y, w, h, D, remaining, indices, result):
    """Helper function to recursively fit a certain area."""
    priority = 6
    for idx in indices:
        for j in range(0, D + 1):
            if priority > 1 and remaining[idx][(0 + j) % 2] == w and remaining[idx][(1 + j) % 2] == h:
                priority, orientation, best = 1, j, idx
                break
            elif priority > 2 and remaining[idx][(0 + j) % 2] == w and remaining[idx][(1 + j) % 2] < h:
                priority, orientation, best = 2, j, idx
            elif priority > 3 and remaining[idx][(0 + j) % 2] < w and remaining[idx][(1 + j) % 2] == h:
                priority, orientation, best = 3, j, idx
            elif priority > 4 and remaining[idx][(0 + j) % 2] < w and remaining[idx][(1 + j) % 2] < h:
                priority, orientation, best = 4, j, idx
            elif priority > 5:
                priority, orientation, best = 5, j, idx
    if priority < 5:
        if orientation == 0:
            omega, d = remaining[best][0], remaining[best][1]
        else:
            omega, d = remaining[best][1], remaining[best][0]
        result[best] = Rectangle(x, y, omega, d)
        indices.remove(best)
        if priority == 2:
            recursive_packing(x, y + d, w, h - d, D, remaining, indices, result)
        elif priority == 3:
            recursive_packing(x + omega, y, w - omega, h, D, remaining, indices, result)
        elif priority == 4:
            min_w = sys.maxsize
            min_h = sys.maxsize
            for idx in indices:
                min_w = min(min_w, remaining[idx][0])
                min_h = min(min_h, remaining[idx][1])
            # Because we can rotate:
            min_w = min(min_h, min_w)
            min_h = min_w
            if w - omega < min_w:
                recursive_packing(x, y + d, w, h - d, D, remaining, indices, result)
            elif h - d < min_h:
                recursive_packing(x + omega, y, w - omega, h, D, remaining, indices, result)
            elif omega < min_w:
                recursive_packing(x + omega, y, w - omega, d, D, remaining, indices, result)
                recursive_packing(x, y + d, w, h - d, D, remaining, indices, result)
            else:
                recursive_packing(x, y + d, omega, h - d, D, remaining, indices, result)
                recursive_packing(x + omega, y, w - omega, h, D, remaining, indices, result)

