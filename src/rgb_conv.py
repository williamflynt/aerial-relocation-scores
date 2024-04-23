"""
This code contributed by user Claude from StackOverflow.
https://stackoverflow.com/users/1207489/claude
@
https://stackoverflow.com/a/74666366
"""

import numpy as np


def rgb_to_hls(rgb_array: np.ndarray) -> np.ndarray:
    """
    Expects an array of shape (X, 3), each row being RGB colours.
    Returns an array of same size, each row being HLS colours.
    Like `colorsys` python module, all values are between 0 and 1.

    NOTE: like `colorsys`, this uses HLS rather than the more usual HSL
    """
    assert rgb_array.ndim == 2
    assert rgb_array.shape[1] == 3
    assert np.max(rgb_array) <= 1
    assert np.min(rgb_array) >= 0

    r, g, b = rgb_array.T.reshape((3, -1, 1))
    maxc = np.max(rgb_array, axis=1).reshape((-1, 1))
    minc = np.min(rgb_array, axis=1).reshape((-1, 1))

    sumc = maxc + minc
    rangec = maxc - minc

    with np.errstate(divide="ignore", invalid="ignore"):
        rgb_c = (maxc - rgb_array) / rangec
    rc, gc, bc = rgb_c.T.reshape((3, -1, 1))

    h = (
        np.where(
            minc == maxc,
            0,
            np.where(
                r == maxc, bc - gc, np.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc)
            ),
        )
        / 6
    ) % 1
    l = sumc / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(
            minc == maxc, 0, np.where(l < 0.5, rangec / sumc, rangec / (2.0 - sumc))
        )

    return np.concatenate((h, l, s), axis=1)


def hls_to_rgb(hls_array: np.ndarray) -> np.ndarray:
    """
    Expects an array of shape (X, 3), each row being HLS colours.
    Returns an array of same size, each row being RGB colours.
    Like `colorsys` python module, all values are between 0 and 1.

    NOTE: like `colorsys`, this uses HLS rather than the more usual HSL
    """
    ONE_THIRD = 1 / 3
    TWO_THIRD = 2 / 3
    ONE_SIXTH = 1 / 6

    def _v(m1, m2, h):
        h = h % 1.0
        return np.where(
            h < ONE_SIXTH,
            m1 + (m2 - m1) * h * 6,
            np.where(
                h < 0.5,
                m2,
                np.where(h < TWO_THIRD, m1 + (m2 - m1) * (TWO_THIRD - h) * 6, m1),
            ),
        )

    assert hls_array.ndim == 2
    assert hls_array.shape[1] == 3
    assert np.max(hls_array) <= 1
    assert np.min(hls_array) >= 0

    h, l, s = hls_array.T.reshape((3, -1, 1))
    m2 = np.where(l < 0.5, l * (1 + s), l + s - (l * s))
    m1 = 2 * l - m2

    r = np.where(s == 0, l, _v(m1, m2, h + ONE_THIRD))
    g = np.where(s == 0, l, _v(m1, m2, h))
    b = np.where(s == 0, l, _v(m1, m2, h - ONE_THIRD))

    return np.concatenate((r, g, b), axis=1)


def _test1():
    import colorsys

    rgb_array = np.array(
        [[0.5, 0.5, 0.8], [0.3, 0.7, 1], [0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5]]
    )
    hls_array = rgb_to_hls(rgb_array)
    for rgb, hls in zip(rgb_array, hls_array):
        assert np.all(abs(np.array(colorsys.rgb_to_hls(*rgb) - hls) < 0.001))
    new_rgb_array = hls_to_rgb(hls_array)
    for hls, rgb in zip(hls_array, new_rgb_array):
        assert np.all(abs(np.array(colorsys.hls_to_rgb(*hls) - rgb) < 0.001))
    assert np.all(abs(rgb_array - new_rgb_array) < 0.001)
    print("tests part 1 done")


def _test2():
    import colorsys

    hls_array = np.array(
        [
            [0.6456692913385826, 0.14960629921259844, 0.7480314960629921],
            [0.3, 0.7, 1],
            [0, 0, 0],
            [0, 1, 0],
            [0.5, 0.5, 0.5],
        ]
    )
    rgb_array = hls_to_rgb(hls_array)
    for hls, rgb in zip(hls_array, rgb_array):
        assert np.all(abs(np.array(colorsys.hls_to_rgb(*hls) - rgb) < 0.001))
    new_hls_array = rgb_to_hls(rgb_array)
    for rgb, hls in zip(rgb_array, new_hls_array):
        assert np.all(abs(np.array(colorsys.rgb_to_hls(*rgb) - hls) < 0.001))
    assert np.all(abs(hls_array - new_hls_array) < 0.001)
    print("All tests done")


def _test():
    _test1()
    _test2()


if __name__ == "__main__":
    _test()
