# tests/test_nnls.py
import math

import pytest

from tokenwatt.nnls import nnls


def test_identity_recovers_exact_solution():
    x, res = nnls([[1.0, 0.0], [0.0, 1.0]], [3.0, 4.0])
    assert x == pytest.approx([3.0, 4.0])
    assert res == pytest.approx(0.0, abs=1e-9)


def test_overdetermined_exact_line_through_origin():
    # b_i = 2 * a_i -> slope 2, zero residual
    x, res = nnls([[1.0], [2.0], [3.0]], [2.0, 4.0, 6.0])
    assert x[0] == pytest.approx(2.0)
    assert res == pytest.approx(0.0, abs=1e-9)


def test_nonnegativity_clamps_a_negative_unconstrained_solution():
    # unconstrained best fit is negative; NNLS must clamp to 0
    x, res = nnls([[1.0], [1.0]], [-2.0, -2.0])
    assert x[0] == pytest.approx(0.0)
    assert res == pytest.approx(math.sqrt(8.0))     # ‖[-2,-2] - 0‖


def test_two_variable_recovers_known_coefficients():
    # b_i = 2*col0 + 0.5*col1 exactly -> [2.0, 0.5]
    A = [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    b = [2.5, 4.5, 6.5]
    x, res = nnls(A, b)
    assert x == pytest.approx([2.0, 0.5], abs=1e-6)
    assert res == pytest.approx(0.0, abs=1e-6)


def test_two_variable_with_one_coefficient_clamped():
    # data wants a negative second coefficient; NNLS zeros it and refits the first
    A = [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    b = [1.0, 2.0, 3.0]           # perfectly slope-1 through origin; the +1 col wants < 0
    x, res = nnls(A, b)
    assert x[1] == pytest.approx(0.0, abs=1e-6)      # second coefficient clamped
    assert x[0] == pytest.approx(1.0, abs=1e-6)
    assert res == pytest.approx(0.0, abs=1e-6)
