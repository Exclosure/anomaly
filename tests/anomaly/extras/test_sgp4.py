"""Test the SGP4 functions."""
from typing import Tuple

import numpy as np
import pytest
from jax.tree_util import tree_flatten, tree_structure, tree_unflatten
from sgp4.api import Satrec
from sgp4.earth_gravity import wgs72
from sgp4.io import twoline2rv
from sgp4.model import WGS72
from sgp4.model import Satrec as PySatrec
from sgp4.propagation import sgp4 as py_sgp4

from anomaly.extras.sgp4 import (
    GRAVITY_MODEL_PARAMETERS,
    SATREC_ALL_FIELDS,
    SATREC_DIFFERENTIABLE_FIELDS,
    TLEParams,
    satrec_to_tle_params,
    sgp4_wrapper,
    sgp4_wrapper_jacobian_exact,
    sgp4_wrapper_jacobian_numerical,
    tle_params_to_pysatrec,
)

# pylint: disable=redefined-outer-name


@pytest.fixture
def tle() -> Tuple[str, str]:
    return (
        "1 00005U 58002B   22034.80043464  .00000230  00000-0  29630-3 0  9998",
        "2 00005  34.2404 191.1967 1843273 156.0081 213.8734 10.84853720270024",
    )


@pytest.fixture
def satrec(tle):
    tle_line1, tle_line2 = tle
    satrec = Satrec.twoline2rv(tle_line1, tle_line2, WGS72)
    return satrec


@pytest.fixture
def satrec_direct(tle):
    tle_line1, tle_line2 = tle
    satrec = twoline2rv(longstr1=tle_line1, longstr2=tle_line2, whichconst=wgs72)
    return satrec


@pytest.fixture
def pysatrec(tle):
    tle_line1, tle_line2 = tle
    satrec = PySatrec.twoline2rv(line1=tle_line1, line2=tle_line2, whichconst=WGS72)
    return satrec


@pytest.fixture
def tle_params(pysatrec) -> TLEParams:
    return satrec_to_tle_params(pysatrec)


def test_satrec_to_tle_params(satrec, satrec_direct, pysatrec):
    """Test that the satrec values are approximately equal.

    Note that we compare only to 8 significant figures because, unlike
    SGP4, we do not truncate ``jdsatepochF`` to 8 significant
    figures.
    """
    equality_fields = SATREC_DIFFERENTIABLE_FIELDS + GRAVITY_MODEL_PARAMETERS
    tests_satrecs = (
        ("c++", satrec),
        ("old twoline2rv", satrec_direct),
        ("python", pysatrec),
    )
    sentinel = object()

    for name, satrec_inst in tests_satrecs:
        params = satrec_to_tle_params(satrec_inst)
        new_satrec = tle_params_to_pysatrec(params)
        for field in equality_fields:
            orig = getattr(satrec, field, sentinel)
            new = getattr(new_satrec, field, sentinel)
            try:
                np.testing.assert_allclose(
                    orig,
                    new,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"{name}:{field} is not equal",
                )
                continue
            except TypeError:
                pass
            assert orig == new, f"{name}:{field} is not equal"


def test_tree_roundtrips(satrec, satrec_direct, pysatrec):
    tests_satrecs = [satrec, satrec_direct, pysatrec]
    sentinel = object()
    for satrec_inst in tests_satrecs:
        leaves, tree = tree_flatten(satrec_inst)
        unflat = tree_unflatten(tree, leaves)
        for slot in SATREC_ALL_FIELDS:
            # Note: Using a sentinel instead of ``None`` ensures
            # missing attributes are missing on both old and new
            orig = getattr(satrec_inst, slot, sentinel)
            new = getattr(unflat, slot, sentinel)
            if orig != new:
                print(slot, orig, new)
        assert isinstance(satrec_inst, type(unflat))


def test_satrec_new(satrec):
    satrec.satnum = 1
    assert satrec.satnum == 1
    satrec.satnum = 99999
    assert satrec.satnum == 99999
    satrec.satnum = 100001
    assert satrec.satnum == 100001
    satrec.satnum = 339999
    assert satrec.satnum == 339999
    with pytest.raises(ValueError):
        satrec.satnum = -1
    with pytest.raises(ValueError):
        satrec.satnum = 340000


def test_sgp4_wrapper(satrec: Satrec, pysatrec: PySatrec):
    tle_params = satrec_to_tle_params(pysatrec)
    for satrec_inst in [satrec, pysatrec]:
        for tsince in [0.0, 1.0, 2.5]:
            r0, v0 = sgp4_wrapper(tle_params, tsince)  # type: ignore
            _, r1, v1 = satrec_inst.sgp4_tsince(tsince)
            r2, v2 = py_sgp4(pysatrec, tsince)  # pylint: disable=invalid-name
            np.testing.assert_allclose(r0, r1, rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(v0, v1, rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(r0, r2, rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(v0, v2, rtol=1e-6, atol=1e-8)


def test_sgp4_wrapper_jacobian_exact(tle_params):
    for tsince in [0.0, 0.1, 1.0, 2.5]:
        r0, v0 = sgp4_wrapper_jacobian_exact(tle_params, tsince)
        assert np.array(r0).shape == (3, len(TLEParams._fields))
        assert np.array(v0).shape == (3, len(TLEParams._fields))


def test_sgp4_wrapper_jacobian_numerical(tle_params):
    for tsince in [0.0, 0.00001, 0.001, 0.1, 1.0, 2.5]:
        numerical = sgp4_wrapper_jacobian_numerical(tle_params, tsince)  # type: ignore
        exact = sgp4_wrapper_jacobian_exact(tle_params, tsince)
        assert tree_structure(numerical) == tree_structure(exact)
        r0, v0 = numerical
        r1, v1 = exact
        np.testing.assert_allclose(r0, r1, rtol=1e-6, atol=1e-4)
        np.testing.assert_allclose(v0, v1, rtol=1e-6, atol=1e-4)
