"""Differentiable wrappers around the SGP4 library.

To use this package, you should install ``anomaly`` with the
``extras`` option via

    ``pip install anomaly[extras]``


Sharp bits:

  - The ``jdsatepochF`` field behaves approximately like the ``Satrec.jdsatepochF``
    field when initialized from a TLE via ``Satrec.twoline2rv``. It is not,
    however, rounded to 8 digits in order to preserve differentiability.

    Moreover, when a Python ``Satrec`` is initialized from a TLE via the
    low-level ``io.twoline2rv``, the ``jdsatepochF`` is unset and the
    ``jdsatepoch`` is set to the Julian date of the epoch without rounding.

    Therefore, we recommend using the ``Satrec.twoline2rv`` method to
    initialize a ``Satrec`` from a TLE.

"""
from __future__ import annotations

from numbers import Real
from typing import Any, NamedTuple, Tuple, Type, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node, tree_map

try:
    import sgp4 as _
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Unable to import sgp4. Install the extras via\n"
        "    ``pip install anomaly[extras]``\n"
    ) from e
from sgp4.api import Satrec
from sgp4.earth_gravity import EarthGravity, wgs72
from sgp4.ext import invjday, jday
from sgp4.model import Satrec as PySatrec
from sgp4.propagation import sgp4 as _py_sgp4
from sgp4.propagation import sgp4init

VectorT = Union[Real, np.ndarray, jnp.ndarray]

JDS_1950 = 2433281.5  # Julian date of January 1, 1950


TLE_ELEMENTS = [
    # This is an ``int`` value, to differentiate we need to convert to float
    #'epochyr',  # ``int`` value, don't differentiate
    "epochdays",  # ``float`` value, differentiable.
    "ndot",
    "nddot",
    "bstar",
    "inclo",
    "nodeo",
    "ecco",
    "argpo",
    "mo",
    "no_kozai",
    # 'no', # An alias for no_kozai, don't include.
]
"""TLE elements.

Documentation:

See ``sgp4/__init__.py:420-434`` and ``sgp4/__init__.py:L384-387``.
"""

JULIAN_DATES = [
    "jdsatepoch",
    "jdsatepochF",
]
"""Julian date fields.

These are computed from epochyr and epochdays, but
they are used in initialization. There are two different
and conflicting ways that these fields are initialized in
the ``sgp4`` library.

  1. In ``sgp4/io.py:L147``, ``twoline2rv``, the ``jdsatepochF``
     value is not set, and the ``jdsatepoch`` value contains the
     full Julian date, including the fractional day.

  2. In ``sgp4/model.py:L60-63``, the ``jdsatepochF`` value is set,
     and the ``jdsatepoch`` value contains the half-integer Julian
     date.

Documentation:

    See ``sgp4/__init__.py:438-439``

"""

COMPUTED_ORBIT_PROPERTIES = [
    "a",
    "altp",
    "alta",
    "argpdot",  # Related to Om, at the very least...
    # Note: ``gsto`` is listed in code as a deep space variable,
    # but it appears in the __init__ documentation as a computed
    # orbit property.
    #
    # It does appear to be used in deep-space mode, but not in
    # near-earth mode.
    #
    # cf. propagation.py:1328
    "gsto",  # deep-space  (TODO: is it deep-space?)
    "mdot",
    "nodedot",
]
"""Computed orbit properties.

Documentation:

See ``__init__.py:441-457``.
"""

#### RESULTS OF MOST RECENT PROPAGATION ####
# __init__.py: 469-478
RESULTS_OF_MOST_RECENT_PROPAGATION = [
    # The time you gave when you most recently asked SGP4
    # to compute this satellite’s position,
    # measured in minutes before (negative) or after (positive)
    # the satellite’s epoch.
    "t",
]


MEAN_ELEMENTS_FROM_MOST_RECENT_PROPAGATION = [
    "Om",
    "am",
    "em",
    "mm",
    "nm",
    "om",
    "im",  # propagation.py:L1799
]
"""Mean elements from most recent propagation.

Documentation:

See ``__init__.py:L494-510``.
"""


GRAVITY_MODEL_PARAMETERS = [
    "tumin",
    "xke",
    "mu",
    "radiusearthkm",
    "j2",
    "j3",
    "j4",
    "j3oj2",
]
"""Gravity model parameters.

Documentation:
See ``__init__.py:512-522`` and ``propagation.py:2027``.

The tuple is often referred to as `whichconst`, except
in model.py, where `whichconst` is an enumerated type.

When a tuple, the structure is:

    ``tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2 = whichconst``

The order here matches the documentation in ``__init__.py``.
"""


UNDOCUMENTED_VARIABLES = [
    "no_unkozai",
]
"""Undocumented variables in the ``sgp4.Satrec`` object.

  - ``no_unkozai``: This value is set in ``propagator.py:1387`` by ``_initl(...)``.
     The value is set in ``propagator.py:1164`` as a rescaling of ``no_kozai``.

"""

NEAR_EARTH_VARIABLES = [
    "aycof",
    "con41",
    "cc1",
    "cc4",
    "cc5",
    "d2",
    "d3",
    "d4",
    "delmo",
    "eta",
    # 'argpdot', # Appears in COMPUTED ORBIT PROPERTIES above.
    "omgcof",
    "sinmao",
    # 't',  # Appears in RESULTS OF MOST RECENT PROPAGATION above.
    "t2cof",
    "t3cof",
    "t4cof",
    "t5cof",
    "x1mth2",
    "x7thm1",
    # 'mdot',  # Appears in COMPUTED ORBIT PROPERTIES above.
    # 'nodedot',  # Appears in COMPUTED ORBIT PROPERTIES above.
    "xlcof",
    "xmcof",
    "nodecf",
]
"""Near-earth variables in the `sgp4.Satrec` object.

Documentation:

See ``propagator.py:1303-1312``.
"""

DEEP_SPACE_VARIABLES = [
    "d2201",
    "d2211",
    "d3210",
    "d3222",
    "d4410",
    "d4422",
    "d5220",
    "d5232",
    "d5421",
    "d5433",
    "dedt",
    "del1",
    "del2",
    "del3",
    "didt",
    "dmdt",
    "dnodt",
    "domdt",
    "e3",
    "ee2",
    "peo",
    "pgho",
    "pho",
    "pinco",
    "plo",
    "se2",
    "se3",
    "sgh2",
    "sgh3",
    "sgh4",
    "sh2",
    "sh3",
    "si2",
    "si3",
    "sl2",
    "sl3",
    "sl4",
    # 'gsto', # Appears in COMPUTED_ORBIT_PROPERTIES
    "xfact",
    "xgh2",
    "xgh3",
    "xgh4",
    "xh2",
    "xh3",
    "xi2",
    "xi3",
    "xl2",
    "xl3",
    "xl4",
    "xlamo",
    "zmol",
    "zmos",
    "atime",
    "xli",
    "xni",
]
"""Deep-space variables in the `sgp4.Satrec` object.

Documentation:

See ``propagator.py:1314-1333``.
"""

SATREC_DIFFERENTIABLE_FIELDS = (
    TLE_ELEMENTS
    + ["jdsatepoch"]
    # + JULIAN_DATES
    # + COMPUTED_ORBIT_PROPERTIES
    # + RESULTS_OF_MOST_RECENT_PROPAGATION
    # + MEAN_ELEMENTS_FROM_MOST_RECENT_PROPAGATION
    # + GRAVITY_MODEL_PARAMETERS
    # + UNDOCUMENTED
    # + NEAR_EARTH_VARIABLES
    # + DEEP_SPACE_VARIABLES
)

SATREC_ALL_FIELDS = PySatrec.__slots__


class TLEParams(NamedTuple):
    """Differentiable orbital parameters that appear in the TLE.

    We use the names that are used in SGP4, see docstrings
    for details on meaning.

    Variables of type VectorT can be differentiated against,
    while others cannot.

    For map between these coeffs and the TLE fields, see:
    https://github.com/brandon-rhodes/python-sgp4/blob/29ca790d228beb81a9ad67062676f37ccb47c118/sgp4/__init__.py#L420-L434
    https://github.com/brandon-rhodes/python-sgp4/blob/18e28ec05948631e8894fcc85bb623177fe913f0/sgp4/io.py#L177-L183
    """

    bstar: VectorT
    """sgp4 type drag coefficient, kg/m2er"""

    ecco: VectorT
    """eccentricity"""

    jdsatepoch: VectorT
    """Julian date at epoch.

    Note: We use the convention that this number
    includes fractional days. This is what happens
    when creating a record with ``sgp4.io.twoline2rv``.
    When created with ``Satrec.twoline2rv``, the
    date is split betwen ``jdsatepoch`` (which is
    always a half-integer) and ``jdsatepochF``
    which is the fractional component.
    """

    argpo: VectorT
    """argument of perigee (output if ds)"""

    inclo: VectorT
    """inclination"""

    mo: VectorT
    """mean anomaly (output if ds)"""

    nodeo: VectorT
    """right ascension of ascending node"""

    ndot: VectorT
    """First derivative of mean motion.

    Loaded from the TLE, but otherwise ignored.
    """

    nddot: VectorT
    """Second derivative of mean motion.

    Loaded from the TLE, but otherwise ignored.
    """

    no_kozai: VectorT
    """Mean motion."""


Satrecs = Union[Satrec, PySatrec]
SatrecT = TypeVar("SatrecT", bound=Satrecs)


def tle_params_to_pysatrec(
    tle_params: TLEParams,
    *,
    whichconst: EarthGravity = wgs72,
    opsmode="i",
) -> PySatrec:
    """Convert TLE parameters to a ``sgp4.model.Satrec`` object."""
    # pylint: disable=no-member
    satrec = PySatrec()
    for field, value in tle_params._asdict().items():
        setattr(satrec, field, value)
    # The ``type: ignore`` avoids unbound variable warnings.
    sgp4init(
        whichconst=whichconst,
        opsmode=opsmode,
        satn=getattr(satrec, "satnum", 0),
        epoch=tle_params.jdsatepoch - JDS_1950,
        xbstar=satrec.bstar,  # type: ignore
        xndot=satrec.ndot,  # type: ignore
        xnddot=satrec.nddot,  # type: ignore
        xecco=satrec.ecco,  # type: ignore
        xargpo=satrec.argpo,  # type: ignore
        xinclo=satrec.inclo,  # type: ignore
        xmo=satrec.mo,  # type: ignore
        xno_kozai=satrec.no_kozai,  # type: ignore
        xnodeo=satrec.nodeo,  # type: ignore
        satrec=satrec,
    )

    # Fix jdsatepoch / jdsatepochF to approximately match the
    # behavior of ``Satrec.twoline2rv``.
    jdsatepoch, jdsatepochF = jnp.divmod(  # pylint: disable=invalid-name
        tle_params.jdsatepoch - JDS_1950,
        1.0,
    )
    satrec.jdsatepoch = jdsatepoch + JDS_1950

    # Note: We do _not_ round this to 8 digits, unlike ``Satrec.twoline2rv``.
    satrec.jdsatepochF = jdsatepochF

    year, *_ = invjday(satrec.jdsatepoch)
    jan0 = jday(year, 1, 0, 0, 0, 0.0)
    satrec.epochyr = year % 100
    satrec.epochdays = satrec.jdsatepoch - jan0 + satrec.jdsatepochF
    return satrec


def tle_params_to_satrec(
    tle_params: TLEParams,
    *,
    whichconst: EarthGravity = wgs72,
    opsmode="i",
) -> Satrec:
    """Convert TLE parameters to a ``sgp4.api.Satrec`` object."""
    return to_satrec(
        tle_params_to_pysatrec(
            tle_params=tle_params,
            whichconst=whichconst,
            opsmode=opsmode,
        )
    )


def to_satrec(satrec: Satrecs) -> Satrec:
    """Convert a ``sgp4.model.Satrec`` object to a ``sgp4.api.Satrec`` object.

    If the object is already a ``sgp4.api.Satrec`` object, a copy is returned.
    """
    flat, extra = flatten_satrec(satrec)
    return _unflatten_all_satrecs(extra, flat, cls_=Satrec)


def to_pysatrec(satrec: Satrecs) -> PySatrec:
    """Convert a ``sgp4.model.Satrec`` object to a ``sgp4.api.PySatrec`` object.

    If the object is already a ``sgp4.model.PySatrec`` object, a copy is returned.
    """
    flat, extra = flatten_satrec(satrec)
    return _unflatten_all_satrecs(extra, flat, cls_=PySatrec)


def satrec_to_tle_params(
    satrec: Satrecs,
) -> TLEParams:
    """Convert a ``sgp4.api.Satrec`` object to a ``TLEParams`` object."""
    kwargs = {field: getattr(satrec, field) for field in TLEParams._fields}
    kwargs["jdsatepoch"] += getattr(satrec, "jdsatepochF", 0.0)
    return TLEParams(**kwargs)


def flatten_satrec(
    satrec: Satrecs,
) -> Tuple[Tuple[VectorT, ...], Tuple[str, Any]]:
    """Flatten a ``sgp4.api.Satrec`` object into a tuple of arrays."""
    slots = PySatrec.__slots__
    fields = SATREC_DIFFERENTIABLE_FIELDS
    extra = []
    for slot in set(slots) - set(fields):
        if hasattr(satrec, slot):
            extra.append((slot, getattr(satrec, slot)))
    flat = tuple(getattr(satrec, field) for field in fields)
    return flat, tuple(extra)


flatten_pysatrec = flatten_satrec


def unflatten_satrec(extra: Tuple[str, Any], flat: Tuple[VectorT, ...]) -> Satrec:
    """Unflatten a tuple of arrays into a ``sgp4.api.Satrec`` object."""
    return _unflatten_all_satrecs(extra=extra, flat=flat, cls_=Satrec)


def unflatten_pysatrec(extra: Tuple[str, Any], flat: Tuple[VectorT, ...]) -> PySatrec:
    """Unflatten a tuple of arrays into a ``sgp4.model.PySatrec`` object."""
    return _unflatten_all_satrecs(extra=extra, flat=flat, cls_=PySatrec)


def _unflatten_all_satrecs(
    extra: Tuple[str, Any], flat: Tuple[VectorT, ...], *, cls_: Type[SatrecT]
) -> SatrecT:
    """Unflatten a tuple of arrays into a object of type ``cls_``."""
    result: SatrecT = cls_()
    fields = SATREC_DIFFERENTIABLE_FIELDS
    for slot, value in zip(fields, flat):
        setattr(result, slot, value)
    for slot, value in extra:
        setattr(result, slot, value)
    return result


try:
    # Sharp bit: JAX tree flattening/unflattening will convert
    # a C++ Satrec to a slower Python Satrec.
    register_pytree_node(
        Satrec,
        flatten_func=flatten_satrec,
        unflatten_func=unflatten_satrec,  # type: ignore
    )
    register_pytree_node(
        PySatrec,
        flatten_func=flatten_pysatrec,
        unflatten_func=unflatten_pysatrec,
    )
except ValueError:  # pragma: no cover
    # With auto-reload in Jupyter, ``register_pytree_node``
    # may be called twice for ``Satrec``.
    import warnings

    warnings.warn(
        "Warning: node registration for Satrec not updated.",
        category=RuntimeWarning,
    )


def sgp4_wrapper(
    tle_params: TLEParams, tsince: Real, *, whichconst: EarthGravity = wgs72
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """SGP4 propagation that is exactly differentiable.

    This differs from ``sgp4.propagation.sgp4`` in the following ways:
      1. The input record is unmodified.
      2. Errors are not returned. In the case of an error, the resulting
         vectors are all zero.
    """
    satrec = tle_params_to_pysatrec(tle_params, whichconst=whichconst)
    result = _py_sgp4(satrec, tsince, whichconst)
    return result


sgp4_wrapper_jacobian_exact = jax.jacfwd(sgp4_wrapper, 0)


def sgp4_wrapper_jacobian_numerical(
    tle_params: TLEParams, tsince: Real, *, whichconst: EarthGravity = wgs72, eps=1e-7
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """SGP4 propagation Jacobian via central-difference."""
    kwargs = {}
    for field in TLEParams._fields:
        params_0 = tle_params._replace(**{field: getattr(tle_params, field) - eps / 2})
        params_1 = tle_params._replace(**{field: getattr(tle_params, field) + eps / 2})
        _, r0, v0 = tle_params_to_pysatrec(params_0, whichconst=whichconst).sgp4_tsince(
            tsince
        )
        _, r1, v1 = tle_params_to_pysatrec(params_1, whichconst=whichconst).sgp4_tsince(
            tsince
        )
        dr_approx = (np.array(r1) - np.array(r0)) / eps  # type: ignore
        dv_approx = (np.array(v1) - np.array(v0)) / eps  # type: ignore
        kwargs[field] = (tuple(dr_approx.tolist()), tuple(dv_approx.tolist()))
    return _tree_transpose(TLEParams(**kwargs))


def _tree_transpose(tle_params: TLEParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return tree_map(lambda *xs: TLEParams(*xs), *tle_params)
