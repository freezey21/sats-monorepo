"""Two-body gravity force and gravity-gradient moment."""

import numpy as np

from arda.common.rotations import Quaternion
from arda.constants import EARTH_GRAVITATIONAL_PARAMETER_M3_PER_S2
from arda.types import FloatArray, ForceMoment, Satellite, State

# Constants
R = 6.3781363 * 10**6  # equatorial radius of earth (m)
G = 6.6743 * 10**-11  # gravitational constant (m^3 kg^-1 s^-2)
M = 5.97219 * 10**24  # mass of earth (kg)
J_2, J_3, J_4 = (
    1.08263 * 10**-3,
    -2.5327 * 10**-6,
    -1.6196 * 10**-6,
)  # J_n coefficients (unitless)


def gravity_force_moment(
    state: State,
    satellite: Satellite,
) -> ForceMoment:
    return ForceMoment(
        force_eci_n=_gravity_force_eci_n(state.position_eci_m, satellite.mass_kg),
        moment_eci_nm=_gravity_gradient_moment_eci_nm(
            state.position_eci_m,
            state.quat_eci_to_body,
            satellite.inertia_body_kgm2,
        ),
    )


def _gravity_force_eci_n(position_eci_m: FloatArray, mass_kg: float) -> FloatArray:
    radius_m = np.linalg.norm(position_eci_m)
    mu = G * M  # gravitational parameter (m^3 s^-2)
    x, y, z = (
        position_eci_m[0],
        position_eci_m[1],
        position_eci_m[2],
    )  # dimensions of position (m)
    pos_vec = np.array([x, y, z])
    a_newton = -1 * (mu / radius_m**3) * pos_vec
    s = z/radius_m #to simplify calculations
    a_J2 = (
        (-3 / 2)
        * (J_2)
        * (mu / radius_m**2)
        * (R**2 / radius_m**2)
        * (
            np.array(
                [
                    (x / radius_m) * (1 - 5 * (s**2)),
                    (y / radius_m) * (1 - 5 * (s**2)),
                    (z / radius_m) * (3 - 5 * (s**2)),
                ]
            )
        )
    )
    a_J3 = (
        (-5 / 2)
        * (J_3)
        * (mu / radius_m**2)
        * (R**3 / radius_m**3)
        * (
            np.array(
                [
                    (x / radius_m) * (3 * (z / radius_m) - 7 * (s**3)),
                    (y / radius_m) * (3 * (y / radius_m) - 7 * (s**3)),
                    (3 / 5) - 6 * (s**2) + 7 * (s**4),
                ]
            )
        )
    )
    a_J4 = (
        (5 / 8)
        * (J_4)
        * (mu / radius_m**2)
        * (R**4 / radius_m**4)
        * (
            np.array(
                [
                    (x / radius_m)
                    * (3 - 42 * (s**2) + 63 * (s**4)),
                    (y / radius_m)
                    * (3 - 42 * (s**2) + 63 * (s**4)),
                    (z / radius_m)
                    * (15 - 70 * (s**2) + 63 * (s**4)),
                ]
            )
        )
    )
    a_tot = a_newton + a_J2 #only J2 for now

    return mass_kg * a_tot


def _gravity_gradient_moment_eci_nm(
    _position_eci_m: FloatArray,
    _quat_eci_to_body: Quaternion,
    _inertia_body_kgm2: FloatArray,
) -> FloatArray:
    """Return zero until gravity-gradient dynamics are implemented."""
    return np.zeros(3)


def test_gravity_force_moment() -> None:
    state = State(
        position_eci_m=np.array([7_000_000.0, 7_000_000.0, 0.0]),
        velocity_eci_m_per_s=np.zeros(3),
        quat_eci_to_body=Quaternion.identity(),
        angular_velocity_body_wrt_eci_eci_rad_per_s=np.zeros(3),
    )
    satellite = Satellite(10.0, np.diag([1.0, 2.0, 3.0]))

    force_moment = gravity_force_moment(state, satellite)

    assert np.all(force_moment.force_eci_n[:2] < 0.0)
    np.testing.assert_array_equal(force_moment.moment_eci_nm, np.zeros(3))
