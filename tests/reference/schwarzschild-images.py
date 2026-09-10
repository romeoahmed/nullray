"""Generate a high-precision, disk-occluded Schwarzschild stellar-image fixture.

Requires mpmath 1.3.0 only for fixture regeneration, not application execution.
Run: python schwarzschild-images.py [decimal_precision] > fixture.json

The independent planar orbit obeys (du/dpsi)^2 = 1/b^2 - u^2 + 2u^3.
Factoring at its outer turn converts the angular sweep to Legendre F and K:
https://dlmf.nist.gov/19.2.E4. mpmath takes m=k^2, not the DLMF modulus k.
"""

import json
import math
import struct
import sys

import mpmath as mp


def binary32(value):
    """Match the application's uploaded frame and catalogue coordinates."""
    return struct.unpack('<f', struct.pack('<f', value))[0]


def generate(precision=80):
    """Enumerate three image orders on both sides of a polar observer."""
    if precision < 40:
        raise ValueError('Use at least 40 decimal digits for this fixture.')
    mp.mp.dps = precision
    radius = mp.mpf(30)
    width, height = 640, 360
    zoom = mp.mpf(binary32(1.2))
    source_y = binary32(0.01 / math.sqrt(1.0001))
    source_z = binary32(-1 / math.sqrt(1.0001))
    offset = mp.atan2(mp.mpf(source_y), -mp.mpf(source_z))
    lapse = mp.sqrt(1 - 2 / radius)

    def orbit(turning_radius):
        """Total great-circle angle from observer to infinity, through one turn."""
        u2 = 1 / turning_radius
        discriminant = mp.sqrt(mp.mpf('.25') + u2 - 3 * u2 * u2)
        u1 = (mp.mpf('.5') - u2 - discriminant) / 2
        u3 = (mp.mpf('.5') - u2 + discriminant) / 2
        parameter = (u2 - u1) / (u3 - u1)
        factor = mp.sqrt(2 / (u3 - u1))
        source_phase = mp.ellipf(mp.asin(mp.sqrt(-u1 / (u2 - u1))), parameter)
        observer_phase = mp.ellipf(
            mp.asin(mp.sqrt((1 / radius - u1) / (u2 - u1))), parameter
        )
        sweep = factor * (2 * mp.ellipk(parameter) - source_phase - observer_phase)
        return sweep, (u1, u2, parameter, factor, observer_phase)

    def impact(turning_radius):
        return turning_radius / mp.sqrt(1 - 2 / turning_radius)

    records = []
    for order in range(3):
        for side in [1, -1]:
            target = (2 * order + 1) * mp.pi - side * offset
            lower, upper = mp.mpf(3) + mp.mpf('1e-30'), radius
            for _ in range(4 * precision):
                middle = (lower + upper) / 2
                if orbit(middle)[0] > target:
                    lower = middle
                else:
                    upper = middle
            turn = (lower + upper) / 2
            sweep, data = orbit(turn)
            b = impact(turn)
            rho = mp.tan(mp.asin(b * lapse / radius))
            derivative = mp.diff(lambda value: orbit(value)[0], turn) / mp.diff(impact, turn)
            impact_slope = radius / (lapse * (1 + rho * rho) ** mp.mpf('1.5'))
            jacobian = abs(mp.sin(sweep) * derivative * impact_slope) / rho * (zoom / height) ** 2
            u1, u2, parameter, factor, observer_phase = data
            crossings = []
            for index in range(2 * order + 1):
                angle = (mp.mpf(index) + mp.mpf('.5')) * mp.pi
                sine = mp.ellipfun('sn', observer_phase + angle / factor, parameter)
                crossings.append(float(1 / (u1 + (u2 - u1) * sine * sine)))
            records.append({
                'order': order,
                'side': side,
                'pixel': [float((width - 1) / 2 + side * rho * height / zoom), (height - 1) / 2],
                'turningRadius': float(turn),
                'impact': float(b),
                'sweep': float(sweep),
                'jacobian': float(jacobian),
                'crossingRadii': crossings,
                'occulted': any(6 <= value <= 64 for value in crossings),
                'angularResidual': mp.nstr(abs(sweep - target), 6),
            })
    return {
        'schema': 1,
        'generator': 'tests/reference/schwarzschild-images.py',
        'reference': 'Independent Schwarzschild planar cubic, Legendre integrals, and implicit image Jacobian.',
        'mpmathVersion': mp.__version__,
        'decimalPrecision': precision,
        'frame': {'width': width, 'height': height, 'radius': float(radius), 'inclination': 0,
                  'spin': 0, 'charge': 0, 'zoom': float(zoom), 'diskInner': 6, 'diskOuter': 64},
        'source': [0, source_y, source_z],
        'sourceOffset': float(offset),
        'energy': float(lapse),
        'cases': records,
    }


if __name__ == '__main__':
    print(json.dumps(generate(int(sys.argv[1]) if len(sys.argv) > 1 else 80), indent=2))
