"""Inspect how a planar 2R robot approaches a Jacobian singularity."""

import numpy as np

from robotics_control_lab import Planar2R, is_singular, jacobian, manipulability


def main() -> None:
    arm = Planar2R(0.6, 0.4)
    for q2 in np.linspace(0.0, np.pi / 2.0, 6):
        q = (0.4, float(q2))
        value = manipulability(arm, *q)
        singular = is_singular(arm, *q)
        print(f"q2={q2:+.3f} rad  manipulability={value:.6f}  singular={singular}")
        if not singular:
            print(jacobian(arm, *q))
            print()


if __name__ == "__main__":
    main()
