"""Compatibility launcher for simulation.end_effector_demo."""


def main(*args, **kwargs):
    from simulation.end_effector_demo import main as implementation
    return implementation(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
