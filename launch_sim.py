"""Compatibility launcher for simulation.launch_sim."""


def main(*args, **kwargs):
    from simulation.launch_sim import main as implementation
    return implementation(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
