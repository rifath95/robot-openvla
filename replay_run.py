"""Compatibility launcher for simulation.replay_run."""


def main(*args, **kwargs):
    from simulation.replay_run import main as implementation
    return implementation(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
