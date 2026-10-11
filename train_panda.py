"""Compatibility launcher for training.train_panda."""


def main(*args, **kwargs):
    from training.train_panda import main as implementation
    return implementation(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
