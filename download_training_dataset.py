"""Compatibility launcher for cloud.download_training_dataset."""


def main(*args, **kwargs):
    from cloud.download_training_dataset import main as implementation
    return implementation(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
