"""Compatibility launcher for cloud.cloud_session."""


def main(*args, **kwargs):
    from cloud.cloud_session import main as implementation
    return implementation(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
