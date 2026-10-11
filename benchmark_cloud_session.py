"""Compatibility launcher for cloud.benchmark_cloud_session."""


def main(*args, **kwargs):
    from cloud.benchmark_cloud_session import main as implementation
    return implementation(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
