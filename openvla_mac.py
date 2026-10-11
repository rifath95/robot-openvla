"""Compatibility launcher for inference.openvla_mac."""


def main(*args, **kwargs):
    from inference.openvla_mac import main as implementation
    return implementation(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
