"""Explicit paired red/blue evaluation; interactive and single-cube defaults stay unchanged."""
from cloud.scripted_touch_session import main

if __name__ == '__main__':
    raise SystemExit(main(default_task='color_touch'))
