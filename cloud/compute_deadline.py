"""Cooperative computation deadline; delivery is bounded separately by supervisor."""
import os
import time

POLICY='compute_then_delivery_v1'

class ComputeDeadlineReached(RuntimeError):
    pass


def expired():
    value=os.environ.get('PANDA_COMPUTE_DEADLINE_MONOTONIC')
    return bool(value) and time.monotonic()>=float(value)


def check():
    if expired():raise ComputeDeadlineReached('Inner computation deadline reached; preserving completed work for delivery')
