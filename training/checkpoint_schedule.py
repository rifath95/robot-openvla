"""Optional explicit checkpoint updates; interval saves remain the default."""
import json
import os


def validate_schedule(updates, maximum):
    if updates is None:
        return None
    if (not isinstance(updates, list) or not updates or
        any(type(update) is not int or update < 0 or update > maximum for update in updates) or
        updates != sorted(set(updates))):
        raise ValueError('checkpoint_updates must be sorted unique integers between 0 and max_steps')
    return updates


def checkpoint_due(update, interval):
    schedule = os.environ.get('PANDA_CHECKPOINT_UPDATES', '')
    if schedule:
        return update in json.loads(schedule)
    return update > 0 and update % interval == 0
