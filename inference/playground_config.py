"""Default model family for the manual playground; unrelated sweeps are unchanged."""

CHECKPOINTS = (6000, 8000, 10000, 12000)
PROFILE_RELATIVE = 'inference/sweep_profiles/color_place_bowl_600_uninterrupted.json'
STAGING_DIRECTORY = '/root/panda-interactive-bowl-uninterrupted'


def validate_host(health, selected=None):
    if (health.get('training_profile') != PROFILE_RELATIVE or
            health.get('checkpoints') != list(CHECKPOINTS) or
            (selected is not None and health.get('training_session') != selected)):
        raise ValueError('Existing playground host uses different checkpoints or a different training run. '
                         'Use a fresh playground pod or explicitly stop the old playground host first.')
