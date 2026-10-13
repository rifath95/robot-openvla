"""Viewer-only instruction overlays, separate from camera rendering."""
import textwrap
import mujoco


def show_instruction(viewer, instruction, *other_overlays):
    """Preserve controls/status while showing the task in the upper right."""
    text = instruction if instruction else 'No recorded instruction'
    viewer.set_texts([*other_overlays,
        (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
         'Instruction\n' + '\n'.join(textwrap.wrap(text, width=42)), '')])
