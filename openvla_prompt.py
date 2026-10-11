"""Inference prompt matching the official OpenVLA RLDS batch transform."""


def action_prompt(instruction):
    # RLDSBatchTransform lowercases language before PurePromptBuilder wraps it.
    # Preserve that convention, including world X/Y becoming world x/y.
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"
