"""Instruction selection and measurements for two-cube touch inference."""
from simulation.color_touch_scene import cube_position, finger_cube_contacts


def select_instruction(instruction=None, target_color=None):
    if target_color is None:
        if instruction is None:
            target_color = 'red'
        else:
            text = instruction.strip().lower().rstrip('.')
            matches = [c for c in ('red', 'blue') if text == f'touch the {c} cube']
            if not matches:
                raise ValueError('Color touch requires a red or blue touch instruction')
            target_color = matches[0]
    if target_color not in ('red', 'blue'):
        raise ValueError('Target color must be red or blue')
    expected = f'touch the {target_color} cube.'
    if instruction is not None and instruction.strip().lower().rstrip('.') != expected.rstrip('.'):
        raise ValueError('Instruction and target color disagree')
    return instruction if instruction is not None else expected, target_color


class ContactOutcome:
    """Track either cube's contact throughout execution, including brief contacts."""
    def __init__(self, env, target_color, color_names=None):
        self.target_color = target_color
        self.color_names=color_names or {'red':'red','blue':'blue'}
        self.initial = {c: cube_position(env, c) for c in ('red', 'blue')}
        self.contacts = {'red': [], 'blue': []}
        self.first_contact = None

    def observe(self, env):
        current = {c: finger_cube_contacts(env.model, env.data, c) for c in self.contacts}
        touching = [c for c in current if current[c]]
        if touching and self.first_contact is None:
            self.first_contact = touching[0] if len(touching) == 1 else 'both'
        for color, contacts in current.items():
            if contacts:
                self.contacts[color] = contacts

    def result(self, env):
        other = 'blue' if self.target_color == 'red' else 'red'
        correct, wrong = bool(self.contacts[self.target_color]), bool(self.contacts[other])
        first=self.color_names.get(self.first_contact,self.first_contact)
        return dict(target_color=self.color_names[self.target_color], correct_cube_contact=correct,
                    wrong_cube_contact=wrong, success=correct and not wrong,
                    first_contact=first,
                    outcome='both_cubes' if correct and wrong else 'wrong_cube' if wrong
                            else 'correct_cube' if correct else 'no_contact',
                    finger_cube_contacts={self.color_names[c]:[{**x,'color':self.color_names[c]} for x in cs]
                                          for c,cs in self.contacts.items()},
                    initial_cubes={self.color_names[c]: p.tolist() for c, p in self.initial.items()},
                    final_cubes={self.color_names[c]: cube_position(env, c).tolist() for c in self.contacts})
