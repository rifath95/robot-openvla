"""Fixed display colors for evaluation; collision geometry stays unchanged."""
import mujoco

PALETTE = {
    'red': (.85, .15, .1, 1), 'blue': (.1, .25, .9, 1),
    'black': (.025, .025, .025, 1), 'purple': (.55, .10, .75, 1),
    'green': (.08, .65, .15, 1), 'yellow': (.95, .85, .05, 1),
    'orange': (.95, .40, .05, 1), 'white': (.92, .92, .92, 1),
    'pink': (.95, .35, .65, 1), 'brown': (.4, .18, .05, 1),
    'cyan': (.05, .8, .85, 1),
}
DEFAULT_PAIRS = ['black:purple', 'green:yellow', 'orange:white']


def color_mapping(colors):
    if len(colors)!=2 or colors[0]==colors[1] or any(c not in PALETTE for c in colors):
        raise ValueError('Choose two distinct named palette colors')
    return dict(zip(('red','blue'),colors))


def configure_colors(model, names):
    if set(names)!= {'red','blue'}:
        raise ValueError('Color mapping must cover both cube identities')
    color_mapping([names['red'],names['blue']])
    for slot,geom_name in [('red','cube_geom'),('blue','blue_cube_geom')]:
        geom=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,geom_name)
        if geom<0:raise ValueError('Cube geometry missing')
        model.geom_rgba[geom]=PALETTE[names[slot]]


def parse_pairs(values=None):
    pairs=[]
    for value in values or DEFAULT_PAIRS:
        colors=value.split(':')
        color_mapping(colors)
        if frozenset(colors) in [frozenset(p) for p in pairs]:
            raise ValueError('Duplicate color pair')
        pairs.append(colors)
    return pairs


def select_instruction(instruction, target_color, colors):
    names=color_mapping(colors)
    if target_color is None:
        if instruction is None:target_color=colors[0]
        else:
            text=instruction.strip().lower().rstrip('.')
            matches=[c for c in colors if text==f'touch the {c} cube']
            if not matches:raise ValueError('Instruction must request a visible cube color')
            target_color=matches[0]
    if target_color not in colors:raise ValueError('Requested color is not present in this scene')
    expected=f'touch the {target_color} cube.'
    if instruction is not None and instruction.strip().lower().rstrip('.')!=expected.rstrip('.'):
        raise ValueError('Instruction and target color disagree')
    slot=next(slot for slot,name in names.items() if name==target_color)
    return instruction if instruction is not None else expected,slot,names
