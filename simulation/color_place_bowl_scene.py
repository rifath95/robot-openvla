"""One/two/four colored cubes with the unchanged placement bowl and controller."""
import itertools
import math
from common.paths import PROJECT_ROOT
from simulation.color_palette import PALETTE
CAMERA_NAME="bowl_camera"


def slots(count):
    if type(count) is not int or count not in (1,2,4):raise ValueError('Require one, two or four cubes')
    return ['cube']+[f'cube_{i}' for i in range(1,count)]


def scene_path(count):
    slots(count)
    return PROJECT_ROOT/('simulation/scenes/scene_place_bowl.xml' if count==1 else f'simulation/scenes/scene_color_place_bowl_{count}.xml')


def positions(seed,count):
    import numpy as np
    names=slots(count)
    if type(seed) is not int or seed<0:raise ValueError('Nonnegative seed required')
    if count==1:
        from inference.place_bowl_task import positions as single
        p=single(seed);return dict(cubes={'cube':p['cube_xy']},bowl=p['bowl_xy'])
    rng=np.random.default_rng([20261014,seed,46])
    # Separated lattice; every cube/bowl distance stays in the training range.
    candidates=np.array([[x,y] for x,y in ((.52,-.125),(.63,-.125),(.41,.005),(.52,.005))])
    candidates+=rng.uniform(-.004,.004,candidates.shape)
    bowl=np.array([.66,.145])+rng.uniform(-.004,.004,2)
    if seed%2:candidates[:,1]*=-1;bowl[1]*=-1
    candidates=candidates[rng.permutation(len(candidates))]
    return dict(cubes={s:xy.tolist() for s,xy in zip(names,candidates[:count])},bowl=bowl.tolist())


def validate_layout(value,count):
    if not isinstance(value,dict) or set(value)!={'cubes','bowl'} or set(value['cubes'])!=set(slots(count)):raise ValueError('Invalid bowl layout slots')
    points=list(value['cubes'].values())+[value['bowl']]
    for xy in points:
        if not isinstance(xy,list) or len(xy)!=2 or any(type(v) not in (float,int) or not math.isfinite(v) for v in xy):raise ValueError('Invalid coordinates')
        if not .40<=xy[0]<=.68 or abs(xy[1])>.16:raise ValueError('Layout outside bounds')
    if any(math.dist(a,b)<.095 for a,b in itertools.combinations(value['cubes'].values(),2)):raise ValueError('Cubes overlap')
    if any(not .18<=math.dist(xy,value['bowl'])<=.32 for xy in value['cubes'].values()):raise ValueError('Cube/bowl separation outside training range')


def configure_colors(model,names):
    import mujoco
    if set(names)!=set(slots(len(names))) or len(set(names.values()))!=len(names) or not set(names.values())<=set(PALETTE):raise ValueError('Invalid distinct colors')
    for slot,color in names.items():
        geom=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,slot+'_geom')
        if geom<0:raise ValueError('Missing cube geometry')
        model.geom_rgba[geom]=PALETTE[color]


def initialize_scene(env,value,names):
    import mujoco
    from simulation.bowl_scene import configure_grip
    from simulation.panda_actions import make_controller
    from simulation.panda_start_poses import initialize_start_pose
    from simulation.touch_scene import START_POSITION
    validate_layout(value,len(names));configure_colors(env.model,names);configure_grip(env.model)
    m,d=env.model,env.data
    home=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_KEY,'scene_home');mujoco.mj_resetDataKeyframe(m,d,home)
    bowl=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'bowl');m.body_pos[bowl]=[*value['bowl'],.254]
    for slot,xy in value['cubes'].items():
        j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,slot+'_joint');q=int(m.jnt_qposadr[j]);v=int(m.jnt_dofadr[j])
        d.qpos[q:q+7]=[*xy,.271,1,0,0,0];d.qvel[v:v+6]=0
    mujoco.mj_forward(m,d);env.controller=make_controller(m,d)
    reference=env.controller.end_effector_pose()[1].copy()
    start=initialize_start_pose(env,START_POSITION,0,reference,gripper=1,environment_bodies=tuple(names))
    return dict(starting_pose=start,cube_positions=value['cubes'],cube_colors=names,bowl_position=m.body_pos[bowl].tolist(),camera=CAMERA_NAME)


def cube_position(env,slot):
    import mujoco
    return env.data.xpos[mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_BODY,slot)].copy()


class PlacementOutcome:
    def __init__(self,env,target,names):
        if target not in names.values():raise ValueError('Requested cube is absent')
        self.target=target;self.names=dict(names);self.initial={s:cube_position(env,s) for s in names}
        self.maximum={s:0. for s in names};self.placed=set()
    def observe(self,env):
        for s in self.names:self.maximum[s]=max(self.maximum[s],float(cube_position(env,s)[2]-self.initial[s][2]))
    def result(self,env):
        import mujoco,numpy as np
        self.observe(env);m,d=env.model,env.data
        bowl=d.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'bowl')]
        hand=env.controller.end_effector_pose()[0];details={}
        for slot,color in self.names.items():
            pos=cube_position(env,slot);body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,slot);velocity=np.empty(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,body,velocity,0)
            distance=float(np.linalg.norm(pos[:2]-bowl[:2]));height=float(abs(pos[2]-.279));away=float(np.linalg.norm(pos-hand))
            settled=bool(np.linalg.norm(velocity[:3])<.1 and np.linalg.norm(velocity[3:])<.02)
            placed=self.maximum[slot]>=.08 and distance<=.045 and height<=.004 and settled and away>=.08
            if placed:self.placed.add(color)
            details[color]=dict(maximum_lift_metres=self.maximum[slot],bowl_center_distance_metres=distance,height_error_metres=height,hand_distance_metres=away,settled=settled,placed=bool(placed),initial_position=self.initial[slot].tolist(),final_position=pos.tolist())
        correct=self.target in self.placed;wrong=bool(self.placed-{self.target})
        lifted=details[self.target]['maximum_lift_metres']>=.08
        other_lifted=any(v['maximum_lift_metres']>=.08 for c,v in details.items() if c!=self.target)
        outcome='both_placed' if correct and wrong else 'wrong_placed' if wrong else 'correct_placed' if correct else 'target_lifted_not_placed' if lifted else 'wrong_cube_lifted' if other_lifted else 'no_lift'
        return dict(success=correct and not wrong,placement_success=correct and not wrong,correct_placement=correct,wrong_placement=wrong,target_color=self.target,outcome=outcome,placed_colors=sorted(self.placed),cubes=details)


def scene_fingerprint(count):
    import hashlib
    files=(scene_path(count),PROJECT_ROOT/'scene.xml',PROJECT_ROOT/'mujoco_menagerie/franka_emika_panda/panda.xml')
    return hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
