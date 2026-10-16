"""Create the frozen 800-trial plan; launch with cloud.cloud_sweep_session."""
import argparse
from datetime import datetime,timezone
import uuid
from common.paths import PROJECT_ROOT
from cloud.bowl_chain_plan import make_uninterrupted_plan,validate_saved
from cloud.scripted_touch_session import write_reports


def prepare(session):
    plan=make_uninterrupted_plan(session);validate_saved(dict(plan=plan,trials=[]))
    name='scripted_uninterrupted_bowl_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    folder=PROJECT_ROOT/'outputs/sweeps'/name;folder.mkdir(parents=True,exist_ok=False)
    write_reports(folder,plan,[])
    return folder


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-session',required=True);a=p.parse_args()
    folder=prepare(a.training_session)
    print('Prepared 800 trials:',folder)
    print('For this experiment, launch on the sweep pod:')
    print('/opt/openvla/bin/python -u -m cloud.color_bowl_sweep --launch --profile inference/sweep_profiles/color_place_bowl_600_uninterrupted.json --training-session',a.training_session)
    print('Delivery: pack/upload on sweep pod, verify HF, then auto-stop. Mac may disconnect.')

if __name__=='__main__':main()
