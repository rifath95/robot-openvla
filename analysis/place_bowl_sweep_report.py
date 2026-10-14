"""Report physical release into the bowl and retreat, separate from execution failures."""
from collections import defaultdict
from common.replay_links import add_trial_replays


def write_report(folder,plan,rows,error=None):
    groups=defaultdict(list)
    for r in rows:groups[r['checkpoint']].append(r)
    lines=['# Scripted bowl-placement evaluation','',
        'Instruction: `pick up the red cube and place it in the blue bowl.` Sampling off; all models use the same unseen cube-and-bowl layouts.',
        'Success requires prior lift of at least 8 cm, settled cube inside the bowl (within 4.5 cm of center and 4 mm of floor resting height), and empty hand at least 8 cm away.',
        'Checkpoint 0 is pretrained base with this dataset’s action normalization. Controller failures remain visible.','',
        '| Model | Trials | Placements | Success rate | Clean successes | Failed trials |','|---|---:|---:|---:|---:|---:|']
    for cp,trials in sorted(groups.items()):
        label='Base' if cp==0 and plan.get('include_base_model') else str(cp)
        success=sum(bool(r.get('placement_success')) for r in trials)
        lines.append(f'| {label} | {len(trials)} | {success} | {success/len(trials):.1%} | '
            f'{sum(bool(r.get("placement_success")) and r["phase"]=="Completed" for r in trials)} | {sum(r["phase"]=="Failed" for r in trials)} |')
    lines+=['',f'Wall time including staging and model loading: {plan.get("elapsed_seconds",0):.1f} seconds.','',
        '| Model | Seed | Placement | Max lift (cm) | Bowl center distance (cm) | Hand distance (cm) | Phase | Details |',
        '|---|---:|---|---:|---:|---:|---|---|']
    for r in rows:
        lines.append(f'| {r.get("model_label",r["checkpoint"])} | {r["seed"]} | {r.get("placement_success",False)} | '
            f'{100*r.get("maximum_lift_metres",0):.2f} | {100*r.get("bowl_center_distance_metres",0):.2f} | '
            f'{100*r.get("empty_hand_cube_distance_metres",0):.2f} | {r["phase"]} | [Run]({r["directory"]}/RUN_REPORT.md) |')
    if error:lines+=['',f'Launcher ended early: {error}']
    (folder/'REPORT.md').write_text(add_trial_replays(folder,rows,'\n'.join(lines)+'\n'))
