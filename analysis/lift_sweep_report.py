"""Report actual held lift success, keeping controller and infrastructure failures visible."""
from collections import defaultdict
from common.replay_links import add_trial_replays


def write_report(folder,plan,rows,error=None):
    groups=defaultdict(list)
    for row in rows:groups[row['checkpoint']].append(row)
    lines=['# Scripted lift evaluation','',
        'Instruction: `pick up the red cube.` Sampling off; all models share the same unseen positions.',
        'Success requires bilateral finger contact, cube near the grasp point, and at least 5 cm of lift for one continuous simulator second. Movement may continue.',
        'Prediction wait time does not count. Checkpoint 0 is the pretrained base with this dataset’s action normalization.','',
        '| Model | Trials | Lift successes | Success rate | Clean successes | Failed trials |',
        '|---|---:|---:|---:|---:|---:|']
    for checkpoint,trials in sorted(groups.items()):
        label='Base' if checkpoint==0 and plan.get('include_base_model') else str(checkpoint)
        successes=sum(bool(r.get('lift_success')) for r in trials)
        clean=sum(bool(r.get('lift_success')) and r['phase']=='Completed' for r in trials)
        lines.append(f'| {label} | {len(trials)} | {successes} | {successes/len(trials):.1%} | {clean} | {sum(r["phase"]=="Failed" for r in trials)} |')
    lines+=['',f'Wall time for staging, loading and trials: {plan.get("elapsed_seconds",0):.1f} seconds.','',
        '| Model | Seed | Lift success | Max lift (cm) | Max held above threshold (s) | Phase | Details |',
        '|---|---:|---|---:|---:|---|---|']
    for r in rows:
        lines.append(f'| {r.get("model_label",r["checkpoint"])} | {r["seed"]} | {r.get("lift_success",False)} | '
            f'{100*r.get("maximum_lift_metres",0):.2f} | {r.get("maximum_grasp_seconds",0):.2f} | {r["phase"]} | [Run]({r["directory"]}/RUN_REPORT.md) |')
    if error:lines+=['',f'Launcher ended early: {error}']
    (folder/'REPORT.md').write_text(add_trial_replays(folder,rows,'\n'.join(lines)+'\n'))
