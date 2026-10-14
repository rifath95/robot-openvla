"""Correct/wrong placement rates and durations for the chained bowl sweep."""
from collections import defaultdict
from common.replay_links import add_trial_replays


def write_report(folder,plan,rows,error=None):
    groups=defaultdict(list)
    for r in rows:groups[r['checkpoint']].append(r)
    lines=['# Color selection and bowl placement sweep','',
        'Requested-color placement requires prior lift ≥8 cm, settled cube within 4.5 cm of bowl center and 4 mm of resting height, and empty hand ≥8 cm away. Wrong-cube placement is not success. Sampling off; each request has a fresh reset.', '',
        '| Model | Trials | Correct placements | Success rate | Clean successes | Wrong/both placements | Target lifted, not placed | Wrong cube lifted | No lift | Controller failures | Infrastructure failures |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cp,trials in sorted(groups.items()):
        count=lambda outcome:sum(r.get('outcome')==outcome for r in trials)
        good=sum(r.get('placement_success',False) for r in trials)
        clean=sum(r.get('placement_success',False) and r.get('phase')=='Completed' for r in trials)
        lines.append(f'| {"Base" if cp==0 else cp} | {len(trials)} | {good} | {good/len(trials):.1%} | {clean} | {count("wrong_placed")+count("both_placed")} | {count("target_lifted_not_placed")} | {count("wrong_cube_lifted")} | {count("no_lift")} | {sum(r.get("controller_failed",False) for r in trials)} | {sum(r.get("infrastructure_failed",False) for r in trials)} |')
    lines+=['','| Model | Scene group | Target | Trials | Correct | Wrong/both | Mean trial seconds |','|---|---|---|---:|---:|---:|---:|']
    sub=defaultdict(list)
    for r in rows:sub[(r['checkpoint'],r['color_set'],r['target_color'])].append(r)
    for (cp,g,target),rs in sorted(sub.items()):
        lines.append(f'| {"Base" if cp==0 else cp} | {g} | {target} | {len(rs)} | {sum(r.get("placement_success",False) and r.get("phase")=="Completed" for r in rs)} | {sum(r.get("wrong_placement",False) for r in rs)} | {sum(r.get("trial_seconds",0) for r in rs)/len(rs):.2f} |')
    lines+=['',f'Wall time: {plan.get("elapsed_seconds",0):.1f} seconds.','',
        '| Model | Group | Seed | Target | Outcome | Phase | Seconds | Details |','|---|---|---:|---|---|---|---:|---|']
    for r in rows:
        lines.append(f'| {r.get("model_label",r["checkpoint"])} | {r["color_set"]} | {r["seed"]} | {r["target_color"]} | {r.get("outcome","unmeasured")} | {r.get("phase","Missing summary")} | {r.get("trial_seconds",0):.2f} | [Run]({r["directory"]}/RUN_REPORT.md) |')
    if error:lines+=['',f'Interrupted: {error}']
    (folder/'REPORT.md').write_text(add_trial_replays(folder,rows,'\n'.join(lines)+'\n'))
