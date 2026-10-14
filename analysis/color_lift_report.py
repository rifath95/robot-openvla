"""Color selection and physical lifting, with conditional choice and no-lift counts."""
from collections import defaultdict
from common.replay_links import add_trial_replays


def write_report(folder,plan,rows,error=None):
    groups=defaultdict(list)
    for r in rows:groups[r['checkpoint']].append(r)
    lines=['# Two-cube color selection and lifting','',
        'Sampling off. Request each color separately in the same layout; color assignments rotate across positions.',
        'Lift requires bilateral contact near the grasp point, at least 5 cm above start for one continuous simulator second.',
        'These adapters were fine-tuned on red-only, single-cube lifting. Choice accuracy counts correct lifts among trials with a lift; no-lift failures remain in overall success.','',
        '| Checkpoint | Trials | Correct-only lift | Wrong/both lift | No lift | Correct choice among lifts |',
        '|---|---:|---:|---:|---:|---:|']
    def counts(trials):
        correct=sum(bool(r.get('correct_lift')) and not r.get('wrong_lift') for r in trials)
        wrong=sum(bool(r.get('wrong_lift')) for r in trials)
        return correct,wrong,len(trials)-correct-wrong
    for cp,trials in sorted(groups.items()):
        c,w,n=counts(trials);choice=f'{c/(c+w):.1%}' if c+w else '—'
        lines.append(f'| {cp} | {len(trials)} | {c} ({c/len(trials):.1%}) | {w} | {n} | {choice} |')
    lines+=['','| Checkpoint | Pair | Target | Trials | Correct-only lifts | Wrong/both | No lift |','|---|---|---|---:|---:|---:|---:|']
    for cp,trials in sorted(groups.items()):
        for pair in plan['color_pairs']:
            for target in pair:
                subset=[r for r in trials if r.get('color_pair')==':'.join(pair) and r.get('target_color')==target]
                if not subset:continue
                c,w,n=counts(subset);lines.append(f'| {cp} | {"/".join(pair)} | {target} | {len(subset)} | {c} | {w} | {n} |')
    lines+=['',f'Wall time including staging and loading: {plan.get("elapsed_seconds",0):.1f} seconds.','',
        '| Checkpoint | Pair | Seed | Target | Outcome | Phase | Details |','|---|---|---:|---|---|---|---|']
    for r in rows:
        lines.append(f'| {r["checkpoint"]} | {r.get("color_pair","")} | {r["seed"]} | {r.get("target_color","")} | {r.get("outcome","unmeasured")} | {r["phase"]} | [Run]({r["directory"]}/RUN_REPORT.md) |')
    if error:lines+=['',f'Launcher ended early: {error}']
    (folder/'REPORT.md').write_text(add_trial_replays(folder,rows,'\n'.join(lines)+'\n'))
