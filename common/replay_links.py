"""Stable workspace-local links for recorded-motion and action playback."""
import json
import os
from pathlib import Path
import secrets
from urllib.parse import urlencode
from common.paths import PROJECT_ROOT,OUTPUTS_ROOT


CONFIG=OUTPUTS_ROOT/'diagnostics/replay/launcher.json'


def context():
    CONFIG.parent.mkdir(parents=True,exist_ok=True)
    if not CONFIG.exists():
        value=dict(port=8765,token=secrets.token_urlsafe(32))
        try:
            fd=os.open(CONFIG,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'w') as stream:json.dump(value,stream)
        except FileExistsError:pass
    return json.loads(CONFIG.read_text())


def run_name(folder):
    folder=Path(folder).resolve()
    for root in (PROJECT_ROOT/'outputs',PROJECT_ROOT/'datasets'):
        if folder.is_relative_to(root):return folder.relative_to(PROJECT_ROOT).as_posix()
    # Cloud sweep reports use the eventual local path, then are rebound on download.
    volume=Path('/workspace/panda-training/scripted-evaluations')
    if folder.is_relative_to(volume):return 'outputs/sweeps/'+folder.relative_to(volume).as_posix()
    return None


def replay_url(folder,mode='poses',*,base_url=None,token=None):
    if mode not in ('poses','actions'):raise ValueError('Unknown replay mode')
    name=run_name(folder)
    if name is None:return None
    if base_url is None or token is None:
        settings=context();base_url=base_url or f"http://127.0.0.1:{settings['port']}";token=token or settings['token']
    return base_url+'/replay?'+urlencode(dict(run=name,mode=mode,token=token))


def replay_buttons(folder):
    folder=Path(folder)
    pose=replay_url(folder)
    if pose is None:return ''
    links=[f'[▶ Replay]({pose})']
    if (folder/'transitions.jsonl').is_file() and (folder/'initial_state.npz').is_file():
        links.append(f'[▶ Action Replay]({replay_url(folder,"actions")})')
    return ' · '.join(links)


def add_trial_replays(folder,rows,text):
    """Append a launch column to existing per-trial tables; retain other content."""
    if run_name(folder) is None:return text
    lines=text.splitlines()
    for row in rows:
        needle=f'[Run]({row["directory"]}/RUN_REPORT.md) |'
        buttons=replay_buttons(Path(folder)/row['directory'])
        if not buttons:continue
        for i,line in enumerate(lines):
            if needle in line:
                lines[i]=line.split(needle)[0]+needle+' '+buttons+' |'
    for i,line in enumerate(lines):
        if line.endswith('| Details |'):
            lines[i]+=' Replay |'
            if i+1<len(lines):lines[i+1]+='---|'
    return '\n'.join(lines)+'\n'
