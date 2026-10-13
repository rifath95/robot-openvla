"""Stage immutable sweep weights once on container disk; never write to source weights."""
import hashlib
import json
import shutil
from pathlib import Path
from cloud.checkpoint_ready import check_ready


def stage_weights(base, adapters, destination):
    base=Path(base);destination=Path(destination)
    sources={'base':base,**{f'checkpoint_{int(n):06d}':Path(p) for n,p in adapters.items()}}
    for name,source in sources.items():
        if name!='base':check_ready(source)
        elif not (source/'config.json').is_file():raise ValueError('Base model export missing')
    files={name:sorted(p for p in source.rglob('*') if p.is_file() and '.cache' not in p.relative_to(source).parts)
           for name,source in sources.items()}
    needed=sum(p.stat().st_size for rows in files.values() for p in rows)
    destination.parent.mkdir(parents=True,exist_ok=True)
    marker=destination/'staging.json'
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved['sources']!={k:str(v.resolve()) for k,v in sources.items()}:raise ValueError('Staging identity differs')
        for row in saved['files']:
            p=destination/row['path']
            if not p.is_file() or p.stat().st_size!=row['bytes']:raise ValueError('Staged weight missing or truncated')
        return destination
    if shutil.disk_usage(destination.parent).free<needed+2*1024**3:raise ValueError('Container disk needs space for base, adapters and 2 GiB reserve')
    destination.mkdir(exist_ok=True)
    records=[]
    for name,source in sources.items():
        for original in files[name]:
            relative=Path(name)/original.relative_to(source)
            target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.copying')
            digest=hashlib.sha256();count=0
            with original.open('rb') as src,temporary.open('wb') as dst:
                while chunk:=src.read(8*1024**2):dst.write(chunk);digest.update(chunk);count+=len(chunk)
            if count!=original.stat().st_size:raise ValueError('Source changed while copying weights')
            temporary.replace(target)
            records.append(dict(path=str(relative),bytes=count,sha256=digest.hexdigest()))
        print(f'Staged {name} from volume to container disk.',flush=True)
    marker.write_text(json.dumps(dict(sources={k:str(v.resolve()) for k,v in sources.items()},files=records),indent=2)+'\n')
    return destination
