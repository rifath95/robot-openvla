"""Copy immutable training inputs to container storage; outputs stay on the volume."""
import hashlib
import json
from pathlib import Path
import shutil
import time


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(8*1024**2):h.update(chunk)
    return h.hexdigest()


def input_files(model,dataset):
    sources={'base':Path(model).resolve(),'dataset':Path(dataset).resolve()}
    if not (sources['base']/'config.json').is_file():raise ValueError('Base model config missing')
    if not json.loads((sources['dataset']/'verification.json').read_text()).get('passed'):
        raise ValueError('Dataset export is not verified')
    files={label:sorted(p for p in root.rglob('*') if p.is_file() and '.cache' not in p.relative_to(root).parts)
           for label,root in sources.items()}
    return sources,files


def stage_inputs(model,dataset,destination):
    """One copy per session, validate copied bytes and dataset checksums before use."""
    started=time.monotonic();sources,files=input_files(model,dataset)
    destination=Path(destination).resolve()
    if destination.is_relative_to(Path('/workspace')) or any(destination.is_relative_to(p) or p.is_relative_to(destination) for p in sources.values()):
        raise ValueError('Stage inputs on separate container storage, outside source directories')
    destination.parent.mkdir(parents=True,exist_ok=True)
    identity={label:str(root) for label,root in sources.items()}
    inventory=[dict(source=label,path=str(p.relative_to(sources[label])),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
               for label,rows in files.items() for p in rows]
    marker=destination/'staging.json'
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved['sources']!=identity or saved['source_inventory']!=inventory:
            raise ValueError('Staging source changed; use a new session cache')
        for record in saved['files']:
            p=destination/record['path']
            if not p.is_file() or p.stat().st_size!=record['bytes'] or digest(p)!=record['sha256']:
                raise ValueError('Staged input is missing or corrupted')
        return destination/'base',destination/Path(dataset).name,saved
    needed=sum(r['bytes'] for r in inventory)
    if shutil.disk_usage(destination.parent).free<needed+2*1024**3:
        raise ValueError('Container disk needs space for model, dataset and 2 GiB reserve')
    destination.mkdir(exist_ok=True);records=[]
    for label,source in sources.items():
        name='base' if label=='base' else Path(dataset).name
        for original in files[label]:
            target=destination/name/original.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.copying');h=hashlib.sha256();count=0
            before=original.stat()
            with original.open('rb') as src,temporary.open('wb') as dst:
                while chunk:=src.read(8*1024**2):dst.write(chunk);h.update(chunk);count+=len(chunk)
            after=original.stat()
            if (count!=before.st_size or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns)
                or digest(temporary)!=h.hexdigest()):
                raise ValueError('Input changed or copy verification failed')
            temporary.replace(target)
            records.append(dict(path=str(target.relative_to(destination)),bytes=count,sha256=h.hexdigest()))
        print(f'Staged {label} on container disk: {destination/name}',flush=True)
    from cloud.download_training_dataset import verify_bundle
    staged_dataset=destination/Path(dataset).name
    verify_bundle(staged_dataset,(staged_dataset/'SHA256SUMS').read_text())
    receipt=dict(sources=identity,source_inventory=inventory,files=records,
        model_path=str(destination/'base'),dataset_path=str(staged_dataset),staging_seconds=time.monotonic()-started)
    provenance=Path(dataset).parent/(Path(dataset).name+'_hf_source.json')
    if provenance.is_file():receipt['dataset_hf_source']=json.loads(provenance.read_text())
    temporary=marker.with_suffix('.tmp');temporary.write_text(json.dumps(receipt,indent=2)+'\n');temporary.replace(marker)
    return destination/'base',staged_dataset,receipt
