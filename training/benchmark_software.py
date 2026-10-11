"""Software provenance for comparing saved GPU benchmark runs."""
import csv
from importlib import metadata as packages
import io
import platform
import subprocess


PACKAGE_NAMES=('torch','torchvision','transformers','tokenizers','peft','accelerate',
               'timm','numpy','tensorflow','tensorflow-datasets','dlimp','huggingface-hub')


def collect_software(context):
    versions={}
    for name in PACKAGE_NAMES:
        try:versions[name]=packages.version(name)
        except packages.PackageNotFoundError:versions[name]='Not installed'
    context=context or {}
    result=dict(python=platform.python_version(),packages=versions,
                cuda_runtime=context.get('cuda_runtime','Unknown'),
                precision=context.get('precision','Unknown'),
                attention=context.get('attention','Unknown'),driver='Unknown',
                benchmark_code_revision='Unknown')
    for key,command in (
        ('driver',['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader,nounits']),
        ('benchmark_code_revision',['git','rev-parse','HEAD'])):
        try:
            output=subprocess.run(command,check=True,capture_output=True,text=True,timeout=10).stdout
            values=set(line.strip() for line in output.splitlines() if line.strip())
            if len(values)==1:result[key]=values.pop()
        except (OSError,subprocess.SubprocessError):pass
    return result


def software_lines(metadata,info=None,host=None):
    """Prefer recorded stack, with explicit fallbacks for historical artifacts."""
    stack=metadata.get('software_stack') or {}
    context=metadata.get('context') or {}
    info=info or {}
    versions=stack.get('packages',{})
    values=[('Python',stack.get('python','Unknown'))]
    for name in PACKAGE_NAMES:
        fallback={'torch':info.get('torch_version',context.get('torch_version','Unknown')),
                  'transformers':context.get('transformers_version','Unknown')}.get(name,'Unknown')
        values.append((name,versions.get(name,fallback)))
    driver=stack.get('driver','Unknown')
    if driver=='Unknown' and host:
        snapshot=list(csv.reader(io.StringIO(host.get('gpu_snapshot',''))))
        if len(snapshot)>1 and 'driver_version' in [c.strip() for c in snapshot[0]]:
            index=[c.strip() for c in snapshot[0]].index('driver_version')
            recorded={row[index].strip() for row in snapshot[1:] if len(row)>index}
            if len(recorded)==1:driver=recorded.pop()
    values.extend([
        ('PyTorch CUDA runtime',stack.get('cuda_runtime',context.get('cuda_runtime','Unknown'))),
        ('NVIDIA driver',driver),
        ('Precision',stack.get('precision',context.get('precision','Unknown'))),
        ('Attention',stack.get('attention',context.get('attention','Unknown'))),
        ('Benchmark code commit',stack.get('benchmark_code_revision','Unknown'))])
    def safe(value):return str(value).replace('|','/').replace('\n',' ')
    return ['','Software stack (recorded versions; missing historical values are Unknown):','',
            '| Component | Version / setting |','| --- | --- |']+[
                f'| {safe(name)} | {safe(value)} |' for name,value in values]+['']
