"""Record NVIDIA ECC modes without changing GPU settings."""
import csv
import io
import os
import subprocess


def collect_ecc(gpu_uuid=None):
    result={'current':'Unknown','pending':'Unknown','source':'nvidia-smi'}
    try:
        output=subprocess.run(
            ['nvidia-smi','--query-gpu=index,uuid,name,ecc.mode.current,ecc.mode.pending',
             '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10)
        devices=[]
        for row in csv.reader(io.StringIO(output.stdout)):
            if len(row)!=5:raise ValueError('Unexpected ECC query response')
            index,uuid,name,current,pending=(value.strip() for value in row)
            devices.append(dict(index=index,uuid=uuid,gpu=name,current=current,pending=pending))
        result['devices']=devices
        # A single visible NVIDIA device is unambiguous. For multiple devices,
        # prefer CUDA's UUID, then an explicitly selected CUDA device.
        selected=next((d for d in devices if gpu_uuid and d['uuid']==gpu_uuid),None)
        if selected is None and len(devices)==1:selected=devices[0]
        if selected is None:
            visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')[0].strip()
            selected=next((d for d in devices if visible and visible in (d['index'],d['uuid'])),None)
        if selected is None:
            result['reason']='Could not identify the benchmark GPU in the ECC query'
        else:
            result.update(selected)
    except (OSError,subprocess.SubprocessError,ValueError) as error:
        result['reason']=f'ECC query unavailable ({type(error).__name__})'
    return result


def ecc_label(record):
    value=str((record or {}).get('current','Unknown')).strip()
    return {'enabled':'Enabled','disabled':'Disabled','n/a':'Unavailable (N/A)',
            '[n/a]':'Unavailable (N/A)','not supported':'Not supported',
            '[not supported]':'Not supported'}.get(value.lower(),'Unknown')


def ecc_details(record):
    record=record or {}
    pending=record.get('pending','Unknown')
    return f'GPU ECC: **{ecc_label(record)}** (current); pending mode: **{pending}**.'
