# Retain RunPod's /start.sh SSH startup; do not replace its entrypoint/CMD.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
COPY requirements-openvla.txt /opt/openvla-requirements.txt
RUN python3.11 -m venv /opt/openvla \
    && /opt/openvla/bin/python -m pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121 \
    && /opt/openvla/bin/python -m pip install --no-cache-dir -r /opt/openvla-requirements.txt \
    && /opt/openvla/bin/python -m pip check \
    && sha256sum /opt/openvla-requirements.txt | cut -d ' ' -f 1 > /opt/openvla/requirements.sha256
# No model weights, credentials, or Mac simulation environment are included.
