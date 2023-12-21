FROM python:3.8.18-slim-bullseye
SHELL ["/bin/bash", "-c"]
## Install esential linux packages
RUN apt-get update && \
    apt-get install -y apt-utils nano && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
## Copy DeepTumour requirements
COPY requirements.txt torch-1.4.0+cpu-cp38-cp38-linux_x86_64.whl /DeepTumour/
## Install DeepTumour venv
RUN pip install --upgrade pip && \
    pip install --no-cache-dir numpy==1.18.1 && \
    pip install /DeepTumour/torch-1.4.0+cpu-cp38-cp38-linux_x86_64.whl && \
    pip install --no-cache-dir -r /DeepTumour/requirements.txt && \
    rm /DeepTumour/torch-1.4.0+cpu-cp38-cp38-linux_x86_64.whl /DeepTumour/requirements.txt
## Copy DeepTumour scripts
COPY *.py /DeepTumour/
## Update PATH
ENV PATH=$PATH:/DeepTumour/
## Entrypoint
WORKDIR /home
ENTRYPOINT ["/bin/bash", "-c"]