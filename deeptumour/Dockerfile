FROM python:3.8.18-slim-bullseye
SHELL ["/bin/bash", "-c"]
## Install esential linux packages
RUN apt-get update && \
    apt-get install -y apt-utils nano && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
## Copy DeepTumour requirements
COPY requirements/* /tmp/
## Install DeepTumour venv
RUN pip install --upgrade pip && \
    pip install --no-cache-dir numpy==1.21.0 && \
    pip install /tmp/torch-1.9.1+cpu-cp38-cp38-linux_x86_64.whl && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    mkdir /.liftover && mv /tmp/hg38ToHg19.over.chain.gz /.liftover && \
    rm /tmp/torch-1.9.1+cpu-cp38-cp38-linux_x86_64.whl /tmp/requirements.txt
## Copy DeepTumour scripts
COPY src/* /DeepTumour/
## Update PATH
ENV PATH=$PATH:/DeepTumour/
## Entrypoint
WORKDIR /home
ENTRYPOINT ["DeepTumour.py"]