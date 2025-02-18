FROM python:3.10.12-slim-bullseye
SHELL ["/bin/bash", "-c"]
## Install esential linux packages
RUN apt-get update && \
    apt-get install -y apt-utils nano && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
## Copy requirements
COPY requirements/* /tmp/
## Install venv
RUN pip install --upgrade pip && \
    pip install /tmp/torch-2.1.0+cpu.cxx11.abi-cp310-cp310-linux_x86_64.whl && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    mkdir /.liftover && mv /tmp/hg19ToHg38.over.chain.gz /.liftover && \
    rm /tmp/torch-2.1.0+cpu.cxx11.abi-cp310-cp310-linux_x86_64.whl /tmp/requirements.txt
## Copy scripts
COPY model /oncoGAN/model
COPY src/* /oncoGAN/
## Update PATH
ENV PATH=$PATH:/oncoGAN/
ENV MPLCONFIGDIR=/tmp
## Entrypoint
WORKDIR /home
ENTRYPOINT ["launcher.py"]