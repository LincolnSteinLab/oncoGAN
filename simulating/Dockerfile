FROM python:3.10.12-slim-bullseye
SHELL ["/bin/bash", "-c"]
## Install esential linux packages
RUN apt-get update && \
    apt-get install -y apt-utils nano && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
## Copy GAN requirements
COPY requirements.txt torch-2.1.0+cpu.cxx11.abi-cp310-cp310-linux_x86_64.whl /genomeGAN/
## Install GAN venv
RUN pip install --upgrade pip && \
    pip install /genomeGAN/torch-2.1.0+cpu.cxx11.abi-cp310-cp310-linux_x86_64.whl && \
    pip install --no-cache-dir -r /genomeGAN/requirements.txt && \
    rm /genomeGAN/torch-2.1.0+cpu.cxx11.abi-cp310-cp310-linux_x86_64.whl /genomeGAN/requirements.txt
## Copy GAN scripts
COPY model /genomeGAN/model
COPY *.py /genomeGAN/
## Update PATH
ENV PATH=$PATH:/genomeGAN/
## Entrypoint
WORKDIR /home
ENTRYPOINT ["launcher.py"]