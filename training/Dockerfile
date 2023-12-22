FROM python:3.10.12-slim-bullseye
SHELL ["/bin/bash", "-c"]
## Install esential linux packages
RUN apt-get update && \
    apt-get install -y apt-utils nano && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
## Copy GAN requirements
COPY requirements.txt /genomeGAN/
## Install GAN venv
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /genomeGAN/requirements.txt && \
    rm /genomeGAN/requirements.txt
## Copy GAN scripts
COPY launcher.py /genomeGAN/launcher.py
COPY ctabgan /genomeGAN/ctabgan
COPY ctgan /genomeGAN/ctgan
## Update PATH
ENV PATH=$PATH:/genomeGAN/
## Entrypoint
WORKDIR /home
ENTRYPOINT ["launcher.py"]