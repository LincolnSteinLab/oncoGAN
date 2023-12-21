FROM python:3.8.18-slim-bullseye
SHELL ["/bin/bash", "-c"]
## Install esential linux packages
RUN apt-get update && \
    apt-get install -y apt-utils nano && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    pip install --no-cache-dir virtualenv
## Copy DeepTumour requirements
COPY deeptumour/requirements.txt /DeepTumour/requirements.txt
## Install DeepTumour venv
RUN python -m venv /DeepTumour/venvDeepTumour && \
    source /DeepTumour/venvDeepTumour/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache-dir numpy==1.18.1 && \
    pip install --no-cache-dir -r /DeepTumour/requirements.txt && \
    deactivate
## Copy GAN requirements
COPY genomegan/requirements.txt /genomeGAN/requirements.txt
## Install GAN venv
RUN python -m venv /genomeGAN/venvGAN && \
    source /genomeGAN/venvGAN/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r /genomeGAN/requirements.txt && \
    deactivate
## Copy DeepTumour files
COPY deeptumour/*.py /DeepTumour/
## Copy GAN files
COPY genomegan/*.py /genomeGAN/
## Update PATH
ENV PATH=$PATH:/genomeGAN/:/DeepTumour/
## Entrypoint
WORKDIR /home
ENTRYPOINT ["/bin/bash", "-c"]