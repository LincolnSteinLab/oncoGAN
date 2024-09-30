FROM python:3.10.12-slim-bullseye
SHELL ["/bin/bash", "-c"]
## Install esential linux packages
RUN apt-get update && \
    apt-get install -y apt-utils nano && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
## Copy GAN requirements
COPY requirements.txt /tmp/
## Install GAN venv
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt
## Copy GAN scripts
COPY src/launcher.py /oncoGAN/launcher.py
COPY src/ctabgan /oncoGAN/ctabgan
COPY src/ctgan /oncoGAN/ctgan
## Give extra permissions to run jupyter notebook
RUN mkdir -p /.local/share/jupyter && chmod -R 777 /.local/share/jupyter
## Update environment variables
ENV JUPYTER_ALLOW_INSECURE_WRITES=1
ENV PATH=$PATH:/oncoGAN/
ENV MPLCONFIGDIR=/tmp
## Entrypoint
WORKDIR /home
ENTRYPOINT ["launcher.py"]