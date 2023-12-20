FROM python:3.10.12-bullseye
## Install esential linux packages
RUN ["/bin/bash", "-c", "apt update && apt install -y apt-utils && apt install -y build-essential curl libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev libbz2-dev nano zlib1g-dev && apt autoremove -y && apt clean && rm -rf /var/lib/apt/lists/*"]
## Copy DeepTumour requirements
COPY deeptumour_requirements.txt /DeepTumour/requirements.txt
COPY deeptumour/Python-3.8.2.tar.xz /DeepTumour/Python-3.8.2.tar.xz
## Install python3.8 for DeepTumour
RUN ["/bin/bash", "-c", "cd /DeepTumour/ && tar -xf Python-3.8.2.tar.xz && cd Python-3.8.2 && ./configure --enable-optimizations && make -j 2 && make altinstall && rm -rf Python-3.8.2 Python-3.8.2.tar.xz && cd /"]
## Install DeepTumour venv
RUN ["/bin/bash", "-c", "python3.8 -m venv /DeepTumour/venvDeepTumour && source /DeepTumour/venvDeepTumour/bin/activate && pip install --upgrade pip && pip install --no-cache-dir --upgrade setuptools wheel numpy==1.18.1 && pip install --no-cache-dir -r /DeepTumour/requirements.txt && deactivate"]
## Copy GAN requirements
COPY gan_requirements.txt /genomeGAN/requirements.txt
## Install GAN venv
RUN ["/bin/bash", "-c", "python -m venv /genomeGAN/venvGAN && source /genomeGAN/venvGAN/bin/activate && pip install --upgrade pip && pip install --no-cache-dir --upgrade setuptools wheel && pip install --no-cache-dir -r /genomeGAN/requirements.txt && deactivate"]
## Copy DeepTumour files
COPY deeptumour/references /DeepTumour/references
COPY deeptumour/*.py /DeepTumour/
## Copy GAN files
COPY simulation /genomeGAN/simulation
COPY training /genomeGAN/training
## Update PATH
ENV PATH=$PATH:/genomeGAN/simulation:/genomeGAN/training/ctabgan:/DeepTumour/
## Entrypoint
WORKDIR /home
ENTRYPOINT ["/bin/bash", "-c"]