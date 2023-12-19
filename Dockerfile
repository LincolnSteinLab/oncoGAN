FROM python:3.10.12-bullseye
## Install esential linux packages
RUN ["/bin/bash", "-c", "apt update && apt install -y apt-utils && apt install -y build-essential curl libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev libbz2-dev nano zlib1g-dev && apt autoremove -y && apt clean && rm -rf /var/lib/apt/lists/*"]
## Copy DeepTumour files
COPY deeptumour_requirements.txt /DeepTumour/requirements.txt
COPY deeptumour /DeepTumour/
## Install python3.8 for DeepTumour
RUN ["/bin/bash", "-c", "cd /DeepTumour/Python-3.8.2 && ./configure --enable-optimizations && make -j 2 && make altinstall && rm -rf /DeepTumour/Python-3.8.2"]
## Install DeepTumour venv
RUN ["/bin/bash", "-c", "python3.8 -m venv /DeepTumour/venvDeepTumour && source /DeepTumour/venvDeepTumour/bin/activate && pip install --upgrade pip && pip install --no-cache-dir --upgrade setuptools wheel numpy==1.18.1 && pip install --no-cache-dir -r /DeepTumour/requirements.txt && deactivate"]
## Copy GAN files
COPY gan_requirements.txt /genomeGAN/requirements.txt
COPY simulation training /genomeGAN/
## Install GAN venv
RUN ["/bin/bash", "-c", "python -m venv /genomeGAN/venvGAN && source /genomeGAN/venvGAN/bin/activate && pip install --upgrade pip && pip install --no-cache-dir --upgrade setuptools wheel && pip install --no-cache-dir -r /genomeGAN/requirements.txt && deactivate"]
## Update PATH
ENV PATH=$PATH:/genomeGAN/:/DeepTumour/
## Entrypoint
WORKDIR /home
ENTRYPOINT ["/bin/bash"]