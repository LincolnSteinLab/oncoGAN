FROM quay.io/biocontainers/insilicoseq:2.0.1--pyh7cba7a3_0
SHELL ["/bin/bash", "-c"]
## Copy requirements
COPY requirements.txt /tmp/
## Install venv
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt
## Copy scripts
COPY src/* /oncoGAN_InSilicoSeq/
## Update PATH
ENV PATH=$PATH:/oncoGAN_InSilicoSeq/
## Entrypoint
WORKDIR /home
ENTRYPOINT ["launcher.py"]
