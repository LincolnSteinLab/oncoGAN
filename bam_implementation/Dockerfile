FROM mambaorg/micromamba:2.1.0

# Install micromamba environment
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.yml /tmp/requirements.yml
RUN micromamba install -y -n base -f /tmp/requirements.yml && \
    micromamba clean --yes --all && \
    rm /tmp/requirements.yml

# Copy the scripts
COPY --chmod=777 --chown=$MAMBA_USER:$MAMBA_USER src /src
COPY --chmod=777 --chown=$MAMBA_USER:$MAMBA_USER bamsurgeon_bin/addsv.py /opt/conda/bin/addsv.py
ENV PATH=$PATH:/src/

# Set working directory
WORKDIR /home
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python3", "/src/launcher.py"]