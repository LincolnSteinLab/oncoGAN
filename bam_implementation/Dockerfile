FROM mambaorg/micromamba:2.1.0

# Install micromamba environment
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.yml /tmp/requirements.yml
RUN micromamba install --yes --name base -f /tmp/requirements.yml && \
    micromamba clean --yes --all && \
    rm /tmp/requirements.yml

# Copy the scripts
COPY --chmod=777 src /src
ENV PATH=$PATH:/src/

# Set working directory
WORKDIR /home
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python3", "/src/launcher.py"]