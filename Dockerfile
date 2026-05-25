FROM mambaorg/micromamba:2.5-debian13-slim

# Install micromamba environments
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements/environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    rm /tmp/environment.yml && \
    micromamba clean --yes --all
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements/dae_environment.yml /tmp/dae_environment.yml 
RUN micromamba create -y -n dae -f /tmp/dae_environment.yml && \
    rm /tmp/dae_environment.yml && \
    micromamba clean --yes --all

# Install .NET SDK for SimChA
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget ca-certificates && \
    wget https://packages.microsoft.com/config/debian/13/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends dotnet-sdk-8.0 && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*
    
# Install R
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements/r_packages.R /tmp/r_packages.R
RUN apt-get update && \
    apt-get install -y --no-install-recommends r-base-dev && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*
RUN Rscript /tmp/r_packages.R && \
    rm /tmp/r_packages.R
USER $MAMBA_USER

# Copy liftover chain file
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements/hg19ToHg38.over.chain.gz /.liftover/hg19ToHg38.over.chain.gz

# Copy models directory
COPY --chmod=777 --chown=$MAMBA_USER:$MAMBA_USER models/ /oncoGAN/models/
COPY --chmod=777 --chown=$MAMBA_USER:$MAMBA_USER src/* /oncoGAN/

# Update PATH
ENV DOTNET_CLI_TELEMETRY_OPTOUT=1
ENV PATH=$PATH:/oncoGAN
ENV MPLCONFIGDIR=/tmp

# Entrypoint
WORKDIR /home/run
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "/oncoGAN/launcher.py"]
