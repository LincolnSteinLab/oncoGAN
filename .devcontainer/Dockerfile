FROM oicr/oncogan:v1.0.0

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*
USER mambauser
