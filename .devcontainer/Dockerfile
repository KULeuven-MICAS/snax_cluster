# Copyright 2020 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

FROM ghcr.io/kuleuven-micas/snax:main

# Install base tools and locales
RUN apt-get update && \
    apt-get install -y bash curl git locales && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Setup shell environment
RUN echo 'source /pixi/entrypoint.sh' >> /etc/bash.bashrc
CMD ["bash"]
