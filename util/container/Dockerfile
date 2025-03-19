# Copyright 2020 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

FROM ghcr.io/prefix-dev/pixi AS build

WORKDIR /pixi
COPY pixi.toml .
COPY pixi.lock .
RUN pixi install --locked
RUN pixi shell-hook -s bash > /pixi/shell-hook
RUN echo "#!/bin/bash" > /pixi/entrypoint.sh
RUN cat /pixi/shell-hook >> /pixi/entrypoint.sh
# extend the shell-hook script to run the command passed to the container
RUN echo 'exec "$@"' >> /pixi/entrypoint.sh

FROM ubuntu:24.04
COPY --from=build /pixi/.pixi/envs/default /pixi/.pixi/envs/default
COPY --from=build /pixi/entrypoint.sh /pixi/entrypoint.sh
RUN chmod +x /pixi/entrypoint.sh

ENTRYPOINT [ "/pixi/entrypoint.sh" ]
CMD ["/bin/bash"]
