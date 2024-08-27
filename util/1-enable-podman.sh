# Copyright 2024 KU Leuven, Belgium
#                INESC-ID, Portugal
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Guilherme Paim <gppaim@ieee.org>
# Abdelrahman Ayman Fathallah <abdelrahman.ayman@esat.kuleuven.be>

#######################
# Automagic Bash      #
#######################

source util/0-env-config.sh
podman run -e TARGET_ROOT=/repo -e PROJECT=$PROJECT -e CFG=$CFG -it -v `pwd`:/repo -w /repo ghcr.io/kuleuven-micas/snax:main
