#!/bin/bash

docker create --name dummy ghcr.io/kuleuven-micas/snax:latest
docker cp dummy:/opt ${PREFIX}/snax-utils
