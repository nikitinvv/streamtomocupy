#!/bin/bash

# install with pip install -e .[dev] then run this

THIS_DIR=$(dirname "$0")
cd $THIS_DIR/..
ruff --config ruff.toml check . --fix
ruff --config ruff.toml format . 