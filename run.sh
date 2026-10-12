#!/usr/bin/env bash
# Opens the main notebook in Jupyter using the local virtual environment.
# Runs install.sh first if the environment does not exist yet.
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    bash ./install.sh
fi

.venv/bin/python -m notebook Neural_Network.ipynb
