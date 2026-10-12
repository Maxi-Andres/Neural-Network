#!/usr/bin/env bash
# Creates a local virtual environment (.venv) and installs everything the
# notebooks need. Only Python 3 has to be installed beforehand.
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 was not found. Install it with your package manager and run this again."
    exit 1
fi

if [ ! -x .venv/bin/python ]; then
    echo "Creating virtual environment in .venv ..."
    python3 -m venv .venv
fi

echo "Installing dependencies, this can take a few minutes ..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Done! Run ./run.sh to open the notebook."
