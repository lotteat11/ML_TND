#!/bin/bash
set -e

if [[ "$(uname)" == "Darwin" ]]; then
    if ! brew list libomp &>/dev/null; then
        brew install libomp
    fi
fi

pip install -r requirements.txt
