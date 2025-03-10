#!/bin/bash

for directory in ./ekf*/; do
    pushd $directory
    pip install -e .
    popd
done
