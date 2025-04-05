#!/bin/bash

echo "Building frontend..."
python build_frontend.py
if [ $? -ne 0 ]; then
    echo "Failed to build frontend."
    exit 1
fi

echo "Starting dashboard..."
python dashboard.py 