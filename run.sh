#!/bin/sh
cd "$(dirname "$0")" && open http://localhost:8765 && exec .venv/bin/python server.py
