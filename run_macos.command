#!/bin/bash
set -e
cd "$(dirname "$0")"
python3 run_all.py
python3 verify_results.py
python3 -m unittest discover -s tests
python3 verify_checksums.py
