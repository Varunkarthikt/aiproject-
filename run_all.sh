#!/usr/bin/env bash
set -e
python data/generate_data.py
python train.py --latent 10 --epochs 20
python evaluate.py
echo 'Done. Check outputs/results.json and outputs/*.pt'
