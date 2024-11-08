#!/bin/bash

# Ensure the script fails on any error
set -e

# Get the number of GPUs
NUM_GPUS=8

# Activate poetry shell first
eval "$(poetry env info --path)/bin/activate"

# Run the distributed training
poetry run torchrun \
  --nproc_per_node=$NUM_GPUS \
  entropix/torch_main.py \
  --model-size "70B" \
  --weights-dir weights/70B-Instruct \
  --prompts-file entropix/data/prompts.csv