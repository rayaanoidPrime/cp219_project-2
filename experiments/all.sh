#!/bin/bash

echo "Starting batch run..."

# Run scripts sequentially
python experiments/run_task3.py  --mode="full" --wandb-name="task3_all_features_FULL"
python experiments/run_task4.py  --mode="full" --wandb-name="task4_all_features_FULL"

echo "All scripts completed!"
