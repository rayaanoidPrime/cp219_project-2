#!/bin/bash

echo "Starting batch run..."

# Run scripts sequentially
python experiments/run_task3.py --wandb-name="task3_full_0.l5" --mode="full"
python experiments/run_task4.py --wandb-name="task4_full_0.l5" --mode="full"

echo "All scripts completed!"
