#!/bin/bash
#SBATCH --job-name=sapbert_norm
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=job_%j_%a.out
#SBATCH --error=job_%j_%a.err
#SBATCH --mem=32G
#SBATCH --array=0-1  # Two tasks, one for each entity type

# Define an array of entity types
entity_types=("conditions" "interventions")

# Select the entity type based on the SLURM_ARRAY_TASK_ID
target_entity_type=${entity_types[$SLURM_ARRAY_TASK_ID]}

# Run the Python script with the specified entity type
python src/sapbert_normalization.py --target_entity_type $target_entity_type
