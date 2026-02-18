#!/usr/bin/env bash
#SBATCH -A cs541
#SBATCH -p academic
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -C A30
#SBATCH -t 03:00:00
#SBATCH --mem 32G
#SBATCH --job-name=aegis-test-loader
#SBATCH --output=/home/tfnu/aegis-data/logs/test_loader_%j.out
#SBATCH --error=/home/tfnu/aegis-data/logs/test_loader_%j.err

# Create log dir if it doesn't exist
mkdir -p /home/tfnu/aegis-data/logs

# Activate conda
source /home/tfnu/miniconda3/etc/profile.d/conda.sh
conda activate aegis

# Go to repo
cd /home/tfnu/aegis-runtime

# Confirm environment
echo "Python: $(which python)"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Starting test..."

# Run test
python test_loader.py

echo "Done."