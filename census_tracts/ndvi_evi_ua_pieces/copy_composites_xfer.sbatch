#!/bin/bash
# Copy the composite chunks + manifest to CFS with sha256 verification against the fetch manifest (xfer QOS).
#SBATCH -A your_nersc_project
#SBATCH -q xfer
#SBATCH -t 03:00:00
#SBATCH -J ndvi_cfs_comp
#SBATCH -o /path/to/scratch/urbveg/ndvi/logs/slurm-%j.out
cd /path/to/scratch/urbveg/ndvi/scripts
source ../../scripts/env.sh
python -W ignore copy_to_cfs.py --composites --threads 8
