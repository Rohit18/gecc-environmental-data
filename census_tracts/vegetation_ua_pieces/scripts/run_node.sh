#!/bin/bash
# Run one command on a compute node with env + timing. Usage (inside salloc/srun): run_node.sh <cmd...>
cd /path/to/scratch/urbveg/scripts; source ./env.sh
echo "=== $* $(date -u +%T) on $(hostname)"
/usr/bin/time -v "$@" 2> >(grep -E 'Elapsed|Maximum resident|Traceback|Error|error' >&2)
echo "exit=$?"
