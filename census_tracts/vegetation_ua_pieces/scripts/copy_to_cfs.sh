#!/bin/bash
# Copy run products to CFS. Logs to logs/.
#  - data/output/ (all run profiles; excluding test_* subset reductions): never overwrites (--ignore-existing)
#  - config/, manifests/, scripts/, REPORT.md: updated; any replaced file is kept as <name>.bak-<UTC date>
set -euo pipefail
SRC=/path/to/scratch/urbveg
DST=/path/to/project/urbveg
SUF=.bak-$(date -u +%Y%m%dT%H%M%SZ)
LOG=$SRC/logs/copy_to_cfs_$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p /path/to/project/urbveg/data
{
rsync -a --ignore-existing --itemize-changes --exclude 'test_*' $SRC/data/output/ $DST/data/output/
for d in config manifests scripts; do
  rsync -a --checksum --backup --suffix=$SUF --itemize-changes --exclude '__pycache__' --exclude '*.bak-*' $SRC/$d/ $DST/$d/
done
rsync -a --checksum --backup --suffix=$SUF --itemize-changes $SRC/REPORT.md $DST/REPORT.md
echo "DONE $(date -u +%FT%TZ)"
} 2>&1 | tee $LOG
