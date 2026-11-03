# Source before running urbveg scripts (equivalent of conda activate for the interim env).
export URBVEG_ENV=${URBVEG_ENV:-/path/to/scratch/urbveg/env}
export PATH=$URBVEG_ENV/bin:$PATH
export PROJ_DATA=$URBVEG_ENV/share/proj
export GDAL_DATA=$URBVEG_ENV/share/gdal
