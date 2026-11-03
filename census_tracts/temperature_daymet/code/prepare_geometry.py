#!/usr/bin/env python3
"""Gate all projected tracts using the representative-point scale."""
import argparse
import json
import numpy as np
import netCDF4
from common import *


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=str(source_path(2020,'hi','tmax')))
    args=parser.parse_args()
    with netCDF4.Dataset(args.source) as ds:
        crs,_,_,_=read_grid(ds,'tmax')
    tracts=load_roster()
    projected,repairs=project_tracts(tracts,crs)
    metrics=geometry_area_metrics(tracts,projected,crs)
    metrics['absolute_scale_corrected_deviation']=abs(metrics.scale_corrected_ratio-1.)
    invalid=(~np.isfinite(metrics.scale_corrected_ratio)) | (metrics.absolute_scale_corrected_deviation > .02)
    by_state=metrics.groupby('StFIPS').raw_planar_geodesic_ratio.agg(['count','min','median','max']).reset_index()
    worst=metrics.loc[metrics.absolute_scale_corrected_deviation.idxmax()].to_dict()
    aleutian=metrics.loc[metrics.GEOID=='02016000100'].iloc[0].to_dict()
    hi_large=metrics.loc[metrics.GEOID=='15003981200'].iloc[0].to_dict()
    audit=dict(created_utc=utc_now(),passed=bool(not invalid.any()),crs=crs.to_wkt(),source_crs_file=args.source,
        roster_sha256=ROSTER_SHA256,tracts=len(tracts),projected_empty=0,projected_invalid=0,
        make_valid_changed=repairs,scale_corrected_threshold=.02,failed_tracts=metrics.loc[invalid].to_dict('records'),
        maximum_deviation=worst,aleutian=aleutian,hawaii_large=hi_large,
        raw_ratio_by_StFIPS=by_state.to_dict('records'),
        off_nersc_reference=dict(invalid_geometries=0,maximum_deviation=.017,maximum_GEOID='15003981200',aleutian_ratio=1.003),
        geodesic_ring_orientation='Temporary measurement geometry uses CCW exterior and CW holes, per Geod requirements; source coordinates and topology unchanged',
        tracts_with_signed_ring_area_correction=metrics.loc[~np.isclose(metrics.geodesic_area_m2,metrics.unoriented_signed_geodesic_area_m2,rtol=1e-10)].to_dict('records'),
        round_trip_is_gate=False)
    (CFS/'audits').mkdir(parents=True,exist_ok=True)
    metrics.to_parquet(CFS/'audits/geometry_area_checks.parquet',index=False)
    by_state.to_csv(CFS/'audits/planar_area_distortion_by_StFIPS.csv',index=False)
    atomic_json(CFS/'audits/geometry_preflight.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('crs','raw_ratio_by_StFIPS')},indent=2),flush=True)
    if invalid.any(): raise SystemExit('Projected geometry gate FAILED; see audit')


if __name__=='__main__':main()
