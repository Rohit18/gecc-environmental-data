"""Merge only complete native-grid parts, validate against frozen histograms, export."""
import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from extract import ROOT,signature,complete
from water_source import status
from lai_common import histogram_stats
from chm_quantiles import stats_from_hist

def baseline(kind):
    if kind=='lai':
        d=pd.read_parquet('/path/to/scratch/gecc_lai_reproducibility_audit_20261012/rerun_work_20261012/lai_rerun_histograms.parquet')
        out={}
        for r in d.itertuples():
            h=np.zeros(256);h[np.asarray(r.hist_values,dtype=int)]=np.asarray(r.hist_frac)*r.wcount
            out[r.GEOID]=h
        return out
    out={}
    paths=sorted(Path('/path/to/project/gecc/chm_1m/work/tract_stats').glob('part_*_hist.parquet'))
    paths.append(Path('/path/to/scratch/gecc_tract_addon/tree_height/chm1m_raw_hist.parquet'))
    for p in paths:
        for r in pd.read_parquet(p).itertuples():
            if r.GEOID not in out:out[r.GEOID]=np.zeros(256)
            h=np.frombuffer(r.hist,dtype=np.float64) if isinstance(r.hist,(bytes,bytearray,memoryview)) else np.asarray(r.hist,dtype=float)
            if h.shape!=(256,) or not np.isfinite(h).all() or (h<0).any():
                raise ValueError(f'invalid archived histogram for {r.GEOID} in {p}')
            out[r.GEOID]+=h
    return out

def merge(kind):
    sig=signature()
    if kind=='lai':
        jobs=pd.read_csv(ROOT/'inputs/lai_jobs.csv')
        paths=[ROOT/f'work/lai/part_{j:05d}.parquet' for j in jobs.job]
        attrs=pd.read_parquet('/path/to/scratch/gecc_lai_reproducibility_audit_20261012/rerun_work_20261012/lai_rerun_candidate.parquet').set_index('GEOID')
    else:
        jobs=pd.read_parquet(ROOT/'inputs/chm_jobs.parquet')
        paths=[ROOT/f'work/chm/tile_{j}.parquet' for j in jobs.tile]
        attrs=pd.concat([pd.read_parquet('/path/to/project/gecc/chm_1m/outputs/CensusTract2020_MetaCHM1m.parquet'),
                         pd.read_parquet('/path/to/scratch/gecc_tract_addon/tree_height/chm1m_raw.parquet')],ignore_index=True).set_index('GEOID')
    assert len(attrs)==85528 and attrs.index.is_unique
    missing=[str(p) for p in paths if not complete(p,sig)]
    if missing:raise RuntimeError(f'{kind}: {len(missing)} missing/stale parts: {missing[:5]}')
    merged={}
    keys=['source_wcount','water_wcount','wc_unknown_wcount','retained_wcount','touched']
    for p in paths:
        for r in pd.read_parquet(p).itertuples():
            if r.GEOID not in merged:
                merged[r.GEOID]={'hist':np.zeros(256),**dict.fromkeys(keys,0)}
            m=merged[r.GEOID];m['hist']+=np.asarray(r.hist)
            for k in keys:m[k]+=getattr(r,k)
    assert set(merged)==set(attrs.index),f'{kind}: incomplete tract coverage'
    old=baseline(kind)
    assert set(old)==set(merged)
    result=[];diffs=[];hist_rows=[]
    max_source_rel=0.;max_bin_excess=0.
    for geoid,m in sorted(merged.items()):
        h=m.pop('hist');before=old[geoid]
        n=h.sum();bcount=before.sum()
        tol=max(1e-5,bcount*1e-9)
        error=abs(m['source_wcount']-bcount)
        assert error<=tol,(kind,geoid,'source count differs',error,tol)
        excess=float(np.max(h-before))
        assert excess<=tol,(kind,geoid,'retained bin exceeds original',excess,tol)
        assert abs(n+m['water_wcount']-bcount)<=tol,(kind,geoid,'water-count conservation')
        max_source_rel=max(max_source_rel,error/max(1,bcount));max_bin_excess=max(max_bin_excess,excess)
        if kind=='lai':
            u=np.flatnonzero(h>0);stats=histogram_stats(u,h[u],m['touched'])
        else:
            stats=stats_from_hist(h)
            if stats is None:
                stats={k:np.nan for k in stats_from_hist(np.ones(256))};stats['n_px']=0.
            stats['n_px_1m']=stats.pop('n_px')
        sw=np.array([m['source_wcount']-m['water_wcount']-m['wc_unknown_wcount'],m['water_wcount'],m['wc_unknown_wcount']])
        row={'GEOID':geoid,**stats,**m,
             'water_mask_status':status(n,sw),
             'water_removed_fraction':m['water_wcount']/bcount if bcount else np.nan,
             'worldcover_unknown_fraction':m['wc_unknown_wcount']/bcount if bcount else np.nan}
        row['mask_source']='ESA WorldCover 2020 v100 Map'
        row['excluded_class']=80
        row['mask_sampling']='nearest neighbour at native source pixel centres'
        row['processing_signature']=sig
        result.append(row)
        mean_col='lai_native_mean' if kind=='lai' else 'chm_mean_m'
        old_mean=float(attrs.loc[geoid,mean_col])
        diffs.append({'GEOID':geoid,'original_mean':old_mean,'water_excluded_mean':stats[mean_col],
                      'mean_change':stats[mean_col]-old_mean if stats[mean_col] is not None else np.nan,
                      'original_weight':bcount,'retained_weight':n,**m})
        hist_rows.append({'GEOID':geoid,'hist':h.tolist(),**m})
    d=pd.DataFrame(result).set_index('GEOID')
    if kind=='lai':
        base=attrs.drop(columns=[c for c in attrs.columns if c.startswith('lai_native_')],errors='ignore')
        base=base.drop(columns=['source_quality_status'],errors='ignore')
        out=base.join(d)
        out['script_version']='nersc-water80-1.0.0'
        out['statistical_support']='native valid source pixels excluding WC2020 class 80; unclassified WC support flagged'
        out['missing_value_policy']='source nodata and water excluded; non-water source zeros valid; no retained pixels yields null statistics'
        out['count_definition']='valid native non-water pixels touched by tract; WorldCover-unclassified pixels retained and flagged'
        name='CensusTract_LAI_2020_WC2020_no_water'
    else:
        meta=[c for c in ['StFIPS','CoFIPS','TrFIPS','Area_Land','Area_Water'] if c in attrs.columns]
        out=attrs[meta].join(d)
        out['boundary_year']=2020
        out['height_units']='m'
        out['canopy_source']='Meta/WRI CHM v1 alsgedi_global_v6_float/chm'
        out['quantile_definition']='coverage-weighted inverse empirical CDF, lower at ties (audited reducer)'
        name='CensusTract_CanopyHeight_1m_WC2020_no_water'
    out=out.reset_index()
    assert out.GEOID.str.fullmatch(r'\d{11}').all() and not out.GEOID.duplicated().any()
    assert len(out)==85528
    if kind=='lai':
        zero=out.lai_native_count==0
        metric=[c for c in out if c.startswith('lai_native_') and c!='lai_native_count']
        assert out.loc[zero,metric].isna().all().all()
        assert (out.lai_native_min.dropna()>=0).all()
        assert ((out.lai_native_min<=out.lai_native_p25)|(zero)).all()
        assert ((out.lai_native_p25<=out.lai_native_median)|(zero)).all()
        assert ((out.lai_native_median<=out.lai_native_p75)|(zero)).all()
        assert ((out.lai_native_p75<=out.lai_native_max)|(zero)).all()
    else:
        zero=out.n_px_1m==0
        metric=[c for c in out if c.startswith('chm_')]
        assert out.loc[zero,metric].isna().all().all()
        for c in ['chm_px_gt0_frac','chm_cover_gt1m_frac','chm_cover_gt5m_frac']:
            assert out[c].dropna().between(0,1).all(),c
        assert ((out.chm_mean_m<=out.chm_mean_gt0_m+1e-10)|out.chm_mean_gt0_m.isna()).all()
        assert ((out.chm_median_m<=out.chm_p95_m)|(zero)).all()
        assert not any(c.endswith('_25m') for c in out.columns)
    out.to_parquet(ROOT/f'outputs/{name}.parquet',index=False)
    out.to_csv(ROOT/f'outputs/{name}.csv',index=False)
    pd.DataFrame(hist_rows).to_parquet(ROOT/f'outputs/{kind}_water_excluded_histograms.parquet',index=False)
    pd.DataFrame(diffs).to_csv(ROOT/f'validation/{kind}_comparison_with_original.csv',index=False)
    report={'kind':kind,'passed':True,'tracts':len(out),'parts':len(paths),'signature':sig,
            'max_source_count_relative_error':max_source_rel,'max_retained_bin_excess_pixels':max_bin_excess,
            'status_counts':out.water_mask_status.value_counts().to_dict(),
            'tracts_with_water_removed':int((out.water_wcount>0).sum()),
            'finished_utc':time.strftime('%FT%TZ',time.gmtime())}
    (ROOT/f'validation/{kind}_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

def package():
    for k in ['lai','chm']:assert json.loads((ROOT/f'validation/{k}_validation.json').read_text())['passed']
    (ROOT/'STATUS.md').write_text('# GECC census-tract water exclusion\n\nCompleted and validated for all 85,528 census tracts, LAI and 1 m canopy height. See outputs/ and validation/. Original releases were preserved. WorldCover-unclassified source support is retained and explicitly flagged; see METHODS.md.\n')
    dest=ROOT/'outputs/GECC_CensusTracts_LAI_CHM1m_WC2020_no_water_20261013.zip'
    files=list((ROOT/'outputs').glob('*.csv'))+list((ROOT/'outputs').glob('*.parquet'))
    files+=list((ROOT/'validation').glob('*.json'))+[ROOT/'METHODS.md',ROOT/'STATUS.md']
    files+=list((ROOT/'code').glob('*.py'))+list((ROOT/'code').glob('*.sh'))
    files+=[ROOT/'inputs/worldcover_inventory.csv',ROOT/'inputs/inventory_summary.json']
    sums=''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in files)
    (ROOT/'outputs/DELIVERABLES.sha256').write_text(sums)
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
        z.write(ROOT/'outputs/DELIVERABLES.sha256','DELIVERABLES.sha256')
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    print('PACKAGE_COMPLETE',dest,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('kind',choices=['lai','chm','package']);a=ap.parse_args()
    package() if a.kind=='package' else merge(a.kind)
