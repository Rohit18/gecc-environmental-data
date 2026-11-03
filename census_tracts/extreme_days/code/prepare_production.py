#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from production_common import *
import calendar,re

def prepare():
 cfg={'scope':'complete separate Stage 3 module; 85528 GEOIDs x 44 periods','years':YEARS,'seasons':SEASONS,'baseline_start_year':2003,'baseline_end_year':2007,'boundary_year':2020,'tmax_threshold_percentile':95,'tmin_threshold_percentile':5,'djf_convention':'same-year','extreme_method_id':METHOD,'mask_policies':POLICIES,'source_product':'SVCM-SP','scaling_c_per_raw_unit':0.1,'raw_nodata':-32768,'quantile':'exact HF7 integer numerator; h=(n-1)*p; widened int64; strict exceedance','baseline_n_min':1,'sampling_step_degrees':P.STEP,'coverage_cutoff':P.CUTOFF,'operation_order':'native threshold/count then fixed-grid nearest-neighbor geographic samples then per-band coverage-weighted tract mean','native_grid':'exact original 2003 grid; all reached native centers must retain row/col identity','antimeridian':'02016000100 tested split/wrap, geographic identities deduplicated','source_routing':{'states_DC':'NorthAmerica','PR_USVI':'SouthAmerica','AS_GU_CNMI':'unsupported_pacific'},'geographic_datum_assumption':'Use verified original stored coordinates directly; base GRS80 unspecified datum; no verified WGS84 retag or datum transformation','daily_time_window':'unknown','gee_parity':'scientific definitions implemented locally; exact numerical Earth Engine parity unverified','source_manifest_identifier':sha(D/'daily_members.csv.gz'),'workers':4,'scratch_reserve_bound_bytes':300000000000,'durable_reserve_bound_bytes':40000000000,'trial_target':'verified available states/DC-plus-territory candidate; published full table equivalence not established'}
 if (R/'config.json').exists():assert config()==cfg
 else:js(R/'config.json',cfg)
 (R/'qa').mkdir(exist_ok=True);S.mkdir(exist_ok=True)
 review=R/'review_evidence/stage2_independent_review';hashes={}
 for line in (review/'REVIEW_SHA256SUMS.txt').read_text().splitlines():
  h,n=line.split(maxsplit=1);assert sha(review/n.lstrip('*'))==h;hashes[n]=h
 rv=json.loads((review/'base_coordinate_comparison.json').read_text());assert rv['tracts_compared']==84414 and rv['source_hashes_match_NERSC_candidate'] and rv['release_hashes_match_supplied_reference'] and not rv['polygon_ring_vertex_structure_mismatches'] and not rv['tracts_above_1e_minus12_degree_delta']
 assert json.loads((review/'independent_return_checks.json').read_text())['status']=='PASS'
 frozen=json.loads((D/'geometry_cache_integrity.json').read_text());source_hashes={}
 for record in frozen['polygon_hashes']:
  assert sha(record['path'])==record['sha256_newly_verified'];source_hashes[record['path']]=record['sha256_newly_verified']
 cv=pd.read_csv(PILOT/'qa/base_geometry_signature_comparison.csv.gz',dtype={'GEOID':str}).set_index('GEOID')
 compare=pd.read_csv(review/'base_coordinate_comparison.csv.gz',dtype={'GEOID':str});log('review comparison columns',list(compare.columns))
 base=P.gpd.read_file(P.BASE);assert len(base)==84414 and base.GEOID.is_unique and base.geometry.is_valid.all()
 assert all(P.gh(g)==cv.loc[k,'candidate_geometry_sha256'] for k,g in zip(base.GEOID,base.geometry))
 # Review has hash-matched the same actual source geometries, not replacement geometries.
 for name in compare.columns:
  if 'source' in name and ('hash' in name or 'sha256' in name):assert compare.set_index('GEOID')[name].reindex(cv.index).equals(cv.candidate_geometry_sha256.rename(name))
 official=pd.read_csv(PILOT/'references/official_territory_geometry_signatures.csv',dtype={'GEOID':str}).set_index('GEOID');territories=[]
 for x in json.loads((PILOT/'references/official_territory_source_archives.json').read_text()):
  p=A/'boundaries'/x['file'];assert sha(p)==x['sha256'];source_hashes[str(p)]=x['sha256'];g=P.gpd.read_file(p);assert g.geometry.is_valid.all() and all(P.gh(q)==official.loc[k,'geometry_sha256'] for k,q in zip(g.GEOID,g.geometry));territories.append(g)
 territory=pd.concat(territories,ignore_index=True);geometries=dict(zip(base.GEOID,base.geometry));geometries.update(dict(zip(territory.GEOID,territory.geometry)))
 reg=pd.read_csv(PILOT/'references/current_allUS_registry_geometry_signatures.csv',dtype={'GEOID':str});assert set(reg.GEOID)==set(geometries) and len(reg)==85528
 reg=reg[['GEOID']].sort_values('GEOID').reset_index(drop=True);reg['source_region']=np.where(reg.GEOID.str[:2].isin(['60','66','69']),'unsupported_pacific',np.where(reg.GEOID.str[:2].isin(['72','78']),'SouthAmerica','NorthAmerica'));reg.to_csv(R/'registry.csv',index=False);idmap={k:i for i,k in enumerate(reg.GEOID)}
 # Calendar validation precedes any date dictionary construction.
 inv=pd.read_csv(D/'daily_members.csv.gz');assert not inv.duplicated(['region','variable','date']).any();sel=inv[inv.season.isin(SEASONS)].copy();checks=[]
 for rec in sel.itertuples():
  m=re.fullmatch(r'(NorthAmerica|SouthAmerica)_Ta_SVCMsp_(\d{4})(TMAX|TMIN)_(\d{3})\.tif',Path(rec.member).name);assert m
  region,y,var,doy=m.groups();date=datetime.date(int(y),1,1)+datetime.timedelta(days=int(doy)-1);assert str(date)==rec.date and date.year==int(y) and region==rec.region and var==rec.variable and int(y) in YEARS
 for (r,y,s,v),d in sel.groupby(['region','year','season','variable']):
  expected={datetime.date(int(y),mo,day) for mo in ([6,7,8] if s=='JJA' else [1,2,12]) for day in range(1,calendar.monthrange(int(y),mo)[1]+1)};actual={datetime.date.fromisoformat(x) for x in d.date};missing=sorted(expected-actual);assert actual<=expected and missing==([datetime.date(int(y),12,31)] if s=='DJF' and calendar.isleap(int(y)) else []);assert len(actual)==(92 if s=='JJA' else 90)
  checks.append({'region':r,'year':int(y),'season':s,'variable':v,'available_days':len(actual),'calendar_days':len(expected),'missing_dates':[str(x) for x in missing]})
 assert len(sel)==16016 and len(checks)==176
 sel.to_csv(R/'qa/selected_members.csv.gz',index=False,compression={'method':'gzip','mtime':0});js(R/'qa/calendar.json',{'passed':True,'records':checks,'baseline_source_days':{'JJA':460,'DJF':450},'baseline_calendar_days':{'JJA':460,'DJF':451}})
 arcs=json.loads((D/'archive_integrity_evidence.json').read_text())
 for x in arcs:
  stat=Path(x['path']).stat();assert stat.st_size==x['bytes'] and stat.st_mtime_ns==x['mtime_ns']
 js(R/'qa/archive_expected.json',arcs)
 info=json.loads((A/'air_temperature/fua_global_index_info.json').read_text());bi=json.loads((B/'work/coverage_index_m1000_o0_info.json').read_text());assert bi['step_deg']==info['step_deg']==P.STEP and bi['origin']==[0.,0.]
 for c in [*info['origin'],bi['window'][0],bi['window'][3]]:assert abs(c/P.STEP-round(c/P.STEP))<1e-8
 frozenidx=json.loads((PILOT/'qa/coverage.json').read_text())['reuse_hashes'];paths=[B/'work/sampling_index_final.npz',B/'work/grid_index_geoids.csv',A/'air_temperature/fua_global_index.npz',A/'air_temperature/index_geoids.json',A/'air_temperature/fua_global_index_info.json',B/'work/coverage_index_m1000_o0_info.json']
 for p in paths:
  h=sha(p);source_hashes[str(p)]=h
  if str(p) in frozenidx:assert h==frozenidx[str(p)]
 ix=np.load(paths[0]);order=pd.read_csv(paths[1],dtype=str).iloc[:,-1].tolist();assert order==base.GEOID.tolist()
 ax=np.load(paths[2]);aids=json.loads(paths[3].read_text());assert set(aids)==set(territory.GEOID)
 cached=P.gpd.read_file(A/'air_temperature/tracts_for_index.gpkg');assert all(P.gh(g)==P.gh(geometries[str(k).zfill(11)]) for k,g in zip(cached.eFUA_ID,cached.geometry))
 # Chunked exact mapping; retain EVERY geographic contributing pair and its weight.
 records=[];mapping_diff=0
 tract=ix['tract'];w=ix['coverage'];lc=ix['cell'];native=ix['native_idx'];outcell=np.empty(len(lc),dtype=np.int64);outnative=np.empty(len(lc),dtype=np.int64)
 for lo in range(0,len(lc),1000000):
  hi=min(len(lc),lo+1000000);lon,lat=P.grid_lonlat(lc[lo:hi],bi,True);outcell[lo:hi]=P.canonical_cell(lon,lat,info);outnative[lo:hi]=P.map_native(lon,lat,P.meta('NorthAmerica'));mapping_diff+=int((outnative[lo:hi]!=native[lo:hi]).sum())
 assert mapping_diff==0,'Exact-header native mapping differs from frozen final index'
 bmap=np.array([idmap[k] for k in order],np.int32);records.append((bmap[tract],outcell,w,outnative))
 acell=ax['cell'];lon,lat=P.grid_lonlat(acell,info);ani=P.map_native(lon,lat,P.meta('SouthAmerica'));amap=np.array([idmap[k] for k in aids],np.int32);at=amap[ax['fua']];ani[reg.source_region.to_numpy()[at]!='SouthAmerica']=-1;records.append((at,acell,ax['coverage'],ani))
 template=S/'global_grid_template.tif';shutil.copyfile(PS/'global_grid_template.tif',template)
 wrap=pd.read_parquet(PS/'wrap_pairs.parquet');assert sha(PS/'wrap_pairs.parquet')==frozenidx[str(PS/'wrap_pairs.parquet')]
 original=P.wrap_parts(geometries['02016000100']);fresh=P.exact_extract(str(template),P.gpd.GeoDataFrame({'GEOID':['02016000100']},geometry=[original],crs='EPSG:4326'),['cell_id','coverage'],output='pandas').iloc[0];cc=np.asarray(fresh.cell_id,np.int64);ww=np.asarray(fresh.coverage,np.float32);keep=ww>=P.CUTOFF;cc=cc[keep];ww=ww[keep];a=np.argsort(cc);b=np.argsort(wrap.cell.to_numpy());assert np.array_equal(cc[a],wrap.cell.to_numpy()[b]) and np.array_equal(ww[a],wrap.coverage.to_numpy()[b]);assert len(cc)==len(np.unique(cc))
 lon,lat=P.grid_lonlat(cc,info);wni=P.map_native(lon,lat,P.meta('NorthAmerica'));records.append((np.full(len(cc),idmap['02016000100'],np.int32),cc,ww,wni));wrap.to_parquet(R/'qa/antimeridian_pairs.parquet',index=False)
 t=np.concatenate([r[0] for r in records]);cell=np.concatenate([r[1] for r in records]);weight=np.concatenate([r[2] for r in records]);ni=np.concatenate([r[3] for r in records]);assert np.all(weight>=P.CUTOFF) and np.all(weight<=1)
 encoded=t.astype(np.int64)*int(np.prod(info['shape']))+cell;assert len(np.unique(encoded))==len(encoded);del encoded
 np.savez(S/'all_pairs.npz',tract=t,cell=cell,coverage=weight,native_idx=ni)
 regional=[]
 for region in REGIONS:
  keep=(reg.source_region.to_numpy()[t]==region)&(ni>=0);cells=np.unique(ni[keep]);np.save(S/f'cells_{region}.npy',cells);pos=np.searchsorted(cells,ni[keep]);assert np.array_equal(cells[pos],ni[keep]);np.savez(S/f'pairs_{region}.npz',tract=t[keep],cell=cell[keep],coverage=weight[keep],pos=pos.astype(np.int32));regional.append({'region':region,'pairs':int(keep.sum()),'unique_native_cells':len(cells),'actual_reached_native_rows':len(np.unique(cells//P.meta(region)['width'])),'daily_selection_bytes':len(cells)*2*364})
 support=pd.DataFrame({'GEOID':reg.GEOID,'coverage_pairs':np.bincount(t,minlength=len(reg)),'total_coverage_weight':np.bincount(t,weights=weight.astype(float),minlength=len(reg)),'inside_grid_pairs':np.bincount(t[ni>=0],minlength=len(reg)),'inside_grid_weight':np.bincount(t[ni>=0],weights=weight[ni>=0].astype(float),minlength=len(reg))});support.to_parquet(R/'qa/registry_coverage.parquet',index=False)
 # Fresh coverage checks on small and large extra tracts outside pilot; frozen signatures validate full index reuse.
 pilot=set(P.ids());extra=[]
 for prefix in ['01','02','06','12','15','36','48','72','78']:
  subset=[k for k in reg.GEOID if k[:2]==prefix and k not in pilot];sizes={k:int(support.loc[idmap[k],'coverage_pairs']) for k in subset};small=sorted(subset,key=lambda k:sizes[k]);extra.extend(small[:1]+small[-1:])
 spatial=[]
 for k in extra:
  fresh=P.exact_extract(str(template),P.gpd.GeoDataFrame({'id':[k]},geometry=[geometries[k]],crs='EPSG:4326'),['cell_id','coverage'],output='pandas').iloc[0];cc=np.asarray(fresh.cell_id,np.int64);ww=np.asarray(fresh.coverage,np.float32);keep=ww>=P.CUTOFF;cc=cc[keep];ww=ww[keep];old=t==idmap[k];aa=np.argsort(cc);bb=np.argsort(cell[old]);assert np.array_equal(cc[aa],cell[old][bb]);assert np.allclose(ww[aa],weight[old][bb],atol=3e-7,rtol=0);spatial.append({'GEOID':k,'pairs':len(cc),'coverage_verified':True})
 js(R/'qa/environment.json',environment());(R/'requirements-pinned.txt').write_text('\n'.join(f'{k}=={v}' for k,v in environment()['packages'].items())+'\n')
 hashes={str(p):sha(p) for p in [S/'all_pairs.npz',*[S/f'pairs_{r}.npz' for r in REGIONS],*[S/f'cells_{r}.npy' for r in REGIONS],R/'registry.csv',R/'native_grids.json',R/'qa/selected_members.csv.gz']}
 js(R/'qa/prepare.json',{'passed':True,'registry_geoids':len(reg),'pairs':len(t),'regions':regional,'legacy_native_mapping_differences':mapping_diff,'no_geographic_pairs_dropped':True,'geometry_gate':'Accepted independent direct comparison, all source polygons locally hash-verified','review_hashes':hashes if False else {str(review/k):v for k,v in hashes.items() if False},'independent_review_artifacts':{str(review/n):h for n,h in {line.split(maxsplit=1)[1]:line.split(maxsplit=1)[0] for line in (review/'REVIEW_SHA256SUMS.txt').read_text().splitlines()}.items()},'geometry_review_summary':rv,'source_hashes':source_hashes,'generated_hashes':hashes,'extra_spatial_checks':spatial,'extra_validation_geoids':extra,'grid':info,'base_grid':bi,'serialized_release_defect':{'PR_USVI_nonfinite_geometries':1013,'blocks_new_module':False,'requires_separate_release_remediation':True}})
 log('national inventory',regional,'pairs',len(t))
if __name__=='__main__':measure('prepare',prepare)
