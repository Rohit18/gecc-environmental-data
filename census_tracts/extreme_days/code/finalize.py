#!/usr/bin/env python3
"""Validate complete shards, assemble separate tables, trial merge and return package."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from production_common import *
import math,zipfile,traceback,io,gzip
MAIN='CensusTract_extreme_air_temperature_2003_2024'
FILTER='CensusTract_extreme_air_temperature_range_filtered_2003_2024'

def native_summary():
 baseline=[];sensitivity=[]
 for region in REGIONS:
  cells=np.load(S/f'cells_{region}.npy');nc=len(cells)
  for season in SEASONS:
   full=460 if season=='JJA' else 450
   for policy in POLICIES:
    root=R/'native'/region/season/policy/'baseline';n=np.load(root/'baseline_n.npy',mmap_mode='r');q=np.load(root/'threshold_qnum.npy',mmap_mode='r');assert n.shape==q.shape==(2,nc) and np.all((n>=0)&(n<=full))
    for v,var in enumerate(['TMAX','TMIN']):
     histogram=np.bincount(n[v],minlength=full+1);baseline.append({'region':region,'season':season,'policy':policy,'variable':var,'native_cells':nc,'no_threshold_cells':int(histogram[0]),'complete_baseline_cells':int(histogram[full]),'partial_baseline_cells':int(histogram[1:full].sum()),'support_n_histogram':histogram.tolist()})
   a=R/'native'/region/season/POLICIES[0]/'baseline';b=R/'native'/region/season/POLICIES[1]/'baseline';an=np.load(a/'baseline_n.npy',mmap_mode='r');bn=np.load(b/'baseline_n.npy',mmap_mode='r');aq=np.load(a/'threshold_qnum.npy',mmap_mode='r');bq=np.load(b/'threshold_qnum.npy',mmap_mode='r')
   for v,var in enumerate(['TMAX','TMIN']):
    both=(an[v]>0)&(bn[v]>0);delta=(aq[v,both].astype(np.float64)-bq[v,both])/1000;sensitivity.append({'region':region,'season':season,'variable':var,'support_n_changed_cells':int((an[v]!=bn[v]).sum()),'threshold_changed_cells':int((both&(aq[v]!=bq[v])).sum()),'primary_only_threshold_cells':int(((an[v]>0)&(bn[v]==0)).sum()),'max_absolute_threshold_difference_c':float(np.abs(delta).max()) if len(delta) else None})
 js(R/'qa/baseline_support_summary.json',{'passed':True,'records':baseline});js(R/'qa/threshold_sensitivity.json',{'passed':True,'records':sensitivity})

def assemble():
 assert json.loads((R/'qa/production_complete.json').read_text())['passed'];reg=registry();idset=set(reg.GEOID);allqa={};frames={};summary=[]
 for policy,name in zip(POLICIES,[MAIN,FILTER]):
  parts=[];supports=[]
  for y,s in PERIODS:
   root=R/'shards/production'/policy;marker=json.loads((root/f'{y}_{s}.complete.json').read_text());assert marker['run_signature']==sha(R/'qa/run_signature.json') and sha(root/f'{y}_{s}.parquet')==marker['metrics_sha256'] and sha(root/f'{y}_{s}_support.parquet')==marker['support_sha256'];d=pd.read_parquet(root/f'{y}_{s}.parquet');sp=pd.read_parquet(root/f'{y}_{s}_support.parquet');assert len(d)==85528 and set(d.GEOID)==idset and not d.duplicated(KEY).any() and not d[KEY].isna().any().any();assert d.observation_year.eq(y).all() and d.season.eq(s).all();assert d.observation_year.dtype=='int64' and d.GEOID.str.fullmatch(r'\d{11}').all()
   for m in MET:
    assert str(d[m].dtype)=='Float64';assert np.array_equal(d[m].isna(),sp[m+'_support_weight'].eq(0));assert d[m].dropna().between(0,92 if s=='JJA' else 90).all()
   for m,bound in [(MET[0],'hot_matched_valid_days_bound'),(MET[1],'cold_matched_valid_days_bound')]:assert (d[m].isna()|(d[m]<=sp[bound]+1e-10)).all()
   parts.append(d);supports.append(sp);x=d.copy();x['state_fips']=x.GEOID.str[:2]
   for st,g in x.groupby('state_fips'):
    record={'policy':policy,'year':y,'season':s,'state_fips':st,'rows':len(g),'supported_all_bands':int(g[MET].notna().all(axis=1).sum()),'partial_or_null':int(g[MET].isna().any(axis=1).sum())}
    for m in MET:record[m+'_null']=int(g[m].isna().sum());record[m+'_zero']=int(g[m].eq(0).sum())
    
    region_support=sp.loc[g.index]
    for var in ['tmax','tmin']:
     record[f'baseline_{var}_absent_rows']=int(region_support[f'baseline_{var}_support_weight'].eq(0).sum())
     record[f'baseline_{var}_partial_rows']=int(region_support[f'baseline_{var}_partial_weight'].gt(0).sum())
    summary.append(record)
  df=pd.concat(parts,ignore_index=True).sort_values(KEY).reset_index(drop=True);del parts;assert len(df)==3763232 and not df.duplicated(KEY).any() and df.groupby('GEOID',observed=True).size().eq(44).all();assert np.array_equal(df.GEOID.to_numpy(),np.repeat(reg.GEOID.to_numpy(),44))
  metadata={'run_signature_sha256':sha(R/'qa/run_signature.json'),'source_integrity':'qa/source_integrity.json','support_table':f'{name}_support.parquet','geometry_review':'review_evidence/stage2_independent_review/base_coordinate_comparison.json'}
  atomic_parquet(df,R/f'{name}.parquet',metadata);tmp=R/f'{name}.csv.gz.partial';df.to_csv(tmp,index=False,compression={'method':'gzip','mtime':0});os.replace(tmp,R/f'{name}.csv.gz')
  support=pd.concat(supports,ignore_index=True).sort_values(KEY).reset_index(drop=True);atomic_parquet(support,R/f'{name}_support.parquet',metadata);del supports
  allqa[policy]={'rows':len(df),'geoids':df.GEOID.nunique(),'periods':44,'nulls':{m:int(df[m].isna().sum()) for m in MET},'supported_zeros':{m:int(df[m].eq(0).sum()) for m in MET},'status_counts':df.support_status.value_counts().to_dict(),'parquet_sha256':sha(R/f'{name}.parquet'),'csv_gz_sha256':sha(R/f'{name}.csv.gz'),'support_sha256':sha(R/f'{name}_support.parquet')};frames[policy]=df[KEY+MET].copy()
  # Streaming CSV reparse verifies exact key order, nulls and float round-trip with tight tolerance.
  csvrows=0;maxdiff=0.
  for chunk in pd.read_csv(R/f'{name}.csv.gz',dtype={'GEOID':str},chunksize=65536,float_precision='round_trip'):
   ref=df.iloc[csvrows:csvrows+len(chunk)];assert np.array_equal(chunk[KEY].to_numpy(),ref[KEY].to_numpy())
   for m in MET:
    a=chunk[m].to_numpy(float);b=ref[m].to_numpy(float,na_value=np.nan);assert np.array_equal(np.isnan(a),np.isnan(b));delta=np.abs(a-b);md=float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.;maxdiff=max(maxdiff,md);assert md<=1e-12
   csvrows+=len(chunk)
  assert csvrows==3763232;allqa[policy]['csv_roundtrip_max_absolute_days']=maxdiff
  wrap=df[df.GEOID.eq('02016000100')].copy();wrapsp=support[support.GEOID.eq('02016000100')];atomic_parquet(wrap,R/f'antimeridian_{policy}.parquet');atomic_parquet(wrapsp,R/f'antimeridian_{policy}_support.parquet');del support,df
 pd.DataFrame(summary).to_csv(R/'coverage_support_by_state_period.csv',index=False)
 dif=frames[POLICIES[0]].merge(frames[POLICIES[1]],on=KEY,validate='one_to_one',suffixes=('_primary','_filtered'));sensitivity={}
 for m in MET:
  a=dif[m+'_primary'];b=dif[m+'_filtered'];dif[m+'_primary_minus_filtered']=a-b;both=a.notna()&b.notna();delta=(a-b).abs();sensitivity[m]={'changed_rows_over_1e_minus10_days':int((delta>1e-10).sum()),'exact_value_difference_rows':int((both&(a!=b)).sum()),'max_absolute_difference_days':float(delta.max()) if both.any() else None,'null_mask_changed_rows':int((a.isna()!=b.isna()).sum())}
 atomic_parquet(dif,R/'sensitivity_comparison.parquet');js(R/'qa/output_validation.json',{'passed':True,'tables':allqa,'sensitivity':sensitivity,'keys':'complete registry x 44 actual periods; no duplicate or null keys','units':'coverage-weighted mean days'});del frames,dif
 # Source integrity: all publisher MD5s fresh, all used members fresh against inherited SHA256.
 arcs=[json.loads(p.read_text()) for p in sorted((R/'qa/archive_hashes').glob('*.json'))];assert len(arcs)==44 and all(a['fresh_full_hash'] and a['md5']==a['publisher_md5'] for a in arcs)
 members=pd.concat([pd.read_parquet(p) for p in sorted((R/'qa/member_checks').glob('*.parquet'))],ignore_index=True);sel=pd.read_csv(R/'qa/selected_members.csv.gz');assert len(members)==16016 and set(members.logical_path)==set(Path(r.archive).name+'/'+r.member for r in sel.itertuples()) and not members.logical_path.duplicated().any();assert members.inherited_match.all() and members.header_verified.all();atomic_parquet(members,R/'source_member_integrity.parquet');members.groupby(['region','year','season','variable'])[['nodata_cells','out_of_range_cells','selected_cells','member_bytes','decoded_full_row_bytes']].sum().reset_index().to_csv(R/'source_support_outliers.csv',index=False)
 for a in arcs:
  st=Path(a['path']).stat();assert st.st_size==a['bytes'] and st.st_mtime_ns==a['mtime_ns']
 js(R/'qa/source_integrity.json',{'passed':True,'new_full_archive_hashes':44,'publisher_md5_matches':44,'archive_bytes':sum(a['bytes'] for a in arcs),'new_full_selected_member_sha256s':len(members),'inherited_member_matches':int(members.inherited_match.sum()),'all_required_reached_rows_read':True,'compressed_selected_member_bytes':int(members.member_bytes.sum()),'decoded_full_row_bytes':int(members.decoded_full_row_bytes.sum()),'out_of_range_native_pixel_days':int(members.out_of_range_cells.sum()),'header_proofs':len(list((R/'qa/header_proofs').glob('*.json'))),'raw_scaling':'int16 value x 0.1 degrees C; TIFF scale1 offset0; publisher physical scaling0.1','archive_details':arcs})
 return allqa

def pilot_and_extra_checks():
 primary=pd.read_parquet(R/f'{MAIN}.parquet');filtered=pd.read_parquet(R/f'{FILTER}.parquet');refs={POLICIES[0]:pd.concat([pd.read_parquet(PILOT/'primary_pilot.parquet'),pd.read_parquet(PILOT/'primary_diagnostics.parquet')]),POLICIES[1]:pd.read_parquet(PILOT/'range_sensitivity.parquet')};results=[]
 for policy,current in zip(POLICIES,[primary,filtered]):
  ref=refs[policy].set_index(KEY);ref=ref[ref.index.get_level_values('GEOID')!='02016000100'];now=current.set_index(KEY).loc[ref.index]
  for m in MET:
   a=now[m].to_numpy(float,na_value=np.nan);b=ref[m].to_numpy(float,na_value=np.nan);assert np.array_equal(np.isnan(a),np.isnan(b));delta=np.abs(a-b);maximum=float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.;assert maximum<=1e-10;results.append({'policy':policy,'metric':m,'max_abs_days':maximum,'exact_null_masks':True,'tolerance_abs_days':1e-10,'tolerance_rel':0})
  wr=pd.read_parquet(PILOT/'antimeridian_diagnostic.parquet');wr=wr[wr.method_id.str.endswith(policy)].set_index(KEY);n=current.set_index(KEY).loc[wr.index]
  for m in MET:assert np.allclose(n[m].to_numpy(float,na_value=np.nan),wr[m].to_numpy(float,na_value=np.nan),atol=1e-10,rtol=0,equal_nan=True)
 # Verify national native pilot-cell arrays EXACTLY, in every accepted period and policy.
 native=[]
 for region in REGIONS:
  cells=np.load(S/f'cells_{region}.npy');pcells=np.load(PS/f'cells_{region}.npy');pos=np.searchsorted(cells,pcells);assert np.array_equal(cells[pos],pcells)
  for s in SEASONS:
   for p in POLICIES:
    old=PS/'computed/run1'/region/s/p;base=R/'native'/region/s/p/'baseline'
    for name in ['threshold_qnum.npy','baseline_n.npy']:assert np.array_equal(np.load(base/name,mmap_mode='r')[:,pos],np.load(old/name,mmap_mode='r'));native.append({'region':region,'season':s,'policy':p,'array':name,'exact_cells':len(pos)})
    for y,se in P.MAND+P.DIAG:
     if se!=s:continue
     for newname,oldname in [('counts.npy',f'counts_{y}.npy'),('support.npy',f'support_{y}.npy')]:assert np.array_equal(np.load(R/'native'/region/s/p/str(y)/newname,mmap_mode='r')[:,pos],np.load(old/oldname,mmap_mode='r'));native.append({'region':region,'season':s,'policy':p,'year':y,'array':newname,'exact_cells':len(pos)})
 # Independent fsum with stored weights on extra tracts; GEOS weights checked separately in preparation.
 import production as Q
 reg=registry();kmap={k:i for i,k in enumerate(reg.GEOID)};checks=[]
 for region in REGIONS:
  pp={k:v for k,v in np.load(S/f'pairs_{region}.npz').items()};gids=json.loads((R/'qa/prepare.json').read_text())['extra_validation_geoids']
  for k in gids:
   if reg.loc[kmap[k],'source_region']!=region:continue
   keep=pp['tract']==kmap[k];pos=pp['pos'][keep];weights=pp['coverage'][keep].astype(float)
   for year,season in [(2003,'DJF'),(2016,'JJA'),(2024,'DJF')]:
    for policy,current in zip(POLICIES,[primary,filtered]):
     path=R/'native'/region/season/policy/str(year);c=np.load(path/'counts.npy',mmap_mode='r');m=np.load(path/'support.npy',mmap_mode='r');record=current[(current.GEOID==k)&(current.observation_year==year)&(current.season==season)].iloc[0]
     for b,name in enumerate(MET):
      valid=m[b,pos];den=math.fsum(weights[valid]);value=math.fsum(float(a)*float(w) for a,w in zip(c[b,pos[valid]],weights[valid]))/den if den else np.nan;actual=record[name];delta=abs(value-actual) if den else None;assert (pd.isna(actual) and not den) or delta<=1e-10;checks.append({'GEOID':k,'year':year,'season':season,'policy':policy,'metric':name,'pairs':int(keep.sum()),'absolute_difference_days':delta})
 js(R/'qa/pilot_and_independent_validation.json',{'passed':True,'mandatory_pilot_rows':256,'diagnostic_pilot_rows':96,'filtered_pilot_rows':352,'antimeridian_against_split_wrap':True,'pilot_metric_checks':results,'exact_native_arrays':native,'extra_tract_independent_fsum_checks':checks})
 # Representative rerun: recompute kernel values without accepted outputs or checkpoint shortcuts.
 reruns=[]
 for region in REGIONS:
  cells=np.load(S/f'cells_{region}.npy');positions=np.linspace(0,len(cells)-1,min(4096,len(cells)),dtype=int)
  for policy in POLICIES:
   for season in SEASONS:
    base=R/'native'/region/season/policy/'baseline';bn=np.load(base/'baseline_n.npy',mmap_mode='r');qq=np.load(base/'threshold_qnum.npy',mmap_mode='r');counts=np.load(R/'native'/region/season/policy/'2003/counts.npy',mmap_mode='r');masks=np.load(R/'native'/region/season/policy/'2003/support.npy',mmap_mode='r');valid=[];ext=[];ns=[]
    for v,var in enumerate(['TMAX','TMIN']):
     baseline=np.concatenate([Q.raw(region,y,season,var)[:,positions] for y in range(2003,2008)]);q,n=P.quantile(baseline,[95,5][v],policy);assert np.array_equal(q,qq[v,positions]) and np.array_equal(n,bn[v,positions]);x=Q.raw(region,2003,season,var)[:,positions];ok=P.validity(x,policy);valid.append(ok.sum(axis=0,dtype=np.int16));comparison=x.astype(np.int64)*100>q if v==0 else x.astype(np.int64)*100<q;ext.append((comparison&ok).sum(axis=0,dtype=np.int16));ns.append(n)
    common=(valid[0]>0)&(valid[1]>0);cp=np.stack([ext[0],ext[1],valid[0],valid[1]]);mp=np.stack([common&(ns[0]>0),common&(ns[1]>0),common,common]);assert np.array_equal(cp,counts[:,positions]) and np.array_equal(mp,masks[:,positions]);reruns.append({'region':region,'policy':policy,'season':season,'native_cells_recomputed':len(positions),'counts_thresholds_support_exact':True})
 js(R/'qa/reproducibility.json',{'passed':True,'representative_reruns':reruns,'run_signature':sha(R/'qa/run_signature.json')})

def merge():
 before={str(p):sha(p) for p in [P.OLD,P.ADD]};old_evidence=json.loads((PILOT/'qa/trial_merge.json').read_text());assert before==old_evidence['source_tables_unchanged_sha256'];target=PS/'trial_candidate_temperature.parquet';assert sha(target)==old_evidence['candidate_sha256'];new=pd.read_parquet(R/f'{MAIN}.parquet');newcols=[c for c in new if c not in KEY];indexed=new.set_index(KEY);assert indexed.index.is_unique;old=pq.ParquetFile(target);schema=old.schema_arrow;original=schema.names;assert not(set(newcols)&set(original));destination=R/'trial_merge/CensusTract_air_temperature_with_extremes_trial.parquet';destination.parent.mkdir(exist_ok=True);tmp=destination.with_suffix('.partial');extra_schema=pa.Table.from_pandas(new[newcols],preserve_index=False).schema;mergedschema=pa.schema(list(schema)+list(extra_schema));seen=np.zeros(len(indexed),bool);nrows=0
 with pq.ParquetWriter(tmp,mergedschema,compression='zstd') as writer:
  for rb in old.iter_batches(batch_size=65536):
   batch=pa.Table.from_batches([rb]);keys=pd.MultiIndex.from_frame(batch.select(KEY).to_pandas());pos=indexed.index.get_indexer(keys);assert (pos>=0).all() and len(np.unique(pos))==len(pos) and not seen[pos].any();seen[pos]=True;add=pa.Table.from_pandas(indexed.iloc[pos][newcols],schema=extra_schema,preserve_index=False);joined=pa.Table.from_arrays(list(batch.columns)+list(add.columns),schema=mergedschema);assert joined.select(original).equals(batch,check_metadata=False);writer.write_table(joined);nrows+=len(batch)
 assert nrows==3763232 and seen.all();os.replace(tmp,destination);rows=0;preserved=hashlib.sha256()
 for rb,mb in zip(old.iter_batches(batch_size=65536),pq.ParquetFile(destination).iter_batches(batch_size=65536),strict=True):
  a=pa.Table.from_batches([rb]);b=pa.Table.from_batches([mb]);assert a.equals(b.select(original),check_metadata=False);keys=pd.MultiIndex.from_frame(a.select(KEY).to_pandas());expected=pa.Table.from_pandas(indexed.loc[keys,newcols],schema=extra_schema,preserve_index=False);assert expected.equals(b.select(newcols),check_metadata=False);sink=pa.BufferOutputStream()
  with pa.ipc.new_stream(sink,a.schema) as writer:writer.write_table(a)
  preserved.update(sink.getvalue().to_pybytes());rows+=len(a)
 assert rows==3763232 and all(sha(p)==h for p,h in before.items()) and sha(target)==old_evidence['candidate_sha256']
 js(R/'qa/trial_merge.json',{'passed':True,'target_path':str(target),'target_sha256':old_evidence['candidate_sha256'],'merged_path':str(destination),'merged_sha256':sha(destination),'rows':nrows,'all_primary_keys_matched':nrows,'original_columns_preserved':original,'original_order_values_nulls_preserved_all_persisted_rows':True,'new_columns_equal_all_persisted_rows':True,'original_semantic_sha256':preserved.hexdigest(),'source_tables_unchanged_sha256':before,'qualification':'Available verified states/DC-plus-official-territory candidate; original 1408 pilot rows match current published reference. Full candidate equality to current published table is not independently established.'})

def package():
 outputs=json.loads((R/'qa/output_validation.json').read_text());source=json.loads((R/'qa/source_integrity.json').read_text());mergeqa=json.loads((R/'qa/trial_merge.json').read_text());bench=json.loads((R/'qa/benchmark.json').read_text());prepare=json.loads((R/'qa/prepare.json').read_text());assert all(json.loads((R/'qa'/x).read_text())['passed'] for x in ['production_complete.json','output_validation.json','source_integrity.json','pilot_and_independent_validation.json','reproducibility.json','trial_merge.json'])
 (R/'METHODS.md').write_text('''# Methods and data dictionary\n\nEach primary and independently filtered table has one record per verified 2020 tract, labelled year 2003–2024 and DJF/JJA. GEOID is eleven-digit text; observation_year is int64. No repeated geometry.\n\n|Field|Meaning|\n|---|---|\n|hot_days_tmax_gt_p95|Nullable float64 coverage-weighted mean count of daily native Tmax strictly above fixed seasonal cell baseline p95; days|\n|cold_days_tmin_lt_p05|Nullable float64 coverage-weighted mean count of daily native Tmin strictly below fixed seasonal cell baseline p05; days|\n|valid_tmax_days, valid_tmin_days|Nullable float64 mean native available valid-day counts on common seasonal support; days|\n|boundary_year|2020|\n|baseline_start_year, baseline_end_year|2003, 2007; pooled by native cell, season and variable|\n|tmax_threshold_percentile, tmin_threshold_percentile|95 and 5; percentile units 0–100|\n|djf_convention|same-year: January, February and December of observation_year|\n|extreme_method_id, mask_policy|Versioned scientific implementation and independent daily mask policy|\n|source_product, source_region|SVCM-SP; fixed NA for states/DC, SA for PR/VI; Pacific unsupported|\n|source_manifest_identifier|SHA256 of immutable Stage 1 daily member inventory; fresh integrity details linked in qa/source_integrity.json|\n|index_policy|Verified final 1000 m grid; tested split/wrap for 02016000100|\n|support_status|ok; partial_band_threshold_support; source_no_common_seasonal_support; outside_selected_source_grid; no_coverage_pairs; unsupported_pacific_source_route|\n\nDaily values are original int16 × 0.1 °C, nodata −32768. Primary masks nodata only. The separate sensitivity additionally masks values outside −95..65 °C and independently recomputes thresholds, support and counts. Native per-cell thresholds pool all available 2003–2007 daily values: JJA 460, DJF 450 (calendar DJF451). There is no new minimum baseline support. Exact HF7 q_num uses widened integers; physical threshold is q_num/1000 °C and comparisons use raw×100 strictly greater/less. Equality is not extreme.\n\nNative seasonal common support requires both variable-specific valid-day counts positive. Hot/cold additionally require their own baseline support. Each band retains its own coverage-weight denominator; supported zeros are real zero, absent support is null. Supporting tables contain those weights, hot/cold matched valid-day upper bounds, baseline weighted mean/minimum days, baseline support and partial-baseline weights. Coverage weights are geographic cell intersection fractions, not geodesic areas. Each contributing geographic cell is retained even when native mappings repeat.\n\nCount and threshold bands use the exact 2003 source native sinusoidal grid (sphere radius6371007.181 m). All original used daily headers are retained and all reached native centers must preserve native row/column identities. Counts then nearest-neighbor sample onto the zero-aligned geographic step0.008983152841195215°, retaining coverage≥1/255, before whole-tract per-band means. No bilinear resampling, gap filling, alternate product, region blending or tract-first daily averaging. Raw 2022 header drift is preserved and documented; production count-band sampling uses the fixed reference grid.\n\nSource-calendar JJA92 and DJF90 files are available. Real February29 mapping is retained; missing leap December31 is not imputed. Available-day counts are separate from calendar expectations. Daily time-window/timezone is unknown. Baseline support is descriptive; partial support is allowed.\n\nOriginal stored coordinates are used directly. The base source has GRS80 with unspecified datum; this does not establish WGS84 identity and no datum transform is silently introduced. Independent reviewed direct-coordinate evidence resolves release serialization differences; five valid official territory originals are used. Existing serialized PR/USVI release1013 nonfinite geometries require a separate release remediation.\n\nThis is a local implementation of City Health scientific definitions. Earth Engine ingest/grid, default percentile approximation and exact reduction parity are unverified. The isolated merge target is the verified available states/DC-plus-territory candidate, not a demonstrated all-row copy of the published current table. No existing release is modified or published.\n''')
 evidence={'status':'STAGE 3 COMPLETE — READY FOR LOCAL REVIEW AND MERGE','required_checks_passed':True,'outputs':outputs,'source_integrity':{k:v for k,v in source.items() if k!='archive_details'},'geometry_gate':'resolved by supplied independent review plus local source hashes','serialized_release_defect':1013,'trial_merge':mergeqa,'publication_limitations':['Earth Engine exact numerical parity unverified','Original base datum unspecified; direct stored coordinates retained','1013 PR/USVI serialized release geometries need separate remediation','Trial target full equality to published table not independently established']};js(R/'validation.json',evidence)
 # Native caches stay outside ZIP; durable content hashes allow exact reuse and review.
 native=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted((R/'native').rglob('*')) if p.is_file()];js(R/'native_cache_manifest.json',native)
 (R/'STAGE3_REPORT.md').write_text('# Stage 3 complete run\n\n**STAGE 3 COMPLETE — READY FOR LOCAL REVIEW AND MERGE**\n\nSeparate primary and filtered modules each contain3,763,232 rows:85,528 verified tract IDs×22years×2seasons. All44period ID sets, key uniqueness, types, seasonal bounds, per-band null/zero masks and matched-support bounds pass.\n\nThe geometry gate is resolved by the supplied independently reviewed coordinate comparison, recorded under review_evidence. All84,414 NERSC original-source hashes and1114 valid official territory originals pass locally.84,412 serialized release digest differences represent at most2.842170943040401e-14degrees, with unchanged polygon/ring/vertex structure. The pilot report remains unchanged.\n\n'+f'Production reached-cell inventory: {json.dumps(prepare["regions"])}. Retained geographic pairs: {prepare["pairs"]:,}.\n\n'+f'Benchmark: {bench["seconds"]:.1f}s; parent peak RSS{bench["maxrss_kib"]:,}KiB, child peak{bench["max_child_rss_kib"]:,}KiB; fresh full reached-cell staging at2003/2022, representative exact kernels, full registry aggregation. Measured scratch reserve bound{bench["scratch_bound_bytes"]:,}bytes, durable bound{bench["durable_bound_bytes"]:,}bytes. See logs, phase manifests and slurm_accounting.txt for all allocation timings and resource measurements.\n\n'+f'Integrity:44 fresh complete archive MD5s match publisher values (also fresh SHA256);16,016 used seasonal member SHA256s match inherited manifests; every used header and reached decoded row chunk passes. Actual headers preserved; fixed native reference grid row/column identity is verified across all production reached cells. Compressed selected members{source["compressed_selected_member_bytes"]:,}bytes; decoded full rows{source["decoded_full_row_bytes"]:,}bytes. Out-of-range selected native pixel-days{source["out_of_range_native_pixel_days"]:,}. Dates and per-year publisher DOI versions are preserved in source evidence.\n\n'+f'Output statistics:\n```json\n{json.dumps(outputs,indent=2)}\n```\n\n'+'Full results reproduce the256 primary,96 diagnostic and352 filtered accepted pilot records within1e-10days absolute, zero relative tolerance, with exact native thresholds/counts and null masks.02016000100 matches the separate split/wrap diagnostic; all44production periods are actually evaluated. Additional tracts across nine state/territory groups have fresh coverage checks and independent math.fsum reductions; extra native scalar checks and representative4096-cell kernel reruns pass. See qa/pilot_and_independent_validation.json, qa/independent_native_checks.json and qa/reproducibility.json.\n\n'+f'Trial merge matches all3,763,232 keys into {mergeqa["target_path"]}. Persisted all-row stream comparison preserves every19original columns, null, value and row order; appended new fields match the module on every row. Original source tables retain their prior SHA256s. Merge path{mergeqa["merged_path"]}, SHA256{mergeqa["merged_sha256"]}. No source, pilot or existing release edited. Full target equality to the published table remains qualified.\n\n'+'The separate support tables, state/year/season coverage summaries, source-date/outlier summaries, sensitivity differences and antimeridian tables record supported zeros, absent/partial baseline and regional coverage. METHODS.md defines units, available-day counts, masks, fixed grid, stored-coordinate/unknown-datum assumption and Earth Engine parity limits.\n\n'+'The return ZIP includes final tables, supporting summaries, reproducible code/config, checksums, pinned environment and compact evidence. Huge daily selections, native caches and full trial merge stay outside ZIP; native_cache_manifest.json and qa/trial_merge.json list durable paths and hashes. Scratch baseline selections are regenerable; cleaned analysis arrays are recorded only after verifying durable counts and tract shards. Allow space for the transferred ZIP plus uncompressed table/support sizes listed in ARTIFACT_SHA256SUMS.txt and return_zip_manifest.json; no raw archives are needed for local module review.\n\n'+'Remaining publication limitations: no claim of exact numerical Earth Engine identity; no silent datum verification;1013nonfinite serialized PR/USVI geometries require independent release remediation; trial candidate full published-table equivalence is not established. Final release integration is left for local review.\n')
 included=[]
 for p in sorted(R.rglob('*')):
  if not p.is_file():continue
  rel=p.relative_to(R)
  if rel.parts[0] in ['native','benchmark_native','shards','trial_merge','logs'] or '__pycache__' in rel.parts or p.name in ['ARTIFACT_SHA256SUMS.txt','return_zip_manifest.json','return_zip_sha256.txt'] or p.suffix=='.zip' or '.partial' in p.name:continue
  included.append(p)
 (R/'ARTIFACT_SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(R)}\n' for p in included));included.append(R/'ARTIFACT_SHA256SUMS.txt');zpath=R/'census_tract_extreme_temperature_stage3_full_return.zip';tmp=zpath.with_suffix('.partial')
 with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
  for p in included:z.write(p,str(p.relative_to(R)))
 os.replace(tmp,zpath)
 with zipfile.ZipFile(zpath) as z:
  assert z.testzip() is None
  for line in z.read('ARTIFACT_SHA256SUMS.txt').decode().splitlines():
   h,n=line.split('  ',1);assert hashlib.sha256(z.read(n)).hexdigest()==h
 digest=sha(zpath);size=zpath.stat().st_size;js(R/'return_zip_manifest.json',{'path':str(zpath),'bytes':size,'sha256':digest,'crc_verified':True,'artifact_hashes_verified':len(included)-1,'uncompressed_artifact_bytes':sum(p.stat().st_size for p in included),'local_transfer_and_extract_bytes':size+sum(p.stat().st_size for p in included),'excluded_native_cache_bytes':sum(x['bytes'] for x in native),'trial_merge_path':mergeqa['merged_path'],'trial_merge_sha256':mergeqa['merged_sha256']});(R/'return_zip_sha256.txt').write_text(f'{digest}  {zpath.name}\n');log('RETURN ZIP',size,digest)

if __name__=='__main__':
 try:
  for name,fn in [('native_summary',native_summary),('assemble',assemble),('pilot_extra',pilot_and_extra_checks),('merge',merge),('package',package)]:measure(name,fn)
 except Exception:traceback.print_exc();sys.exit(1)
