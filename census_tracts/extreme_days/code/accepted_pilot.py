#!/usr/bin/env python3
"""Isolated Stage 2: native raw integer thresholds/counts; reviewed pilot only.
Never edits source archives, source tables, or releases. All heavy phases run in Slurm.
"""
import os
os.environ.setdefault('PROJ_DATA','/path/to/home/.conda/envs/gecc-lai/share/proj')
os.environ.setdefault('GDAL_CACHEMAX','64')
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import sys,json,hashlib,time,calendar,datetime,resource,platform,zipfile,shutil,traceback,re,math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely,pyproj,rasterio,pyarrow as pa,pyarrow.parquet as pq
from shapely.geometry import MultiPolygon,box
from shapely.affinity import translate
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.io import MemoryFile
from exactextract import exact_extract
R=Path('/path/to/project/gecc/extreme_temperature_tract2020/stage2_pilot')
S=Path('/path/to/scratch/extreme_temperature_tract2020/stage2_work')
D=Path('/path/to/project/gecc/extreme_temperature_discovery_20261011_stage1')
B=Path('/path/to/scratch/gecc_census_tract_airtemp/continuation_2021_2024')
A=Path('/path/to/scratch/gecc_tract_addon')
REF=R/'references'; STEP=.008983152841195215; CUTOFF=1/255; NODATA=-32768
KEY=['GEOID','observation_year','season']
MET=['hot_days_tmax_gt_p95','cold_days_tmin_lt_p05','valid_tmax_days','valid_tmin_days']
MAND=[(y,s) for y in [2003,2004,2020,2024] for s in ['DJF','JJA']]
DIAG=[(2010,'DJF'),(2022,'JJA'),(2023,'DJF')]
PERIODS=sorted(set(MAND+DIAG+[(y,s) for y in range(2003,2008) for s in ['DJF','JJA']]))
POLICIES=['nodata_only','nodata_range_m95_p65']
BASE=Path('/path/to/home/GECC/Census_2020_US_Tract_Polygons/Census_2020_US_Tract_Polygons.shp')
OLD=Path('/path/to/scratch/gecc_deliverables/stage/CensusTract_air_temperature_2003_2024/data/CensusTract_air_temperature_2003_2024.parquet')
ADD=A/'air_temperature/air_temperature_addon.parquet'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def js(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')

def log(*x): print(datetime.datetime.now(datetime.timezone.utc).isoformat(),*x,flush=True)

def gh(g):
 g=shapely.force_2d(g)
 if g.geom_type=='Polygon':g=MultiPolygon([g])
 return hashlib.sha256(shapely.to_wkb(shapely.normalize(g),byte_order=1,output_dimension=2,include_srid=False)).hexdigest()

def semantic(df):
 d=df.sort_values(KEY).reset_index(drop=True)
 return hashlib.sha256(d.to_json(orient='split',index=False,double_precision=15).encode()).hexdigest()

def table(df,p):
 d=df.copy();d['GEOID']=d.GEOID.astype('string');d['observation_year']=d.observation_year.astype('int64')
 for m in MET:
  if m in d:d[m]=pd.array(d[m],dtype='Float64')
 arrow=pa.Table.from_pandas(d,preserve_index=False); metadata=dict(arrow.schema.metadata or {}); metadata[b'stage2_method_config']= (R/'config.json').read_bytes(); metadata[b'stage2_code_sha256']=sha(Path(__file__)).encode(); metadata[b'stage2_source_manifest_sha256']=sha(R/'qa/staged_input_manifest.csv.gz').encode(); metadata[b'stage2_threshold_manifest']=b'qa/threshold_count_cache_manifests.json'; pq.write_table(arrow.replace_schema_metadata(metadata),p,compression='zstd'); d.to_csv(str(p).replace('.parquet','.csv.gz'),index=False,compression='gzip')

def ids():return pd.read_csv(D/'pilot_tract_selection.csv',dtype={'GEOID':str}).GEOID.tolist()

def meta(region):return json.loads((R/'native_grids.json').read_text())[region]

def map_native(lon,lat,m):
 a,_,x0,_,e,y0=m['transform'];x=6371007.181*np.radians(lon)*np.cos(np.radians(lat));y=6371007.181*np.radians(lat)
 col=np.floor((x-x0)/a).astype('int64');row=np.floor((y-y0)/e).astype('int64')
 ok=(col>=0)&(col<m['width'])&(row>=0)&(row<m['height'])
 return np.where(ok,row*m['width']+col,-1)

def grid_lonlat(cell,info,base=False):
 rr,cc=np.divmod(cell,info['shape'][1]);lon0,top=(info['window'][0],info['window'][3]) if base else info['origin']
 return lon0+(cc+.5)*STEP,top-(rr+.5)*STEP

def canonical_cell(lon,lat,info):
 cc=np.rint((lon-info['origin'][0])/STEP-.5).astype('int64');rr=np.rint((info['origin'][1]-lat)/STEP-.5).astype('int64')
 return rr*info['shape'][1]+cc

def wrap_parts(g):
 parts=list(g.geoms) if g.geom_type=='MultiPolygon' else [g];out=[]
 for p in parts:
  if p.bounds[2]-p.bounds[0]>180:
   p=shapely.transform(p,lambda xy:np.column_stack((np.where(xy[:,0]<0,xy[:,0]+360,xy[:,0]),xy[:,1])))
   for lo,hi,shift in [(0,180,0),(180,360,-360)]:
    q=p.intersection(box(lo,-90,hi,90))
    if not q.is_empty:out.append(translate(q,xoff=shift))
  else:out.append(p)
 return shapely.union_all(out)

def prepare():
 t=time.time(); S.mkdir(parents=True,exist_ok=True)
 # Byte and coordinate identity, without coordinate transformation.
 refs={}
 for line in (REF/'SHA256SUMS.txt').read_text().splitlines():
  h,f=line.split('  ',1);assert sha(REF/f)==h;refs[f]=h
 reg=pd.read_csv(REF/'current_allUS_registry_geometry_signatures.csv',dtype={'GEOID':str}).set_index('GEOID')
 official=pd.read_csv(REF/'official_territory_geometry_signatures.csv',dtype={'GEOID':str}).set_index('GEOID')
 frozen=json.loads((D/'geometry_cache_integrity.json').read_text())
 for x in frozen['polygon_hashes']:assert sha(x['path'])==x['sha256_newly_verified']
 base=gpd.read_file(BASE);assert len(base)==84414 and base.GEOID.is_unique and base.geometry.is_valid.all()
 diff=[str(k) for k,g in zip(base.GEOID,base.geometry) if gh(g)!=reg.loc[str(k),'geometry_sha256']]
 pd.DataFrame({'GEOID':base.GEOID.astype(str),'candidate_geometry_sha256':[gh(g) for g in base.geometry],'reference_geometry_sha256':[reg.loc[str(k),'geometry_sha256'] for k in base.GEOID]}).to_csv(R/'qa/base_geometry_signature_comparison.csv.gz',index=False); pd.DataFrame({'GEOID':diff}).to_csv(R/'qa/base_unmatched_geometry_ids.csv',index=False)
 territories=[];zip_checks=[]
 for x in json.loads((REF/'official_territory_source_archives.json').read_text()):
  p=A/'boundaries'/x['file'];assert sha(p)==x['sha256'];g=gpd.read_file(p)
  assert g.geometry.is_valid.all() and not g.geometry.is_empty.any()
  bad=[str(k) for k,q in zip(g.GEOID,g.geometry) if gh(q)!=official.loc[str(k),'geometry_sha256']]
  assert not bad,bad;territories.append(g)
  zip_checks.append({'path':str(p),'sha256':x['sha256'],'count':len(g),'crs':str(g.crs),'geometry_signature_mismatches':bad})
 territory=pd.concat(territories,ignore_index=True)
 candidate=set(base.GEOID)|set(territory.GEOID);assert not(set(base.GEOID)&set(territory.GEOID))
 assert candidate==set(reg.index) and len(candidate)==85528
 pilot=ids();assert sorted(pilot)==json.loads((REF/'current_release_evidence.json').read_text())['pilot_geoids']
 # Analysis reuses numerical stored coordinates; no forced datum transform.
 geometries=dict(zip(base.GEOID,base.geometry));geometries.update(dict(zip(territory.GEOID,territory.geometry)))
 gpd.GeoDataFrame({'GEOID':pilot},geometry=[geometries[k] for k in pilot],crs='EPSG:4326').to_parquet(S/'pilot_geometry.parquet',index=False)
 # Attributes required by the current table: centroid/Pop absent in add-on remain null.
 attrs=pd.DataFrame({'GEOID':territory.GEOID,'StFIPS':territory.STATEFP,'CoFIPS':territory.COUNTYFP,'TrFIPS':territory.TRACTCE,
 'LAT_GeoCen':np.nan,'LON_GeoCen':np.nan,'Area_Land':territory.ALAND.astype(float),'Area_Water':territory.AWATER.astype(float),'Pop':pd.array([None]*len(territory),dtype='Int64')})
 attrs.to_parquet(S/'territory_attributes.parquet',index=False)
 js(R/'qa/registry_geometry.json',{'passed':not diff,'candidate_source_geometry_passed':True,'registry_ids_passed':True,'references':refs,'registry_count':len(candidate),'base_geometry_matches':len(base)-len(diff),'base_mismatch_count':len(diff),'base_mismatch_ids_file':'qa/base_unmatched_geometry_ids.csv','base_coordinate_comparison_file':'qa/base_geometry_signature_comparison.csv.gz','unresolved_gate':'Current-release stored base coordinate equivalence cannot be established from hash-only evidence; candidate frozen bytes verified; full valid reference coordinates are absent.' if diff else None,'territory_geometry_matches':len(territory),'official_archives':zip_checks,'base_crs_wkt':base.crs.to_wkt(),'analysis_crs_assignment':'Stored geographic coordinates sampled directly; no datum transform. Coordinate identity is independent of CRS identity. Base unknown GRS80 datum, originals NAD83, serialized release EPSG:4326.','known_serialized_release_defect':1013,'reference_registry_digest':hashlib.sha256(('\n'.join(sorted(candidate))+'\n').encode()).hexdigest()})
 # Dates explicitly audited; rejects duplicates rather than a dictionary overwrite.
 inventory=pd.read_csv(D/'daily_members.csv.gz');inventory['date']=pd.to_datetime(inventory.date)
 assert not inventory.duplicated(['region','variable','date']).any()
 selected=inventory.loc[[(int(y),s) in PERIODS for y,s in zip(inventory.year,inventory.season)]].copy()
 calendar_checks=[]
 for (region,y,s,var),d in selected.groupby(['region','year','season','variable']):
  months=[6,7,8] if s=='JJA' else [1,2,12];expected={datetime.date(int(y),mo,day) for mo in months for day in range(1,calendar.monthrange(int(y),mo)[1]+1)}
  actual=set(d.date.dt.date);missing=sorted(expected-actual)
  assert actual<=expected and missing==([datetime.date(int(y),12,31)] if s=='DJF' and calendar.isleap(int(y)) else [])
  calendar_checks.append({'region':region,'year':int(y),'season':s,'variable':var,'source_days':len(d),'calendar_days':len(expected),'missing_dates':[str(x) for x in missing]})
 for r in ['NorthAmerica','SouthAmerica']:
  for s,total in [('JJA',460),('DJF',450)]:
   for v in ['TMAX','TMIN']:assert len(selected[(selected.region==r)&(selected.season==s)&(selected.variable==v)&selected.year.between(2003,2007)])==total
 for record in selected.itertuples():
  match=re.fullmatch(r'(NorthAmerica|SouthAmerica)_Ta_SVCMsp_(\d{4})(TMAX|TMIN)_(\d{3})\.tif',Path(record.member).name); assert match
  region,y,var,doy=match.groups(); actual=datetime.date(int(y),1,1)+datetime.timedelta(days=int(doy)-1)
  assert actual.year==int(y) and region==record.region and var==record.variable and actual==record.date.date()
 selected['date']=selected.date.dt.strftime('%Y-%m-%d');selected.to_csv(R/'qa/selected_members.csv.gz',index=False)
 js(R/'qa/calendar.json',{'passed':True,'duplicate_keys':0,'records':calendar_checks})
 # Coverage/grid signature verification for all pilot pairs; cache for other IDs is not computed.
 info=json.loads((A/'air_temperature/fua_global_index_info.json').read_text());bi=json.loads((B/'work/coverage_index_m1000_o0_info.json').read_text())
 assert bi['step_deg']==info['step_deg']==STEP and bi['origin']==[0.,0.]
 for coord in [*info['origin'],bi['window'][0],bi['window'][3]]:assert abs(coord/STEP-round(coord/STEP))<1e-8
 template=S/'global_grid_template.tif'
 if not template.exists():
  with rasterio.open(template,'w',driver='GTiff',width=info['shape'][1],height=info['shape'][0],count=1,dtype='uint8',crs='EPSG:4326',transform=Affine(STEP,0,info['origin'][0],0,-STEP,info['origin'][1]),tiled=True,compress='deflate',SPARSE_OK='YES'):pass
 ix=np.load(B/'work/sampling_index_final.npz');order=pd.read_csv(B/'work/grid_index_geoids.csv',dtype=str);orderids=order.iloc[:, -1].tolist()
 assert orderids==base.GEOID.tolist(),order.columns.tolist()
 ax=np.load(A/'air_temperature/fua_global_index.npz');aids=json.loads((A/'air_temperature/index_geoids.json').read_text());assert set(aids)==set(territory.GEOID)
 # The global index was built with NAD83->WGS84; verify its geometry numerically.
 cached=gpd.read_file(A/'air_temperature/tracts_for_index.gpkg');cached_map=dict(zip(cached.eFUA_ID.astype(str).str.zfill(11),cached.geometry))
 transformed_mismatch=[k for k in aids if gh(cached_map[k])!=gh(geometries[k])]
 assert not transformed_mismatch, f'Cached territory coordinate mismatch: {transformed_mismatch}'
 coverage_rows=[];verification=[];geometry_checks=[];mapping_corrections=[]
 for pi,k in enumerate(pilot):
  if k in set(base.GEOID):
   ti=orderids.index(k);keep=ix['tract']==ti;localcell=ix['cell'][keep];cov=ix['coverage'][keep]
   lon,lat=grid_lonlat(localcell,bi,True);cell=canonical_cell(lon,lat,info)
   native=map_native(lon,lat,meta('NorthAmerica'));old_native=ix['native_idx'][keep];changed=native!=old_native
   for c,old,new in zip(cell[changed],old_native[changed],native[changed]):mapping_corrections.append({'GEOID':k,'cell':int(c),'old_native_idx':int(old),'exact_header_native_idx':int(new)})
   region='NorthAmerica'
  else:
   ti=aids.index(k);keep=ax['fua']==ti;cell=ax['cell'][keep];cov=ax['coverage'][keep];lon,lat=grid_lonlat(cell,info)
   region='SouthAmerica' if k[:2] in ['72','78'] else 'unsupported_pacific';native=map_native(lon,lat,meta('SouthAmerica')) if region=='SouthAmerica' else np.full(len(cell),-1,dtype='int64')
  assert len(cell)==len(np.unique(cell)) and np.all(cov>=CUTOFF)
  if k!='02016000100':
   geom=geometries[k];gf=gpd.GeoDataFrame({'GEOID':[k]},geometry=[geom],crs='EPSG:4326')
   fresh=exact_extract(str(template),gf,['cell_id','coverage'],output='pandas').iloc[0]
   cc=np.asarray(fresh.cell_id,dtype='int64');ww=np.asarray(fresh.coverage,dtype='float32');kk=ww>=CUTOFF;cc=cc[kk];ww=ww[kk]
   oldsort=np.argsort(cell);newsort=np.argsort(cc);assert np.array_equal(cell[oldsort],cc[newsort]),f'Coverage cell mismatch {k}'
   assert np.allclose(cov[oldsort],ww[newsort],rtol=0,atol=3e-7),f'Coverage weights {k}'
   # Independent GEOS intersections for all cells of small real polygons.
   if len(cell)<500:
    testcells=cell;tlon,tlat=grid_lonlat(testcells,info)
    geos=np.array([geom.intersection(box(x-STEP/2,y-STEP/2,x+STEP/2,y+STEP/2)).area/(STEP*STEP) for x,y in zip(tlon,tlat)])
    assert np.allclose(geos,cov,atol=2e-6,rtol=0);geometry_checks.append({'GEOID':k,'cells':len(cell),'max_abs_error':float(np.max(np.abs(geos-cov))) if len(cell) else 0})
  else:assert len(cell)==0
  coverage_rows.append(pd.DataFrame({'pilot_index':pi,'GEOID':k,'cell':cell,'coverage':cov,'native_idx':native,'source_region':region,'index_policy':'legacy_compatibility'}))
  verification.append({'GEOID':k,'pairs':len(cell),'inside_source_grid_pairs':int((native>=0).sum()),'source_region':region})
 # Split/wrap includes both signs. Positive longitude can be outside selected NA grid.
 geom=wrap_parts(geometries['02016000100']);fresh=exact_extract(str(template),gpd.GeoDataFrame({'GEOID':['02016000100']},geometry=[geom],crs='EPSG:4326'),['cell_id','coverage'],output='pandas').iloc[0]
 cell=np.asarray(fresh.cell_id,dtype='int64');cov=np.asarray(fresh.coverage,dtype='float32');keep=cov>=CUTOFF;cell=cell[keep];cov=cov[keep];assert len(cell)==len(np.unique(cell))
 lon,lat=grid_lonlat(cell,info);assert np.any(lon<0) and np.any(lon>0)
 native=map_native(lon,lat,meta('NorthAmerica'))
 pd.DataFrame({'pilot_index':pilot.index('02016000100'),'GEOID':'02016000100','cell':cell,'coverage':cov,'native_idx':native,'source_region':'NorthAmerica','index_policy':'split_wrap_diagnostic'}).to_parquet(S/'wrap_pairs.parquet',index=False)
 pairs=pd.concat(coverage_rows,ignore_index=True);pairs.to_parquet(S/'pilot_pairs.parquet',index=False)
 allpairs=pd.concat([pairs,pd.read_parquet(S/'wrap_pairs.parquet')],ignore_index=True)
 signatures={}
 for region in ['NorthAmerica','SouthAmerica']:
  ni=np.unique(allpairs.loc[(allpairs.source_region==region)&(allpairs.native_idx>=0),'native_idx'].to_numpy('int64'));assert np.all(np.diff(ni)>0)
  # Nearest-neighbor needs precisely reached cells, radius zero: no neighboring-value interpolation.
  np.save(S/f'cells_{region}.npy',ni);signatures[region]={'selected_native_cells':len(ni),'native_cell_identity_sha256':hashlib.sha256(ni.tobytes()).hexdigest(),'halo_radius':0}
  pos=allpairs[(allpairs.source_region==region)&(allpairs.native_idx>=0)]
  assert np.array_equal(ni[np.searchsorted(ni,pos.native_idx)],pos.native_idx)
  lon0,lat0=grid_lonlat(pos.cell.to_numpy()[::max(1,len(pos)//1024)],info)
  transform=pyproj.Transformer.from_crs('EPSG:4326','+proj=sinu +R=6371007.181 +units=m +no_defs',always_xy=True)
  xx,yy=transform.transform(lon0,lat0);m=meta(region);a,_,x0,_,e,y0=m['transform']
  direct=np.floor((yy-y0)/e).astype('int64')*m['width']+np.floor((xx-x0)/a).astype('int64')
  assert np.array_equal(direct,map_native(lon0,lat0,m))
 js(R/'qa/coverage.json',{'passed':True,'grid':info,'base_grid':bi,'verified_tracts':verification,'independent_GEOS':geometry_checks,'source_transform_metadata':'Exact original TIFF header; legacy work metadata rounded to six decimals is not used.','legacy_native_map_corrections':mapping_corrections,'cached_territory_coordinate_mismatches':transformed_mismatch,'wrap_pairs':len(cell),'wrap_positive_pairs':int((lon>0).sum()),'wrap_negative_pairs':int((lon<0).sum()),'wrap_inside_NA':int((native>=0).sum()),'source_cells':signatures,'reuse_hashes':{str(p):sha(p) for p in [B/'work/sampling_index_final.npz',B/'work/grid_index_geoids.csv',A/'air_temperature/fua_global_index.npz',S/'pilot_pairs.parquet',S/'wrap_pairs.parquet']}})
 js(R/'qa/environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'shapely':shapely.__version__,'GEOS':shapely.geos_version_string,'rasterio':rasterio.__version__,'GDAL':rasterio.__gdal_version__,'pyproj':pyproj.__version__,'proj_data':pyproj.datadir.get_data_dir(),'pyarrow':pa.__version__,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'prepare_seconds':time.time()-t})
 log('prepare passed',signatures)

# Validity and exact percentile are applied independently in each policy.
def validity(x,policy):
 v=x!=NODATA
 if policy!=POLICIES[0]:v &= (x>=-950)&(x<=650)
 return v

def quantile(x,p,policy):
 v=validity(x,policy);n=v.sum(axis=0,dtype=np.int32);sorted_x=np.sort(np.where(v,x,np.int16(32767)),axis=0).astype(np.int64)
 h=np.maximum(n-1,0).astype(np.int64)*p;i=h//100;r=h%100;c=np.arange(x.shape[1]);j=np.minimum(i+1,np.maximum(n-1,0))
 q=100*sorted_x[i,c]+r*(sorted_x[j,c]-sorted_x[i,c]);q[n==0]=0
 return q.astype(np.int32),n.astype(np.int16)

def hand_tests():
 from fractions import Fraction
 arrays=[[],[3],[-3,-3,-3],[0,1,2,3,4],[650,32766,-32767,-951,651],[-32768,-32768],[10,-32768,20],[-32767,32766]]
 for policy in POLICIES:
  for p in [5,95]:
   for a in arrays:
    x=np.full((max(1,len(a)),1),NODATA,dtype=np.int16)
    if a:x[:len(a),0]=a
    q,n=quantile(x,p,policy);valid=sorted([v for v in a if v!=NODATA and (policy==POLICIES[0] or -950<=v<=650)])
    assert n[0]==len(valid)
    if valid:
     h=Fraction((len(valid)-1)*p,100);i=h.numerator//h.denominator;r=h-i;expected=valid[i]+r*(valid[min(i+1,len(valid)-1)]-valid[i]);assert Fraction(int(q[0]),100)==expected
     for raw in [-32767,-950,-3,0,3,650,32766]:
      assert (100*int(raw)>int(q[0]))==(Fraction(raw)>expected)
      assert (100*int(raw)<int(q[0]))==(Fraction(raw)<expected)
    else:assert q[0]==0
 # Strict exact equality and null versus real zero; independently weighted masked bands.
 q,n=quantile(np.array([[100],[100]],np.int16),95,POLICIES[0]);assert q[0]==10000 and not(100*100>q[0])
 v=np.array([0.,2.,np.nan]);w=np.array([.5,.25,.75]);good=np.isfinite(v);assert np.sum(v[good]*w[good])/np.sum(w[good])==2/3
 try:
  sample=pd.DataFrame({'r':['a','a'],'d':['2004-02-29']*2});assert not sample.duplicated(['r','d']).any()
 except AssertionError:pass
 else:raise AssertionError('duplicate test did not reject')
 js(R/'qa/hand_tests.json',{'passed':True,'policies':POLICIES,'arrays':len(arrays),'ranks':[5,95],'exact_equality_zero':True,'overflow_extremes':True,'nodata_n0_n1_partial_ties':True,'duplicate_rejection':True,'weighted_null_zero_fraction':True})

# Full member read once: hash compressed file, then decode selected full-row chunks.
# No full raster duplicated on disk. Selected int16 values persisted losslessly.
def verify_header_grid(actual, reference, cells, region, year):
 actual=np.asarray(actual,dtype=float);ref=np.asarray(reference['transform'],dtype=float)
 assert actual[1]==actual[3]==0 and actual[0]>0 and actual[4]<0
 corners=np.array([[0.,0.],[reference['width'],0.],[0.,reference['height']],[reference['width'],reference['height']]])
 dx=(actual[2]-ref[2])+corners[:,0]*(actual[0]-ref[0]);dy=(actual[5]-ref[5])+corners[:,1]*(actual[4]-ref[4]);drift=float(np.hypot(dx,dy).max())
 assert drift<=.002, f'Native grid extent drift {drift} m: {region} {year}'
 rr,cc=np.divmod(cells,reference['width']);x=ref[2]+(cc+.5)*ref[0];y=ref[5]+(rr+.5)*ref[4]
 native=np.floor((y-actual[5])/actual[4]).astype('int64')*reference['width']+np.floor((x-actual[2])/actual[0]).astype('int64');assert np.array_equal(native,cells),'Native cell center identity changed'
 pairs=pd.concat([pd.read_parquet(S/'pilot_pairs.parquet'),pd.read_parquet(S/'wrap_pairs.parquet')],ignore_index=True);pairs=pairs[pairs.source_region==region];info=json.loads((R/'qa/coverage.json').read_text())['grid'];lon,lat=grid_lonlat(pairs.cell.to_numpy(),info)
 m=dict(reference);m['transform']=actual.tolist();new=map_native(lon,lat,m);different=new!=pairs.native_idx.to_numpy();count=int(different.sum())
 result={'region':region,'year':year,'actual_transform':actual.tolist(),'reference_transform':ref.tolist(),'max_extent_corner_drift_m':drift,'selected_native_cell_centers_checked':len(cells),'native_center_identity_mismatches':0,'geographic_pairs_checked':len(pairs),'geographic_sample_mapping_mismatches':count,'no_reprojection_rounding_or_interpolation':True}
 js(R/'qa'/f'header_grid_equivalence_{region}_{year}.json',result)
 if count:pairs.loc[different,['GEOID','cell','native_idx']].assign(actual_native_idx=new[different]).to_csv(R/'qa'/f'header_grid_mismatches_{region}_{year}.csv',index=False)
 result['count_band_sampling_grid']='Fixed exact baseline native reference grid; daily row/column cells co-registered by verified native center identity. Raw-header geographic mapping differences saved separately.';result['native_cell_coregistration_passed']=True;js(R/'qa'/f'header_grid_equivalence_{region}_{year}.json',result)
 return result

def stage_task(task):
 region,year,rows,expected=task;start=time.time();cells=np.load(S/f'cells_{region}.npy');m=meta(region);nr,nc=m['height'],m['width']
 native_rows,native_cols=np.divmod(cells,nc); chunks=[]
 for chunk in np.unique(native_rows//64):
  first=int(chunk*64);last=min(first+64,nr);pos=np.flatnonzero((native_rows>=first)&(native_rows<last));chunks.append((first,last,pos,native_rows[pos]-first,native_cols[pos]))
 outdir=S/'raw'/region;outdir.mkdir(parents=True,exist_ok=True);records=[];handles={};verified_transforms={}
 grouped={}
 for rec in rows:grouped.setdefault((rec['season'],rec['variable']),[]).append(rec)
 outputs={};lookup={}
 for (season,var),rr in grouped.items():
  rr.sort(key=lambda r:r['date']);name=f'{year}_{season}_{var}';p=outdir/f'{name}.npy';done=outdir/f'{name}.complete.json'
  signature=hashlib.sha256(json.dumps(rr,sort_keys=True).encode()+cells.tobytes()).hexdigest()
  if done.exists():
   old=json.loads(done.read_text());assert old['input_signature']==signature and sha(p)==old['values_sha256'];records.extend(old['records']);continue
  temporary=Path(str(p)+'.'+str(os.environ.get('SLURM_JOB_ID','local'))+'.partial');arr=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.int16,shape=(len(rr),len(cells)));outputs[name]=(arr,p,done,signature,[],temporary)
  for j,rec in enumerate(rr):lookup[rec['member']]=(name,j)
 try:
  for rec in sorted(rows,key=lambda r:int(r['offset_data'])):
   if rec['member'] not in lookup:continue
   file_started=time.time()
   archive=rec['archive']
   if archive not in handles:handles[archive]=open(archive,'rb')
   f=handles[archive];f.seek(int(rec['offset_data']));content=f.read(int(rec['member_bytes']));assert len(content)==int(rec['member_bytes'])
   digest=hashlib.sha256(content).hexdigest();logical=Path(archive).name+'/'+rec['member'];inherited=expected.get(logical)
   if inherited:assert digest==inherited,logical
   name,j=lookup[rec['member']];arr=outputs[name][0];decoded=0
   with MemoryFile(content) as mf,mf.open() as ds:
    assert ds.width==nc and ds.height==nr and ds.count==1 and ds.dtypes==('int16',) and ds.nodata==NODATA
    assert ds.crs.to_dict().get('proj')=='sinu' and abs(ds.crs.to_dict().get('R',6371007.181)-6371007.181)<1e-6
    actual_transform=tuple(list(ds.transform)[:6])
    if actual_transform not in verified_transforms:
     if np.allclose(actual_transform,m['transform'],rtol=0,atol=1e-7):verified_transforms[actual_transform]={'max_extent_corner_drift_m':None,'geographic_sample_mapping_mismatches':0,'strict_transform_match':True}
     else:verified_transforms[actual_transform]=verify_header_grid(actual_transform,m,cells,region,year)
    grid_check=verified_transforms[actual_transform]
    assert ds.scales==(1.0,) and ds.offsets==(0.0,)
    assert ds.block_shapes[0]==(1,nc),ds.block_shapes
    for first,last,pos,rr,cc in chunks:
     block=ds.read(1,window=Window(0,first,nc,last-first));assert block.shape==(last-first,nc)
     arr[j,pos]=block[rr,cc];decoded+=block.nbytes
   staged=arr[j];bad=(staged!=NODATA)&((staged<-950)|(staged>650))
   record={'region':region,'year':year,'season':rec['season'],'variable':rec['variable'],'date':rec['date'],'logical_path':logical,'member_bytes':len(content),'full_file_sha256_new':digest,'inherited_sha256':inherited,'inherited_match':bool(inherited),'selected_values_sha256':hashlib.sha256(staged.tobytes()).hexdigest(),'decoded_full_row_bytes':decoded,'selected_cells':len(cells),'nodata_cells':int((staged==NODATA).sum()),'outside_range_cells':int(bad.sum()),'raw_min':int(staged[staged!=NODATA].min()) if np.any(staged!=NODATA) else None,'raw_max':int(staged[staged!=NODATA].max()) if np.any(staged!=NODATA) else None,'header_verified':True,'actual_transform':list(actual_transform),'grid_equivalence_check':grid_check,'read_started_unix':file_started,'read_finished_unix':time.time()}
   records.append(record);outputs[name][4].append(record)
  for arr,p,done,signature,recs,temporary in outputs.values():
   arr.flush();del arr;os.replace(temporary,p)
   js(done,{'input_signature':signature,'values_sha256':sha(p),'records':recs})
 finally:
  for f in handles.values():f.close()
 elapsed=time.time()-start;result={'region':region,'year':year,'elapsed_seconds':elapsed,'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'records':records,'member_bytes':sum(x['member_bytes'] for x in records),'decoded_bytes':sum(x['decoded_full_row_bytes'] for x in records)}
 js(R/'qa'/f'stage_{region}_{year}.json',result);log('staged',region,year,len(records),round(elapsed,1),'seconds');return {k:v for k,v in result.items() if k!='records'}

def stage():
 start=time.time()
 if (R/'qa/staging_summary.json').exists():
  existing=json.loads((R/'qa/staging_summary.json').read_text());assert existing['passed']
  for row in json.loads((R/'qa/archive_stat_preflight.json').read_text())['records']:
   original=pd.read_csv(R/'qa/stage1_archive_inventory.csv').set_index('path').loc[row['path']];stat=Path(row['path']).stat();assert stat.st_size==original['bytes'] and stat.st_mtime_ns==original['mtime_ns']
  for region in ['NorthAmerica','SouthAmerica']:
   cells=np.load(S/f'cells_{region}.npy')
   for done in (S/'raw'/region).glob('*.complete.json'):
    cache=json.loads(done.read_text());data=done.with_name(done.name.replace('.complete.json','.npy'));assert sha(data)==cache['values_sha256'];array=np.load(data,mmap_mode='r');assert array.shape[1]==len(cells)
  assert len(list((S/'raw').rglob('*.complete.json')))==68
  assert len(pd.read_csv(R/'qa/staged_input_manifest.csv.gz'))==existing['full_member_sha256_new']
  log('verified complete staging checkpoint reuse');return {'checkpoint_reused':True}
 d=pd.read_csv(R/'qa/selected_members.csv.gz');expected={}
 for p in [B/'outputs_full_series/source_input_manifest.csv',A/'provenance/svcmsp_source_input_manifest.csv']:
  df=pd.read_csv(p,usecols=['logical_path','sha256']);assert not df.logical_path.duplicated().any()
  for k,h in zip(df.logical_path,df.sha256):
   if k in expected:assert expected[k]==h
   expected[k]=h
 tasks=[(region,int(y),df.to_dict('records'),expected) for (region,y),df in d.groupby(['region','year'])]
 # Representative benchmark before launching remaining workers.
 benchmark=tasks.pop(0);result=[stage_task(benchmark)]
 if not (R/'qa/initial_io_benchmark.json').exists():js(R/'qa/initial_io_benchmark.json',result[0])
 workers=int(os.environ.get('PILOT_WORKERS','4'))
 with ProcessPoolExecutor(workers) as ex:
  for f in as_completed([ex.submit(stage_task,x) for x in tasks]):result.append(f.result())
 allrecords=[]
 for x in sorted((R/'qa').glob('stage_*.json')):allrecords.extend(json.loads(x.read_text())['records'])
 manifest=pd.DataFrame(allrecords);assert len(manifest)==len(d) and not manifest.logical_path.duplicated().any();manifest=manifest.sort_values(['region','year','variable','date']).reset_index(drop=True);manifest.to_csv(R/'qa/staged_input_manifest.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 js(R/'qa/staging_summary.json',{'passed':True,'workers':workers,'elapsed_seconds':time.time()-start,'tasks':result,'all_used_headers_verified':len(manifest),'full_member_sha256_new':len(manifest),'inherited_member_hash_matches':int(manifest.inherited_match.sum()),'compressed_member_bytes_read':int(manifest.member_bytes.sum()),'decoded_full_row_bytes':int(manifest.decoded_full_row_bytes.sum()),'read_span_seconds':float(manifest.read_finished_unix.max()-manifest.read_started_unix.min()),'serial_file_processing_seconds':float((manifest.read_finished_unix-manifest.read_started_unix).sum()),'selected_cache_bytes':sum(p.stat().st_size for p in (S/'raw').rglob('*.npy')),'archive_integrity':'Inherited publisher hashes and Stage1 listing/size evidence; full annual archives not rehashed. All used compressed TIFF members newly hashed and all used row chunks read.'})
 log('stage passed')

def load_raw(region,y,s,v):return np.load(S/'raw'/region/f'{y}_{s}_{v}.npy',mmap_mode='r')

def compute_task(args):
 region,season,policy,run=args;start=time.time();cells=np.load(S/f'cells_{region}.npy');nc=len(cells);pcts=[95,5];vars=['TMAX','TMIN'];tile=8192
 out=S/'computed'/f'run{run}'/region/season/policy;out.mkdir(parents=True,exist_ok=True)
 signature={'runtime':{'numpy':np.__version__,'rasterio':rasterio.__version__,'GDAL':rasterio.__gdal_version__},'native_grid':meta(region),'code_sha256':sha(Path(__file__)),'policy':policy,'region':region,'season':season,'baseline':[2003,2007],'cell_sha256':sha(S/f'cells_{region}.npy'),'source_manifest_sha256':sha(R/'qa/staged_input_manifest.csv.gz'),'quantile':'HF7 exact q_num/100 raw; int64 arithmetic; Celsius=q_num/1000','tile_cells':tile}
 if (out/'cache_manifest.json').exists():
  existing=json.loads((out/'cache_manifest.json').read_text())
  assert all(existing.get(k)==v for k,v in signature.items()), f'Immutable cache signature changed: {out}; preserve it and use a new cache directory.'
  assert all(sha(out/name)==digest for name,digest in existing['files'].items())
  log('verified checkpoint reuse',region,season,policy,run);return str(out)
 q=np.lib.format.open_memmap(out/'threshold_qnum.npy',mode='w+',dtype=np.int32,shape=(2,nc));n=np.lib.format.open_memmap(out/'baseline_n.npy',mode='w+',dtype=np.int16,shape=(2,nc))
 baselines={v:[load_raw(region,y,season,v) for y in range(2003,2008)] for v in vars}
 for lo in range(0,nc,tile):
  hi=min(nc,lo+tile)
  for vi,v in enumerate(vars):
   cube=np.concatenate([a[:,lo:hi] for a in baselines[v]],axis=0);q[vi,lo:hi],n[vi,lo:hi]=quantile(cube,pcts[vi],policy)
 q.flush();n.flush();periods=[(y,s) for y,s in MAND+DIAG if s==season];qa=[]
 for y,s in periods:
  daily=[load_raw(region,y,s,v) for v in vars];counts=np.lib.format.open_memmap(out/f'counts_{y}.npy',mode='w+',dtype=np.int16,shape=(4,nc))
  masks=np.lib.format.open_memmap(out/f'support_{y}.npy',mode='w+',dtype=bool,shape=(4,nc));real_zero=[0]*4;null=[0]*4
  for lo in range(0,nc,tile):
   hi=min(nc,lo+tile);valid=[];exceeds=[]
   for vi,a in enumerate(daily):
    raw=np.array(a[:,lo:hi]);v=validity(raw,policy);valid.append(v.sum(axis=0,dtype=np.int16));scaled=raw.astype(np.int64)*100
    comparison=(scaled>q[vi,lo:hi]) if vi==0 else (scaled<q[vi,lo:hi]);exceeds.append((comparison&v).sum(axis=0,dtype=np.int16));assert np.all(exceeds[vi]<=valid[vi])
   common=(valid[0]>0)&(valid[1]>0);ms=np.stack([common&(n[0,lo:hi]>0),common&(n[1,lo:hi]>0),common,common]);c=np.stack([exceeds[0],exceeds[1],valid[0],valid[1]])
   counts[:,lo:hi]=c;masks[:,lo:hi]=ms
   for b in range(4):real_zero[b]+=int(((c[b]==0)&ms[b]).sum());null[b]+=int((~ms[b]).sum())
  counts.flush();masks.flush()
  qa.append({'region':region,'season':season,'year':y,'policy':policy,'cells':nc,'real_zero_by_band':real_zero,'null_by_band':null,'days':[len(a) for a in daily]})
 # Independent direct scalar calculations, including first valid and first nodata and deterministic samples.
 directchecks=0
 positions=np.unique(np.r_[np.linspace(0,max(0,nc-1),min(16,nc),dtype=int),np.flatnonzero(n[0]>0)[:4],np.flatnonzero(n[0]==0)[:4]])
 for vi,v in enumerate(vars):
  for pos in positions:
   raw=np.concatenate([a[:,pos] for a in baselines[v]]).tolist();good=sorted([int(a) for a in raw if a!=NODATA and (policy==POLICIES[0] or -950<=a<=650)])
   if good:
    h=(len(good)-1)*pcts[vi];i,r=divmod(h,100);expected=100*good[i]+r*(good[min(i+1,len(good)-1)]-good[i]);assert q[vi,pos]==expected
   assert n[vi,pos]==len(good)
   for y,s in periods:
    a=load_raw(region,y,s,v)[:,pos].tolist();valid=[int(x) for x in a if x!=NODATA and (policy==POLICIES[0] or -950<=x<=650)]
    counts=np.load(out/f'counts_{y}.npy',mmap_mode='r');masks=np.load(out/f'support_{y}.npy',mmap_mode='r')
    assert counts[vi+2,pos]==len(valid)
    if good:
     count=sum((100*x>expected if vi==0 else 100*x<expected) for x in valid);assert counts[vi,pos]==count
    common=all(np.load(out/f'counts_{y}.npy',mmap_mode='r')[j+2,pos]>0 for j in [0,1]);assert masks[vi,pos]==(common and len(good)>0);assert masks[vi+2,pos]==common
    directchecks+=1
 js(out/'cache_manifest.json',{**signature,'files':{p.name:sha(p) for p in out.glob('*.npy')},'direct_scalar_checks':directchecks,'elapsed_seconds':time.time()-start,'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'pixel_qa':qa})
 for path in out.glob('*.npy'):path.chmod(0o444)
 log('computed',region,season,policy,run,round(time.time()-start,1));return str(out)

def reduce_run(run):
 pilot=ids();pairs=pd.read_parquet(S/'pilot_pairs.parquet');wrap=pd.read_parquet(S/'wrap_pairs.parquet');rows=[];supports=[];threshold_qa=[]
 for policy in POLICIES:
  for index_policy,pp in [('legacy_compatibility',pairs),('split_wrap_diagnostic',wrap)]:
   gids=pilot if index_policy=='legacy_compatibility' else ['02016000100']
   for k in gids:
    pairs_k=pp[pp.GEOID==k];region='unsupported_pacific' if k[:2] in ['60','66','69'] else ('SouthAmerica' if k[:2] in ['72','78'] else 'NorthAmerica')
    good=pairs_k.native_idx>=0;weight=pairs_k.loc[good,'coverage'].to_numpy('float64');native=pairs_k.loc[good,'native_idx'].to_numpy('int64')
    if region!='unsupported_pacific':
     cells=np.load(S/f'cells_{region}.npy');pos=np.searchsorted(cells,native);assert np.array_equal(cells[pos],native)
    else:pos=np.array([],dtype=int)
    for y,s in MAND+DIAG:
     rec={'GEOID':k,'observation_year':int(y),'season':s,'boundary_year':2020,'source_region':region,'method_id':'svcmsp_native_hf7_v1_'+policy,'index_policy':index_policy,'baseline_start_year':2003,'baseline_end_year':2007,'tmax_percentile':95,'tmin_percentile':5,'DJF_label':'Jan_Feb_Dec_same_year','registry_status':'candidate_registry_only_base_hash_unresolved' if not json.loads((R/'qa/registry_geometry.json').read_text())['passed'] else 'current_registry_original_geometry_verified'}
     values=[np.nan]*4;denoms=[0.]*4;matched=[np.nan]*2
     baseline_mean=[np.nan]*2;baseline_min=[None]*2;baseline_partial=[0.]*2
     if len(pos):
      out=S/'computed'/f'run{run}'/region/s/policy;c=np.load(out/f'counts_{y}.npy',mmap_mode='r')[:,pos];m=np.load(out/f'support_{y}.npy',mmap_mode='r')[:,pos];bn=np.load(out/'baseline_n.npy',mmap_mode='r')[:,pos];qq=np.load(out/'threshold_qnum.npy',mmap_mode='r')[:,pos]
      for b in range(4):
       mask=m[b];denoms[b]=float(weight[mask].sum());values[b]=float(np.dot(c[b,mask].astype(float),weight[mask])/denoms[b]) if denoms[b] else np.nan
       if b<2 and denoms[b]:
        matched[b]=float(np.dot(c[b+2,mask].astype(float),weight[mask])/denoms[b]);assert -1e-12<=values[b]<=matched[b]+1e-12
      for v in range(2):
       bmask=bn[v]>0;den=weight[bmask].sum()
       if den:
        baseline_mean[v]=float(np.dot(bn[v,bmask],weight[bmask])/den);baseline_min[v]=int(bn[v,bmask].min());baseline_partial[v]=float(weight[bmask&(bn[v]<(460 if s=='JJA' else 450))].sum())
     for b,metric in enumerate(MET):rec[metric]=values[b]
     status='ok'
     if region=='unsupported_pacific':status='unsupported_pacific_source_route'
     elif k=='02016000100' and index_policy=='legacy_compatibility':status='legacy_antimeridian_index_limitation'
     elif not len(pairs_k):status='no_coverage_pairs'
     elif not len(pos):status='outside_selected_source_grid'
     elif not any(denoms):status='source_no_common_seasonal_support'
     elif not all(denoms):status='partial_band_threshold_support'
     rec['support_status']=status;rows.append(rec)
     support={'GEOID':k,'observation_year':y,'season':s,'method_id':rec['method_id'],'index_policy':index_policy,'source_region':region,'coverage_pairs':len(pairs_k),'inside_grid_pairs':len(pos),'total_coverage_weight':float(pairs_k.coverage.sum()),'inside_grid_weight':float(weight.sum()),'support_status':status}
     for b in range(4):support[MET[b]+'_support_weight']=denoms[b]
     support.update({'hot_matched_valid_days_bound':matched[0],'cold_matched_valid_days_bound':matched[1],'baseline_tmax_mean_days':baseline_mean[0],'baseline_tmin_mean_days':baseline_mean[1],'baseline_tmax_min_days':baseline_min[0],'baseline_tmin_min_days':baseline_min[1],'baseline_tmax_partial_weight':baseline_partial[0],'baseline_tmin_partial_weight':baseline_partial[1]})
     supports.append(support)
 df=pd.DataFrame(rows);support=pd.DataFrame(supports)
 primary=df[(df.method_id.str.endswith(POLICIES[0]))&(df.index_policy=='legacy_compatibility')]
 mandatory=primary.loc[[(y,s) in MAND for y,s in zip(primary.observation_year,primary.season)]];diag=primary.loc[[(y,s) in DIAG for y,s in zip(primary.observation_year,primary.season)]]
 sens=df[(df.method_id.str.endswith(POLICIES[1]))&(df.index_policy=='legacy_compatibility')];wd=df[df.index_policy=='split_wrap_diagnostic']
 assert len(mandatory)==256 and len(diag)==96 and len(sens)==352 and len(wd)==22
 for dd,periods in [(mandatory,MAND),(diag,DIAG),(sens,MAND+DIAG)]:
  assert not dd.duplicated(KEY).any() and set(dd.GEOID)==set(pilot)
  assert set(zip(dd.observation_year,dd.season))==set(periods)
  assert dd.groupby(['observation_year','season']).GEOID.nunique().eq(32).all()
  for metric in MET:
   assert dd[metric].dropna().between(0,92).all()
 # Freeze output metadata and linked manifests in Parquet file metadata.
 root=R if run==1 else S/'repeat_outputs';root.mkdir(exist_ok=True)
 for name,dd in [('primary_pilot',mandatory),('primary_diagnostics',diag),('range_sensitivity',sens),('antimeridian_diagnostic',wd)]:table(dd,root/f'{name}.parquet')
 support.to_parquet(root/'support_qa.parquet',index=False);support.to_csv(root/'support_qa.csv.gz',index=False,compression='gzip')
 result={'passed':True,'run':run,'semantic_digests':{n:semantic(d) for n,d in [('primary_pilot',mandatory),('primary_diagnostics',diag),('range_sensitivity',sens),('antimeridian_diagnostic',wd),('support_qa',support)]},'rows':{'mandatory':len(mandatory),'diagnostic':len(diag),'sensitivity':len(sens),'wrap':len(wd)},'primary_status_counts':primary.support_status.value_counts().to_dict(),'primary_nulls':{m:int(primary[m].isna().sum()) for m in MET},'primary_valid_zeros':{m:int(primary[m].eq(0).sum()) for m in MET}}
 js(root/'output_validation.json',result);return result

def independent_spatial_reduction():
 info=json.loads((R/'qa/coverage.json').read_text())['grid'];geo=gpd.read_parquet(S/'pilot_geometry.parquet').set_index('GEOID');pairs=pd.read_parquet(S/'pilot_pairs.parquet')
 metrics=pd.read_parquet(R/'primary_pilot.parquet').set_index(KEY);checks=[]
 for k,df in pairs.groupby('GEOID'):
  if len(df)>500 or not (df.native_idx>=0).any():continue
  region=df.source_region.iloc[0];lon,lat=grid_lonlat(df.cell.to_numpy(),info);geom=geo.loc[k,'geometry']
  geos=np.array([geom.intersection(box(x-STEP/2,y-STEP/2,x+STEP/2,y+STEP/2)).area/(STEP*STEP) for x,y in zip(lon,lat)])
  inside=df.native_idx.to_numpy()>=0;cells=np.load(S/f'cells_{region}.npy');pos=np.searchsorted(cells,df.native_idx.to_numpy()[inside]);weights=geos[inside]
  for year,season in [(2003,'DJF'),(2024,'JJA')]:
   root=S/'computed/run1'/region/season/POLICIES[0];c=np.load(root/f'counts_{year}.npy',mmap_mode='r')[:,pos];support=np.load(root/f'support_{year}.npy',mmap_mode='r')[:,pos]
   for b,m in enumerate(MET):
    good=support[b];den=math.fsum(float(w) for w in weights[good]);expected=math.fsum(float(v)*float(w) for v,w in zip(c[b,good],weights[good]))/den if den else np.nan;actual=metrics.loc[(k,year,season),m]
    if den:assert abs(expected-actual)<1e-5,(k,year,season,m,expected,actual)
    else:assert pd.isna(actual)
    checks.append({'GEOID':k,'year':year,'season':season,'metric':m,'absolute_difference':float(abs(expected-actual)) if den else None})
 # Independent cutoff probes are wholly inside one cell and straddle the cutoff.
 center=canonical_cell(np.array([-75.]),np.array([40.]),info);x,y=grid_lonlat(center,info);left=x[0]-STEP/2;bottom=y[0]-STEP/2
 probes=[]
 for fraction in [CUTOFF-1e-6,CUTOFF+1e-6]:
  polygon=box(left,bottom,left+STEP*fraction,bottom+STEP)
  result=exact_extract(str(S/'global_grid_template.tif'),gpd.GeoDataFrame({'id':[1]},geometry=[polygon],crs='EPSG:4326'),['cell_id','coverage'],output='pandas').iloc[0]
  coverage=np.asarray(result.coverage,dtype=np.float32);cells_=np.asarray(result.cell_id);desired=cells_==center[0];assert desired.sum()==1
  measured=float(coverage[desired][0]);assert (measured>=CUTOFF)==(fraction>=CUTOFF);probes.append({'fraction':fraction,'measured':measured,'included':measured>=CUTOFF})
 js(R/'qa/independent_spatial_reduction.json',{'passed':True,'weighted_real_polygon_checks':checks,'coverage_cutoff_probes':probes,'absolute_day_tolerance':1e-5,'geometry_engine':'Independent GEOS intersections and math.fsum, compared with float32 cached coverage weighted production results.'})

def header_grid_sensitivity():
 differences=pd.read_csv(R/'qa/header_grid_mismatches_NorthAmerica_2022.csv',dtype={'GEOID':str});pairs=pd.read_parquet(S/'pilot_pairs.parquet');cells=np.load(S/'cells_NorthAmerica.npy');primary=pd.read_parquet(R/'primary_diagnostics.parquet');filtered=pd.read_parquet(R/'range_sensitivity.parquet');rows=[]
 for policy in POLICIES:
  path=S/'computed/run1/NorthAmerica/JJA'/policy;counts=np.load(path/'counts_2022.npy',mmap_mode='r');masks=np.load(path/'support_2022.npy',mmap_mode='r');reference=primary if policy==POLICIES[0] else filtered
  for k in differences.GEOID.unique():
   p=pairs[pairs.GEOID==k].copy();replacement=differences[differences.GEOID==k].set_index('cell').actual_native_idx
   for cell,index in replacement.items():p.loc[p.cell==cell,'native_idx']=index
   p=p[p.native_idx>=0];pos=np.searchsorted(cells,p.native_idx.to_numpy());assert np.array_equal(cells[pos],p.native_idx);weights=p.coverage.to_numpy('float64');record={'GEOID':k,'observation_year':2022,'season':'JJA','method_id':'2022_actual_header_sampling_'+policy,'changed_geographic_pairs':len(replacement)}
   base=reference[(reference.GEOID==k)&(reference.observation_year==2022)&(reference.season=='JJA')].iloc[0]
   for b,metric in enumerate(MET):
    valid=masks[b,pos];den=weights[valid].sum();value=float(np.dot(counts[b,pos[valid]],weights[valid])/den) if den else np.nan;record[metric]=value;record[metric+'_fixed_count_grid']=base[metric];record[metric+'_actual_minus_fixed']=value-base[metric] if pd.notna(base[metric]) else np.nan;record[metric+'_support_weight']=float(den)
   rows.append(record)
 df=pd.DataFrame(rows);df.to_parquet(R/'header_grid_sensitivity.parquet',index=False);df.to_csv(R/'header_grid_sensitivity.csv.gz',index=False,compression='gzip');js(R/'qa/header_grid_sensitivity.json',{'passed':True,'rows':len(df),'affected_geographic_pairs':len(differences),'native_cell_identity_unchanged':True,'fixed_count_grid':'Exact 2003 baseline native reference','alternative':'Nearest-neighbor sample same native count/support arrays with original 2022 TIFF transform','max_abs_actual_minus_fixed_days':{m:float(df[m+'_actual_minus_fixed'].abs().max()) if df[m+'_actual_minus_fixed'].notna().any() else None for m in MET}})

def compute():
 hand_tests();tasks=[(r,s,p,run) for run in [1,2] for r in ['NorthAmerica','SouthAmerica'] for s in ['DJF','JJA'] for p in POLICIES]
 results=[]
 for run in [1,2]:
  with ProcessPoolExecutor(int(os.environ.get('PILOT_WORKERS','4'))) as ex:
   for f in as_completed([ex.submit(compute_task,x) for x in tasks if x[-1]==run]):f.result()
  results.append(reduce_run(run))
 assert results[0]['semantic_digests']==results[1]['semantic_digests']
 js(R/'qa/reproducibility.json',{'passed':True,'identical_inputs_code_config':True,'semantic_digests':results[0]['semantic_digests'],'comparison':'Canonical sorted values, nulls and support; Parquet bytes not compared.'})
 # Threshold/support changes independently recomputed in filtered branch.
 rows=[]
 for region in ['NorthAmerica','SouthAmerica']:
  for s in ['DJF','JJA']:
   out=S/'computed/run1'/region/s;qp=np.load(out/POLICIES[0]/'threshold_qnum.npy');qf=np.load(out/POLICIES[1]/'threshold_qnum.npy');np_=np.load(out/POLICIES[0]/'baseline_n.npy');nf=np.load(out/POLICIES[1]/'baseline_n.npy')
   for v in [0,1]:
    both=(np_[v]>0)&(nf[v]>0);different=both&(qp[v]!=qf[v]);delta=(qp[v,both].astype(float)-qf[v,both])/1000
    rows.append({'region':region,'season':s,'variable':['TMAX','TMIN'][v],'cells':len(both),'threshold_changed':int(different.sum()),'baseline_support_changed':int((np_[v]!=nf[v]).sum()),'primary_only_threshold_cells':int(((np_[v]>0)&(nf[v]==0)).sum()),'max_abs_threshold_difference_c':float(np.max(np.abs(delta))) if len(delta) else None})
 pd.DataFrame(rows).to_csv(R/'qa/threshold_sensitivity.csv',index=False)
 primary=pd.concat([pd.read_parquet(R/'primary_pilot.parquet'),pd.read_parquet(R/'primary_diagnostics.parquet')]);filtered=pd.read_parquet(R/'range_sensitivity.parquet')
 dif=primary[KEY+MET].merge(filtered[KEY+MET],on=KEY,validate='one_to_one',suffixes=('_primary','_filtered'))
 for m in MET:dif[m+'_primary_minus_filtered']=dif[m+'_primary']-dif[m+'_filtered']
 support=pd.read_parquet(R/'support_qa.parquet');sp=support[(support.index_policy=='legacy_compatibility')&support.method_id.str.endswith(POLICIES[0])];sf=support[(support.index_policy=='legacy_compatibility')&support.method_id.str.endswith(POLICIES[1])]
 for m in MET:
  col=m+'_support_weight';lookup=sf.set_index(KEY)[col];dif[col+'_primary_minus_filtered']=sp.set_index(KEY)[col].subtract(lookup).reindex(pd.MultiIndex.from_frame(dif[KEY])).to_numpy()
 dif.to_parquet(R/'sensitivity_comparison.parquet',index=False);dif.to_csv(R/'sensitivity_comparison.csv.gz',index=False,compression='gzip')
 cachemanifests=[]
 for p in (S/'computed/run1').rglob('cache_manifest.json'):cachemanifests.append({'directory':str(p.parent),**json.loads(p.read_text())})
 js(R/'qa/threshold_count_cache_manifests.json',cachemanifests)
 independent_spatial_reduction(); header_grid_sensitivity(); log('computational pilot passed')

# Stream candidate and demonstration; original table order and Arrow values retained.
def merge():
 start=time.time();before={str(p):sha(p) for p in [OLD,ADD]};assert before[str(OLD)]==json.loads((D/'target_integrity.json').read_text())['sha256_newly_verified']
 old=pq.ParquetFile(OLD);schema=old.schema_arrow;cols=schema.names;assert len(cols)==19
 addon=pd.read_parquet(ADD);attrs=pd.read_parquet(S/'territory_attributes.parquet');assert attrs.GEOID.is_unique
 addon=addon.merge(attrs,on='GEOID',how='left',validate='many_to_one');addon=addon[cols]
 addtable=pa.Table.from_pandas(addon,schema=schema,preserve_index=False);assert len(addtable)==49016
 new=pd.read_parquet(R/'primary_pilot.parquet');ndiag=pd.read_parquet(R/'primary_diagnostics.parquet')
 assert not new.duplicated(KEY).any() and not ndiag.duplicated(KEY).any()
 newcols=MET+['boundary_year','method_id','source_region','support_status','index_policy','registry_status']
 assert not(set(newcols)&set(cols))
 maps={tuple(v):{c:row[c] for c in newcols} for v,row in zip(new[KEY].itertuples(index=False,name=None),new.to_dict('records'))}
 diagmap=set(ndiag[KEY].itertuples(index=False,name=None));matched=set();dmatched=set();uniqueids=set();counts={};pilotrows=[];nrows=0;old_digest=hashlib.sha256();merged_orig_digest=hashlib.sha256()
 candidate=S/'trial_candidate_temperature.parquet';demonstration=S/'trial_merge_demonstration.parquet'
 mergeschema=pa.schema(list(schema)+[pa.field(m,pa.float64()) for m in MET]+[pa.field('boundary_year',pa.int64()),*[pa.field(c,pa.large_string()) for c in ['method_id','source_region','support_status','index_policy','registry_status']]])
 with pq.ParquetWriter(candidate,schema,compression='zstd') as cw,pq.ParquetWriter(demonstration,mergeschema,compression='zstd') as mw:
  def consume(batch):
   nonlocal nrows
   df=batch.select(KEY).to_pandas(); keys=list(df.itertuples(index=False,name=None)); assert len(keys)==len(set(keys))
   for k,y,s in keys:
    uniqueids.add(k);bit=(int(y)-2003)*2+(0 if s=='DJF' else 1);assert 0<=bit<44 and s in ['DJF','JJA'];mask=1<<bit;assert not counts.get(k,0)&mask;counts[k]=counts.get(k,0)|mask
   pp=batch.filter(pa.compute.is_in(batch['GEOID'],value_set=pa.array(ids())));pilotrows.extend(pp.to_pylist())
   cw.write_table(batch); additions=[]
   for col in newcols:
    additions.append(pa.array([maps[k][col] if k in maps else None for k in keys],type=mergeschema.field(col).type,from_pandas=True))
   merged=pa.Table.from_arrays(list(batch.columns)+additions,schema=mergeschema);assert merged.select(cols).equals(batch,check_metadata=False)
   # Arrow IPC semantic bytes on the same streamed batches establishes preserved order/null/value identity.
   sink=pa.BufferOutputStream()
   with pa.ipc.new_stream(sink,batch.schema) as writer:writer.write_table(batch)
   old_digest.update(sink.getvalue().to_pybytes())
   sink=pa.BufferOutputStream()
   with pa.ipc.new_stream(sink,batch.schema) as writer:writer.write_table(merged.select(cols))
   merged_orig_digest.update(sink.getvalue().to_pybytes())
   mw.write_table(merged);matched.update(k for k in keys if k in maps);dmatched.update(k for k in keys if k in diagmap);nrows+=len(batch)
  oldids=set()
  for rb in old.iter_batches(batch_size=65536):
   batch=pa.Table.from_batches([rb]);oldids.update(batch['GEOID'].to_pylist());consume(batch)
  assert not oldids&set(addon.GEOID)
  for batch in addtable.to_batches(max_chunksize=65536):consume(pa.Table.from_batches([batch]))
 assert nrows==3763232 and len(uniqueids)==85528 and all(v==(1<<44)-1 for v in counts.values())
 assert matched==set(maps) and len(matched)==256 and dmatched==diagmap and len(dmatched)==96
 assert old_digest.hexdigest()==merged_orig_digest.hexdigest()
 persisted_rows=0;persisted_matches=0
 candidate_reader=pq.ParquetFile(candidate).iter_batches(batch_size=65536)
 demonstration_reader=pq.ParquetFile(demonstration).iter_batches(batch_size=65536)
 for cb,mb in zip(candidate_reader,demonstration_reader,strict=True):
  ct=pa.Table.from_batches([cb]);mt=pa.Table.from_batches([mb]);assert ct.equals(mt.select(cols),check_metadata=False)
  method=mt['method_id'];present=pa.compute.is_valid(method).to_numpy();keys_=list(mt.select(KEY).to_pandas().itertuples(index=False,name=None));expected=np.array([key in maps for key in keys_]);assert np.array_equal(present,expected)
  absent=pa.array(~expected)
  for field in newcols:assert mt[field].filter(absent).null_count==int((~expected).sum())
  persisted_rows+=len(ct);persisted_matches+=int(present.sum())
 assert persisted_rows==nrows and persisted_matches==256
 reference=pd.DataFrame(json.loads((REF/'current_allUS_pilot_temperature_reference.json').read_text())).sort_values(KEY).reset_index(drop=True)
 actual=pd.DataFrame(pilotrows).sort_values(KEY).reset_index(drop=True);assert len(actual)==1408
 failures=[]
 for c in cols:
  x=actual[c];y=reference[c];equal=(x.isna()&y.isna())|(x==y)
  if not equal.all():failures.extend(actual.loc[~equal,KEY].assign(column=c).to_dict('records'))
 js(R/'qa/pilot_record_comparison.json',{'passed':not failures,'columns':cols,'rows':1408,'exact_value_and_null_comparison':True,'failures':failures})
 assert not failures, f'Pilot reference mismatches {failures[:10]}'
 # Read persisted demonstrations for all expected keys; all nonpilot rows have null appended fields.
 projected=pq.read_table(demonstration,filters=[('GEOID','in',ids())]);demodf=projected.to_pandas()
 pilotdemo=demodf.merge(new[KEY],on=KEY,how='inner',validate='one_to_one');assert len(pilotdemo)==256
 for m in MET:
  a=pilotdemo.sort_values(KEY)[m].reset_index(drop=True);b=new.sort_values(KEY)[m].reset_index(drop=True);assert ((a.isna()&b.isna())|(a==b)).all()
 # Diagnostic separate demonstration, limited to 96 selected candidate keys.
 diagoriginal=demodf[cols].merge(ndiag[KEY],on=KEY,how='inner',validate='one_to_one');diagdemo=diagoriginal.merge(ndiag[KEY+newcols],on=KEY,how='left',validate='one_to_one',sort=False);assert len(diagdemo)==96
 diagdemo.to_parquet(R/'trial_merge_diagnostics.parquet',index=False);pilotdemo.to_parquet(R/'trial_merge_pilot_extract.parquet',index=False)
 assert all(sha(p)==h for p,h in before.items())
 result={'passed':True,'candidate_path':str(candidate),'demonstration_path':str(demonstration),'candidate_sha256':sha(candidate),'demonstration_sha256':sha(demonstration),'rows':nrows,'unique_geoids':len(uniqueids),'periods_per_id':44,'duplicate_keys':0,'disjoint_base_territory_ids':True,'mandatory_expected_keys':256,'mandatory_matched_keys':len(matched),'missing_expected_keys':0,'nonpilot_unmatched_keys':nrows-256,'diagnostic_matched_keys':len(dmatched),'original_order_values_nulls_preserved':True,'persisted_allrow_original_column_comparison':True,'persisted_rows_checked':persisted_rows,'persisted_pilot_matches':persisted_matches,'original_semantic_digest':old_digest.hexdigest(),'merged_original_semantic_digest':merged_orig_digest.hexdigest(),'source_tables_unchanged_sha256':before,'qualified_target':'Available states/DC + official-attribute territory add-on; registry IDs/original frozen source geometry verified; base current-release coordinate hashes unresolved; all 1408 pilot rows exactly match all 19 current-release reference columns. Full-table current-release numerical/byte equivalence not established.','addon_attribute_mapping':{'StFIPS':'STATEFP','CoFIPS':'COUNTYFP','TrFIPS':'TRACTCE','Area_Land':'ALAND as float64','Area_Water':'AWATER as float64','LAT_GeoCen':'absent: null preserved','LON_GeoCen':'absent: null preserved','Pop':'absent: int64 null preserved','temperature_fields':'copied unchanged by same name','keys':'copied unchanged'},'elapsed_seconds':time.time()-start}
 js(R/'qa/trial_merge.json',result);log('trial merge passed')

def main():
 phase=sys.argv[1];start=time.time();log('start',phase,os.environ.get('SLURM_JOB_ID'))
 result={'prepare':prepare,'stage':stage,'compute':compute,'merge':merge}[phase]()
 phase_file=f'phase_{phase}_reuse_{os.environ.get("SLURM_JOB_ID")}.json' if isinstance(result,dict) and result.get('checkpoint_reused') else f'phase_{phase}.json'
 js(R/'qa'/phase_file,{'passed':True,'elapsed_seconds':time.time()-start,'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'max_child_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'slurm_job_id':os.environ.get('SLURM_JOB_ID')})
if __name__=='__main__':
 try:main()
 except Exception:
  traceback.print_exc();sys.exit(1)
