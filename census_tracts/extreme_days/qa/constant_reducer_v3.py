"""Keep native/storage signatures; version geographic output reduction separately.
A weighted mean of equal contributing counts equals that count exactly. No clipping,
threshold rounding, mask change, data repair or scientific definition change.
"""
import importlib.util,sys,json,shutil,math,time
from pathlib import Path
R=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('storage_adapter',R/'qa/storage_compatibility_v2.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
C=A.C;Q=A.Q;np=C.np
OLD=R.parent/'storage_v2'
_original_grouped=Q.grouped;_original_js=C.js;_original_table=C.atomic_parquet

def reducer_signature():return C.sha(Path(__file__))
def js(path,obj):
 if 'shards/production' in str(path) and str(path).endswith('.complete.json'):obj={**obj,'reduction_signature_sha256':reducer_signature(),'constant_band_means_exact':True}
 _original_js(path,obj)
C.js=js;Q.js=js

def table(df,path,metadata=None):
 md={**(metadata or {}),'reduction_signature_sha256':reducer_signature(),'reduction_arithmetic':'Coverage-weighted Float64 means; exactly constant contributing bands return their integer value without clipping.'}
 _original_table(df,path,md)
C.atomic_parquet=table;Q.atomic_parquet=table

def grouped(c,m,bn,pairs,nt):
 result=list(_original_grouped(c,m,bn,pairs,nt));v,den,bounds=result[:3]
 low=np.full((6,nt),32767,np.int16);high=np.full((6,nt),-1,np.int16)
 for lo in range(0,len(pairs['tract']),1000000):
  sl=slice(lo,lo+1000000);t=pairs['tract'][sl].astype(np.int64);pos=pairs['pos'][sl];cc=c[:,pos];mm=m[:,pos]
  assert np.all(pairs['coverage'][sl]>0)
  if not len(t):continue
  starts=np.flatnonzero(np.r_[True,t[1:]!=t[:-1]]);groups=t[starts]
  for b in range(6):
   source=b if b<4 else b-2;mask=mm[b] if b<4 else mm[b-4]
   lows=np.minimum.reduceat(np.where(mask,cc[source],32767),starts);highs=np.maximum.reduceat(np.where(mask,cc[source],-1),starts)
   np.minimum.at(low[b],groups,lows);np.maximum.at(high[b],groups,highs)
 for b in range(6):
  exact=(high[b]>=0)&(low[b]==high[b]);target=v[b] if b<4 else bounds[b-4]
  target[exact]=low[b,exact].astype(np.float64)
 return tuple(result)
Q.grouped=grouped

_oldfreeze=Q.freeze
def freeze():
 sig=_oldfreeze()
 manifest={'native_signature_sha256':sig,'reducer_code_sha256':reducer_signature(),'algorithm':'Exact value for a band whose positive-weight eligible native counts are all equal; ordinary reviewed Float64 weighted mean otherwise','thresholds_counts_masks_unchanged':True,'no_clipping_or_gate_relaxation':True,'prior_bound_diagnosis':C.sha(R/'qa/bounds_failure_diagnosis.json')}
 path=R/'qa/reduction_signature.json'
 if path.exists():assert json.loads(path.read_text())==manifest
 else:_original_js(path,manifest)
 for path in (R/'shards/production').glob('*/*.complete.json'):assert json.loads(path.read_text())['reduction_signature_sha256']==reducer_signature()
 return sig
Q.freeze=freeze

def gate():
 freeze();native=[]
 for old in sorted((OLD/'native').glob('*/*/*/*/complete.json')):
  data=json.loads(old.read_text());assert data['identity']['run']==C.sha(R/'qa/run_signature.json')
  target=R/'native'/old.parent.relative_to(OLD/'native');target.mkdir(parents=True,exist_ok=True)
  for name,h in data['files'].items():
   assert C.sha(old.parent/name)==h
   if not (target/name).exists():shutil.copyfile(old.parent/name,target/name)
   assert C.sha(target/name)==h;(target/name).chmod(0o444)
  if not (target/'complete.json').exists():shutil.copyfile(old,target/'complete.json')
  assert C.sha(target/'complete.json')==C.sha(old)
  native.append({'source':str(old.parent),'destination':str(target),'marker_sha256':C.sha(old),'files':data['files']})
 assert len(native)==16
 reg=C.registry();comparisons=[]
 for region in Q.REGIONS:
  pairs={k:v for k,v in np.load(C.S/f'pairs_{region}.npz').items()}
  for season in Q.SEASONS:
   grouped.baseline_days=460 if season=='JJA' else 450;limit=92 if season=='JJA' else 90
   for policy in Q.POLICIES:
    root=R/'native'/region/season/policy;counts=np.load(root/'2003/counts.npy',mmap_mode='r');masks=np.load(root/'2003/support.npy',mmap_mode='r');bn=np.load(root/'baseline/baseline_n.npy',mmap_mode='r');old=_original_grouped(counts,masks,bn,pairs,len(reg));new=grouped(counts,masks,bn,pairs,len(reg));delta=np.abs(old[0]-new[0]);maximum=float(np.nanmax(delta)) if np.isfinite(delta).any() else 0;assert maximum<=1e-10
    assert np.array_equal(np.isnan(old[0]),np.isnan(new[0]));assert np.all((np.isnan(new[0]))|((new[0]>=0)&(new[0]<=limit)))
    for i in range(1,len(new)):
     if i==2:continue
     assert np.array_equal(old[i],new[i],equal_nan=True)
    # Direct independent scalar calculation on each previously offending tract.
    failures=C.pd.read_csv(R/'qa/bounds_failure_rows.csv.gz',dtype={'GEOID':str});failures=failures[(failures.region==region)&(failures.season==season)&(failures.policy==policy)];kmap={k:i for i,k in enumerate(reg.GEOID)}
    for rec in failures.itertuples():
     ti=kmap[rec.GEOID];b=C.MET.index(rec.metric);keep=pairs['tract']==ti;pos=pairs['pos'][keep];good=masks[b,pos];values=counts[b,pos[good]];assert len(values) and np.all(values==values[0]);assert new[0][b,ti]==int(values[0])
    comparisons.append({'region':region,'season':season,'policy':policy,'max_change_days':maximum,'strict_day_bounds_pass':True,'null_masks_and_all_support_fields_unchanged':True,'offending_tracts_checked_independently':len(failures)})
 _original_js(R/'qa/constant_reducer_gate.json',{'passed':True,'native_arrays_content_verified_and_unchanged':native,'full_2003_registry_comparisons':comparisons,'absolute_tolerance_days':1e-10,'relative_tolerance':0,'reducer_code_sha256':reducer_signature()});C.log('CONSTANT REDUCER GATE PASSED')

def production():
 freeze();assert json.loads((R/'qa/constant_reducer_gate.json').read_text())['passed'];Q.production()
def finalize():
 freeze();import finalize as F
 for name,fn in [('native_summary',F.native_summary),('assemble',F.assemble),('pilot_extra',F.pilot_and_extra_checks),('merge',F.merge),('package',F.package)]:C.measure(name,fn)

if __name__=='__main__':C.measure(sys.argv[1],{'gate':gate,'production':production,'finalize':finalize}[sys.argv[1]])
