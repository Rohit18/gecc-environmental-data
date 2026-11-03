"""Versioned IO-only recovery: CFS arrays use ordinary reads/writes, scratch mmap.
Original scientific kernels remain unchanged; all new caches have a new namespace.
"""
import sys,os,json,hashlib,shutil,time
from pathlib import Path
R=Path(__file__).resolve().parent.parent
S=Path('/path/to/scratch/extreme_temperature_tract2020/stage3_work/storage_v2')
OLD=R.parents[1];OSCRATCH=S.parent
sys.path.insert(0,str(R/'code'))
import production_common as C
C.R=R;C.S=S;C.P.R=R;C.P.S=S;C.PILOT=OLD.parent/'stage2_pilot';C.PS=OSCRATCH.parent/'stage2_work'
np=C.np
_orig_open=np.lib.format.open_memmap;_orig_load=np.load
class DurableHeapArray(np.ndarray):
 def __new__(cls,path,dtype,shape,fortran=False):
  obj=np.empty(shape,dtype=dtype,order='F' if fortran else 'C').view(cls);obj._durable_path=Path(path);return obj
 def __array_finalize__(self,obj):self._durable_path=getattr(obj,'_durable_path',None)
 def flush(self):
  with self._durable_path.open('wb') as f:np.save(f,np.asarray(self),allow_pickle=False);f.flush();os.fsync(f.fileno())
def safe_open(filename,mode='r+',dtype=None,shape=None,fortran_order=False,version=None,max_header_size=10000):
 if str(filename).startswith('/global/cfs/'):
  assert mode=='w+' and dtype is not None and shape is not None
  return DurableHeapArray(filename,dtype,shape,fortran_order)
 return _orig_open(filename,mode=mode,dtype=dtype,shape=shape,fortran_order=fortran_order,version=version,max_header_size=max_header_size)
def safe_load(file,*args,**kwargs):
 if str(file).startswith('/global/cfs/'):kwargs.pop('mmap_mode',None)
 return _orig_load(file,*args,**kwargs)
np.lib.format.open_memmap=safe_open;np.load=safe_load
import production as Q
Q.R=R;Q.S=S
# Default raw loader root was imported after common state was set above.

def freeze():
 sig=C.signature();sig['storage_adapter_sha256']=C.sha(Path(__file__));sig['storage_backend']='ordinary CFS NumPy array IO; mmap only on scratch';sig['prior_signature_sha256']=C.sha(R/'qa/prior_run_signature.json')
 p=R/'qa/run_signature.json'
 if p.exists():assert json.loads(p.read_text())==sig,'Storage-v2 implementation signature mismatch'
 else:C.js(p,sig)
 for path,h in sig['indices'].items():assert C.sha(path)==h
 return C.sha(p)
Q.freeze=freeze

def migrate():
 freeze();prior=json.loads((R/'qa/prior_run_signature.json').read_text());current=json.loads((R/'qa/run_signature.json').read_text());assert prior['code']==current['code'] and prior['config']==current['config'] and prior['environment']==current['environment'] and prior['indices']==current['indices']
 # Independent round-trip of exact dtypes, negative/extreme values, zeros and Boolean masks.
 checks=[]
 for dtype in [np.int16,np.int32,np.bool_]:
  values=np.array([[-32768,-32767,0,32767],[1,2,3,4]],dtype=np.int32)
  if dtype==np.bool_:values=values!=0
  else:values=values.astype(dtype)
  p=R/'qa'/f'ordinary_io_test_{np.dtype(dtype).name}.npy';a=safe_open(p,mode='w+',dtype=dtype,shape=values.shape);a[:]=values;a.flush();result=safe_load(p,mmap_mode='r');assert result.dtype==values.dtype and np.array_equal(result,values);checks.append({'dtype':str(np.dtype(dtype)),'shape':list(values.shape),'array_sha256':hashlib.sha256(result.tobytes()).hexdigest(),'passed':True})
 inventory=C.pd.read_csv(R/'qa/selected_members.csv.gz');newhash=C.sha(R/'qa/run_signature.json');oldhash=C.sha(R/'qa/prior_run_signature.json');reused=[]
 for marker in sorted((OSCRATCH/'raw').glob('*/*.complete.json')):
  cached=json.loads(marker.read_text());p=marker.with_name(marker.name.replace('.complete.json','.npy'));region=marker.parent.name;year,season,var=p.stem.split('_');year=int(year);cells=np.load(S/f'cells_{region}.npy');rows=inventory[(inventory.region==region)&(inventory.year==year)&(inventory.season==season)&(inventory.variable==var)].sort_values('date').to_dict('records');arc=json.loads((R/'qa/archive_hashes'/f'{region}_{year}.json').read_text());st=Path(arc['path']).stat();assert st.st_size==arc['bytes'] and st.st_mtime_ns==arc['mtime_ns'] and arc['md5']==arc['publisher_md5']
  oldsig=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()+cells.tobytes()+oldhash.encode()+arc['sha256'].encode()).hexdigest();assert cached['signature']==oldsig and cached['run_signature']==oldhash;assert p.exists() and C.sha(p)==cached['sha256'];a=np.load(p,mmap_mode='r');assert a.shape==(len(rows),len(cells)) and a.dtype==np.int16
  target=S/'raw'/region/p.name;target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():os.link(p,target)
  newsig=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()+cells.tobytes()+newhash.encode()+arc['sha256'].encode()).hexdigest();newmarker=target.with_name(marker.name);data={**cached,'signature':newsig,'run_signature':newhash,'verified_reuse_from':str(p),'prior_marker_sha256':C.sha(marker),'storage_change_only':True}
  if newmarker.exists():assert json.loads(newmarker.read_text())==data
  else:C.js(newmarker,data)
  reused.append({'path':str(target),'source':str(p),'sha256':cached['sha256'],'bytes':p.stat().st_size,'full_content_verified':True})
 assert len(reused)==48
 C.js(R/'qa/storage_compatibility_gate.json',{'passed':True,'independent_ordinary_IO_roundtrips':checks,'scientific_kernels_unchanged':True,'inputs_config_environment_cell_order_unchanged':True,'reuse_files':reused,'code_adapter_sha256':C.sha(Path(__file__)),'new_signature':newhash,'preserved_old_signature':oldhash})
 C.log('STORAGE V2 GATE PASSED',len(reused),'full content-verified arrays')

def production():
 assert json.loads((R/'qa/storage_compatibility_gate.json').read_text())['passed'];Q.production()

def finalize():
 freeze();import finalize as F
 for name,fn in [('native_summary',F.native_summary),('assemble',F.assemble),('pilot_extra',F.pilot_and_extra_checks),('merge',F.merge),('package',F.package)]:C.measure(name,fn)

if __name__=='__main__':C.measure(sys.argv[1],{'migrate':migrate,'production':production,'finalize':finalize}[sys.argv[1]])
