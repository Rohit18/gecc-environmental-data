"""Frozen production methods; imports an unchanged, hash-pinned accepted pilot."""
import os
os.environ.setdefault('PROJ_DATA','/path/to/home/.conda/envs/gecc-lai/share/proj')
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,importlib.util,json,hashlib,time,datetime,resource,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('accepted_pilot',HERE/'accepted_pilot.py');P=importlib.util.module_from_spec(spec);sys.modules[spec.name]=P;spec.loader.exec_module(P)
np=P.np;pd=P.pd;pa=P.pa;pq=P.pq
R=HERE.parent;S=Path('/path/to/scratch/extreme_temperature_tract2020/stage3_work');PILOT=R.parent/'stage2_pilot';PS=S.parent/'stage2_work';D=P.D;A=P.A;B=P.B
KEY=P.KEY;MET=P.MET;REGIONS=['NorthAmerica','SouthAmerica'];POLICIES=P.POLICIES;YEARS=list(range(2003,2025));SEASONS=['DJF','JJA'];PERIODS=[(y,s) for y in YEARS for s in SEASONS]
METHOD='svcmsp_native_hf7_fixed2003grid_wrap_v1'
P.R=R;P.S=S;P.PERIODS=PERIODS
sha=P.sha;log=P.log

def js(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.partial');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n');os.replace(tmp,p)
P.js=js

def config():return json.loads((R/'config.json').read_text())
def registry():return pd.read_csv(R/'registry.csv',dtype={'GEOID':str})
def environment():
 import importlib.metadata as im
 return {'python':sys.version,'packages':{k:im.version(k) for k in ['numpy','pandas','geopandas','shapely','rasterio','pyproj','pyarrow','exactextract']},'GDAL':P.rasterio.__gdal_version__,'GEOS':P.shapely.geos_version_string,'PROJ':P.pyproj.proj_version_str,'PROJ_DATA':P.pyproj.datadir.get_data_dir()}
def signature():return {'config':sha(R/'config.json'),'environment':environment(),'code':{p.name:sha(p) for p in sorted(HERE.glob('*.py'))},'indices':json.loads((R/'qa/prepare.json').read_text())['generated_hashes']}
def measure(name,fn,*args):
 t=time.time();log('start',name);result=fn(*args);js(R/'qa'/f'phase_{name}_{os.environ.get("SLURM_JOB_ID","local")}.json',{'passed':True,'seconds':time.time()-t,'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'max_child_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'job_id':os.environ.get('SLURM_JOB_ID')});log('passed',name);return result

def atomic_parquet(df,path,metadata=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.partial');table=pa.Table.from_pandas(df,preserve_index=False)
 md=dict(table.schema.metadata or {});md[b'extreme_method_config']=(R/'config.json').read_bytes();md[b'input_manifest_sha256']=sha(R/'qa/selected_members.csv.gz').encode()
 if metadata:md.update({str(k).encode():json.dumps(v).encode() for k,v in metadata.items()})
 pq.write_table(table.replace_schema_metadata(md),tmp,compression='zstd');assert pq.ParquetFile(tmp).metadata.num_rows==len(df);os.replace(tmp,path)
