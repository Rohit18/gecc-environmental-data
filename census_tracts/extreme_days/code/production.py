#!/usr/bin/env python3
"""Stage 3 bounded staging, exact native kernels, grouped geographic reduction.
Each completion marker pins inputs, cell identity, code, config and environment.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from production_common import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import math,traceback

def freeze():
 sig=signature();p=R/'qa/run_signature.json'
 if p.exists():assert json.loads(p.read_text())==sig,'Implementation changed; preserve old caches and use a new run namespace.'
 else:js(p,sig)
 for p,h in sig['indices'].items():assert sha(p)==h
 return sha(R/'qa/run_signature.json')

def expected_members():
 expected={}
 for p in [B/'outputs_full_series/source_input_manifest.csv',A/'provenance/svcmsp_source_input_manifest.csv']:
  d=pd.read_csv(p,usecols=['logical_path','sha256']);assert not d.logical_path.duplicated().any()
  for k,h in zip(d.logical_path,d.sha256):
   if k in expected:assert expected[k]==h
   expected[k]=h
 return expected

def archive_hash(region,year):
 row=next(r for r in json.loads((R/'qa/archive_expected.json').read_text()) if r['region']==region and r['year']==year);p=Path(row['path']);st=p.stat();assert st.st_size==row['bytes'] and st.st_mtime_ns==row['mtime_ns']
 target=R/'qa/archive_hashes'/f'{region}_{year}.json'
 if target.exists():
  old=json.loads(target.read_text());assert old['bytes']==st.st_size and old['mtime_ns']==st.st_mtime_ns and old['publisher_md5']==row['publisher_md5_inherited'] and old['md5']==row['publisher_md5_inherited'];return old
 start=time.time();md5=hashlib.md5();h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(16<<20),b''):md5.update(chunk);h.update(chunk)
 assert md5.hexdigest()==row['publisher_md5_inherited'],f'Publisher archive MD5 mismatch: {p}'
 result={'region':region,'year':year,'path':str(p),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'md5':md5.hexdigest(),'sha256':h.hexdigest(),'publisher_md5':row['publisher_md5_inherited'],'publisher_doi':row['catalog_doi'],'fresh_full_hash':True,'seconds':time.time()-start,'job_id':os.environ.get('SLURM_JOB_ID')};js(target,result);log('archive hash passed',region,year,round(result['seconds'],1));return result

def header_proof(actual,reference,cells,region,year):
 a=np.asarray(actual,float);ref=np.asarray(reference['transform'],float);assert a[1]==a[3]==0 and a[0]>0 and a[4]<0
 corners=np.array([[0,0],[reference['width'],0],[0,reference['height']],[reference['width'],reference['height']]])
 dx=a[2]-ref[2]+corners[:,0]*(a[0]-ref[0]);dy=a[5]-ref[5]+corners[:,1]*(a[4]-ref[4]);drift=float(np.hypot(dx,dy).max());assert drift<=.002,(region,year,drift)
 for lo in range(0,len(cells),1000000):
  rr,cc=np.divmod(cells[lo:lo+1000000],reference['width']);xx=ref[2]+(cc+.5)*ref[0];yy=ref[5]+(rr+.5)*ref[4];rnew=np.floor((yy-a[5])/a[4]).astype(np.int64);cnew=np.floor((xx-a[2])/a[0]).astype(np.int64);assert np.array_equal(rr,rnew) and np.array_equal(cc,cnew),'Full reached-cell center identity failure'
 pairs={k:v for k,v in np.load(S/f'pairs_{region}.npz').items()};grid=json.loads((R/'qa/prepare.json').read_text())['grid'];newref={**reference,'transform':a.tolist()};count=0;changed=[]
 for lo in range(0,len(pairs['cell']),1000000):
  cell=pairs['cell'][lo:lo+1000000];lon,lat=P.grid_lonlat(cell,grid);new=P.map_native(lon,lat,newref);old=cells[pairs['pos'][lo:lo+1000000]];diff=new!=old;count+=int(diff.sum())
  if diff.any():changed.append(pd.DataFrame({'tract':pairs['tract'][lo:lo+1000000][diff],'cell':cell[diff],'fixed_native_idx':old[diff],'actual_header_native_idx':new[diff]}))
 token=hashlib.sha256(a.tobytes()).hexdigest()[:12];out={'region':region,'year':year,'actual_transform':a.tolist(),'reference_transform':ref.tolist(),'max_extent_drift_m':drift,'native_centers_checked':len(cells),'native_center_mismatches':0,'reached_geographic_pairs_checked':len(pairs['cell']),'raw_header_geographic_mapping_differences':count,'count_grid':'exact 2003 reference; unchanged native cell row/column identities','coordinate_rounding':False}
 js(R/'qa/header_proofs'/f'{region}_{year}_{token}.json',out)
 if changed:pd.concat(changed).to_csv(R/'qa/header_proofs'/f'{region}_{year}_{token}_differences.csv.gz',index=False)
 return out

def stage_year(task):
 region,year=task;start=time.time();arc=archive_hash(region,year);cells=np.load(S/f'cells_{region}.npy');m=P.meta(region);nc=m['width'];nr=m['height'];rr,cc=np.divmod(cells,nc);chunks=[]
 # Only rows reached by production cells; chunking keeps scanline decompression bounded.
 for block in np.unique(rr//64):
  first=int(block*64);last=min(first+64,nr);pos=np.flatnonzero((rr>=first)&(rr<last));chunks.append((first,last,pos,rr[pos]-first,cc[pos]))
 d=pd.read_csv(R/'qa/selected_members.csv.gz');d=d[(d.region==region)&(d.year==year)];assert len(d)==364;expected=expected_members();out=S/'raw'/region;out.mkdir(parents=True,exist_ok=True);outputs={};lookup={};records=[];run=sha(R/'qa/run_signature.json')
 for (season,var),group in d.groupby(['season','variable']):
  group=group.sort_values('date');rows=group.to_dict('records');name=f'{year}_{season}_{var}';p=out/f'{name}.npy';marker=out/f'{name}.complete.json';sig=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()+cells.tobytes()+run.encode()+arc['sha256'].encode()).hexdigest()
  if marker.exists():
   cached=json.loads(marker.read_text());assert cached['signature']==sig
   if p.exists():assert sha(p)==cached['sha256'];records.extend(cached['records']);continue
   raise RuntimeError(f'Raw selection previously cleaned; explicit regeneration namespace required: {p}')
  tmp=p.with_name(p.name+f'.{os.environ.get("SLURM_JOB_ID")}.{os.getpid()}.partial');arr=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.int16,shape=(len(rows),len(cells)));outputs[name]=(arr,p,marker,sig,[],tmp)
  for j,row in enumerate(rows):lookup[row['member']]=(name,j)
 proof={}
 with open(arc['path'],'rb') as f:
  for rec in d.sort_values('offset_data').to_dict('records'):
   if rec['member'] not in lookup:continue
   t=time.time();f.seek(int(rec['offset_data']));content=f.read(int(rec['member_bytes']));assert len(content)==int(rec['member_bytes']);h=hashlib.sha256(content).hexdigest();logical=Path(arc['path']).name+'/'+rec['member'];assert logical in expected and h==expected[logical],f'Member hash mismatch/missing: {logical}';name,j=lookup[rec['member']];arr=outputs[name][0];decoded=0
   with P.MemoryFile(content) as mf,mf.open() as ds:
    assert ds.width==nc and ds.height==nr and ds.count==1 and ds.dtypes==('int16',) and ds.nodata==P.NODATA and ds.scales==(1.,) and ds.offsets==(0.,) and ds.block_shapes[0]==(1,nc)
    assert ds.crs.to_dict().get('proj')=='sinu' and abs(ds.crs.to_dict().get('R',6371007.181)-6371007.181)<1e-6
    transform=tuple(list(ds.transform)[:6])
    if transform not in proof:proof[transform]=header_proof(transform,m,cells,region,year)
    for first,last,pos,rsel,csel in chunks:
     values=ds.read(1,window=P.Window(0,first,nc,last-first));assert values.shape==(last-first,nc);arr[j,pos]=values[rsel,csel];decoded+=values.nbytes
   selected=arr[j];bad=(selected!=P.NODATA)&((selected<-950)|(selected>650));valid=selected[selected!=P.NODATA];record={'region':region,'year':year,'season':rec['season'],'variable':rec['variable'],'date':rec['date'],'logical_path':logical,'member_bytes':len(content),'full_member_sha256_new':h,'inherited_sha256':expected[logical],'inherited_match':True,'selected_values_sha256':hashlib.sha256(selected.tobytes()).hexdigest(),'selected_cells':len(cells),'decoded_full_row_bytes':decoded,'nodata_cells':int((selected==P.NODATA).sum()),'out_of_range_cells':int(bad.sum()),'raw_min':int(valid.min()) if len(valid) else None,'raw_max':int(valid.max()) if len(valid) else None,'actual_transform':list(transform),'read_started_unix':t,'read_finished_unix':time.time(),'header_verified':True};records.append(record);outputs[name][4].append(record)
 for arr,p,marker,sig,recs,tmp in outputs.values():
  arr.flush();del arr;os.replace(tmp,p);h=sha(p);js(marker,{'signature':sig,'sha256':h,'shape':[len(recs),len(cells)],'records':recs,'run_signature':run,'archive_sha256':arc['sha256']});p.chmod(0o444)
 assert len(records)==364
 manifest=R/'qa/member_checks'/f'{region}_{year}.parquet';atomic_parquet(pd.DataFrame(records),manifest)
 result={'passed':True,'region':region,'year':year,'members':len(records),'compressed_bytes':sum(x['member_bytes'] for x in records),'decoded_bytes':sum(x['decoded_full_row_bytes'] for x in records),'selected_bytes':sum(p.stat().st_size for p in out.glob(f'{year}_*.npy')),'seconds':time.time()-start,'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'member_manifest_sha256':sha(manifest)};js(R/'qa/staging'/f'{region}_{year}.json',result);log('national staged',region,year,round(result['seconds'],1));return result

def parallel(fn,tasks,workers=4):
 results=[]
 with ProcessPoolExecutor(workers) as ex:
  for future in as_completed([ex.submit(fn,x) for x in tasks]):results.append(future.result())
 return results

def raw(region,y,s,v,root=S):return np.load(root/'raw'/region/f'{y}_{s}_{v}.npy',mmap_mode='r')

def completion(out,identity):
 p=out/'complete.json'
 if not p.exists():return False
 old=json.loads(p.read_text());assert old['identity']==identity,f'Checkpoint signature mismatch {out}'
 for n,h in old['files'].items():assert sha(out/n)==h
 return True

def finish(out,identity,extras=None):
 files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix in ['.npy','.parquet']};assert files
 js(out/'complete.json',{'identity':identity,'files':files,**(extras or {})})
 for n in files:(out/n).chmod(0o444)

def native_baseline(task):
 region,season,policy=task;start=time.time();cells=np.load(S/f'cells_{region}.npy');nc=len(cells);out=R/'native'/region/season/policy/'baseline';out.mkdir(parents=True,exist_ok=True);identity={'run':sha(R/'qa/run_signature.json'),'region':region,'season':season,'policy':policy,'stage_inputs':{f'{y}_{v}':sha(S/'raw'/region/f'{y}_{season}_{v}.complete.json') for y in range(2003,2008) for v in ['TMAX','TMIN']}}
 if completion(out,identity):return str(out)
 q=np.lib.format.open_memmap(out/'threshold_qnum.npy.partial',mode='w+',dtype=np.int32,shape=(2,nc));n=np.lib.format.open_memmap(out/'baseline_n.npy.partial',mode='w+',dtype=np.int16,shape=(2,nc));inputs=[[raw(region,y,season,v) for y in range(2003,2008)] for v in ['TMAX','TMIN']]
 for lo in range(0,nc,8192):
  hi=min(nc,lo+8192)
  for v,pct in enumerate([95,5]):cube=np.concatenate([a[:,lo:hi] for a in inputs[v]],axis=0);q[v,lo:hi],n[v,lo:hi]=P.quantile(cube,pct,policy)
 q.flush();n.flush();del q,n
 for name in ['threshold_qnum.npy','baseline_n.npy']:os.replace(out/(name+'.partial'),out/name)
 finish(out,identity,{'seconds':time.time()-start,'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss});log('baseline',region,season,policy,round(time.time()-start,1));return str(out)

def native_count(task):
 region,year,season,policy=task;start=time.time();nc=len(np.load(S/f'cells_{region}.npy'));base=R/'native'/region/season/policy/'baseline';out=R/'native'/region/season/policy/str(year);out.mkdir(parents=True,exist_ok=True);identity={'run':sha(R/'qa/run_signature.json'),'baseline':sha(base/'complete.json'),'year':year,'inputs':{v:sha(S/'raw'/region/f'{year}_{season}_{v}.complete.json') for v in ['TMAX','TMIN']}}
 if completion(out,identity):return str(out)
 q=np.load(base/'threshold_qnum.npy',mmap_mode='r');n=np.load(base/'baseline_n.npy',mmap_mode='r');daily=[raw(region,year,season,v) for v in ['TMAX','TMIN']];c=np.lib.format.open_memmap(out/'counts.npy.partial',mode='w+',dtype=np.int16,shape=(4,nc));m=np.lib.format.open_memmap(out/'support.npy.partial',mode='w+',dtype=bool,shape=(4,nc));zero=np.zeros(4,np.int64);null=np.zeros(4,np.int64)
 for lo in range(0,nc,16384):
  hi=min(nc,lo+16384);valid=[];ext=[]
  for v in [0,1]:
   x=np.array(daily[v][:,lo:hi]);ok=P.validity(x,policy);valid.append(ok.sum(axis=0,dtype=np.int16));scaled=x.astype(np.int64)*100;ex=(scaled>q[v,lo:hi]) if v==0 else (scaled<q[v,lo:hi]);ext.append((ex&ok).sum(axis=0,dtype=np.int16));assert np.all(ext[v]<=valid[v]) and np.all(valid[v]<=len(daily[v]))
  common=(valid[0]>0)&(valid[1]>0);mask=np.stack([common&(n[0,lo:hi]>0),common&(n[1,lo:hi]>0),common,common]);counts=np.stack([ext[0],ext[1],valid[0],valid[1]]);c[:,lo:hi]=counts;m[:,lo:hi]=mask;zero+=((counts==0)&mask).sum(axis=1);null+=(~mask).sum(axis=1)
 c.flush();m.flush();del c,m
 for name in ['counts.npy','support.npy']:os.replace(out/(name+'.partial'),out/name)
 finish(out,identity,{'seconds':time.time()-start,'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'supported_zero_cells_by_band':zero.tolist(),'absent_cells_by_band':null.tolist(),'source_days':[len(a) for a in daily]});log('counted',region,year,season,policy,round(time.time()-start,1));return str(out)

def grouped(c,m,bn,pairs,ntracts):
 """Bounded pair chunks; duplicate native positions keep independent geographic weights."""
 den=np.zeros((4,ntracts));num=np.zeros_like(den);bounds=np.zeros((2,ntracts));bden=np.zeros((2,ntracts));bnum=np.zeros_like(bden);bpartial=np.zeros_like(bden);bmin=np.full((2,ntracts),32767,np.int16)
 partial_max=grouped.baseline_days
 for lo in range(0,len(pairs['tract']),1000000):
  sl=slice(lo,lo+1000000);t=pairs['tract'][sl].astype(np.int64);pos=pairs['pos'][sl];w=pairs['coverage'][sl].astype(np.float64);cc=c[:,pos];mm=m[:,pos];nn=bn[:,pos]
  for b in range(4):
   mask=mm[b];tt=t[mask];ww=w[mask];den[b]+=np.bincount(tt,weights=ww,minlength=ntracts);num[b]+=np.bincount(tt,weights=ww*cc[b,mask],minlength=ntracts)
   if b<2:bounds[b]+=np.bincount(tt,weights=ww*cc[b+2,mask],minlength=ntracts)
  for v in [0,1]:
   mask=nn[v]>0;tt=t[mask];ww=w[mask];bden[v]+=np.bincount(tt,weights=ww,minlength=ntracts);bnum[v]+=np.bincount(tt,weights=ww*nn[v,mask],minlength=ntracts);partial=mask&(nn[v]<partial_max);bpartial[v]+=np.bincount(t[partial],weights=w[partial],minlength=ntracts);np.minimum.at(bmin[v],tt,nn[v,mask])
 values=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0);matched=np.divide(bounds,den[:2],out=np.full_like(bounds,np.nan),where=den[:2]>0);means=np.divide(bnum,bden,out=np.full_like(bnum,np.nan),where=bden>0)
 assert np.all((values[:2]<=matched+1e-10)|np.isnan(values[:2]));return values,den,matched,means,bmin,bpartial,bden


def reduce_period(task,mode='production',custom=None):
 year,season,policy=task;start=time.time();reg=registry();nt=len(reg);cov=pd.read_parquet(R/'qa/registry_coverage.parquet');values=np.full((4,nt),np.nan);den=np.zeros((4,nt));matched=np.full((2,nt),np.nan);bmean=np.full((2,nt),np.nan);bmin=np.full((2,nt),32767,np.int16);bpartial=np.zeros((2,nt));bden=np.zeros((2,nt));grouped.baseline_days=460 if season=='JJA' else 450
 for region in REGIONS:
  pairs={k:v for k,v in np.load(S/f'pairs_{region}.npz').items()};base=R/'native'/region/season/policy/'baseline';out=R/'native'/region/season/policy/str(year)
  if custom:c,m,bn=custom(region,year,season,policy)
  else:c=np.load(out/'counts.npy',mmap_mode='r');m=np.load(out/'support.npy',mmap_mode='r');bn=np.load(base/'baseline_n.npy',mmap_mode='r')
  result=grouped(c,m,bn,pairs,nt);sel=reg.source_region.eq(region).to_numpy()
  for target,array in zip([values,den,matched,bmean,bmin,bpartial,bden],result):target[:,sel]=array[:,sel]
 df=reg.copy();df['observation_year']=np.int64(year);df['season']=season
 for b,name in enumerate(MET):df[name]=pd.array(values[b],dtype='Float64');assert df[name].dropna().between(0,92 if season=='JJA' else 90).all()
 df['boundary_year']=np.int64(2020);df['baseline_start_year']=np.int64(2003);df['baseline_end_year']=np.int64(2007);df['tmax_threshold_percentile']=np.int64(95);df['tmin_threshold_percentile']=np.int64(5);df['djf_convention']='same-year';df['extreme_method_id']=METHOD+'_'+policy;df['mask_policy']=policy;df['source_product']='SVCM-SP';df['source_manifest_identifier']=config()['source_manifest_identifier'];df['index_policy']=np.where(df.GEOID.eq('02016000100'),'split_wrap','verified_final_1000m')
 status=np.full(nt,'ok',dtype=object);status[~np.all(den>0,axis=0)]='partial_band_threshold_support';status[~np.any(den>0,axis=0)]='source_no_common_seasonal_support';status[cov.inside_grid_pairs.eq(0)]='outside_selected_source_grid';status[cov.coverage_pairs.eq(0)]='no_coverage_pairs';status[reg.source_region.eq('unsupported_pacific')]='unsupported_pacific_source_route';df['support_status']=status
 support=cov.copy();support['observation_year']=np.int64(year);support['season']=season;support['mask_policy']=policy;support['source_region']=reg.source_region;support['support_status']=status
 for b,name in enumerate(MET):support[name+'_support_weight']=den[b];assert np.array_equal(np.isnan(values[b]),den[b]==0)
 for v,var in enumerate(['tmax','tmin']):
  support[f'baseline_{var}_mean_days']=bmean[v];support[f'baseline_{var}_min_days']=pd.array(np.where(bmin[v]==32767,np.nan,bmin[v]),dtype='Float64');support[f'baseline_{var}_partial_weight']=bpartial[v];support[f'baseline_{var}_support_weight']=bden[v]
 support['hot_matched_valid_days_bound']=matched[0];support['cold_matched_valid_days_bound']=matched[1]
 dest=R/'shards'/mode/policy;atomic_parquet(df,dest/f'{year}_{season}.parquet');atomic_parquet(support,dest/f'{year}_{season}_support.parquet');js(dest/f'{year}_{season}.complete.json',{'run_signature':sha(R/'qa/run_signature.json'),'rows':nt,'metrics_sha256':sha(dest/f'{year}_{season}.parquet'),'support_sha256':sha(dest/f'{year}_{season}_support.parquet'),'seconds':time.time()-start});log('reduced full registry',mode,year,season,policy,round(time.time()-start,1));return df


def scalar_native_checks(region,year,season,policy,root=R/'native'):
 cells=np.load(S/f'cells_{region}.npy');base=root/region/season/policy/'baseline';out=root/region/season/policy/str(year);q=np.load(base/'threshold_qnum.npy',mmap_mode='r');bn=np.load(base/'baseline_n.npy',mmap_mode='r');counts=np.load(out/'counts.npy',mmap_mode='r');masks=np.load(out/'support.npy',mmap_mode='r');positions=np.unique(np.r_[np.linspace(0,len(cells)-1,32,dtype=int),np.flatnonzero(bn[0]>0)[:4],np.flatnonzero(bn[0]==0)[:4]]);ncheck=0
 for vi,var in enumerate(['TMAX','TMIN']):
  for pos in positions:
   vals=np.concatenate([raw(region,y,season,var)[:,pos] for y in range(2003,2008)]).tolist();good=sorted([int(v) for v in vals if v!=P.NODATA and (policy==POLICIES[0] or -950<=v<=650)]);assert bn[vi,pos]==len(good)
   daily=raw(region,year,season,var)[:,pos].tolist();valid=[int(v) for v in daily if v!=P.NODATA and (policy==POLICIES[0] or -950<=v<=650)];assert counts[vi+2,pos]==len(valid)
   if good:
    h=(len(good)-1)*[95,5][vi];i,r=divmod(h,100);expected=100*good[i]+r*(good[min(i+1,len(good)-1)]-good[i]);assert q[vi,pos]==expected;assert counts[vi,pos]==sum(100*v>expected if vi==0 else 100*v<expected for v in valid)
   common=counts[2,pos]>0 and counts[3,pos]>0;assert masks[vi,pos]==(common and bool(good)) and masks[vi+2,pos]==common;ncheck+=1
 return {'region':region,'year':year,'season':season,'policy':policy,'scalar_cells_variables_checked':ncheck,'passed':True}


def benchmark():
 freeze();P.hand_tests();start=time.time();reg=registry();nt=len(reg)
 # Actual full production reached cells and both geographic reducers; fresh staging at baseline and drift year.
 staging=parallel(stage_year,[(r,y) for y in [2003,2022] for r in REGIONS],4)
 # Recompute exact kernels from signature-verified pilot selections on representative tiles.
 checks=[];native_records=[]
 for region in REGIONS:
  pcells=np.load(PS/f'cells_{region}.npy');cells=np.load(S/f'cells_{region}.npy');pos=np.searchsorted(cells,pcells);assert np.array_equal(cells[pos],pcells)
  for s in SEASONS:
   for v in ['TMAX','TMIN']:
    for y in range(2003,2008):
     done=json.loads((PS/'raw'/region/f'{y}_{s}_{v}.complete.json').read_text());assert sha(PS/'raw'/region/f'{y}_{s}_{v}.npy')==done['values_sha256'];assert done['input_signature']==hashlib.sha256(json.dumps(sorted(pd.read_csv(PILOT/'qa/selected_members.csv.gz').query('region==@region and year==@y and season==@s and variable==@v').to_dict('records'),key=lambda r:r['date']),sort_keys=True).encode()+pcells.tobytes()).hexdigest()
  for policy in POLICIES:
   for season in SEASONS:
    bvals=[];bns=[];pout=PS/'computed/run1'/region/season/policy;bench=R/'benchmark_native'/region/season/policy;bench.mkdir(parents=True,exist_ok=True)
    q=np.zeros((2,len(cells)),np.int32);bn=np.zeros((2,len(cells)),np.int16)
    for vi,var in enumerate(['TMAX','TMIN']):
     inputs=[raw(region,y,season,var,PS) for y in range(2003,2008)];qn=np.empty(len(pcells),np.int32);nn=np.empty(len(pcells),np.int16)
     for lo in range(0,len(pcells),8192):
      hi=min(len(pcells),lo+8192);qn[lo:hi],nn[lo:hi]=P.quantile(np.concatenate([a[:,lo:hi] for a in inputs]),[95,5][vi],policy)
     assert np.array_equal(qn,np.load(pout/'threshold_qnum.npy',mmap_mode='r')[vi]) and np.array_equal(nn,np.load(pout/'baseline_n.npy',mmap_mode='r')[vi]);q[vi,pos]=qn;bn[vi,pos]=nn;checks.append({'region':region,'season':season,'policy':policy,'variable':var,'exact_threshold_cells':len(pcells)})
    np.save(bench/'baseline_n.npy',bn);np.save(bench/'threshold_qnum.npy',q)
    for y,s in P.MAND+P.DIAG:
     if s!=season:continue
     for var in ['TMAX','TMIN']:
      p=PS/'raw'/region/f'{y}_{s}_{var}.npy';done=json.loads(p.with_name(p.name.replace('.npy','.complete.json')).read_text());assert sha(p)==done['values_sha256']
     c=np.zeros((4,len(cells)),np.int16);m=np.zeros((4,len(cells)),bool);valid=[];ext=[]
     for vi,var in enumerate(['TMAX','TMIN']):
      a=raw(region,y,s,var,PS);ok=P.validity(a,policy);valid.append(ok.sum(axis=0,dtype=np.int16));scaled=a.astype(np.int64)*100;ex=(scaled>q[vi,pos]) if vi==0 else (scaled<q[vi,pos]);ext.append((ex&ok).sum(axis=0,dtype=np.int16))
     common=(valid[0]>0)&(valid[1]>0);cp=np.stack([ext[0],ext[1],valid[0],valid[1]]);mp=np.stack([common&(bn[0,pos]>0),common&(bn[1,pos]>0),common,common]);assert np.array_equal(cp,np.load(pout/f'counts_{y}.npy',mmap_mode='r')) and np.array_equal(mp,np.load(pout/f'support_{y}.npy',mmap_mode='r'));c[:,pos]=cp;m[:,pos]=mp
     np.save(bench/f'counts_{y}.npy',c);np.save(bench/f'support_{y}.npy',m)
     # Verify fresh full-scope staging subset equals pilot caches at both measured years.
     if y in [2003,2022]:
      for var in ['TMAX','TMIN']:assert np.array_equal(raw(region,y,s,var)[:,pos],raw(region,y,s,var,PS))
 def custom(region,y,s,p):
  b=R/'benchmark_native'/region/s/p;return np.load(b/f'counts_{y}.npy',mmap_mode='r'),np.load(b/f'support_{y}.npy',mmap_mode='r'),np.load(b/'baseline_n.npy',mmap_mode='r')
 dfs={}
 for policy in POLICIES:
  frames=[reduce_period((y,s,policy),mode='benchmark',custom=custom) for y,s in P.MAND+P.DIAG];dfs[policy]=pd.concat(frames,ignore_index=True)
 primary=pd.concat([pd.read_parquet(PILOT/'primary_pilot.parquet'),pd.read_parquet(PILOT/'primary_diagnostics.parquet')]);refs={POLICIES[0]:primary,POLICIES[1]:pd.read_parquet(PILOT/'range_sensitivity.parquet')};tol=1e-10;comparisons=[]
 for policy,ref in refs.items():
  current=dfs[policy].set_index(KEY);old=ref.set_index(KEY);old=old.loc[old.index.get_level_values('GEOID')!='02016000100'];now=current.loc[old.index]
  for metric in MET:
   a=now[metric].to_numpy(float,na_value=np.nan);b=old[metric].to_numpy(float,na_value=np.nan);assert np.array_equal(np.isnan(a),np.isnan(b));delta=np.abs(a-b);maximum=float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.;assert maximum<=tol,(metric,maximum);comparisons.append({'policy':policy,'metric':metric,'rows':len(ref),'antimeridian_reference':'separate split/wrap diagnostic','maximum_absolute_difference_days':maximum,'absolute_tolerance_days':tol,'relative_tolerance':0.})
  wrap=pd.read_parquet(PILOT/'antimeridian_diagnostic.parquet');wrap=wrap[wrap.method_id.str.endswith(policy)].set_index(KEY);actual=current.loc[wrap.index]
  for metric in MET:assert np.array_equal(actual[metric].isna(),wrap[metric].isna());assert np.allclose(actual[metric].to_numpy(float,na_value=np.nan),wrap[metric].to_numpy(float,na_value=np.nan),atol=tol,rtol=0,equal_nan=True)
 # Independent math.fsum on several real tracts, including outside-pilot IDs.
 independent=[]
 for region in REGIONS:
  pp={k:v for k,v in np.load(S/f'pairs_{region}.npz').items()};choices=list(pd.read_csv(D/'pilot_tract_selection.csv',dtype={'GEOID':str}).GEOID)+json.loads((R/'qa/prepare.json').read_text())['extra_validation_geoids'];kmap={k:i for i,k in enumerate(reg.GEOID)};c,m,bn=custom(region,2003,'DJF',POLICIES[0]);actual=dfs[POLICIES[0]].query('observation_year==2003 and season=="DJF"').set_index('GEOID')
  for k in choices:
   if reg.loc[kmap[k],'source_region']!=region:continue
   take=pp['tract']==kmap[k];pos=pp['pos'][take];w=pp['coverage'][take].astype(float)
   for b,metric in enumerate(MET):
    ok=m[b,pos];den=math.fsum(w[ok]);expected=math.fsum(float(v)*float(z) for v,z in zip(c[b,pos[ok]],w[ok]))/den if den else np.nan;value=actual.loc[k,metric];assert (pd.isna(value) and not den) or abs(value-expected)<=tol;independent.append({'GEOID':k,'metric':metric,'pairs':int(take.sum()),'passed':True})
 # Resource bound derives from actual cell inventory, not pilot wall-time extrapolation.
 inventory=json.loads((R/'qa/prepare.json').read_text());daily=sum(x['daily_selection_bytes'] for x in inventory['regions']);scratch_bound=daily*9+2000000000;durable_bound=sum(x['unique_native_cells'] for x in inventory['regions'])*(24*44+48)+3000000000;assert scratch_bound<=config()['scratch_reserve_bound_bytes'] and durable_bound<=config()['durable_reserve_bound_bytes'];assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<100000000
 result={'passed':True,'benchmark_scope':'fresh full reached-cell staging 2003 and 2022; exact representative/pilot thresholds and counts; all 85528 keys reduced for 11 periods x two masks. Cells outside tested native subset intentionally lack benchmark support. Full native baseline follows this gate.','staging':staging,'exact_threshold_checks':checks,'pilot_comparisons':comparisons,'independent_reducer_checks':independent,'scratch_bound_bytes':scratch_bound,'durable_bound_bytes':durable_bound,'seconds':time.time()-start,'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'max_child_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'job_id':os.environ.get('SLURM_JOB_ID')};js(R/'qa/benchmark.json',result);log('BENCHMARK PASSED; automatic production authorized')


def clean_raw(region,year):
 if year<=2007:return
 markers=[]
 for season in SEASONS:
  for policy in POLICIES:
   marker=R/'native'/region/season/policy/str(year)/'complete.json';assert marker.exists();data=json.loads(marker.read_text())
   for n,h in data['files'].items():assert sha(marker.parent/n)==h
   assert (R/'shards/production'/policy/f'{year}_{season}.complete.json').exists();markers.append(str(marker))
 removed=[]
 for p in (S/'raw'/region).glob(f'{year}_*.npy'):
  marker=p.with_name(p.name.replace('.npy','.complete.json'));data=json.loads(marker.read_text());assert sha(p)==data['sha256'];removed.append({'path':str(p),'sha256':data['sha256'],'bytes':p.stat().st_size});p.unlink()
 js(R/'qa/cleanup'/f'{region}_{year}.json',{'durable_completions_verified':markers,'removed_only_this_run_regenerable_arrays':removed})


def production():
 freeze();assert json.loads((R/'qa/benchmark.json').read_text())['passed'];start=time.time()
 tasks=[]
 for year in range(2003,2008):
  for r in REGIONS:
   if not all((S/'raw'/r/f'{year}_{s}_{v}.npy').exists() for s in SEASONS for v in ['TMAX','TMIN']):tasks.append((r,year))
 parallel(stage_year,tasks,4);parallel(native_baseline,[(r,s,p) for r in REGIONS for s in SEASONS for p in POLICIES],4)
 checks=[]
 for year in YEARS:
  # All 4 masks and all registry keys have separate durable period shards.
  done=all((R/'shards/production'/p/f'{year}_{s}.complete.json').exists() for p in POLICIES for s in SEASONS)
  if done:
   for p in POLICIES:
    for s in SEASONS:
     marker=R/'shards/production'/p/f'{year}_{s}.complete.json';d=json.loads(marker.read_text());assert d['run_signature']==sha(R/'qa/run_signature.json') and sha(marker.with_name(f'{year}_{s}.parquet'))==d['metrics_sha256'] and sha(marker.with_name(f'{year}_{s}_support.parquet'))==d['support_sha256']
   continue
  parallel(stage_year,[(r,year) for r in REGIONS],2)
  parallel(native_count,[(r,year,s,p) for r in REGIONS for s in SEASONS for p in POLICIES],4)
  for p in POLICIES:
   for s in SEASONS:reduce_period((year,s,p))
  if year in [2003,2008,2016,2022,2024]:
   for r in REGIONS:
    for s in SEASONS:
     for p in POLICIES:checks.append(scalar_native_checks(r,year,s,p))
   js(R/'qa/independent_native_checks.json',{'passed':True,'checks':checks})
  for r in REGIONS:clean_raw(r,year)
 js(R/'qa/production_complete.json',{'passed':True,'period_shards':88,'seconds_this_resume':time.time()-start,'job_id':os.environ.get('SLURM_JOB_ID')})

if __name__=='__main__':
 try:measure(sys.argv[1],{'benchmark':benchmark,'production':production}[sys.argv[1]])
 except Exception:traceback.print_exc();sys.exit(1)
