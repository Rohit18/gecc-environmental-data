from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv, io, json, re, zipfile
import pymupdf as fitz
import split_gridmet_prism as split
ROOT=Path('/path/to/GECC')
DEST=ROOT/'uploadGECC/census_tracts_12_metrics_20261030'
SOURCE=ROOT/'stakeholder_packages/census_tracts_20261018'
HERE=Path(__file__).resolve().parent
with (DEST/'FILE_MANIFEST.csv').open(newline='') as f: manifest=list(csv.DictReader(f))
with (DEST/'metadata/COLUMN_DICTIONARY.csv').open(newline='') as f: dictionary=list(csv.DictReader(f))
fields={}
for r in dictionary:
    fields.setdefault(r['table'],[]).append(r['field'])
assert len(dictionary)==6431 and len(manifest)==34
paths={r['path']:r for r in manifest}
assert len(paths)==34
members=[]
for archive in sorted(DEST.glob('*.zip')):
    assert archive.stat().st_size<2_000_000_000
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            row=paths[info.filename]
            assert row['archive']==archive.name
            assert info.file_size==int(row['bytes'])
            members.append(info.filename)
            if info.filename.endswith('.csv'):
                with z.open(info) as f:
                    header=next(csv.reader(io.TextIOWrapper(f,encoding='utf-8')))
                assert len(header)==int(row['columns']), info.filename
                assert set(header)==set(fields[row['table']]), info.filename
                assert len(header)==len(fields[row['table']]), info.filename
assert len(members)==len(set(members))==34 and set(members)==set(paths)
with (DEST/'metadata/GLC_ANNUAL_TILE_AFFECTED.csv').open(newline='') as f:
    glc_rows=list(csv.DictReader(f))
assert list(glc_rows[0])==fields['glc_annual_tile_affected']
assert sum(r['table']=='land_cover_glc' for r in glc_rows)==6424 and sum(r['table']=='vegetation_ua_pieces' for r in glc_rows)==6380
files=[p for p in DEST.rglob('*') if p.is_file()]
assert len(files)==21
bad=re.compile(r'era5|checksum|sha256|import_schema|8,252|11 environmental|gridmet_prism|GRIDMET/PRISM',re.I)
links=0
for p in files:
    assert not bad.search(p.name),p
    assert p.suffix!='.json' and p.name!='.DS_Store'
    if p.suffix in {'.md','.csv'}:
        text=p.read_text()
        assert not bad.search(text),p
        if p.suffix=='.md':
            for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
                if target.startswith(('http:','https:','#')): continue
                assert (p.parent/target.split('#')[0]).is_file(),(p,target)
                links+=1
    elif p.suffix=='.pdf':
        with fitz.open(p) as doc:
            assert not bad.search(''.join(page.get_text() for page in doc)),p

def compare_archive(archive):
    compared=0
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            with z.open(name) as a,split.source_path(name).open('rb') as b:
                while True:
                    block=a.read(4*1024*1024)
                    assert block==b.read(4*1024*1024),name
                    if not block: break
                    compared+=len(block)
    print('VERIFIED',archive.name,flush=True)
    return compared
with ThreadPoolExecutor(max_workers=3) as pool:
    compared=sum(pool.map(compare_archive,sorted(DEST.glob('*.zip'))))
split_rows=split.verify()
report={'status':'PASS','gridmet_prism_split_rows_checked_against_combined_source':split_rows,'upload_files':len(files),'upload_bytes':sum(p.stat().st_size for p in files),'largest_archive_bytes':max(p.stat().st_size for p in DEST.glob('*.zip')),'data_files_compared_byte_for_byte':34,'data_bytes_compared':compared,'csv_headers_checked':32,'dictionary_fields':len(dictionary),'markdown_links_checked':links,'pdfs_text_checked':2,'pdf_pages':sum(len(fitz.open(p)) for p in DEST.glob('documentation/*.pdf')),'excluded_content_present':False,'uploaded':False}
(HERE/'VERIFICATION_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
