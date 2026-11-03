"""Export the exact complement of the priority CityHealth tract GLC run.

Uses the existing 2022 national export to pin all 84,414 tract IDs, then removes
the 47,105 selected IDs. The 37,309 remaining IDs are exported for 25 missing
years with default priority 100. Custom priorities are unavailable on this
noncommercial project; the previously submitted selected jobs stay in place.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import ee
import submit_cityhealth_selected_tract_glc as selected

base = selected.base
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/remaining_tract_glc'
PROJECT = selected.PROJECT
FOLDER = 'GECC_RemainingTracts_GLC_FCS30D'
YEARS = selected.YEARS
PRIORITY = 100
N_BATCHES = 10


def complement(all_ids, selected_ids):
    if len(all_ids) != len(set(all_ids)) or len(selected_ids) != len(set(selected_ids)):
        raise ValueError('Duplicate IDs in input sets')
    if not set(selected_ids) <= set(all_ids):
        raise ValueError('Selected IDs not present in national tract universe')
    result = sorted(set(all_ids) - set(selected_ids))
    assert not set(result) & set(selected_ids)
    assert set(result) | set(selected_ids) == set(all_ids)
    return result


def source_rows():
    with zipfile.ZipFile(selected.ZIP) as archive:
        for name in sorted(archive.namelist()):
            with archive.open(name) as raw:
                for row in csv.DictReader(io.TextIOWrapper(raw)):
                    assert int(row['year']) == 2022
                    yield row


def plan():
    national = [row['GEOID'] for row in source_rows()]
    with selected.LOOKUP.open() as f:
        picked = sorted({row['GEOID'] for row in csv.DictReader(f)})
    assert len(national) == 84414 and len(picked) == 47105
    ids = complement(national, picked)
    assert len(ids) == 37309
    digest = hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()
    batches = selected.split_batches(ids, N_BATCHES)
    prefix = f'GLC_RemainingTract_{digest[:10]}_v1'
    jobs = [{'year': y, 'batch': b, 'tract_count': len(batch),
             'description': f'{prefix}_{y}_b{b:03d}of{N_BATCHES:03d}'}
            for y in YEARS for b, batch in enumerate(batches)]
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {
        'builder_sha256': Path(__file__), 'selected_builder_sha256': Path(selected.__file__),
        'base_builder_sha256': Path(base.__file__), 'national_2022_zip_sha256': selected.ZIP,
        'selected_lookup_sha256': selected.LOOKUP}.items()}
    result = {'project': PROJECT, 'drive_folder': FOLDER, 'priority': PRIORITY,
              'prefix': prefix, 'selection_sha256': digest, 'tract_count': len(ids),
              'national_tract_count': len(national), 'excluded_selected_count': len(picked),
              'years': YEARS, 'batch_sizes': list(map(len, batches)), 'jobs': jobs, **hashes}
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'plan.json').exists():
        old = json.loads((OUT/'plan.json').read_text())
        assert old['jobs'] == jobs and old['selection_sha256'] == digest
    selected.save(OUT/'plan.json', result)
    with (OUT/'remaining_tract_batch_lookup.csv').open('w') as f:
        writer=csv.writer(f); writer.writerow(['GEOID','batch'])
        for b,batch in enumerate(batches): writer.writerows((geoid,b) for geoid in batch)
    return result, national, picked, ids, batches


def table(image, ids, year, batch):
    return selected.table(image, ids, year, batch).map(
        lambda feature: feature.set('selection', 'remaining_2020_tracts_outside_CityHealth_selection'))


def validate(config, national, picked, ids, batches):
    remote = ee.FeatureCollection(base.TRACT_ASSET).aggregate_array('GEOID').getInfo()
    assert len(remote) == len(set(remote)) == 84414
    assert set(remote) == set(national)
    assert complement(remote, picked) == ids
    remote_batches = ee.Dictionary({str(b): selected.batch_tracts(batch,b).aggregate_array('GEOID')
                                    for b,batch in enumerate(batches)}).getInfo()
    assert all(sorted(remote_batches[str(b)]) == batch for b,batch in enumerate(batches))
    print('PASS: exact complement and all 10 batch memberships verified against Earth Engine.', flush=True)
    remaining = set(ids)
    controls = {}
    positive = zero = None
    for row in source_rows():
        if row['GEOID'] not in remaining: continue
        if positive is None and float(row['total_area_m2']) > 0 and float(row['source_area_m2']) < 5e6:
            positive=row['GEOID']; controls[positive]=row
        if zero is None and float(row['total_area_m2']) == 0:
            zero=row['GEOID']; controls[zero]=row
        if positive and zero: break
    assert positive is not None
    images=base.year_images()
    requests={f'2022_{geoid}': ee.Feature(table(images[2022],[geoid],2022,0).first()).toDictionary(selected.selectors())
              for geoid in controls}
    for year in [1985,1990,1995,2000,2021]:
        requests[f'{year}_{positive}']=ee.Feature(table(images[year],[positive],year,0).first()).toDictionary(selected.selectors())
    rows=ee.Dictionary(requests).getInfo()
    for geoid, reference in controls.items():
        actual=rows[f'2022_{geoid}']
        for field in base.output_selectors():
            if field in ['GEOID','batch']: continue
            if float(reference['total_area_m2']) == 0 and field in selected.FRACTIONS:
                assert actual.get(field) is None
            else:
                assert math.isclose(float(actual[field]),float(reference[field]),rel_tol=1e-7,abs_tol=1e-4),(geoid,field)
    for row in rows.values():
        assert row['GEOID'] in remaining
        assert row['selection']=='remaining_2020_tracts_outside_CityHealth_selection'
        assert math.isclose(sum(row[f'area_class_{c}'] for c in base.GLC_CLASSES),row['total_area_m2'],rel_tol=1e-10,abs_tol=1e-4)
        assert all(row[f'area_class_{c}']>=0 for c in base.GLC_CLASSES)
        if row['total_area_m2'] == 0:
            assert not row['has_glc_sampled_area']
            assert all(row.get(field) is None for field in selected.FRACTIONS)
        else:
            assert row['has_glc_sampled_area']
            assert all(0 <= row[field] <= 1 for field in selected.FRACTIONS)
    selected.save(OUT/'previews.json',rows)
    report={'status':'PASS','validated_at':selected.now(),
            **{k:v for k,v in config.items() if k.endswith('_sha256')},
            'remaining_tract_count':len(ids),'selected_overlap_count':0,'union_tract_count':len(remote),
            'exact_batch_membership_verified':True,'preview_count':len(rows),
            'positive_control':positive,'zero_coverage_control':zero,'existing_2022_values_reproduced':True}
    selected.save(OUT/'validation.json',report)
    print(json.dumps(report,indent=2),flush=True)


def status(config):
    operations=ee.data.listOperations(project='projects/'+PROJECT)
    names={j['description'] for j in config['jobs']}
    relevant=[o for o in operations if o['metadata'].get('description') in names]
    found={o['metadata']['description'] for o in relevant}
    priority_plan=json.loads((selected.OUT/'plan.json').read_text())
    priority_names={j['description'] for j in priority_plan['jobs']}
    priority_ops=[o for o in operations if o['metadata'].get('description') in priority_names]
    report={'checked_at':selected.now(),'project':PROJECT,'planned_tasks':len(names),
            'registered_descriptions':len(found),'missing_descriptions':sorted(names-found),
            'counts':dict(collections.Counter(o['metadata']['state'] for o in relevant)),
            'selected_run_counts':dict(collections.Counter(o['metadata']['state'] for o in priority_ops)),
            'selected_registered_descriptions':len({o['metadata']['description'] for o in priority_ops}),
            'operations':relevant}
    selected.save(OUT/'status.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['operations','missing_descriptions']},indent=2),flush=True)
    return operations,report


def submit(config,batches):
    checked=json.loads((OUT/'validation.json').read_text())
    assert checked['status']=='PASS'
    assert all(checked[k]==v for k,v in config.items() if k.endswith('_sha256'))
    operations,prior=status(config)
    assert prior['selected_registered_descriptions']==250
    assert not prior['selected_run_counts'].get('FAILED',0)
    active={o['metadata'].get('description') for o in operations if o['metadata']['state'] in ['PENDING','RUNNING','SUCCEEDED']}
    images=base.year_images();started=0
    for job in config['jobs']:
        if job['description'] in active: continue
        task=ee.batch.Export.table.toDrive(
            collection=table(images[job['year']],batches[job['batch']],job['year'],job['batch']),
            description=job['description'],folder=FOLDER,fileNamePrefix=job['description'],
            fileFormat='CSV',selectors=selected.selectors())
        task.start()
        with (OUT/'submitted_tasks.jsonl').open('a') as f:
            f.write(json.dumps({**job,'project':PROJECT,'priority':PRIORITY,'task_id':task.id,
                               'submitted_at':selected.now(),'selection_sha256':config['selection_sha256']})+'\n')
        started+=1
        if started%10==0:print(f'Submitted {started} new tasks; last year {job["year"]}',flush=True)
    _,report=status(config)
    assert not report['missing_descriptions'] and not report['counts'].get('FAILED',0)
    assert all(o['metadata']['priority']==PRIORITY for o in report['operations'])
    print(f'PASS: all 250 remaining-tract tasks registered, priority {PRIORITY}.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['plan','validate','submit','status'],default='plan')
    args=parser.parse_args()
    config,national,picked,ids,batches=plan()
    if args.mode=='plan':
        print(json.dumps({k:v for k,v in config.items() if k!='jobs'},indent=2));return
    ee.Initialize(project=PROJECT)
    if args.mode=='validate':validate(config,national,picked,ids,batches)
    elif args.mode=='submit':submit(config,batches)
    else:status(config)


if __name__=='__main__':main()
