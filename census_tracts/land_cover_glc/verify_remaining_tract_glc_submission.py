"""Audit the saved batch partition and final server-reported export identities."""
import collections,csv,json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/remaining_tract_glc'

def main():
    plan=json.loads((OUT/'plan.json').read_text())
    status=json.loads((OUT/'status.json').read_text())
    submitted=[json.loads(line) for line in (OUT/'submitted_tasks.jsonl').read_text().splitlines()]
    with (OUT/'remaining_tract_batch_lookup.csv').open() as f:remaining=list(csv.DictReader(f))
    with (ROOT/'outputs/cityhealth_selected_tract_glc/selected_tract_batch_lookup.csv').open() as f:selected=list(csv.DictReader(f))
    remaining_ids={r['GEOID'] for r in remaining};selected_ids={r['GEOID'] for r in selected}
    assert len(remaining)==len(remaining_ids)==37309
    assert len(selected)==len(selected_ids)==47105
    assert not remaining_ids & selected_ids
    assert len(remaining_ids|selected_ids)==84414
    batch_counts=collections.Counter(int(r['batch']) for r in remaining)
    assert [batch_counts[i] for i in range(10)]==plan['batch_sizes']
    names={j['description'] for j in plan['jobs']}
    assert len(submitted)==len({r['task_id'] for r in submitted})==250
    assert {r['description'] for r in submitted}==names
    assert {o['metadata']['description'] for o in status['operations']}==names
    assert {o['name'].split('/')[-1] for o in status['operations']}=={r['task_id'] for r in submitted}
    assert all(o['name'].startswith('projects/your-ee-project/operations/') for o in status['operations'])
    assert all(o['metadata']['priority']==100 for o in status['operations'])
    assert set(collections.Counter(j['year'] for j in submitted).values())=={10}
    assert sum(j['tract_count'] for j in submitted)==932725
    assert all(j['year']!=2022 for j in submitted)
    assert not status['missing_descriptions']
    assert not status['counts'].get('FAILED',0)
    assert status['selected_registered_descriptions']==250
    assert not status['selected_run_counts'].get('FAILED',0)
    report={'status':'PASS','checked_server_snapshot_at':status['checked_at'],
            'remaining_tract_count':37309,'selected_tract_count':47105,'overlap_count':0,
            'union_tract_count':84414,'submitted_tasks':250,'verified_task_ids':250,
            'years':plan['years'],'remaining_expected_tract_year_rows':932725,
            'project':plan['project'],'effective_priority':100,
            'custom_priority_supported':False,'remaining_run_states':status['counts'],
            'selected_run_states':status['selected_run_counts'],'completed_2022_reused':True,
            'paused_worldcover_and_ghsl_resumed':False}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    for source,dest in [('/tmp/remaining_tract_glc_validation.log','validation.log'),('/tmp/remaining_tract_glc_submission.log','submission.log')]:
        shutil.copyfile(source,OUT/dest)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
