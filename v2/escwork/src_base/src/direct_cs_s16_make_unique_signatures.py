import json, importlib.util, sys
from collections import Counter
from pathlib import Path
all_entries=[]
# Escape physical singles: use RAW physical logical commutation after fault-free ZXZ+handoff,
# not the preserved recovery-frame label (different convention for nonzero syndrome).
for line in open('/mnt/data/s16_work/s16_escape_single_effects.jsonl'):
    r=json.loads(line)
    fr=int(r['d13_lx']) | (int(r['d13_lz'])<<1)
    all_entries.append({'stage':'escape','source_id':int(r['fault_id']), 'live':int(r['actual171']),
      'gdet':int(r['gdet']),'sx':int(r['d13_sx']),'sz':int(r['d13_sz']),'frame':fr})
# Effective canonical incoming Steane w1 sectors, also use raw physical commutation.
for r in json.load(open('/mnt/data/s16_work/s16_sector_effects.json')):
    if int(r['weight'])==1:
      fr=int(r['d13_lx']) | (int(r['d13_lz'])<<1)
      all_entries.append({'stage':'sector','source_id':int(r['syndrome']), 'live':int(r['actual171']),
        'gdet':int(r['gdet']),'sx':int(r['d13_sx']),'sz':int(r['d13_sz']),'frame':fr})
# Growth physical singles already expose raw final logical commutations.
spec=importlib.util.spec_from_file_location('s15','/mnt/data/direct_cs_S15_growth_audit_2026-08-21/src/direct_cs_s15_single_fault_audit.py')
m=importlib.util.module_from_spec(spec);sys.modules['s15']=m;spec.loader.exec_module(m)
for idx,f in enumerate(m.G57_FAULTS+m.G713_FAULTS):
    r=m.simulate_fault(f)
    all_entries.append({'stage':'g57' if f.line<=m.G57_END else 'g713','source_id':idx,'live':0,
      'gdet':int(r['detector_bits']),'sx':int(r['sx_bits']),'sz':int(r['sz_bits']),
      'frame':int(r['logical_x_commutation']) | (int(r['logical_z_commutation'])<<1)})
# Dedupe including raw logical truth label.
by={}
for e in all_entries:
    k=(e['live'],e['gdet'],e['sx'],e['sz'],e['frame'])
    if k not in by: by[k]={'count':0,'stages':Counter(),'examples':[]}
    z=by[k];z['count']+=1;z['stages'][e['stage']]+=1
    if len(z['examples'])<4:z['examples'].append((e['stage'],e['source_id']))
rows=[]
for k,z in by.items():
    live,gdet,sx,sz,fr=k;words=[]
    for v,n in [(live,3),(gdet,3),(sx,2),(sz,2)]:words += [(v>>(64*i))&((1<<64)-1) for i in range(n)]
    rows.append({'words':words,'frame':fr,'count':z['count'],'stages':dict(z['stages']),'examples':z['examples']})
rows.sort(key=lambda r:(r['words'],r['frame']))
out=Path('/mnt/data/s16_work/s16_unique_single_signatures_raw.tsv')
with out.open('w') as f:
    f.write('# live[3], growth[3], sx[2], sz[2], raw logical commutation\n')
    for r in rows:f.write(' '.join(f'{w:016x}' for w in r['words'])+f" {r['frame']}\n")
meta={'physical_or_effective_single_entries_before_dedupe':len(all_entries),'unique_generalized_single_signatures':len(rows),'stage_entry_counts':dict(Counter(e['stage'] for e in all_entries)),'stage_unique_presence':dict(Counter(s for r in rows for s in r['stages'])),'truth_label':'raw final d13 logical commutation; comparable only within identical final syndrome key, exactly as required for logical-coset ambiguity'}
Path('/mnt/data/s16_work/s16_unique_single_signatures_raw_meta.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
