import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=json.load(open(ROOT/'results/s16_summary.json'))
m=(ROOT/'results/s16_order2_exact.meta').read_text()
vals={}
for line in m.splitlines():
    p=line.split()
    if len(p)==2: vals[p[0]]=int(p[1])
assert s['interface']['stabilizer_isomorphism_exact']
assert s['interface']['logical_X_representative_exact']
assert s['interface']['logical_Z_representative_equivalent_mod_Z_stabilizers']
assert s['growth_detector_stitching']['external_pre_growth_record_references']==0
assert s['canonical_sector_coverage']['missing_weight2_signatures']==0
assert vals['signatures']==7529
assert vals['records']==28346686
assert vals['ambiguous_exact_keys']==0
assert vals['hash_groups_mixed_frame']==0
assert s['order_le2_exact_overapprox']['all_zero_operational_key_identity_only']
print('S16 SELF-TEST PASS')
