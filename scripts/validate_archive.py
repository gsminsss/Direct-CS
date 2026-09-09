#!/usr/bin/env python3
from pathlib import Path
import csv,json,math
ROOT=Path(__file__).resolve().parents[1]
def J(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
m=J('data/final_results/s21opt5a_matrix_audit.json')
assert m['H_shape']==[263,3339] and m['H_rank']==263
assert m['H_aug_shape']==[265,3339] and m['H_aug_rank']==265
assert m['raw_effect_classes']==3343 and m['closure_conditioned_decoder_effect_classes']==3340 and m['effect_columns']==3339
assert m['latent_residual_filter']['latent_zero_syndrome_nonidentity_excluded']==3
assert m['model_meta']['operational_detectors']==263 and m['model_meta']['truth_bits_per_probe']==168 and m['model_meta']['truth_bits_decoder_visible'] is False
assert m['model_meta']['X_recovery_meta']['unique_physical_pauli_mechanisms']==18330
assert m['model_meta']['concrete_pauli_mechanisms_per_patch']==19590
p=J('data/final_results/s21opt5a_preflight.json');fd=p['frozen_decoder']
assert fd['gap']==10.0 and fd['sector_candidates']==8 and fd['bp_max_iter']==30 and fd['bp_method']=='minimum_sum'
assert math.isclose(fd['ms_scaling_factor'],.625) and fd['osd_method']=='osd_cs' and fd['osd_order']==2 and fd['max_correction_effects']==8 and fd['threshold_sweep'] is False
assert p['detector_contract']=={'operational':263,'future_truth':168,'truth_visible':False} and p['full_factory_potential_locations']==5154
with (ROOT/'data/tables/a3_binary_vs_infidelity_4point.csv').open(newline='',encoding='utf-8') as f:r=list(csv.DictReader(f))[0]
assert math.isclose(float(r['bin']),2364837.59616,rel_tol=1e-12)
with (ROOT/'data/tables/matched_ler_binary_resource.csv').open(newline='',encoding='utf-8') as f:rr=list(csv.DictReader(f))
assert [int(float(x['threeT_gap'])) for x in rr]==[25,21,21,21]
assert [float(x['Direct_saving_pct'])>0 for x in rr]==[True,True,False,False]
print('PASS: archive contracts internally consistent')
print('263 operational / 168 future-truth; H rank 263; H_aug rank 265')
print('18,330 mechanisms per probe; 19,590 joint universe')
print('frozen decoder gap10, sectors8, BP30 minimum-sum scale0.625, OSDcs order2, max8')
