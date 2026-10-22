S16 reproducibility notes
=========================

Core exact enumerator:
  g++ -O3 -std=c++20 src/direct_cs_s16_order2_exact.cpp -o s16_order2_exact

The signature-builder source documents the corrected raw-commutation convention.
It expects the preserved project source and S15 paths used in this project bundle.

Expected exact-enumerator result:
  signatures = 7529
  records = 28346686
  ambiguous_exact_keys = 0
  hash_groups_mixed_frame = 0

Scientific decoder key:
  live171 + growth131 + final d13 X/Z syndrome

Do not compare the old S12 recovery-frame label directly to raw logical
commutation for nonzero-syndrome events. That was the preliminary S16 bug.
