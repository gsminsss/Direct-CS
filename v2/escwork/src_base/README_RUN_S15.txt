S15 reproduces the low-order appended-growth audit without requiring stim/pymatching.

Expected preserved Sahay source:
  reference/fd3_baseline_ckt.stim
SHA-256:
  6fc1dd197569183db8283fa403283a4fceb6bff6a0272acd5ed105c655541047

The scripts in src/ currently point to the project-local preserved source path used in this run.
For a portable rerun, edit SRC at the top of direct_cs_s15_single_fault_audit.py.

Run order:
  python src/direct_cs_s15_single_fault_audit.py > results/s15_single_fault_audit.json
  python src/direct_cs_s15_order2_signature_audit.py
  python src/direct_cs_s15_mixed_boundary_order2.py
  python src/direct_cs_s15_growth_finite_p_mc.py
  python src/direct_cs_s15_growth_order3_decoder_mc_p1e3.py

Important:
- The order<=2 certificate is exact under the stated Clifford/local-Pauli model.
- The finite-p higher-order upper bracket treats unresolved syndromes pessimistically.
- Do not quote the 9.28e-3 upper bracket as an estimated logical error rate.
