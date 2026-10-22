"""Create a generalized copy of OPT3B that supports escape distance D_ESC in {3,5}.
The original file is not modified.  Each edit is an exact, asserted replacement."""
import sys
src = open(sys.argv[1]).read()
def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (src.count(old), old[:70])
    src = src.replace(old, new)

rep("WAIT_ROUNDS = 2\n", "WAIT_ROUNDS = 2\nD_ESC = 5\nD_FINAL = 13\n\n"
    "def _make_mapping(util):\n"
    "    m = util.QubitMapping(30, 30)\n"
    "    if D_ESC == 5:\n        return m\n"
    "    # Append extra coordinates above the grid (negative y) without changing existing ids.\n"
    "    nid = len(m.mapping)\n"
    "    ylo = 17 - 2 * (D_FINAL - D_ESC) - 2\n"
    "    for x in range(30):\n"
    "        for y in range(ylo, 0):\n"
    "            if (x % 2 == 0 and y % 2 == 0) or (x % 2 == 1 and y % 2 != 0):\n"
    "                m.mapping.append((nid, (x, y))); nid += 1\n"
    "    return m\n")
# runner
rep("""    mapping = util.QubitMapping(30, 30)
    initial_value = lsed.InitialValue.Plus if channel == "X" else lsed.InitialValue.Zero
    runner = lsed.SteanePlusSurfaceCode(
        mapping, 5, initial_value, lsed.SteaneSyndromeExtractionPattern.ZXZ, 0.0, True
    )""", """    mapping = _make_mapping(util)
    initial_value = lsed.InitialValue.Plus if channel == "X" else lsed.InitialValue.Zero
    runner = lsed.SteanePlusSurfaceCode(
        mapping, D_ESC, initial_value, lsed.SteaneSyndromeExtractionPattern.ZXZ, 0.0, True
    )""")
# replace canonical cross-check with an independent full run of the same generator (WAIT_ROUNDS must be 0 for d!=5)
rep("""    # Cross-check against the historical canonical prefix builder.
    full, _wrapper, init_range2, probe_start2, online_ids, terminal_ids = s7b._build_with_wait_rounds(
        hirano_src, channel, WAIT_ROUNDS
    )
    canonical_pre = full[:probe_start2]
    if str(canonical_pre) != str(pre):
        raise RuntimeError("retained-runner prefix differs from canonical S17/S18 Hirano prefix")
    if tuple(rec["init_range"]) != tuple(init_range2):
        raise RuntimeError(("init range drift", rec["init_range"], init_range2))
    if len(online_ids) != EXPECTED_HIRANO_LIVE_DETECTORS or len(terminal_ids) != 12:
        raise AssertionError((len(online_ids), len(terminal_ids)))
""", """    if D_ESC == 5:
        full, _wrapper, init_range2, probe_start2, online_ids, terminal_ids = s7b._build_with_wait_rounds(
            hirano_src, channel, WAIT_ROUNDS
        )
        canonical_pre = full[:probe_start2]
        if str(canonical_pre) != str(pre):
            raise RuntimeError("retained-runner prefix differs from canonical S17/S18 Hirano prefix")
        if tuple(rec["init_range"]) != tuple(init_range2):
            raise RuntimeError(("init range drift", rec["init_range"], init_range2))
        if len(online_ids) != EXPECTED_HIRANO_LIVE_DETECTORS or len(terminal_ids) != 12:
            raise AssertionError((len(online_ids), len(terminal_ids)))
    else:
        if WAIT_ROUNDS != 0:
            raise AssertionError("d!=5 path requires WAIT_ROUNDS=0 for the full-run cross-check")
        r2 = lsed.SteanePlusSurfaceCode(
            _make_mapping(util), D_ESC, initial_value, lsed.SteaneSyndromeExtractionPattern.ZXZ, 0.0, True
        )
        r2.run()
        full = r2.circuit.circuit
        if str(full[:len(pre)]) != str(pre):
            raise RuntimeError("full independent run prefix differs from stopped live prefix")
""")
rep("""    if not hirano_pre or len(hirano_term) != 5:""", """    if not hirano_pre or len(hirano_term) != D_ESC:""")
# splice contract
rep("""    mapping = util.QubitMapping(30, 30)
    refc = util.Circuit(mapping, 0.0)
    ref = sce.SurfaceCodePatch(refc, 5, 13, 1, sce.ExpansionPattern.UPWARD)
    if tuple(ref.offset) != (1, 17):
        raise AssertionError(("unexpected UPWARD initial offset", ref.offset))
    ref._setup_initial_state()

    live = runner.surface_syndrome_measurements
    if len(live) != 24 or len(ref.syndrome_measurements) != 24:""", """    mapping = _make_mapping(util)
    refc = util.Circuit(mapping, 0.0)
    ref = sce.SurfaceCodePatch(refc, D_ESC, D_FINAL, 1, sce.ExpansionPattern.UPWARD)
    ref.offset = (1, 17)
    ref._setup_initial_state()

    live = runner.surface_syndrome_measurements
    if len(live) != D_ESC * D_ESC - 1 or len(ref.syndrome_measurements) != D_ESC * D_ESC - 1:""")
rep("""    expected_data = {(1 + 2*j, 17 + 2*i) for i in range(5) for j in range(5)}""",
    """    expected_data = {(1 + 2*j, 17 + 2*i) for i in range(D_ESC) for j in range(D_ESC)}""")
rep("""        "expected_live_data_coordinates": 25,""", """        "expected_live_data_coordinates": D_ESC * D_ESC,""")
# expansion
rep("""    patch = sce.SurfaceCodePatch(
        runner.circuit, 5, 13, 1, sce.ExpansionPattern.UPWARD
    )
    if tuple(patch.offset) != (1, 17):
        raise AssertionError(patch.offset)
""", """    patch = sce.SurfaceCodePatch(
        runner.circuit, D_ESC, D_FINAL, 1, sce.ExpansionPattern.UPWARD
    )
    patch.offset = (1, 17)
""")
rep("""    if tuple(patch.offset) != (1, 1):
        raise AssertionError(("final offset", patch.offset))""", """    fy = 17 - 2 * (D_FINAL - D_ESC)
    if tuple(patch.offset) != (1, fy):
        raise AssertionError(("final offset", patch.offset))""")
rep("""            q = runner.circuit.mapping.get_id(1, 1 + 2*i)""", """            q = runner.circuit.mapping.get_id(1, fy + 2*i)""")
rep("""            q = runner.circuit.mapping.get_id(1 + 2*j, 25)""", """            q = runner.circuit.mapping.get_id(1 + 2*j, fy + 24)""")
open(sys.argv[2], "w").write(src)
print("ok")
