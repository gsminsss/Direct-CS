
from dataclasses import dataclass
from collections import Counter
import itertools

# ============================================================
# Direct-|CS> Step-2 reproducibility checker
#
# Scope:
#   * encoded-input [[7,1,3]]x[[7,1,3]] cultivation circuit
#   * verified CAT_7 preparation/verification
#   * proof-first midpoint full Steane syndrome detector
#   * adversarial Pauli-after-location fault enumeration
#   * explicit two-late-fault obstruction to standalone core d_f=3
#
# Not included:
#   * encoded |+> preparation
#   * idling/leakage/coherent noise
#   * Steane->surface escape
# ============================================================

@dataclass(frozen=True)
class Gate:
    name: str
    qs: tuple

@dataclass(frozen=True)
class Op:
    typ: str
    qs: tuple
    basis: str | None = None
    label: str | None = None

def cnot_prop(x, z, c, t):
    # Forward conjugation through CNOT.
    # X_c -> X_c X_t ; Z_t -> Z_c Z_t
    if (x >> c) & 1:
        x ^= 1 << t
    if (z >> t) & 1:
        z ^= 1 << c
    return x, z

# ------------------------------------------------------------
# 1) Verified CAT_7
# ------------------------------------------------------------
# Balanced-tree B7:
#   L1: 0->4
#   L2: 0->2, 4->6
#   L3: 0->1, 2->3, 4->5
#
# Ancilla CAT_4:
#   L1: 7->9
#   L2: 7->8, 9->10
#
# Peham-style partial-transversal verification:
#   0->7, 2->9, 4->8, 5->10
# Measure 7,8,9,10 in Z and accept iff all outcomes are equal.
CAT7_GATES = [
    (0,4),(0,2),(4,6),(0,1),(2,3),(4,5),
    (7,9),(7,8),(9,10),
    (0,7),(2,9),(4,8),(5,10),
]
CAT7_W = 7
CAT7_WP = 4
CAT7_NQ = 11

def cat7_propagate_fault(x, z, start_gate):
    for c,t in CAT7_GATES[start_gate:]:
        x,z = cnot_prop(x,z,c,t)
    return x,z

# Physical locations.
# Prep: adversarial X/Y/Z after preparation.
# CX: all 15 non-identity two-qubit Paulis after the gate.
# Measurement: classical bit flip.
CAT7_LOCATIONS = []
for q in range(CAT7_NQ):
    opts=[]
    for p in "XYZ":
        x=(1<<q) if p in "XY" else 0
        z=(1<<q) if p in "YZ" else 0
        x,z=cat7_propagate_fault(x,z,0)
        opts.append((x,z,0,p))
    CAT7_LOCATIONS.append((f"prep{q}",opts))

for gi,(c,t) in enumerate(CAT7_GATES):
    opts=[]
    for pc in "IXYZ":
        for pt in "IXYZ":
            if pc=="I" and pt=="I":
                continue
            x=z=0
            if pc in "XY": x |= 1<<c
            if pc in "YZ": z |= 1<<c
            if pt in "XY": x |= 1<<t
            if pt in "YZ": z |= 1<<t
            x,z=cat7_propagate_fault(x,z,gi+1)
            opts.append((x,z,0,pc+pt))
    CAT7_LOCATIONS.append((f"cx{gi}",opts))

for i,q in enumerate(range(7,11)):
    CAT7_LOCATIONS.append((f"mz{q}",[(0,0,1<<i,"M")]))

def cat7_eval(faults):
    # faults = [(location_index, option_index), ...]
    x=z=mf=0
    for li,oi in faults:
        a,b,c,_ = CAT7_LOCATIONS[li][1][oi]
        x ^= a; z ^= b; mf ^= c
    data_x = x & 0x7f
    data_z = z & 0x7f
    verifier_x = (x >> 7) & 0xf
    measured = verifier_x ^ mf
    accept = measured in (0, 0xf)
    x_weight = min(data_x.bit_count(), 7-data_x.bit_count())
    phase_flip = data_z.bit_count() & 1
    return accept, x_weight, phase_flip

cat_single_total=cat_single_acc=cat_single_bad=0
cat_single_dist=Counter()
for li,(_,opts) in enumerate(CAT7_LOCATIONS):
    for oi in range(len(opts)):
        cat_single_total += 1
        a,w,p = cat7_eval([(li,oi)])
        if a:
            cat_single_acc += 1
            cat_single_dist[(w,p)] += 1
            if w > 1:
                cat_single_bad += 1

cat_pair_total=cat_pair_acc=cat_pair_bad=0
cat_pair_dist=Counter()
for i in range(len(CAT7_LOCATIONS)):
    for j in range(i+1,len(CAT7_LOCATIONS)):
        for oi in range(len(CAT7_LOCATIONS[i][1])):
            for oj in range(len(CAT7_LOCATIONS[j][1])):
                cat_pair_total += 1
                a,w,p = cat7_eval([(i,oi),(j,oj)])
                if a:
                    cat_pair_acc += 1
                    cat_pair_dist[(w,p)] += 1
                    if w > 2:
                        cat_pair_bad += 1

# ------------------------------------------------------------
# 2) Proof-first midpoint detector
# ------------------------------------------------------------
# One block:
#   (i)  three Hirano-style superdense Z-stabilizer gadgets
#   (ii) transversal H on all 7 data
#   (iii) repeat the same three gadgets
#   (iv) transversal H back
#
# This measures both Z and X Steane stabilizers while returning
# the logical data to the original frame.
def z_snapshot_ops(anc0=7,prefix="Z"):
    ops=[]
    aidx=anc0
    # support 0145
    a,b=aidx,aidx+1; aidx+=2
    ops += [
        Op("PREP",(a,),"X",prefix+"0145_pa"),
        Op("PREP",(b,),"Z",prefix+"0145_pb"),
        Op("CX",(a,b),label=prefix+"0145_ent"),
        Op("CX",(1,b),label=prefix+"0145_d1"),
        Op("CX",(4,a),label=prefix+"0145_d4"),
        Op("CX",(5,b),label=prefix+"0145_d5"),
        Op("CX",(0,b),label=prefix+"0145_d0"),
        Op("CX",(a,b),label=prefix+"0145_dis"),
        Op("MEAS",(a,),"X",prefix+"0145_flag"),
        Op("MEAS",(b,),"Z",prefix+"0145_syn"),
    ]
    # support 0235
    a,b=aidx,aidx+1; aidx+=2
    ops += [
        Op("PREP",(a,),"X",prefix+"0235_pa"),
        Op("PREP",(b,),"Z",prefix+"0235_pb"),
        Op("CX",(a,b),label=prefix+"0235_ent"),
        Op("CX",(5,b),label=prefix+"0235_d5"),
        Op("CX",(3,b),label=prefix+"0235_d3"),
        Op("CX",(0,b),label=prefix+"0235_d0"),
        Op("CX",(2,a),label=prefix+"0235_d2"),
        Op("CX",(a,b),label=prefix+"0235_dis"),
        Op("MEAS",(a,),"X",prefix+"0235_flag"),
        Op("MEAS",(b,),"Z",prefix+"0235_syn"),
    ]
    # support 0246
    a,b,c=aidx,aidx+1,aidx+2; aidx+=3
    ops += [
        Op("PREP",(a,),"Z",prefix+"0246_pa"),
        Op("PREP",(b,),"X",prefix+"0246_pb"),
        Op("PREP",(c,),"X",prefix+"0246_pc"),
        Op("CX",(b,a),label=prefix+"0246_entb"),
        Op("CX",(c,a),label=prefix+"0246_entc"),
        Op("CX",(0,a),label=prefix+"0246_d0"),
        Op("CX",(4,b),label=prefix+"0246_d4"),
        Op("CX",(2,c),label=prefix+"0246_d2"),
        Op("CX",(6,b),label=prefix+"0246_d6"),
        Op("CX",(b,a),label=prefix+"0246_disb"),
        Op("CX",(c,a),label=prefix+"0246_disc"),
        Op("MEAS",(a,),"Z",prefix+"0246_syn"),
        Op("MEAS",(b,),"X",prefix+"0246_flagb"),
        Op("MEAS",(c,),"X",prefix+"0246_flagc"),
    ]
    return ops,aidx

ED_OPS,next_anc = z_snapshot_ops(7,"Z")
for q in range(7):
    ED_OPS.append(Op("H",(q,),label=f"H1_{q}"))
second,next_anc = z_snapshot_ops(next_anc,"XviaH")
ED_OPS += second
for q in range(7):
    ED_OPS.append(Op("H",(q,),label=f"H2_{q}"))

def pauli_options(op):
    if op.typ in ("PREP","H"):
        q=op.qs[0]
        out=[]
        for p in "XYZ":
            x=(1<<q) if p in "XY" else 0
            z=(1<<q) if p in "YZ" else 0
            out.append((x,z,p))
        return out
    if op.typ=="CX":
        c,t=op.qs
        out=[]
        for pc in "IXYZ":
            for pt in "IXYZ":
                if pc=="I" and pt=="I":
                    continue
                x=z=0
                if pc in "XY": x |= 1<<c
                if pc in "YZ": z |= 1<<c
                if pt in "XY": x |= 1<<t
                if pt in "YZ": z |= 1<<t
                out.append((x,z,pc+pt))
        return out
    if op.typ=="MEAS":
        return [(0,0,"M")]
    raise ValueError(op.typ)

ED_OPTIONS=[pauli_options(o) for o in ED_OPS]

STAB_SUPPORTS=[{0,1,4,5},{0,2,3,5},{0,2,4,6}]
STAB_GENS=[]
for s in STAB_SUPPORTS:
    m=sum(1<<q for q in s)
    STAB_GENS.append((m,0))
for s in STAB_SUPPORTS:
    m=sum(1<<q for q in s)
    STAB_GENS.append((0,m))

STAB_GROUP=[(0,0)]
for gx,gz in STAB_GENS:
    old=list(STAB_GROUP)
    STAB_GROUP += [(x^gx,z^gz) for x,z in old]

def code_min_weight(x,z):
    return min(((x^sx)|(z^sz)).bit_count() for sx,sz in STAB_GROUP)

def run_ed(faults, input_pauli=(0,0)):
    x,z=input_pauli
    clicks=[]
    for idx,op in enumerate(ED_OPS):
        if op.typ=="PREP":
            q=op.qs[0]
            x &= ~(1<<q); z &= ~(1<<q)
            if idx in faults:
                fx,fz,_=ED_OPTIONS[idx][faults[idx]]
                x ^= fx; z ^= fz
        elif op.typ=="H":
            q=op.qs[0]
            xb=(x>>q)&1
            zb=(z>>q)&1
            if xb != zb:
                x ^= 1<<q
                z ^= 1<<q
            if idx in faults:
                fx,fz,_=ED_OPTIONS[idx][faults[idx]]
                x ^= fx; z ^= fz
        elif op.typ=="CX":
            x,z=cnot_prop(x,z,*op.qs)
            if idx in faults:
                fx,fz,_=ED_OPTIONS[idx][faults[idx]]
                x ^= fx; z ^= fz
        elif op.typ=="MEAS":
            q=op.qs[0]
            flip=((x>>q)&1) if op.basis=="Z" else ((z>>q)&1)
            if idx in faults:
                flip ^= 1
            clicks.append(flip)
            x &= ~(1<<q); z &= ~(1<<q)
    x &= 0x7f; z &= 0x7f
    return all(v==0 for v in clicks), code_min_weight(x,z), x, z

# Check every incoming weight-1 Pauli is detected fault-free.
incoming_undetected=[]
for p in "XYZ":
    for q in range(7):
        x=(1<<q) if p in "XY" else 0
        z=(1<<q) if p in "YZ" else 0
        if run_ed({},(x,z))[0]:
            incoming_undetected.append((p,q))

ed_single_total=sum(len(x) for x in ED_OPTIONS)
ed_single_acc=ed_single_bad=0
ed_single_dist=Counter()
for i,opts in enumerate(ED_OPTIONS):
    for oi in range(len(opts)):
        a,w,_,_=run_ed({i:oi})
        if a:
            ed_single_acc += 1
            ed_single_dist[w] += 1
            if w>1:
                ed_single_bad += 1

ed_pair_total=ed_pair_acc=ed_pair_bad=0
ed_pair_dist=Counter()
for i in range(len(ED_OPTIONS)):
    for j in range(i+1,len(ED_OPTIONS)):
        for oi in range(len(ED_OPTIONS[i])):
            for oj in range(len(ED_OPTIONS[j])):
                ed_pair_total += 1
                a,w,_,_=run_ed({i:oi,j:oj})
                if a:
                    ed_pair_acc += 1
                    ed_pair_dist[w] += 1
                    if w>2:
                        ed_pair_bad += 1

cross_acc=cross_bad=0
cross_dist=Counter()
for p in "XYZ":
    for q in range(7):
        ix=(1<<q) if p in "XY" else 0
        iz=(1<<q) if p in "YZ" else 0
        for i,opts in enumerate(ED_OPTIONS):
            for oi in range(len(opts)):
                a,w,_,_=run_ed({i:oi},(ix,iz))
                if a:
                    cross_acc += 1
                    cross_dist[w] += 1
                    if w>2:
                        cross_bad += 1

# ------------------------------------------------------------
# 3) Explicit late-two-fault obstruction
# ------------------------------------------------------------
def z_syndrome(q):
    return tuple(int(q in s) for s in STAB_SUPPORTS)

s1=z_syndrome(1)
s5=z_syndrome(5)
s3=z_syndrome(3)
pair_syn=tuple(a^b for a,b in zip(s1,s5))

# Decoder sees Z1 Z5 as the same syndrome as Z3, so a minimum-weight
# correction Z3 leaves Z1 Z3 Z5.  This has zero stabilizer syndrome and
# odd overlap with logical X=X^7, hence is a logical Z representative.
residual={1,3,5}
residual_syn=tuple(
    sum(int(q in s) for q in residual) & 1
    for s in STAB_SUPPORTS
)
logical_x_overlap=len(residual)&1

print("=== DIRECT-CS STEP 2 CHECKER ===")
print()
print("[CAT_7]")
print("physical locations:",len(CAT7_LOCATIONS))
print("single Pauli cases:",cat_single_total,
      "accepted:",cat_single_acc,
      "accepted with X-support >1:",cat_single_bad)
print("accepted single distribution (X-weight, phase-parity):",dict(cat_single_dist))
print("two-fault Pauli cases:",cat_pair_total,
      "accepted:",cat_pair_acc,
      "accepted with X-support >2:",cat_pair_bad)
print("accepted pair distribution (X-weight, phase-parity):",dict(cat_pair_dist))
print()
print("[MIDPOINT FULL STEANE ED: one block]")
print("physical locations:",len(ED_OPS),
      dict(Counter(o.typ for o in ED_OPS)))
print("incoming weight-1 Paulis undetected (fault-free):",incoming_undetected)
print("single Pauli cases:",ed_single_total,
      "accepted:",ed_single_acc,
      "accepted residual code-weight >1:",ed_single_bad)
print("accepted single code-weight distribution:",dict(ed_single_dist))
print("two-fault Pauli cases:",ed_pair_total,
      "accepted:",ed_pair_acc,
      "accepted residual code-weight >2:",ed_pair_bad)
print("accepted pair code-weight distribution:",dict(ed_pair_dist))
print("incoming weight-1 + one ED fault: accepted:",cross_acc,
      "residual code-weight >2:",cross_bad)
print("cross distribution:",dict(cross_dist))
print()
print("[LATE TWO-FAULT COUNTEREXAMPLE]")
print("s(Z1) =",s1)
print("s(Z5) =",s5)
print("s(Z1 Z5) =",pair_syn,"= s(Z3) =",s3)
print("minimum-weight correction Z3 leaves Z1 Z3 Z5")
print("residual stabilizer syndrome:",residual_syn)
print("odd overlap with logical X^7:",bool(logical_x_overlap))
print("=> logical-Z failure after ideal minimum-weight decoding.")
print()
print("[ENCODED-INPUT CULTIVATION LOCATION COUNT, through midpoint only]")
print("logical checks: 35 CZZ + 56 CX wrappers + 196 composite 1Q")
print("5 CAT_7 preparations: 65 CX + 55 prep + 20 verifier measurements")
print("5 CAT readouts: 35 X measurements")
print("midpoint full ED on two blocks: 80 CX + 28 H + 28 prep + 28 measurement")
print("TOTAL: 35 CZZ + 201 CX + 224 1Q + 83 prep + 83 measurement = 626 locations")
print()
print("Conclusion: CAT_7 and midpoint ED satisfy the required nested local")
print("support contracts, but the standalone d=3 cultivation core cannot have")
print("fault distance 3 because two late data-only faults after the final C2")
print("check can be accepted and decode to a logical error.  The escape must be")
print("part of the certified gadget.")
