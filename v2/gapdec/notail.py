"""Direct-convention boundary for the 3T circuit.
notail0 : ideal tail round kept for evaluation only (withheld from decoder and gap).
notail1 : one NOISY d13 round inserted (visible, charged), then the ideal round
          (withheld), mirroring Direct's single faultable d13 round + ideal continuation."""
import re, numpy as np

def _split(L):
    # tail block = from the first line after the last noisy TICK-group (the H on ancillas
    # that follows line 'DEPOLARIZE1' at the final growth tick) up to SHIFT_COORDS before MY
    i_my = max(i for i, l in enumerate(L) if l.startswith("MY "))
    i_shift = max(i for i in range(i_my) if L[i].startswith("SHIFT_COORDS"))
    i_mr = max(i for i in range(i_shift) if L[i].startswith("MR "))
    # walk back from MR to the start of the ideal round: first 'H' after the last noisy line
    last_noise = max(i for i in range(i_mr) if re.match(r"^(DEPOLARIZE|X_ERROR|Z_ERROR)", L[i]))
    i_start = last_noise + 1
    while not L[i_start].startswith(("H ", "TICK")): i_start += 1
    if L[i_start].startswith("TICK"): i_start += 1
    return i_start, i_mr, i_shift, i_my

def noisy_round(block, p, idle, data):
    # preparation error at the start of the round (its own reset) and a
    # measurement flip before MR; the reset performed by MR belongs to the
    # following ideal round and stays noiseless.
    anc = [l for l in block if l.startswith("MR ")][0].split()[1:]
    out = [f"X_ERROR({p}) " + " ".join(anc)]
    for l in block:
        nm = l.split()[0] if l.strip() else ""
        if nm == "MR":
            out.append(f"X_ERROR({p}) " + " ".join(anc)); out.append(l)
            continue
        out.append(l)
        if nm == "H":
            out.append(f"DEPOLARIZE1({p}) " + " ".join(l.split()[1:]))
        elif nm == "CX":
            out.append(f"DEPOLARIZE2({p}) " + " ".join(l.split()[1:]))
            if idle:
                busy = set(l.split()[1:]); idl = [q for q in data if q not in busy]
                if idl: out.append(f"DEPOLARIZE1({p}) " + " ".join(idl))
    return out

def transform(L, p, idle, extra_round):
    i0, i_mr, i_shift, i_my = _split(L)
    ops = L[i0:i_mr + 1]; dets = L[i_mr + 1:i_shift + 1]
    nmeas = len(L[i_mr].split()) - 1
    ndet = sum(1 for l in dets if l.startswith("DETECTOR"))
    assert nmeas == ndet == 168, (nmeas, ndet)
    if not extra_round:
        return L, ndet
    anc = set(L[i_mr].split()[1:])
    data = sorted({q for l in ops if l.startswith("CX") for q in l.split()[1:]} - anc, key=int)
    ideal_dets = []
    for l in dets:
        if l.startswith("DETECTOR"):
            head = l.split(")")[0] + ")"
            k = int(re.search(r"rec\[-(\d+)\]", l).group(1)); assert k <= 168
            ideal_dets.append(f"{head} rec[-{k}] rec[-{k + 168}]")
        else:
            ideal_dets.append(l)
    new = L[:i0] + noisy_round(ops, p, idle, data) + dets + ["TICK"] + ops + ideal_dets + L[i_shift + 1:]
    return new, ndet

def visible(ndet_total, n_withheld):
    v = np.ones(ndet_total, bool); v[ndet_total - n_withheld:] = False
    return v
