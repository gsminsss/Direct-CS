import json,sys,numpy as np
r=json.load(open(sys.argv[1])); pub=json.load(open(sys.argv[2]))
I=np.array(r['I']);X=np.array(r['X']); kept0=I.sum()
pubd={g:(k,e,l,a) for g,k,e,l,a in pub}
print(f"discard mine {r['discards']/r['shots']:.4f} pub {1-1/pubd[0][3]:.4f}")
for g in (0,2,4,6,8,10,12,15,18,21,25,30):
    k=I[g:].sum(); e=X[g:].sum()
    print(f"gap {g:2d}: kept frac mine {k/kept0:.4f} pub {pubd[g][0]/pubd[0][0]:.4f} | LER mine {e/max(k,1):.2e} ({e}) pub {pubd[g][2]:.2e}")
