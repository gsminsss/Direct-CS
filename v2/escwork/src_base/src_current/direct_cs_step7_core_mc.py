import numpy as np, math, random, collections, sys, csv, time
I2=np.eye(2,dtype=complex)
X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]],complex); Z=np.diag([1,-1]).astype(complex)
H=(X+Z)/np.sqrt(2); S=np.diag([1,1j]).astype(complex); Sd=S.conj().T
def Ry(th): return np.array([[np.cos(th/2),-np.sin(th/2)],[np.sin(th/2),np.cos(th/2)]],complex)
R=Ry(np.pi/4); K=H@S; KR=K@R
def kron(a,b): return np.kron(a,b)
CZ=np.diag([1,1,1,-1]).astype(complex)
CX_AB=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],complex)
CX_BA=np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]],complex)
Hm=(X-Y)/np.sqrt(2)
c1=kron(X,Sd)@CZ; c2=kron(Sd,X)@CZ; c12=kron(Hm,Hm)
p2={'I':I2,'X':X,'Y':Y,'Z':Z}
def op1(g,q): return kron(g,I2) if q==0 else kron(I2,g)
seqs={
'C1':[('1Q',1,K.conj().T),('CX',None,CX_AB),('1Q',0,KR.conj().T),('1Q',1,R.conj().T),('CZZ',None,None),('1Q',0,KR),('1Q',1,R),('CX',None,CX_AB),('1Q',1,K)],
'C2':[('1Q',0,K.conj().T),('CX',None,CX_BA),('1Q',0,R.conj().T),('1Q',1,KR.conj().T),('CZZ',None,None),('1Q',0,R),('1Q',1,KR),('CX',None,CX_BA),('1Q',0,K)],
'C12':[('1Q',0,KR.conj().T),('1Q',1,KR.conj().T),('CZZ',None,None),('1Q',0,KR),('1Q',1,KR)]}
def branch_mat(check,b,fd=None):
    U=np.eye(4,dtype=complex); rf=0
    for gi,(typ,q,g) in enumerate(seqs[check]):
        if typ=='1Q': U=op1(g,q)@U
        elif typ=='CX': U=g@U
        else: U=(np.eye(4) if b==0 else kron(Z,Z))@U
        if fd and gi in fd:
            for F,flip in fd[gi]: U=F@U; rf^=flip
    return U,rf
ideal_u={ck:(branch_mat(ck,0)[0],branch_mat(ck,1)[0]) for ck in seqs}
def opts(ck,gi):
    typ,q,g=seqs[ck][gi]; out=[]
    if typ=='1Q':
        for s in 'XYZ': out.append((op1(p2[s],q),0,s))
    elif typ=='CX':
        for a in 'IXYZ':
            for b in 'IXYZ':
                if a!='I' or b!='I': out.append((kron(p2[a],p2[b]),0,a+b))
    else:
        for anc in 'IXYZ':
            for a in 'IXYZ':
                for b in 'IXYZ':
                    if anc!='I' or a!='I' or b!='I': out.append((kron(p2[a],p2[b]), int(anc in 'ZY'), anc+a+b))
    return out
opt_cache={(ck,i):opts(ck,i) for ck in seqs for i in range(len(seqs[ck]))}
sequence=['C1','C2','C12','C1','C2','C12']
locations=[]
for ci,ck in enumerate(sequence):
    for j in range(7):
        for gi in range(len(seqs[ck])): locations.append((ci,ck,j,gi))
# code states
supports=[{0,1,4,5},{0,2,3,5},{0,2,4,6}]; gens=[sum(1<<q for q in s) for s in supports]
C={0}
for g in gens: C |= {x^g for x in list(C)}
zero=np.zeros(128,complex)
for x in C: zero[x]=1/np.sqrt(8)
one=np.zeros(128,complex)
for x in C: one[x^127]=1/np.sqrt(8)
plus=(zero+one)/np.sqrt(2)
def pidx(xa,xb):
    idx=0
    for j in range(7): idx=idx*4+2*((xa>>j)&1)+((xb>>j)&1)
    return idx
def pair(a,b):
    out=np.zeros(4**7,complex)
    for xa in np.nonzero(abs(a)>1e-15)[0]:
        for xb in np.nonzero(abs(b)>1e-15)[0]: out[pidx(xa,xb)]+=a[xa]*b[xb]
    return out
psi0=pair(plus,plus); B=np.stack([pair(a,b) for a in (zero,one) for b in (zero,one)],axis=1)
CSg=np.diag([1,1,1,1j]).astype(complex); cs=CSg@(np.ones(4)/2)
def target(a,b): return B@(cs*np.array([1,(-1)**b,(-1)**a,(-1)**(a+b)]))
def local4(st,M,j):
    A=np.moveaxis(st.reshape([4]*7),j,0); A=np.tensordot(M,A,axes=(1,0)); return np.moveaxis(A,0,j).reshape(-1)
def prod(st,mats):
    for j,M in enumerate(mats):
        if M is not None and not np.allclose(M,np.eye(4),rtol=0,atol=1e-13): st=local4(st,M,j)
    return st
def proj(st):
    amps=B.conj().T@st; out=B@amps; pr=float(np.vdot(out,out).real)
    if pr>1e-16: out/=np.sqrt(pr)
    return out,pr
def coord(ck,b,fs):
    fd=collections.defaultdict(list); rf=0
    for gi,oi in fs:
        F,f,_=opt_cache[(ck,gi)][oi]; fd[gi].append((F,f)); rf^=f
    U,_=branch_mat(ck,b,fd); return U,rf
def check(st,ck,fbc,rng,q):
    m0=[];m1=[];rf=0
    for j in range(7):
        fs=fbc.get(j,[])
        if fs:
            u0,r0=coord(ck,0,fs);u1,r1=coord(ck,1,fs);rf^=r0
        else:u0,u1=ideal_u[ck]
        m0.append(u0);m1.append(u1)
    b0=prod(st,m0);b1=prod(st,m1);kp=(b0+b1)/2;km=(b0-b1)/2
    p0=float(np.vdot(kp,kp).real);p1=float(np.vdot(km,km).real); s=p0+p1;p0/=s
    r=0 if rng.random()<p0 else 1; st=kp if r==0 else km; st/=np.linalg.norm(st)
    return st, r^rf^(rng.random()<q)
def valid(r): return r[2]==(r[0]^r[1]) and r[3]==r[0] and r[4]==r[1] and r[5]==r[2]
def shot(p,rng):
    fb=[collections.defaultdict(list) for _ in sequence]
    nf=0
    for idx,(ci,ck,j,gi) in enumerate(locations):
        if rng.random()<p:
            nf+=1; fb[ci][j].append((gi,rng.randrange(len(opt_cache[(ck,gi)]))))
    q=(1-(1-2*p)**7)/2; st=psi0.copy(); rec=[]
    for ci,ck in enumerate(sequence):
        st,m=check(st,ck,fb[ci],rng,q);rec.append(int(m))
        if ci==2:
            st,pr=proj(st)
            if rng.random()>=pr:return False,0.,'mid',nf
    if not valid(rec): return False,0.,'record',nf
    st,pr=proj(st)
    if rng.random()>=pr:return False,0.,'final',nf
    F=abs(np.vdot(target(rec[0],rec[1]),st))**2
    return True,max(0.,1-F),'acc',nf

def run(p,n,seed):
    rng=random.Random(seed);acc=0;pl=0.;why=collections.Counter();nf=collections.Counter()
    for _ in range(n):
        ok,e,w,k=shot(p,rng);acc+=ok;pl+=e if ok else 0.;why[w]+=1;nf[k]+=1
    return dict(p=p,n=n,accept=acc/n,cond_PL=(pl/acc if acc else 0),reasons=dict(why),fault_count=dict(nf))
if __name__=='__main__':
    n=int(sys.argv[1]) if len(sys.argv)>1 else 1000
    ps=[float(x) for x in (sys.argv[2:] if len(sys.argv)>2 else ['0.0005','0.001','0.002'])]
    out=[];t=time.time()
    for i,p in enumerate(ps):
        r=run(p,n,12345+i);out.append(r);print(r,flush=True)
    print('elapsed',time.time()-t)
    from pathlib import Path
    out_path=Path(__file__).resolve().parent/'direct_cs_step7_core_mc.csv'
    with open(out_path,'w',newline='') as f:
        w=csv.writer(f);w.writerow(['p','shots','accept','cond_PL','reasons','fault_count'])
        for r in out:w.writerow([r['p'],r['n'],r['accept'],r['cond_PL'],r['reasons'],r['fault_count']])
