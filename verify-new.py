import sympy as sp, mpmath as mp
mp.mp.dps = 30
t, r = sp.symbols('t r', positive=True)
M, k, Q = sp.symbols('M k Q', real=True)
c1, c2, c3 = 0.31, -0.07, 0.02
a = 1 + c1*t + c2*t**2 + c3*t**3
H = sp.diff(a, t)/a
Hd = sp.diff(H, t)

def build(Qv):
    w = k*r**2
    nu = sp.sqrt(1+w)/(2*a*r)
    mu, lam = M*nu, Qv*nu
    Xi = (1+mu)**2 - lam**2
    N = (1 - mu**2 + lam**2)/Xi        # new form of lapse
    B = a*Xi/(1+w)
    R = B*r
    Y = (N-w)/(1+w)
    return w, mu, lam, Xi, N, B, R, Y

def num(e, s): return mp.mpf(str(sp.N(e.subs(s), 30)))

print("=== 1. kappa_A formulas on marginal roots (neutral and charged)")
for (Mv, kv, Qv, t0) in [(0.3, 0.05, 0, 0.7), (0.25, -0.004, 0, 1.1), (0.3, 0.05, 0.12, 0.7), (0.2, -0.003, 0.1, 0.4)]:
    w, mu, lam, Xi, N, B, R, Y = build(Qv)
    s = {M: Mv, k: kv, t: t0}
    G = sp.lambdify(r, (Y - H*R).subs(s), 'mpmath'); Nf = sp.lambdify(r, N.subs(s), 'mpmath')
    rL = mp.findroot(lambda x: Nf(x), 0.4*Mv)
    xs = [rL*(1+1e-6) + i*0.002 for i in range(3000)]; vals = [G(x) for x in xs]
    roots = [mp.findroot(G, (xs[i], xs[i+1]), solver='bisect') for i in range(len(xs)-1) if vals[i]*vals[i+1] < 0]
    for rA in roots[:2]:
        sA = dict(s); sA[r] = rA
        kap_new = num(sp.Rational(1, 2)*(sp.diff(Y, t)/N + sp.diff(Y, r)/B - R*(Hd/N + 2*H**2)), sA)
        kap_D5 = num((1/(2*N*B))*sp.diff(N*(N-w)/(1+w), r) - H**2*R - R/(2*N)*Hd, sA)
        line = f"  Q={Qv} k={kv}: kappa(new)={float(kap_new):+.10f} kappa(D5)={float(kap_D5):+.10f}"
        if Qv == 0:
            NA, wA, RA, muA = num(N, sA), num(w, sA), num(R, sA), num(mu, sA)
            aA, HdA = num(a, sA), num(Hd, sA)
            kap_cf = (2 + wA*(5*NA - 4*wA - 4) - 6*NA**2)/(4*RA*(1+wA)**2) - RA/(2*NA)*(HdA + kv/(2*aA**2*(1+muA)**4))
            line += f" closed form={float(kap_cf):+.10f}"
        print(line)

print("\n=== 2. merger identity: 8 pi rho R^2 + q^2/R^2 = 1 and M_W = R/3 + 2 q^2/(3 R)")
def merger_case(Mv, kv, Qv, av=1.0):
    # static time slice, constant a; unknowns r_A and H
    rr = sp.symbols('rr', positive=True)
    ww = kv*rr**2; nu_ = sp.sqrt(1+ww)/(2*av*rr); mu_, lam_ = Mv*nu_, Qv*nu_
    Xi_ = (1+mu_)**2 - lam_**2; N_ = (1 - mu_**2 + lam_**2)/Xi_
    B_ = av*Xi_/(1+ww); R_ = B_*rr; Y_ = (N_-ww)/(1+ww)
    F = sp.lambdify(rr, sp.diff(Y_, rr)*R_ - Y_*sp.diff(R_, rr), 'mpmath')
    Nf = sp.lambdify(rr, N_, 'mpmath')
    rL = mp.findroot(lambda x: Nf(x), 0.4*Mv)
    xs = [rL*(1+1e-4) + i*0.001 for i in range(6000)]
    fv = [F(x) for x in xs]
    out = []
    for i in range(len(xs)-1):
        if fv[i]*fv[i+1] < 0:
            rA = mp.findroot(F, (xs[i], xs[i+1]), solver='bisect')
            RA = sp.lambdify(rr, R_, 'mpmath')(rA); YA = sp.lambdify(rr, Y_, 'mpmath')(rA)
            HA = YA/RA
            if HA <= 0: continue
            wA = kv*rA**2; nuA = mp.sqrt(1+wA)/(2*av*rA); muA, lamA = Mv*nuA, Qv*nuA
            XiA = (1+muA)**2 - lamA**2
            rho8 = 3*HA**2 + 12*kv*(1+muA)/(av**2*XiA**3)
            qA = Qv/(1+wA)**1.5
            MW = Mv/(1+wA)**2.5
            out.append((float(rA), float(HA), float(rho8*RA**2 + qA**2/RA**2), float(MW - (RA/3 + 2*qA**2/(3*RA))), float(Mv*HA)))
    return out
for (Mv, kv, Qv) in [(0.3, 0.02, 0.0), (0.3, -0.004, 0.0), (0.3, 0.02, 0.15), (0.25, -0.003, 0.2), (0.3, 0.0, 0.2)]:
    res = merger_case(Mv, kv, Qv)
    print(f"  M={Mv} k={kv} Q={Qv}: (r_A, H, 8pi rho R^2+q^2/R^2, M_W-R/3-2q^2/3R, MH) =", [tuple(round(x, 8) for x in rr_) for rr_ in res])

print("\n=== 3. uniqueness of merger position for Omega_k <= 1")
def mum(w): return (2-2*w-w**2 - mp.sqrt(3-12*w-6*w**2))/(1+w)**2
def YA(w): m = mum(w); return ((1-w)-(1+w)*m)/((1+w)*(1+m))
def McH(w): return (1+w)**mp.mpf(2.5)/3*YA(w)
def ell(w): m = mum(w); return w*m**2/(1+w)
def Omk(w): return -16*ell(w)/McH(w)**2
ws = [mp.mpf(x)/1000 for x in range(-1000+1, 200)]
Om = [Omk(w) for w in ws]
mono_ok = all(Om[i+1] < Om[i] for i in range(len(ws)-1) if ws[i] > -0.0523)
print("  Omega_k strictly decreasing for w in (-0.0523, 0.2):", mono_ok)
print("  min Omega_k over w<-0.0523:", float(min(Om[i] for i in range(len(ws)) if ws[i] < -0.0523)))
print("  Omega_k at w=-0.9999:", float(Omk(mp.mpf('-0.9999'))))

print("\n=== 4. constant-H smaller sphere: leading-order kappa, pressure and Gamma")
def smaller_sphere(eps, ell_, Mv=1.0, av=1.0):
    kv = 4*ell_*av**2/Mv**2; Hv = eps/Mv
    def parts(rr):
        w = kv*rr**2; mu = Mv*mp.sqrt(1+w)/(2*av*rr); N = (1-mu)/(1+mu)
        R = av*rr*(1+mu)**2/(1+w); Y = (N-w)/(1+w)
        return w, mu, N, R, Y
    rL = Mv/mp.sqrt(4*av**2 - Mv**2*kv)
    G = lambda rr: parts(rr)[4] - Hv*parts(rr)[3]
    xs = [rL*(1+1e-9)*(1 + 0.00002*i) for i in range(1, 60000)]
    prev = G(xs[0]); root = None
    for i in range(1, len(xs)):
        cur = G(xs[i])
        if prev < 0 <= cur:
            root = mp.findroot(G, (xs[i-1], xs[i]), solver='bisect'); break
        prev = cur
    w, mu, N, R, Y = parts(root)
    Hd0 = 0
    kap = (2 + w*(5*N - 4*w - 4) - 6*N**2)/(4*R*(1+w)**2) - R/(2*N)*(Hd0 + kv/(2*av**2*(1+mu)**4))
    dRdt = Hv*R/(6*mu - (1+mu)**2*(1+w)**2)*(2*w*(3*mu-2))
    rho8 = 3*Hv**2 + 12*kv/(av**2*(1+mu)**5)
    p8 = -3*Hv**2 - 4*kv/(av**2*N*(1+mu)**6)
    return dict(kap4M=4*Mv*kap, p8M2=p8*Mv**2, Gamma=abs(dRdt)/(abs(kap)*R), N=N)
for (eps, ell_) in [(0.02, 0.0005), (0.01, 0.0005), (0.01, 0.005), (0.005, 0.002), (0.002, 0.002), (0.01, -0.00001), (0.02, -0.00002)]:
    d = smaller_sphere(eps, ell_)
    pred4Mk = 1 - ell_/(2*(ell_+2*eps))
    predp = -ell_/(4*(ell_+2*eps))
    predG = 8*eps*abs(ell_)*(ell_+2*eps)/(ell_+4*eps)
    print(f"  eps={eps} ell={ell_}: 4Mkappa exact {float(d['kap4M']):.5f} vs {pred4Mk:.5f} | 8piM^2 p exact {float(d['p8M2']):+.6f} vs {predp:+.6f} | Gamma exact {float(d['Gamma']):.3e} vs {predG:.3e}")

print("\n=== 5. H=0 smaller sphere relations")
for wv in [0.01, 0.1, 0.2]:
    wv = mp.mpf(wv)
    ell_ = wv*(1-wv)**2/(1+wv)**3
    mu = (1-wv)/(1+wv)
    Rs = 2/((1-wv)*(1+wv)**1.5)
    # direct: R = a r (1+mu)^2/(1+w) with r from mu: mu = M sqrt(1+w)/(2 a r), M=a=1
    r_ = mp.sqrt(1+wv)/(2*mu); R_direct = r_*(1+mu)**2/(1+wv)
    print(f"  w={float(wv)}: ell={float(ell_):.6f} (w=ell(1+w)^3/(1-w)^2 check {float(ell_*(1+wv)**3/(1-wv)**2 - wv):.1e}) R_s: {float(Rs):.8f} vs {float(R_direct):.8f}")
print("  ell_max = ", float(mp.mpf('0.2')*(0.8)**2/(1.2)**3), " 2/27 =", 2/27)