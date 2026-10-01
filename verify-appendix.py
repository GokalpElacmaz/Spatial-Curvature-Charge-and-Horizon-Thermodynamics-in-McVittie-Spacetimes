"""Check the Appendix values that were not yet verified:
 (1) d_t of the Ricci scalar on the lapse-zero surface, neutral and charged;
 (2) the closed form of dR_A/dt with Z in the (mu, lambda) notation.
"""
import mpmath as mp
mp.mp.dps = 70

def make_R(k, M, Q, C, c):
    """Ricci scalar 8 pi (rho - 3p) with H-dot = -c/a^2 exactly, a(t) = sqrt(c/C) sinh(sqrt(C) t)."""
    b = mp.sqrt(C); al = mp.sqrt(c / C)
    a  = lambda t: al * mp.sinh(b * t)
    H  = lambda t: b * mp.coth(b * t)
    Hd = lambda t: -c / a(t)**2
    def Rscalar(t, r):
        A = a(t)
        mu = M * mp.sqrt(1 + k*r**2) / (2*A*r)
        lam = Q * mp.sqrt(1 + k*r**2) / (2*A*r)
        Xi = (1 + mu)**2 - lam**2
        N = (1 - mu**2 + lam**2) / Xi
        h = H(t)
        rho8 = 3*h**2 + 12*k*(1 + mu)/(A**2 * Xi**3)
        p8 = -2*Hd(t)/N - 3*h**2 - 4*k/(A**2 * N * Xi**3)
        return rho8 - 3*p8
    return a, H, Rscalar

def dR_dt_on_surface(k, M, Q, t0):
    delta = mp.sqrt(1 - (Q/M)**2)
    c = k * delta**3 / (4*(1 + delta)**3)
    C = mp.mpf('0.37')
    a, H, Rs = make_R(k, M, Q, C, c)
    A0 = a(t0)
    r0 = M*delta / mp.sqrt(4*A0**2 - k*M**2*delta**2)     # mu = 1/delta at (t0, r0)
    h = mp.mpf('1e-22')
    d = (Rs(t0 + h, r0) - Rs(t0 - h, r0)) / (2*h)
    paper = -9*k*H(t0)*delta**2*(1 + 4*delta) / (4*A0**2*(1 + delta)**3)
    R0 = Rs(t0 + h, r0)
    paper_R0 = None
    return d, paper, R0, H(t0), A0, delta

print("(1) d_t (Ricci scalar) on the lapse-zero surface, fixed r")
for (k, M, Q) in [(mp.mpf('0.3'), mp.mpf(1), mp.mpf(0)),
                  (mp.mpf('-0.05'), mp.mpf(1), mp.mpf(0)),
                  (mp.mpf('0.3'), mp.mpf(1), mp.mpf('0.6')),
                  (mp.mpf('-0.05'), mp.mpf(1), mp.mpf('0.8'))]:
    d, paper, R0, H0, A0, delta = dR_dt_on_surface(k, M, Q, mp.mpf('2.0') if k < 0 else mp.mpf('6.0'))
    # neutral form too
    neutral = -45*k*H0/(32*A0**2)
    print(f"  k={float(k):+.2f} Q/M={float(Q/M):.1f}: numeric {mp.nstr(d, 12)}  paper(charged form) {mp.nstr(paper, 12)}"
          + (f"  paper(neutral form) {mp.nstr(neutral, 12)}" if Q == 0 else ""))
    # Ricci scalar value itself on the surface
    print(f"     R on surface = {mp.nstr(R0, 12)},  12H^2 + 3k/(2a^2) = {mp.nstr(12*H0**2 + 3*k/(2*A0**2), 12)} (neutral formula)")

# ---------------------------------------------------------------- (2) Z in (mu, lambda)
print("\n(2) dR_A/dt: implicit differentiation vs closed form with Z in (mu, lambda)")
import mpmath as mp
mp.mp.dps = 40
def chi_parts(t, r, k, M, Q, a, H):
    A = a(t)
    mu = M*mp.sqrt(1+k*r**2)/(2*A*r); lam = Q*mp.sqrt(1+k*r**2)/(2*A*r)
    Xi = (1+mu)**2 - lam**2
    N = (1-mu**2+lam**2)/Xi
    Y = (N - k*r**2)/(1+k*r**2)
    R = A*r*Xi/(1+k*r**2)
    return N, Y, R, mu, lam, Xi

k = mp.mpf('0.02'); M = mp.mpf(1); Q = mp.mpf('0.5')
# generic history: a(t) = t^(2/3) matter dominated (H = 2/(3t), Hdot = -3H^2/2)
a = lambda t: t**(mp.mpf(2)/3)
H = lambda t: mp.mpf(2)/(3*t)
Hd = lambda t: -mp.mpf(2)/(3*t**2)
t0 = mp.mpf(60)
resid = lambda t, r: chi_parts(t, r, k, M, Q, a, H)[1] - H(t)*chi_parts(t, r, k, M, Q, a, H)[2]
# smaller root near r ~ M/(2a)
def first_root(t):
    prev = None
    n = 4000
    for i in range(1, n):
        r = mp.mpf('0.02') + (mp.mpf('0.6') - mp.mpf('0.02'))*i/n
        Nn, Yy, Rr, *_ = chi_parts(t, r, k, M, Q, a, H)
        if Nn <= 0:
            continue
        v = Yy - H(t)*Rr
        if prev is not None and prev[1] < 0 <= v:
            return mp.findroot(lambda x: resid(t, x), (prev[0], r), solver='anderson', tol=1e-30)
        prev = (r, v)
    raise RuntimeError('no root')
r_small = first_root(t0)
N0, Y0, R0, mu0, lam0, Xi0 = chi_parts(t0, r_small, k, M, Q, a, H)
print("   marginal sphere r_A =", mp.nstr(r_small, 10), " N_A =", mp.nstr(N0, 8), " mu_A =", mp.nstr(mu0, 8))
# numerical trapping horizon evolution
def rA(t):
    return mp.findroot(lambda r: resid(t, r), r_small, tol=1e-30)
dt = mp.mpf('1e-8')
RA = lambda t: chi_parts(t, rA(t), k, M, Q, a, H)[2]
num = (RA(t0+dt) - RA(t0-dt))/(2*dt)
w = k*r_small**2
Z = (mu0 + 2*(mu0**2 - lam0**2) + mu0*(mu0**2 - lam0**2)) / Xi0**2
# closed form as in the paper's Appendix D (charged), reconstructed:
Hh = H(t0); Hdd = Hd(t0)
# denominator 2Z - 2 w (1+N) - (1+w)^2 H^2 R^2 ; numerator H R [ Hdot R^2 (1+w)^2 + 2 w (Z - N(1+N)) ]
closed = Hh*R0*(Hdd*R0**2*(1+w)**2 + 2*w*(Z - N0*(1+N0))) / (2*Z - 2*w*(1+N0) - (1+w)**2*Hh**2*R0**2)
print("   dR_A/dt numeric =", mp.nstr(num, 14))
print("   closed form     =", mp.nstr(closed, 14))