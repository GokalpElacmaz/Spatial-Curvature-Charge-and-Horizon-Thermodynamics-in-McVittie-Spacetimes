"""Explicit charged mergers: two marginal spheres of the Vaidya-Shah solution coincide as M grows, Sect. 4.1.

Units a = 1 at the comparison time, constant H. For fixed H, k and Q/M (so lambda = (Q/M) mu), the function
G(r) = Y - HR is followed on the regular chart region N > 0, which for M > |Q| is mu < 1/gamma, gamma = sqrt(1-Q^2/M^2).
  1. The critical mass M_c is the value of M at which max_r G = 0, a double zero at r_A.
  2. Root structure: two simple zeros for M slightly below M_c and none above; at the double zero dG/dr = 0,
     d^2G/dr^2 < 0 and dG/dM < 0, so two simple zeros approach each other and disappear as M grows (a genuine merger).
  3. Eq. (charged-merger): 8 pi rho_A R_A^2 + q_A^2/R_A^2 = 1 and M_W = M/(1+kr_A^2)^{5/2} - q_A^2/R_A = (R_A^2 - q_A^2)/(3R_A);
     kappa_Lambda of Eq. (kappa-lambda-charged) vanishes and kappa_A = 2 pi R_A (rho_A + p_A), Eq. (kappa-merger), with
     kappa_A from the Hayward-Kodama expression, Eq. (kappa-charged-compact).
  4. d_t chi != 0 at the merger sphere, so the double zero is an instant at which two marginal spheres meet (Sect. 2.3).
Cases: Q/M = 0.5 and 0.9 for k > 0, k < 0 with Omega_k <= 1, and k = 0 against R_A^2 - 3 M R_A + 2 Q^2 = 0.
Run: python3 checks/verify_charged_merger.py"""
import mpmath as mp

mp.mp.dps = 30
pi = mp.pi
a = mp.mpf(1)


def fields(r, M, Q, k, H):
    u = 1 + k * r * r
    mu = M * mp.sqrt(u) / (2 * a * r)
    lam = Q * mp.sqrt(u) / (2 * a * r)
    Xi = (1 + mu)**2 - lam**2
    N = (1 - mu**2 + lam**2) / Xi
    B = a * Xi / u
    R = B * r
    Y = (N - k * r * r) / u
    rho = (3 * H * H + 12 * k * (1 + mu) / (a * a * Xi**3)) / (8 * pi)
    p = (-3 * H * H - 4 * k / (a * a * N * Xi**3)) / (8 * pi)          # constant H
    q = Q / u**mp.mpf(1.5)
    return dict(u=u, mu=mu, lam=lam, Xi=Xi, N=N, B=B, R=R, Y=Y, rho=rho, p=p, q=q)


def G(r, M, Q, k, H):
    F = fields(r, M, Q, k, H)
    return F['Y'] - H * F['R']


def r_lapse_zero(M, Q, k):
    """mu = 1/gamma, i.e. r^2 = (gamma M)^2 (1 + k r^2)/(4a^2)."""
    gam = mp.sqrt(1 - Q * Q / (M * M))
    return gam * M / (2 * a * mp.sqrt(1 - k * gam * gam * M * M / (4 * a * a)))


def r_end(M, Q, k, H):
    """Outer end of the search interval: for k > 0 the sphere of largest R (Y = 0), otherwise a few Hubble radii."""
    r0 = r_lapse_zero(M, Q, k)
    if k > 0:
        hi = r0 * 2
        while fields(hi, M, Q, k, H)['Y'] > 0:
            hi *= 2
        return mp.findroot(lambda r: fields(r, M, Q, k, H)['Y'], (hi / 2, hi), solver='bisect')
    r1 = max(5 / (a * H), 10 * r0)
    if k < 0:
        r1 = min(r1, mp.mpf('0.999') / mp.sqrt(-k))        # stay inside the chart 1 + k r^2 > 0
    return r1


def max_G(M, Q, k, H, n=400):
    r0, r1 = r_lapse_zero(M, Q, k) * (1 + mp.mpf('1e-12')), r_end(M, Q, k, H)
    xs = [r0 * (r1 / r0)**(mp.mpf(i) / n) for i in range(n + 1)]
    vals = [G(x, M, Q, k, H) for x in xs]
    i = max(range(1, n), key=lambda j: vals[j])
    rmax = mp.findroot(lambda r: mp.diff(lambda x: G(x, M, Q, k, H), r), (xs[i - 1], xs[i + 1]), solver='bisect')
    return rmax, G(rmax, M, Q, k, H)


def count_zeros(M, Q, k, H, n=4000):
    r0, r1 = r_lapse_zero(M, Q, k) * (1 + mp.mpf('1e-12')), r_end(M, Q, k, H)
    xs = [r0 * (r1 / r0)**(mp.mpf(i) / n) for i in range(n + 1)]
    vals = [G(x, M, Q, k, H) for x in xs]
    zeros = []
    for j in range(n):
        if vals[j] == 0 or vals[j] * vals[j + 1] < 0:
            zeros.append(mp.findroot(lambda r: G(r, M, Q, k, H), (xs[j], xs[j + 1]), solver='bisect'))
    return zeros


def kappa_HK(r, M, Q, k, H):
    """Hayward-Kodama surface gravity at constant H, Eq. (kappa-charged-compact): d_r(N Y)/(2 N B) - H^2 R."""
    F = fields(r, M, Q, k, H)
    dNY = mp.diff(lambda x: fields(x, M, Q, k, H)['N'] * fields(x, M, Q, k, H)['Y'], r)
    return dNY / (2 * F['N'] * F['B']) - H * H * F['R']


def merger(QoverM, k, H, M_lo, M_hi):
    """Bisection in M for max_r G(r; M) = 0."""
    f = lambda M: max_G(M, QoverM * M, k, H)[1]
    assert f(M_lo) > 0 > f(M_hi), "bracket must have two zeros at M_lo and none at M_hi"
    Mc = mp.findroot(f, (M_lo, M_hi), solver='bisect', tol=mp.mpf(10)**(-24))
    rA, _ = max_G(Mc, QoverM * Mc, k, H)
    return Mc, rA


print("Charged mergers of the Vaidya-Shah solution at constant H (a = 1)")
cases = [  # (Q/M, k, H)
    (mp.mpf('0.5'), mp.mpf(0), mp.mpf('0.05')),
    (mp.mpf('0.5'), mp.mpf('0.0004'), mp.mpf('0.05')),
    (mp.mpf('0.9'), mp.mpf('0.0004'), mp.mpf('0.05')),
    (mp.mpf('0.5'), mp.mpf('-0.0002'), mp.mpf('0.05')),
    (mp.mpf('0.9'), mp.mpf('-0.0004'), mp.mpf('0.05')),
]
for QoverM, k, H in cases:
    Omega_k = -4 * k / (a * a * H * H)
    # bracket: flat neutral value M_c H = 1/(3 sqrt3) ~ 0.19; charge lowers M_c H for fixed Q/M
    Mc, rA = merger(QoverM, k, H, mp.mpf('0.05') / H, mp.mpf('0.25') / H)
    Q = QoverM * Mc
    F = fields(rA, Mc, Q, k, H)
    RA, qA, w = F['R'], F['q'], k * rA * rA
    print(f"\nQ/M = {QoverM}, k = {k}, H = {H}, Omega_k = {mp.nstr(Omega_k, 4)}")
    print(f"  M_c H = {mp.nstr(Mc * H, 12)}   r_A = {mp.nstr(rA, 10)}   k r_A^2 = {mp.nstr(w, 6)}   R_A/M_c = {mp.nstr(RA / Mc, 10)}   "
          f"mu_A = {mp.nstr(F['mu'], 8)}   N_A = {mp.nstr(F['N'], 8)}")
    # root structure and derivatives
    d1 = mp.diff(lambda r: G(r, Mc, Q, k, H), rA)
    d2 = mp.diff(lambda r: G(r, Mc, Q, k, H), rA, 2)
    dM = mp.diff(lambda m: G(rA, m, QoverM * m, k, H), Mc)
    below = count_zeros(Mc * (1 - mp.mpf('1e-3')), QoverM * Mc * (1 - mp.mpf('1e-3')), k, H)
    above = count_zeros(Mc * (1 + mp.mpf('1e-3')), QoverM * Mc * (1 + mp.mpf('1e-3')), k, H)
    print(f"  at the double zero: G = {mp.nstr(G(rA, Mc, Q, k, H), 3)}, dG/dr = {mp.nstr(d1, 3)}, "
          f"r_A^2 d^2G/dr^2 = {mp.nstr(rA * rA * d2, 8)}, M dG/dM (Q/M fixed) = {mp.nstr(Mc * dM, 8)}")
    print(f"  zeros of G in the region N > 0: {len(below)} at M = 0.999 M_c "
          f"(R/M = {[mp.nstr(fields(z, Mc*(1-mp.mpf('1e-3')), Q*(1-mp.mpf('1e-3')), k, H)['R']/(Mc*(1-mp.mpf('1e-3'))), 7) for z in below]}), "
          f"{len(above)} at M = 1.001 M_c")
    assert d2 < 0 and dM < 0 and len(below) == 2 and len(above) == 0
    # Eq. (charged-merger)
    lhs = 8 * pi * F['rho'] * RA * RA + qA * qA / (RA * RA)
    MW = Mc / F['u']**mp.mpf(2.5) - qA * qA / RA
    MW_paper = (RA * RA - qA * qA) / (3 * RA)
    kapL = 1 / (2 * RA) - 4 * pi * RA * F['rho'] - qA * qA / (2 * RA**3)
    kapA = kappa_HK(rA, Mc, Q, k, H)
    kap_paper = 2 * pi * RA * (F['rho'] + F['p'])
    print(f"  8 pi rho_A R_A^2 + q_A^2/R_A^2 = {mp.nstr(lhs, 15)}   M_W = {mp.nstr(MW, 12)}  (R_A^2 - q_A^2)/(3R_A) = {mp.nstr(MW_paper, 12)}")
    print(f"  kappa_Lambda = {mp.nstr(kapL, 3)}   kappa_A R_A = {mp.nstr(kapA * RA, 10)}   2 pi R_A^2 (rho_A + p_A) = {mp.nstr(kap_paper * RA, 10)}")
    assert abs(lhs - 1) < mp.mpf('1e-15') and abs(MW - MW_paper) < mp.mpf('1e-15') and abs(kapA - kap_paper) < mp.mpf('1e-12')
    if k == 0:
        print(f"  k = 0 check: R_A^2 - 3 M R_A + 2 Q^2 = {mp.nstr(RA * RA - 3 * Mc * RA + 2 * Q * Q, 3)}")
        assert abs(RA * RA - 3 * Mc * RA + 2 * Q * Q) < mp.mpf('1e-15')
    # d_t chi at the merger sphere (constant H): dY/dt = r H N', dR/dt = H R N
    Nprime = mp.diff(lambda r: fields(r, Mc, Q, k, H)['N'], rA)
    dchi_dt = 2 * F['Y'] * rA * H * Nprime - 2 * H**3 * RA * RA * F['N']
    print(f"  d_t chi at the merger sphere = {mp.nstr(dchi_dt, 8)}  ({'nonzero: the double zero is an instant at which the two spheres meet' if abs(dchi_dt) > mp.mpf('1e-12') else 'zero: no merger in time'})")
print("\nAll charged merger checks passed.")
