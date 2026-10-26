"""Check Eq. (shell-first-law) of Sect. 5.6 and its consequences, for the cavity wall of Appendix G placed in the
evolving Vaidya-Shah solution, by direct numerical differentiation of the exact metric functions.

  delta E_B = T_Lambda delta S + (1/sqrt(chi_B)) int_{R_A}^{R_B} 4 pi R^2 delta rho_tot|_R dR,
  T_Lambda = kappa_Lambda/(2 pi sqrt(chi_B)),  kappa_Lambda = 1/(2R_A) - 4 pi R_A rho_A - q_A^2/(2R_A^3)  [Eq. (kappa-lambda-charged)],
with rho_tot = rho + q^2/(8 pi R^4) and delta at fixed areal radius R_B and fixed a, H, Hdot, k.
Wall quantities of Appendix G: E_B = R_B (sqrt(1 - R_B^2/R_H^2) - sqrt(chi_B)), R_H^-2 = H^2 + 4k/a^2, S = pi R_A^2,
q_B = Q/(1+k r_B^2)^{3/2}, Phi_B = Y_B Q/(N_B sqrt(chi_B)) [1/(R_A sqrt(1+k r_A^2)) - 1/(R_B sqrt(1+k r_B^2))].
Part 1: k = 0 and Hdot != 0, neutral and charged. With T_Lambda the boundary first law holds exactly,
        delta E_B = T_Lambda delta S + Phi_B delta q_B, for variations of M and of Q.
Part 2: Eq. (shell-first-law) for both signs of k, with and without charge and with Hdot != 0.
Part 3: neutral, k != 0 at constant H: (delta E_B - T_Lambda delta S) sqrt(chi_B)/delta M -> -(15/2) k r_B^2 as k r_B^2 -> 0,
        and T_Lambda - T_B accounts for the term l/(4 eps) of Eq. (k-residual).
kappa_A is computed independently from the Hayward-Kodama formula, Eq. (Hayward-Kodama-sg).
Run: python3 checks/verify_shell_first_law.py"""
import sympy as sp
import mpmath as mp

mp.mp.dps = 40
t, r = sp.symbols('t r', real=True)
M, Q, k, a0, H, Hd = sp.symbols('M Q k a0 H Hd', real=True)

a = a0 * (1 + H * t + (Hd + H**2) * t**2 / 2)          # a, adot and addot are exact at t = 0
mu = M * sp.sqrt(1 + k * r**2) / (2 * a * r)
lam = Q * sp.sqrt(1 + k * r**2) / (2 * a * r)
Xi = (1 + mu)**2 - lam**2
N = (1 - mu**2 + lam**2) / Xi
B = a * Xi / (1 + k * r**2)
R = B * r
Rt, Rr = sp.diff(R, t), sp.diff(R, r)
chi = -(Rt / N)**2 + (Rr / B)**2
kap = (-sp.diff(B * Rt / N, t) + sp.diff(N * Rr / B, r)) / (2 * N * B)       # Eq. (Hayward-Kodama-sg)
rho8pi = 3 * H**2 + 12 * k * (1 + mu) / (a**2 * Xi**3)                         # Eq. (charged-density)
q = Q / (1 + k * r**2)**sp.Rational(3, 2)                                       # Eq. (enclosed-charge)
kapL = 1 / (2 * R) - R * rho8pi / 2 - q**2 / (2 * R**3)                         # Eq. (kappa-lambda-charged)
Y = Rr / B

args = (r, M, Q, k, a0, H, Hd)
L = lambda e: sp.lambdify(args, e.subs(t, 0), 'mpmath')
fR, fchi, fkap, fkapL, fY, fN = map(L, (R, chi, kap, kapL, Y, N))

# integrand of Eq. (shell-first-law): 4 pi R^2 d(rho_tot)/dX at fixed areal R, times dR/dr, for X = M and X = Q
Rs, Rr0 = R.subs(t, 0), sp.diff(R.subs(t, 0), r)
rho_tot = (rho8pi / (8 * sp.pi) + q**2 / (8 * sp.pi * R**4)).subs(t, 0)
fixed_R = lambda f, X: sp.diff(f, X) - sp.diff(f, r) * sp.diff(Rs, X) / Rr0
dens = {X: sp.lambdify(args, 4 * sp.pi * Rs**2 * fixed_R(rho_tot, X) * Rr0, 'mpmath') for X in (M, Q)}


def wall_data(Mv, Qv, kv, av, Hv, Hdv, RB):
    p = (Mv, Qv, kv, av, Hv, Hdv)
    r_edge = mp.findroot(lambda x: fN(x, *p), Mv / (2 * av))      # lapse-zero surface
    rs = [r_edge * (1 + mp.mpf(j) / 400) for j in range(1, 4000)]
    rA = None
    for x0, x1 in zip(rs, rs[1:]):                                 # smaller marginal sphere: first zero of chi
        if fchi(x0, *p) < 0 < fchi(x1, *p):
            rA = mp.findroot(lambda x: fchi(x, *p), (x0, x1), solver='anderson')
            break
    rB = mp.findroot(lambda x: fR(x, *p) - RB, (rA * mp.mpf('1.0001'), 3 * RB), solver='anderson')
    RA, chiB = fR(rA, *p), fchi(rB, *p)
    RH = 1 / mp.sqrt(Hv**2 + 4 * kv / av**2)
    return dict(rA=rA, rB=rB, RA=RA, chiB=chiB,
                E=RB * (mp.sqrt(1 - RB**2 / RH**2) - mp.sqrt(chiB)),
                S=mp.pi * RA**2,
                qB=Qv / (1 + kv * rB**2)**mp.mpf('1.5'),
                Phi=fY(rB, *p) * Qv / (fN(rB, *p) * mp.sqrt(chiB))
                * (1 / (RA * mp.sqrt(1 + kv * rA**2)) - 1 / (RB * mp.sqrt(1 + kv * rB**2))),
                TA=fkap(rA, *p) / (2 * mp.pi * mp.sqrt(chiB)),
                TL=fkapL(rA, *p) / (2 * mp.pi * mp.sqrt(chiB)))


def variations(c, RB, h=mp.mpf('1e-15')):
    """Central differences in M and in Q at fixed R_B, a, H, Hdot, k."""
    d0 = wall_data(*c, RB)
    out = {}
    for i, name in ((0, 'M'), (1, 'Q')):
        cp, cm = list(c), list(c)
        cp[i] += h
        cm[i] -= h
        p, m = wall_data(*cp, RB), wall_data(*cm, RB)
        out[name] = {x: (p[x] - m[x]) / (2 * h) for x in ('E', 'S', 'qB')}
    return d0, out


print("Part 1: k = 0, Hdot != 0. Boundary first law with T_Lambda in place of T_B")
for label, c, RB in (("neutral, matter-dominated", (1, 0, 0, 1, '0.01', '-0.00015'), 6),
                     ("Q/M = 0.5, matter-dominated", (1, '0.5', 0, 1, '0.01', '-0.00015'), 6),
                     ("Q/M = 0.9, Hdot = +0.3 H^2", (1, '0.9', 0, 1, '0.02', '0.00012'), 10)):
    c = [mp.mpf(x) for x in c]
    d0, D = variations(c, RB)
    for name in ('M', 'Q'):
        if name == 'Q' and c[1] == 0:
            continue
        dE = D[name]['E']
        resA = dE - d0['TA'] * D[name]['S'] - d0['Phi'] * D[name]['qB']
        resL = dE - d0['TL'] * D[name]['S'] - d0['Phi'] * D[name]['qB']
        print(f"  {label}, delta {name}: residual/delta E_B with T_B = {mp.nstr(resA / dE, 6)}, with T_Lambda = {mp.nstr(resL / dE, 3)}")
        assert abs(resL / dE) < mp.mpf('1e-20')

print("\nPart 2: Eq. (shell-first-law), sqrt(chi_B)(delta E_B - T_Lambda delta S) against the shell integral")
for label, c, RB in (("k > 0, neutral, Hdot = 0", (1, 0, '1e-5', 1, '0.01', 0), 10),
                     ("k > 0, Q/M = 0.5, Hdot != 0", (1, '0.5', '1e-5', 1, '0.01', '-0.00015'), 10),
                     ("k < 0, Q/M = 0.5, Hdot != 0", (1, '0.5', '-2e-5', 1, '0.01', '-0.00015'), 10),
                     ("k < 0, Q/M = 0.9, Hdot = 0", (1, '0.9', '-1e-5', 1, '0.02', 0), 8)):
    c = [mp.mpf(x) for x in c]
    d0, D = variations(c, mp.mpf(RB))
    sq = mp.sqrt(d0['chiB'])
    for name, X in (('M', M), ('Q', Q)):
        if name == 'Q' and c[1] == 0:
            continue
        lhs = sq * (D[name]['E'] - d0['TL'] * D[name]['S'])
        rhs = mp.quad(lambda x: dens[X](x, *c), [d0['rA'], d0['rB']])
        print(f"  {label}, delta {name}: lhs = {mp.nstr(lhs, 12)}, shell integral = {mp.nstr(rhs, 12)}, difference {mp.nstr(lhs - rhs, 2)}")
        assert abs(lhs - rhs) < mp.mpf('1e-20')

print("\nPart 3: neutral, k != 0 at constant H. Leading order of the remaining wall term")
ratios = []
for kv, RB in (('1e-5', 10), ('1e-5', 5), ('4e-5', 5)):
    c = [mp.mpf(x) for x in (1, 0, kv, 1, '0.01', 0)]
    d0, D = variations(c, mp.mpf(RB))
    sq = mp.sqrt(d0['chiB'])
    w = c[2] * d0['rB']**2
    eps, ell = c[0] * c[4], c[2] * c[0]**2 / (4 * c[3]**2)
    resL = sq * (D['M']['E'] - d0['TL'] * D['M']['S'])
    shift = sq * (d0['TL'] - d0['TA']) * D['M']['S']
    ratios.append(resL / (-mp.mpf(15) / 2 * w))
    print(f"  k r_B^2 = {mp.nstr(w, 4)}: (delta E_B - T_Lambda delta S) sqrt(chi_B)/delta M = {mp.nstr(resL, 6)}, "
          f"ratio to -(15/2) k r_B^2 = {mp.nstr(ratios[-1], 5)}; (T_Lambda - T_B) delta S sqrt(chi_B)/delta M = {mp.nstr(shift, 5)}, "
          f"l/(4 eps) = {mp.nstr(ell / (4 * eps), 5)}")
    assert abs(shift / (ell / (4 * eps)) - 1) < mp.mpf('0.01')
assert abs(min(ratios, key=lambda x: abs(x - 1)) - 1) < mp.mpf('0.01')

print("\nAll shell first law checks passed.")
