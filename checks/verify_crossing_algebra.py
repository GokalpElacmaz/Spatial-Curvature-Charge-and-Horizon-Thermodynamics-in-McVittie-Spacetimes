"""Check the algebra of the crossing condition, Appendix I, symbolically with sympy and numerically with 40 digits.

Variables of Sect. 5.4: X = R_A/R_B, eta = H^2 R_B^2, D = 1 - eta (1 + X + X^2), C = sqrt(chi_B)/(1 - X),
C_pm = 2 sqrt(1-eta) pm sqrt(1 + eta (X^2 + 2X - 1)). The wall data follow from Eqs. (dS-mass), (dS-charge), (dS-chi)
and (dS-TB), which are taken as the starting point.
  1. Omega_B/R_B = (C_+ - C)(C - C_-)/(4C), Eq. (Omega-factor), for every black hole in the cavity (sympy, exact).
  2. On each root: (1 - Phi_B^2) X = 1 - D/C_pm^2 and pi X^2 T_cross R_B = sqrt(1-eta) - D/C_pm, Eqs. (exact-Phi), (exact-T).
  3. d Phi_B^2/dX along each root, Eq. (dPhi2dX), and Z_pm > 0.
  4. (d(T_B R_B)/dX)_{Phi_B, eta} at a crossing = F_pm/(2 pi X^3 D C_pm), Eq. (dTdX-cross); F_+ F_-, Eq. (F-product), exact
     with sympy; positivity of J, G, F_+F_-, Z_pm on 0 < X < 1, 0 < eta < 1, so (dT_B/dX) > 0 on both roots.
  5. The neutral curve, Eq. (neutral-curve), its branches eta_-+(X), Eq. (neutral-eta), the Nariai value of T_cross R_B,
     and the small-X behaviour of Omega_B used for the sign of the grand potential.
Run: python3 checks/verify_crossing_algebra.py"""
import random

import mpmath as mp
import sympy as sp

mp.mp.dps = 40
random.seed(1)

# ---------------------------------------------------------------- symbols
X, eta, P, C = sp.symbols('X eta P C', positive=True)          # P = Phi_B^2
D = 1 - eta * (1 + X + X**2)
root = sp.sqrt(1 + eta * (X**2 + 2 * X - 1))
Cp = 2 * sp.sqrt(1 - eta) + root
Cm = 2 * sp.sqrt(1 - eta) - root


def wall_data_P(X, eta, P):
    """Eqs. (dS-charge), (dS-chi), (dS-TB) in terms of X, eta and P = Phi_B^2. Returns Q^2/R_B^2, chi_B, kappa_A R_B."""
    Dv = 1 - eta * (1 + X + X**2)
    den = 1 - (1 - P) * X
    Q2 = P * X**2 * Dv / den
    chiB = (1 - X)**2 * Dv / den
    kapRB = (1 - Q2 / X**2 - 3 * eta * X**2) / (2 * X)
    return Q2, chiB, kapRB


def omega_P(X, eta, P, sqrt=sp.sqrt, pi=sp.pi):
    """Omega_B/R_B = E_B/R_B - T_B S/R_B - Phi_B Q/R_B, Eq. (crossing-X), from the P-form of the wall data."""
    Q2, chiB, kapRB = wall_data_P(X, eta, P)
    EB = sqrt(1 - eta) - sqrt(chiB)
    TS = X**2 * kapRB / (2 * sqrt(chiB))               # pi X^2 T_B R_B with T_B R_B = kappa_A R_B/(2 pi sqrt(chi_B))
    PhiQ = Q2 * (1 - X) / (X * sqrt(chiB))
    return EB - TS - PhiQ


print("1. Factorization of the grand potential, Eq. (Omega-factor)")
# In terms of C: chi_B = (1-X)^2 C^2 and, from Eq. (dS-chi), 1-(1-P)X = D/C^2, so Eq. (dS-charge) gives Eq. (Q-of-C).
Q2_C = X * (D - (1 - X) * C**2)
EB_C = sp.sqrt(1 - eta) - (1 - X) * C
TS_C = (X - 3 * eta * X**3 - Q2_C / X) / (4 * (1 - X) * C)
PhiQ_C = Q2_C / (X * C)
Omega_C = sp.together(EB_C - TS_C - PhiQ_C)
factor_form = (Cp - C) * (C - Cm) / (4 * C)
diff1 = sp.simplify(sp.expand(sp.cancel(Omega_C - factor_form)))
print("   Omega_B/R_B - (C_+ - C)(C - C_-)/(4C) simplifies to:", diff1)
assert diff1 == 0
# Independent numerical check from the P-form, without the C substitution
worst = mp.mpf(0)
for _ in range(200):
    Xv, ev = mp.mpf(random.uniform(0.05, 0.95)), mp.mpf(random.uniform(0.01, 0.95))
    Dv = 1 - ev * (1 + Xv + Xv**2)
    if Dv <= 0:
        continue
    Pv = mp.mpf(random.uniform(0, 0.9)) / (1 - ev)                  # keeps 1-(1-P)X > 0 for X < 1
    Q2, chiB, kapRB = wall_data_P(Xv, ev, Pv)
    if chiB <= 0 or kapRB <= 0:
        continue
    Cv = mp.sqrt(chiB) / (1 - Xv)
    Cpv = 2 * mp.sqrt(1 - ev) + mp.sqrt(1 + ev * (Xv**2 + 2 * Xv - 1))
    Cmv = 2 * mp.sqrt(1 - ev) - mp.sqrt(1 + ev * (Xv**2 + 2 * Xv - 1))
    direct = omega_P(Xv, ev, Pv, sqrt=mp.sqrt, pi=mp.pi)
    worst = max(worst, abs(direct - (Cpv - Cv) * (Cv - Cmv) / (4 * Cv)))
print(f"   largest |direct - factorized| over random black holes in the cavity: {mp.nstr(worst, 3)}")
assert worst < mp.mpf(10)**(-30)

print("\n2. The two roots, Eqs. (exact-Phi) and (exact-T)")
for sign, Cr in (('+', Cp), ('-', Cm)):
    P_root = 1 - (1 - D / Cr**2) / X                                 # from (1-P)X = 1 - D/C^2
    worst_T = mp.mpf(0)
    for _ in range(100):
        Xv, ev = mp.mpf(random.uniform(0.05, 0.95)), mp.mpf(random.uniform(0.01, 0.95))
        Dv = 1 - ev * (1 + Xv + Xv**2)
        Pv = mp.mpf(sp.N(P_root.subs({X: sp.Float(str(Xv), 45), eta: sp.Float(str(ev), 45)}), 45))
        if Dv <= 0 or Pv < 0:
            continue
        Q2, chiB, kapRB = wall_data_P(Xv, ev, Pv)
        if chiB <= 0:
            continue
        Crv = mp.mpf(sp.N(Cr.subs({X: sp.Float(str(Xv), 45), eta: sp.Float(str(ev), 45)}), 45))
        T_from_data = Xv**2 * kapRB / (2 * mp.sqrt(chiB))           # pi X^2 T_B R_B
        T_formula = mp.sqrt(1 - ev) - Dv / Crv
        worst_T = max(worst_T, abs(T_from_data - T_formula))
        assert abs(omega_P(Xv, ev, Pv, sqrt=mp.sqrt, pi=mp.pi)) < mp.mpf(10)**(-30)   # it is a crossing
    print(f"   root {sign}: Omega_B = 0 there, and |pi X^2 T_B R_B - (sqrt(1-eta) - D/C_pm)| <= {mp.nstr(worst_T, 3)}")

print("\n3. Monotonicity of Phi_B^2 along each root, Eq. (dPhi2dX)")
z = sp.symbols('z', positive=True)
for sign, Cr, s in (('+', Cp, 1), ('-', Cm, -1)):
    P_root = 1 - (1 - D / Cr**2) / X
    dP = sp.diff(P_root, X)
    zX = sp.sqrt(1 + eta * X * (X + 2) / (1 - eta))
    Zpm = 3 * (z + s)**2 + (3 * z**2 + s * 4 * z + 5) * X + (z**2 + 2) * X**2
    formula = s * 2 * (z + s)**2 * Zpm / (z * (2 + s * z)**3 * X**2 * (X + 2)**2)
    worst = mp.mpf(0)
    for _ in range(60):
        Xv, ev = random.uniform(0.05, 0.95), random.uniform(0.01, 0.9)
        sub = {X: sp.Float(str(Xv), 45), eta: sp.Float(str(ev), 45)}
        zv = zX.subs(sub)
        if s < 0 and sp.N(2 - zv) <= 0:
            continue                                                 # C_- > 0 needed on the second root
        a = sp.N(dP.subs(sub), 40)
        b = sp.N(formula.subs({z: zv}).subs(sub), 40)
        worst = max(worst, abs(mp.mpf(str(a)) - mp.mpf(str(b))))
    print(f"   root {sign}: |d(Phi_B^2)/dX - Eq. (dPhi2dX)| <= {mp.nstr(worst, 3)}")
    assert worst < mp.mpf(10)**(-30)
# Z_pm > 0: all coefficients positive for z > 0; the X coefficient of Z_- has discriminant 16 - 60 < 0.
disc = sp.discriminant(3 * z**2 - 4 * z + 5, z)
print(f"   Z_-: discriminant of its X coefficient 3z^2-4z+5 is {disc} < 0, so Z_pm > 0 for z > 0, X > 0.")
assert disc < 0

print("\n4. Thermal stability at the crossings, Eqs. (dTdX-cross) and (F-product)")
J = 3 * (1 - eta) + 3 * eta * X + eta * X**2
G = 6 * (1 - eta)**2 + eta * X * (12 * (1 - eta) + (5 - 2 * eta) * X + 3 * eta * X**2 + eta * X**3)
Fp = G + 2 * J * sp.sqrt((1 - eta) * (1 + eta * (X**2 + 2 * X - 1)))
Fm = G - 2 * J * sp.sqrt((1 - eta) * (1 + eta * (X**2 + 2 * X - 1)))
prod_paper = eta**2 * X**3 * (12 * (1 - eta)**2 + 9 * X + (22 - 4 * eta) * eta * X**2
                              + (6 + 9 * eta) * eta * X**3 + 6 * eta**2 * X**4 + eta**2 * X**5)
dprod = sp.expand(sp.expand(Fp * Fm) - prod_paper)
print("   F_+ F_- minus Eq. (F-product) expands to:", dprod)
assert dprod == 0
Q2P, chiBP, kapP = wall_data_P(X, eta, P)
TRB = kapP / (2 * sp.pi * sp.sqrt(chiBP))
dT_dX_fixedP = sp.diff(TRB, X)
for sign, Cr, F in (('+', Cp, Fp), ('-', Cm, Fm)):
    P_root = 1 - (1 - D / Cr**2) / X
    worst = mp.mpf(0)
    for _ in range(60):
        Xv, ev = random.uniform(0.05, 0.95), random.uniform(0.01, 0.9)
        sub = {X: sp.Float(str(Xv), 45), eta: sp.Float(str(ev), 45)}
        Pv = P_root.subs(sub)
        Dv = D.subs(sub)
        if sp.N(Dv) <= 0 or sp.N(Pv) < 0 or sp.N(Cr.subs(sub)) <= 0:
            continue
        lhs = sp.N(dT_dX_fixedP.subs(sub).subs(P, Pv), 40)
        rhs = sp.N((F / (2 * sp.pi * X**3 * D * Cr)).subs(sub), 40)
        worst = max(worst, abs(mp.mpf(str(lhs)) - mp.mpf(str(rhs))))
    print(f"   root {sign}: |(dT_B R_B/dX)_(Phi_B,eta) at the crossing - F_pm/(2 pi X^3 D C_pm)| <= {mp.nstr(worst, 3)}")
    assert worst < mp.mpf(10)**(-28)
# Positivity on the domain: J and G have positive coefficients for 0<eta<1; so does the product polynomial.
poly = sp.Poly(sp.expand(prod_paper / (eta**2 * X**3)), X)
print("   coefficients of F_+F_-/(eta^2 X^3) in X:", poly.all_coeffs()[::-1])
print("   each is positive for 0 < eta < 1 (12(1-eta)^2, 9, (22-4eta)eta, (6+9eta)eta, 6eta^2, eta^2),")
print("   so F_+F_- > 0, and with F_+ > 0 also F_- > 0. With D > 0 and C_pm > 0 at a crossing, (dT_B/dX) > 0.")
gmin = min(sp.N(G.subs({X: sp.Rational(i, 50), eta: sp.Rational(j, 50)})) for i in range(1, 50) for j in range(1, 50))
fmin = min(sp.N(Fm.subs({X: sp.Rational(i, 50), eta: sp.Rational(j, 50)})) for i in range(1, 50) for j in range(1, 50))
print(f"   grid check (49 x 49 interior points): min G = {gmin:.4f}, min F_- = {fmin:.5f}")
assert gmin > 0 and fmin > 0
F0 = sp.simplify(Fm.subs(eta, 0)), sp.simplify(Fp.subs(eta, 0))
print(f"   at eta = 0: F_- = {F0[0]}, F_+ = {F0[1]} (the marginal case of the extremal black holes).")

print("\n5. Neutral crossing, Eqs. (neutral-curve) and (neutral-eta)")
curve = eta**2 * X**2 * (8 + X**3) + 2 * eta * (4 - 4 * X**2 - 3 * X**3) + 9 * X - 8
for sign, s in (('-', -1), ('+', 1)):
    eta_branch = ((3 * X**3 + 4 * X**2 - 4) + s * 4 * (1 - X) * sp.sqrt(1 + 2 * X + 5 * X**2 + 2 * X**3)) / (X**2 * (8 + X**3))
    res = sp.simplify(sp.expand(curve.subs(eta, eta_branch)))
    print(f"   eta_{sign}(X) inserted in Eq. (neutral-curve) gives {res}")
    assert res == 0
# Phi_B = 0 on a root means (1 - D/C_pm^2) = X. Check which branch belongs to which root.
for sign, Cr in (('-', Cm), ('+', Cp)):
    cond = sp.simplify(((1 - D / Cr**2) - X).subs({X: sp.Rational(1, 2)}))
    sols = [sp.N(sv, 20) for sv in sp.solve(sp.Eq((1 - D / Cr**2), X).subs(X, sp.Rational(1, 2)), eta) if sp.N(sv).is_real and 0 < sp.N(sv) < 1]
    print(f"   root {sign} at X = 1/2 has a neutral crossing at eta = {sols}")
eta_minus_89 = sp.N(((3 * X**3 + 4 * X**2 - 4) - 4 * (1 - X) * sp.sqrt(1 + 2 * X + 5 * X**2 + 2 * X**3)) / (X**2 * (8 + X**3)), 20).subs(X, sp.Rational(8, 9))
print(f"   eta_-(8/9) = {sp.N(eta_minus_89, 15)} (the Lambda = 0 crossing X_0 = 8/9)")
X2 = 2 - sp.sqrt(2)
print(f"   Eq. (neutral-curve) at eta = 1/2, X = 2 - sqrt2: {sp.simplify(curve.subs({eta: sp.Rational(1, 2), X: X2}))}")
T_nariai = sp.limit(((sp.sqrt(1 - eta) - D / Cp) / (sp.pi * X**2)).subs(eta, sp.Rational(1, 3)), X, 1)
print(f"   T_cross R_B as X -> 1 at eta = 1/3: {T_nariai} = {sp.N(T_nariai, 12)}  (sqrt2/(sqrt3 pi) = {sp.N(sp.sqrt(2) / (sp.sqrt(3) * sp.pi), 12)})")
assert sp.simplify(T_nariai - sp.sqrt(2) / (sp.sqrt(3) * sp.pi)) == 0
small = sp.series(omega_P(X, eta, P), X, 0, 2).removeO()
print(f"   Omega_B/R_B for small X: {sp.simplify(small)}  (paper: X [1 - (1-eta) Phi_B^2]/(4 sqrt(1-eta)))")
assert sp.simplify(small - X * (1 - (1 - eta) * P) / (4 * sp.sqrt(1 - eta))) == 0
print("\nAll checks of Appendix I passed.")
