"""Checks behind Sec. V D (both roots of the equal-grand-potential condition).
Needs sympy and mpmath. Run: python3 verify_two_roots.py   (about one minute)

Notation: X = R_A/R_B, eta = H^2 R_B^2, P2 = Phi_B^2,
C_pm = 2 sqrt(1-eta) +- sqrt(1+eta(X^2+2X-1)),  A = 1 - eta(1+X+X^2).
"""
import random
import sympy as sp
import mpmath as mp

mp.mp.dps = 40
A_ = lambda X, e: 1 - e*(1 + X + X*X)
C_ = lambda X, e, s: 2*mp.sqrt(1 - e) + s*mp.sqrt(1 + e*(X*X + 2*X - 1))
P2c = lambda X, e, s: 1 - (1 - A_(X, e)/C_(X, e, s)**2)/X                     # Eq. (exact-Phi)
Tc = lambda X, e, s: (mp.sqrt(1 - e) - A_(X, e)/C_(X, e, s))/(mp.pi*X*X)      # Eq. (exact-T)


def state(X, P2, e):                      # Eqs. (dS-charge), (dS-chi), (dS-TB)
    D = 1 - (1 - P2)*X; A = A_(X, e)
    u = P2*X*X*A/D; chi = (1 - X)**2*A/D
    return u, chi, (1 - u/(X*X) - 3*e*X*X)/(2*X)


def Om(X, P2, e):                         # Omega_B/R_B, Eq. (crossing-X), no root chosen
    u, chi, kap = state(X, P2, e); r = mp.sqrt(chi)
    return mp.sqrt(1 - e) - r - X*X*kap/(2*r) - u*(1 - X)/(X*r)


def TB(X, P2, e):
    u, chi, kap = state(X, P2, e)
    return kap/(2*mp.pi*mp.sqrt(chi))


def physical(X, e, s):
    return A_(X, e) > 0 and C_(X, e, s) > 0 and P2c(X, e, s) >= 0


print("1. Exact identities (sympy)")
X, eta, s_ = sp.symbols("X eta s", positive=True)
a = sp.sqrt(1 - eta); P2 = sp.Symbol("P2", positive=True)
A = 1 - eta*(1 + X + X**2); D = 1 - (1 - P2)*X
u = P2*X**2*A/D; kap = (1 - u/X**2 - 3*eta*X**2)/(2*X)
P2s = sp.solve(sp.Eq((1 - X)**2*A/D, s_**2*(1 - X)**2), P2)[0]
quad = sp.simplify(((a*s_*(1 - X) - s_**2*(1 - X)**2) - (X**2*kap/2 + u*(1 - X)/X).subs(P2, P2s))*4/(1 - X))
print("   crossing condition is s^2 - 4 sqrt(1-eta) s + 3 - eta(3+2X+X^2) = 0 :",
      sp.simplify(quad + (s_**2 - 4*a*s_ + 3 - eta*(3 + 2*X + X**2))) == 0)
L = 3*(1 - eta) + 3*eta*X + eta*X**2
P = 6*(1 - eta)**2 + eta*X*(12*(1 - eta) + (5 - 2*eta)*X + 3*eta*X**2 + eta*X**3)
W = 12*(1 - eta)**2 + 9*X + (22 - 4*eta)*eta*X**2 + (6 + 9*eta)*eta*X**3 + 6*eta**2*X**4 + eta**2*X**5
Pold = 6 + eta*(5*X**2 + 12*X - 12) + eta**2*(X**4 + 3*X**3 - 2*X**2 - 12*X + 6)
print("   P equals the polynomial part of the previous F                 :", sp.expand(P - Pold) == 0)
print("   F_+ F_- = eta^2 X^3 [12(1-eta)^2 + 9X + ...]                     :",
      sp.expand(P**2 - 4*(1 - eta)*(1 + eta*(X**2 + 2*X - 1))*L**2 - eta**2*X**3*W) == 0)
b2 = 1 + eta*(X**2 + 2*X - 1)
print("   T_cross > 0 on the second root, (a^2+eta X(1+X))^2 - a^2 b^2 = eta X^2 [1+eta X(2+X)] :",
      sp.expand((1 - eta + eta*X*(1 + X))**2 - (1 - eta)*b2 - eta*X**2*(1 + eta*X*(2 + X))) == 0)
print("   neutral curve at eta=1/2, X=2-sqrt2 :",
      sp.simplify((eta**2*X**2*(8 + X**3) + 2*eta*(4 - 4*X**2 - 3*X**3) + 9*X - 8).subs({eta: sp.Rational(1, 2), X: 2 - sp.sqrt(2)})) == 0)

print("2. Both roots against a direct solution of the unsquared condition (40-digit arithmetic)")
worst = {1: mp.mpf(0), -1: mp.mpf(0)}; n = {1: 0, -1: 0}
for ie in range(1, 20):
    e = mp.mpf(ie)/20
    for ix in range(1, 40):
        X0 = mp.mpf(ix)/40
        for s in (1, -1):
            if not physical(X0, e, s): continue
            p2 = P2c(X0, e, s)
            Xs = mp.findroot(lambda x: Om(x, p2, e), X0*(1 + mp.mpf("1e-3")))
            worst[s] = max(worst[s], abs(Xs/X0 - 1), abs(TB(Xs, p2, e)/Tc(X0, e, s) - 1)); n[s] += 1
print(f"   first root: {n[1]} points, max relative deviation {mp.nstr(worst[1], 3)};  second root: {n[-1]} points, {mp.nstr(worst[-1], 3)}")

print("3. Number of crossings at fixed (Phi_B, eta), scanning 0 < X < 1")
Phi = lambda e, s: mp.sqrt(1 - 3*e)/(2*mp.sqrt(1 - e) + s*mp.sqrt(1 + 2*e))


def count(p2, e, N=4000):
    c = 0; prev = None
    for i in range(1, N):
        x = mp.mpf(i)/N
        u, chi, kap = state(x, p2, e)
        ok = A_(x, e) > 0 and chi > 0 and kap > 0
        v = mp.sign(Om(x, p2, e)) if ok else None
        if v is not None and prev is not None and v != prev: c += 1
        prev = v
    return c
for e in (mp.mpf("0.1"), mp.mpf("0.3")):
    lo, hi, top = Phi(e, 1), Phi(e, -1), 1/mp.sqrt(1 - e)
    row = [(lab, count(f**2, e)) for lab, f in (("0.5 Phi_+", lo/2), ("0.99 Phi_+", 0.99*lo), ("1.01 Phi_+", 1.01*lo), ("gap middle", (lo + hi)/2),
                                               ("0.99 Phi_-", 0.99*hi), ("midway to top", (hi + top)/2), ("1.01 top", 1.01*top))]
    print(f"   eta={mp.nstr(e, 2)}: Phi_+={mp.nstr(lo, 6)} Phi_-={mp.nstr(hi, 6)} top={mp.nstr(top, 6)} ->", row)
for e in (mp.mpf("0.4"), mp.mpf("0.7")):
    top = 1/mp.sqrt(1 - e)
    print(f"   eta={mp.nstr(e, 2)}: top={mp.nstr(top, 6)} ->", [(lab, count(f**2, e)) for lab, f in (("0", 0), ("0.5 top", top/2), ("0.99 top", 0.99*top), ("1.01 top", 1.01*top))])

print("4. Monotonic potential and positive heat capacity on both roots (random physical points)")
F = lambda X, e, s: (6*(1 - e)**2 + e*X*(12*(1 - e) + (5 - 2*e)*X + 3*e*X**2 + e*X**3)
                     + s*2*mp.sqrt((1 - e)*(1 + e*(X*X + 2*X - 1)))*(3*(1 - e) + 3*e*X + e*X*X))
random.seed(7); dev = mp.mpf(0); cnt = {1: 0, -1: 0}; sgn = {1: set(), -1: set()}
while min(cnt.values()) < 300:
    s = random.choice([1, -1])
    e = mp.mpf(random.uniform(1e-3, 0.333 if s == 1 else 0.999)); x = mp.mpf(random.uniform(0.85 if s == 1 else 1e-3, 0.999))
    if not physical(x, e, s): continue
    p2 = P2c(x, e, s)
    form = F(x, e, s)/(2*mp.pi*x**3*A_(x, e)*C_(x, e, s))
    dev = max(dev, abs(form/mp.diff(lambda y: TB(y, p2, e), x) - 1))
    assert form > 0
    sgn[s].add(int(mp.sign(mp.diff(lambda y: P2c(y, e, s), x)))); cnt[s] += 1
print(f"   points: {cnt}; max |formula / numerical derivative - 1| = {mp.nstr(dev, 3)}; all dT_B/dX > 0")
print(f"   sign of dPhi_B^2/dX: first root {sgn[1]}, second root {sgn[-1]}")

print("5. Limits")
for e in (mp.mpf("0.1"), mp.mpf("0.2"), mp.mpf("0.3")):
    x1 = 1 - mp.mpf("1e-25")
    print(f"   eta={mp.nstr(e, 2)}: X->1 |Phi_B| = {mp.nstr(mp.sqrt(P2c(x1, e, 1)), 10)}, {mp.nstr(mp.sqrt(P2c(x1, e, -1)), 10)}  vs Phi_+- = {mp.nstr(Phi(e, 1), 10)}, {mp.nstr(Phi(e, -1), 10)};"
          f"  X->0 second root Phi^2 = {mp.nstr(P2c(mp.mpf('1e-20'), e, -1), 10)} vs 1/(1-eta) = {mp.nstr(1/(1 - e), 10)}")
en = lambda x, s: (4*x*x + 3*x**3 - 4 + s*4*(1 - x)*mp.sqrt(1 + 2*x + 5*x*x + 2*x**3))/(x*x*(8 + x**3))
x1 = 1 - mp.mpf("1e-20")
print(f"   neutral, eta->1/3: T_cross R_B = {mp.nstr(Tc(x1, en(x1, -1), 1), 12)} (first), {mp.nstr(Tc(x1, en(x1, 1), -1), 12)} (second), sqrt2/(sqrt3 pi) = {mp.nstr(mp.sqrt(2)/(mp.sqrt(3)*mp.pi), 12)}")
print(f"   neutral curve: X=8/9 -> eta = {mp.nstr(en(mp.mpf(8)/9, -1), 3)} and {mp.nstr(en(mp.mpf(8)/9, 1), 8)};  X->0 -> eta = {mp.nstr(en(mp.mpf('1e-12'), 1), 8)}")
e = mp.mpf("1e-4"); xs = [mp.mpf(i)/10 for i in range(1, 10)]
print(f"   second root at eta=1e-4: (Phi_B^2-1)/eta in [{mp.nstr(min((P2c(x, e, -1) - 1)/e for x in xs), 6)}, {mp.nstr(max((P2c(x, e, -1) - 1)/e for x in xs), 6)}],"
      f" 2 pi T_cross R_B/eta in [{mp.nstr(min(2*mp.pi*Tc(x, e, -1)/e for x in xs), 6)}, {mp.nstr(max(2*mp.pi*Tc(x, e, -1)/e for x in xs), 6)}]")
