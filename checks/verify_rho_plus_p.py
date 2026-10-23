"""Check when rho+p vanishes on a whole surface of constant t (Sect. 3).

With a central mass or charge the compactness mu (or lambda) takes a continuous range of values on the surface,
so rho+p = 0 there requires every coefficient of a polynomial in mu to vanish. The result is k = 0 and dH/dt = 0.
For M = Q = 0 (FLRW) rho+p = 0 requires only dH/dt = 4k/a^2, which is de Sitter space in a curved slicing."""
import sympy as sp

mu, k, Hd, a, q, L = sp.symbols('mu k Hdot a q L', real=True)


def coefficient_conditions(expr, var):
    poly = sp.Poly(sp.expand(sp.numer(sp.together(expr * a**2))), var)
    return [sp.factor(c) for c in poly.all_coeffs()]


# neutral, Eq. (rho-p)
N = (1 - mu) / (1 + mu)
e_neutral = 12 * k / (a**2 * (1 + mu)**5) - 2 * Hd / N - 4 * k / (a**2 * N * (1 + mu)**6)
print('neutral, M != 0:', sp.solve(coefficient_conditions(e_neutral, mu), [k, Hd], dict=True))
print('neutral, M = 0 :  8 pi (rho+p) =', sp.simplify(e_neutral.subs(mu, 0)))

# charged, Eqs. (charged-density), (charged-pressure), with lambda = q mu
lam = q * mu
Xi = (1 + mu)**2 - lam**2
Nc = (1 - mu**2 + lam**2) / Xi
e_charged = 12 * k * (1 + mu) / (a**2 * Xi**3) - 2 * Hd / Nc - 4 * k / (a**2 * Nc * Xi**3)
print('charged, M != 0:', sp.solve(coefficient_conditions(e_charged, mu), [k, Hd], dict=True))

# charged with M = 0: mu = 0 and lambda varies
Xi0 = 1 - L**2
N0 = (1 + L**2) / Xi0
e_m0 = 12 * k / (a**2 * Xi0**3) - 2 * Hd / N0 - 4 * k / (a**2 * N0 * Xi0**3)
print('M = 0, Q != 0 :', sp.solve(coefficient_conditions(e_m0, L), [k, Hd], dict=True))

# de Sitter in closed slicing: a = (2 sqrt(k)/h) cosh(h t) gives Hdot = 4k/a^2
t, h, kk = sp.symbols('t h kk', positive=True)
aa = 2 * sp.sqrt(kk) / h * sp.cosh(h * t)
HH = sp.diff(aa, t) / aa
print('closed de Sitter: Hdot - 4k/a^2 =', sp.simplify(sp.diff(HH, t) - 4 * kk / aa**2))
