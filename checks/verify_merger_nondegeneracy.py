"""Check that a double zero of Y-HR on the curve of double zeros (Sect. 2.3) is a genuine merger.

Units a = 1, r_A = 1, k = kr_A^2. Along the curve of double zeros we evaluate d_r^2 (Y-HR) and d_M (Y-HR) at the
double zero. The double zero is a merger (two simple zeros disappear as M grows) where d_r^2 (Y-HR) < 0. The script
locates the sign change, compares it with the maximum of M_c H and the stationary point of Omega_k, and counts the
zeros of Y-HR in the region N>0 just below and just above M_c on both sides of it.

The last part derives the closed forms of Sect. 2.3 with sympy, in the variable u = (1 + kr_A^2)(1 + mu_-):
r_A^2 d_r^2 (Y-HR) = (5u^2 - 24u + 18)/(6u), Eq. (second-derivative), the same factor in dOmega_k/du and d(M_c H)/du,
and the proportionality of d_t chi at the double zero to the numerator of Eq. (dRdt)."""
import mpmath as mp
import numpy as np

mp.mp.dps = 40


def mu_minus(w):
    return (2 - 2 * w - w**2 - mp.sqrt(3 - 12 * w - 6 * w**2)) / (1 + w)**2


def background(w):
    muA = mu_minus(w)
    M = 2 * muA / mp.sqrt(1 + w)
    YA = (1 - w - (1 + w) * muA) / ((1 + w) * (1 + muA))
    H = YA / ((1 + muA)**2 / (1 + w))
    return M, H, YA, muA


def Y_minus_HR(r, M, H, w):
    u = 1 + w * r * r
    mu = M * mp.sqrt(u) / (2 * r)
    N = (1 - mu) / (1 + mu)
    return (N - w * r * r) / u - H * r * (1 + mu)**2 / u


def second_derivative(w):
    M, H, _, _ = background(w)
    return mp.diff(lambda r: Y_minus_HR(r, M, H, w), 1, 2)


def McH(w):
    _, _, YA, _ = background(w)
    return (1 + w)**mp.mpf(2.5) / 3 * YA


def Omega_k(w):
    _, _, YA, muA = background(w)
    return -4 * w * (1 + muA)**4 / ((1 + w)**2 * YA**2)


for w in ['-0.9', '-0.5', '-0.2', '-0.052', '0', '0.1', '0.19']:
    w = mp.mpf(w)
    M, H, _, _ = background(w)
    dM = mp.diff(lambda m: Y_minus_HR(mp.mpf(1), m, H, w), M)
    print(f'kr_A^2={mp.nstr(w, 4):>7}:  r_A^2 d_r^2(Y-HR) = {mp.nstr(second_derivative(w), 6):>10}   d_M(Y-HR) = {mp.nstr(dM, 5)}')
w_star = mp.findroot(second_derivative, mp.mpf('-0.19'))
w_max = mp.findroot(lambda w: mp.diff(McH, w), mp.mpf('-0.187'))
print('d_r^2(Y-HR) = 0 at kr_A^2 =', mp.nstr(w_star, 10), '; max of M_c H at', mp.nstr(w_max, 10),
      '; dOmega_k/dkr_A^2 there =', mp.nstr(mp.diff(Omega_k, w_star), 3), '; Omega_k =', mp.nstr(Omega_k(w_star), 5))


def count_zeros(w, factor):
    M, H, _, _ = background(w)
    M = M * mp.mpf(factor)
    rmax = 1 / mp.sqrt(-w) * mp.mpf('0.9999') if w < 0 else mp.mpf(30)
    rs = [mp.mpf(x) for x in np.linspace(0.05, float(rmax), 40001)]
    vals = []
    for r in rs:
        mu = M * mp.sqrt(1 + w * r * r) / (2 * r)
        vals.append(None if mu >= 1 else Y_minus_HR(r, M, H, w))
    return sum(1 for i in range(len(vals) - 1)
               if vals[i] is not None and vals[i + 1] is not None and vals[i] * vals[i + 1] < 0)


for w in ['0.1', '-0.5']:
    print(f'kr_A^2={w}: zeros for M = 0.98 M_c: {count_zeros(mp.mpf(w), "0.98")}, for M = 1.02 M_c: {count_zeros(mp.mpf(w), "1.02")}')


# closed forms in u (Sect. 2.3) ------------------------------------------------------------------------------------
import sympy as sp

u, r = sp.symbols('u r', positive=True)
w_u = -(u**2 - 6 * u + 6) / (u**2 + 6)
mu_u = u**2 / 6
YA_u = (1 - w_u - (1 + w_u) * mu_u) / ((1 + w_u) * (1 + mu_u))
H_u = YA_u * (1 + w_u) / (1 + mu_u)**2
M_u = 2 * mu_u / sp.sqrt(1 + w_u)
mu_r = M_u * sp.sqrt(1 + w_u * r**2) / (2 * r)
F = ((1 - mu_r) / (1 + mu_r) - w_u * r**2) / (1 + w_u * r**2) - H_u * r * (1 + mu_r)**2 / (1 + w_u * r**2)
assert sp.simplify(F.subs(r, 1)) == 0 and sp.simplify(sp.diff(F, r).subs(r, 1)) == 0
F2 = sp.factor(sp.simplify(sp.diff(F, r, 2).subs(r, 1)))
print('r_A^2 d_r^2(Y-HR) =', F2)
McH_u = sp.factor(sp.simplify((1 + w_u)**sp.Rational(5, 2) / 3 * YA_u))
Om_u = sp.factor(sp.simplify(-4 * w_u * (1 + mu_u)**4 / ((1 + w_u)**2 * YA_u**2)))
print('M_c H =', McH_u, '  d/du =', sp.factor(sp.diff(McH_u, u)))
print('Omega_k =', Om_u, '  d/du =', sp.factor(sp.diff(Om_u, u)))
u_star = (12 - 3 * sp.sqrt(6)) / 5
print('turning point: kr_A^2 =', sp.nsimplify(sp.simplify(w_u.subs(u, u_star))), '=', sp.N(w_u.subs(u, u_star), 12),
      ' M_c H =', sp.N(McH_u.subs(u, u_star), 12), ' Omega_k =', sp.N(Om_u.subs(u, u_star), 12))

# d_t chi at the double zero versus the numerator of Eq. (dRdt), for a(t) = exp(H0 t + Hd t^2/2)


def dt_chi_and_numerator(w, Hd):
    M, H0, _, muA = background(w)
    RA = (1 + muA)**2 / (1 + w)

    def chi(t, rr):
        a = mp.e**(H0 * t + Hd * t * t / 2)
        H = H0 + Hd * t
        uu = 1 + w * rr * rr
        mu = M * mp.sqrt(uu) / (2 * a * rr)
        N = (1 - mu) / (1 + mu)
        return ((N - w * rr * rr) / uu)**2 - H**2 * (a * rr * (1 + mu)**2 / uu)**2
    numerator = RA**2 * Hd * (1 + w)**2 * (1 + muA)**2 + 2 * w * (3 * muA - 2)
    return mp.diff(lambda t: chi(t, mp.mpf(1)), 0), numerator


for w, Hd in [('0', '0'), ('0', '-0.01'), ('0.1', '0'), ('0.1', '-0.01'), ('-0.03', '0')]:
    dtc, num = dt_chi_and_numerator(mp.mpf(w), mp.mpf(Hd))
    ratio = mp.nstr(dtc / num, 8) if num != 0 else 'numerator 0, d_t chi = ' + mp.nstr(dtc, 3)
    print(f'kr_A^2={w:>5}, Hdot={Hd:>6}: d_t chi / numerator of Eq. (dRdt) = {ratio}')
