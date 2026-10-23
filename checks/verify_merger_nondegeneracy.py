"""Check that a double zero of Y-HR on the curve of double zeros (Sect. 2.3) is a genuine merger.

Units a = 1, r_A = 1, k = kr_A^2. Along the curve of double zeros we evaluate d_r^2 (Y-HR) and d_M (Y-HR) at the
double zero. The double zero is a merger (two simple zeros disappear as M grows) where d_r^2 (Y-HR) < 0. The script
locates the sign change, compares it with the maximum of M_c H and the stationary point of Omega_k, and counts the
zeros of Y-HR in the region N>0 just below and just above M_c on both sides of it."""
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
