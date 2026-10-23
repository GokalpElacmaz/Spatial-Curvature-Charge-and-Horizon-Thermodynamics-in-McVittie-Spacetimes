"""Check the split of the Hayward-Kodama surface gravity, Eqs. (kappa-split) and (kappa-lambda-charged),
and the merger relation Eq. (dchi-kappa), with 50-digit arithmetic.

On a marginal sphere of the Vaidya-Shah metric (curved, charged, with dH/dt != 0):
  kappa_A        = kappa_Lambda + 2 pi R_A (rho_A + p_A),
  kappa_Lambda   = 1/(2 R_A) - 4 pi R_A rho_A - q_A^2/(2 R_A^3),
  d_r chi |_A    = 2 kappa_Lambda d_r R |_A   (on a surface of constant t).
kappa is computed directly from its definition, Eq. (Hayward-Kodama-sg). The script also checks the k=0 identity
2 pi R_A (rho_A + p_A) = -Hdot/(2H) used in Sect. 5.6."""
import mpmath as mp

mp.mp.dps = 50
pi = mp.pi


def fields(r, t, P):
    a = P['a0'] * mp.e**(P['H0'] * t + P['Hd'] * t * t / 2)
    H = P['H0'] + P['Hd'] * t
    k, M, Q = P['k'], P['M'], P['Q']
    u = 1 + k * r * r
    mu = M * mp.sqrt(u) / (2 * a * r)
    lam = Q * mp.sqrt(u) / (2 * a * r)
    Xi = (1 + mu)**2 - lam**2
    N = (1 - mu**2 + lam**2) / Xi
    B = a * Xi / u
    rho = (3 * H * H + 12 * k * (1 + mu) / (a * a * Xi**3)) / (8 * pi)
    p = (-2 * P['Hd'] / N - 3 * H * H - 4 * k / (a * a * N * Xi**3)) / (8 * pi)
    return dict(N=N, B=B, R=B * r, rho=rho, p=p, q=Q / u**mp.mpf(1.5), H=H)


def check(P, r_guess):
    t = mp.mpf(0)
    R = lambda r, tt: fields(r, tt, P)['R']

    def Y_minus_HR(r):
        F = fields(r, t, P)
        return mp.diff(lambda x: R(x, t), r) / F['B'] - F['H'] * F['R']

    rA = mp.findroot(Y_minus_HR, mp.mpf(r_guess))
    F = fields(rA, t, P)
    g1 = lambda r, tt: fields(r, tt, P)['N'] * mp.diff(lambda x: R(x, tt), r) / fields(r, tt, P)['B']
    g2 = lambda r, tt: fields(r, tt, P)['B'] * mp.diff(lambda s: R(r, s), tt) / fields(r, tt, P)['N']
    kappa = (mp.diff(lambda x: g1(x, t), rA) - mp.diff(lambda s: g2(rA, s), t)) / (2 * F['N'] * F['B'])
    kL = 1 / (2 * F['R']) - 4 * pi * F['R'] * F['rho'] - F['q']**2 / (2 * F['R']**3)
    chi = lambda r: ((mp.diff(lambda x: R(x, t), r) / fields(r, t, P)['B'])**2
                     - (mp.diff(lambda s: R(r, s), t) / fields(r, t, P)['N'])**2)
    print(f"k={mp.nstr(P['k'], 3)} Q={mp.nstr(P['Q'], 3)} Hdot={mp.nstr(P['Hd'], 3)}: "
          f"kappa - [kappa_Lambda + 2piR(rho+p)] = {mp.nstr(kappa - (kL + 2 * pi * F['R'] * (F['rho'] + F['p'])), 3)}, "
          f"d_r chi - 2 kappa_Lambda d_r R = {mp.nstr(mp.diff(chi, rA) - 2 * kL * mp.diff(lambda x: R(x, t), rA), 3)}")
    if P['k'] == 0:
        print('   k=0: 2 pi R (rho+p) + Hdot/(2H) =', mp.nstr(2 * pi * F['R'] * (F['rho'] + F['p']) + P['Hd'] / (2 * F['H']), 3))


m = mp.mpf
check(dict(a0=m(1), H0=m('0.1'), Hd=m('-0.006'), k=m('0.003'), M=m(1), Q=m('0.8')), '2.2')
check(dict(a0=m(1), H0=m('0.09'), Hd=m('-0.004'), k=m('-0.0015'), M=m(1), Q=m('0.5')), '2.2')
check(dict(a0=m(1), H0=m('0.1'), Hd=m('-0.007'), k=m(0), M=m(1), Q=m('0.6')), '2.2')
