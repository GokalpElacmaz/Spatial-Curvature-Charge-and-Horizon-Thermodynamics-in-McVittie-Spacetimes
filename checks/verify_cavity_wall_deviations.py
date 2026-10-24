"""Check the deviations from the boundary first law outside the flat stationary case, Eqs. (Hdot-residual) and
(k-residual) of Sect. 5.6, derived in Appendix G, by direct numerical differentiation at 30 digits.

Neutral curved McVittie metric at a comparison time t0 with a, H, Hdot and k given. Wall quantities of Appendix G:
  chi_B = Y_B^2 - H^2 R_B^2,   T_B = kappa_A/(2 pi sqrt(chi_B)),   S = pi R_A^2,
  E_B = R_B (sqrt(1 - R_B^2/R_H^2) - sqrt(chi_B)),   R_H^2 = 1/(H^2 + 4k/a^2),
with kappa_A from Eq. (kappa-closed) on the smaller marginal sphere. delta compares the solutions with M - h and M + h
at fixed areal radius R_B of the wall and fixed a, H, Hdot, k (central differences, h = 1e-8 M).
Part 1, k = 0 and Hdot != 0:  delta E_B - T_B delta S = Hdot R_A delta R_A/(2 H sqrt(chi_B)), Eq. (Hdot-residual), exactly.
Part 2, k != 0 at constant H:  delta E_B - T_B delta S ~ -(delta M/sqrt(chi_B)) (15/2 k r_B^2 - l/(4 eps)), Eq. (k-residual),
  at first order in k and leading order in eps = MH, M/R_B and l/eps. The two terms are checked separately against
  their un-expanded forms, the wall factor (1 - 5kr_B^2)/(1 + kr_B^2)^{5/2} from Eq. (response) and the horizon factor
  1 - l/[2(l + 2 eps)] from Eq. (kappa-leading), and the ratio of the numerical residual to Eq. (k-residual) is followed
  along a sequence of parameters in which eps, M/R_B and k r_B^2 shrink together.
Run: python3 checks/verify_cavity_wall_deviations.py"""
import mpmath as mp

mp.mp.dps = 30
pi = mp.pi


def metric(r, a, M, k):
    u = 1 + k * r * r
    mu = M * mp.sqrt(u) / (2 * a * r)
    N = (1 - mu) / (1 + mu)
    B = a * (1 + mu)**2 / u
    R = B * r
    Y = (N - k * r * r) / u
    return mu, N, B, R, Y


def smaller_marginal_sphere(a, M, k, H):
    """Zero of Y - HR just outside the lapse-zero surface mu = 1, which lies at r = M/(2a sqrt(1-l))."""
    l = k * M * M / (4 * a * a)
    r0 = M / (2 * a * mp.sqrt(1 - l))
    G = lambda r: metric(r, a, M, k)[4] - H * metric(r, a, M, k)[3]
    lo, hi = r0 * (1 + mp.mpf('1e-25')), r0 * (1 + mp.mpf('1e-6'))
    while G(hi) < 0:
        hi = r0 + 2 * (hi - r0)
    return mp.findroot(G, (lo, hi), solver='bisect', tol=mp.mpf(10)**(-28))


def kappa_A(rA, a, M, k, H, Hd):
    """Eq. (kappa-closed), evaluated on a marginal sphere with Y = HR."""
    mu, N, B, R, Y = metric(rA, a, M, k)
    w = k * rA * rA
    return (2 + w * (5 * N - 4 * w - 4) - 6 * N * N) / (4 * R * (1 + w)**2) - R / (2 * N) * (Hd + k / (2 * a * a * (1 + mu)**4))


def wall_radius(RB, a, M, k, rA):
    """Isotropic radius r_B with R(r_B) = R_B on the branch where R increases with r."""
    f = lambda r: metric(r, a, M, k)[3] - RB
    hi = rA * 2
    while f(hi) < 0:
        hi *= 2
    return mp.findroot(f, (rA, hi), solver='bisect', tol=mp.mpf(10)**(-28))


def wall_quantities(RB, a, M, k, H, Hd):
    rA = smaller_marginal_sphere(a, M, k, H)
    RA = metric(rA, a, M, k)[3]
    rB = wall_radius(RB, a, M, k, rA)
    _, _, _, _, YB = metric(rB, a, M, k)
    chiB = YB * YB - H * H * RB * RB
    RH2 = 1 / (H * H + 4 * k / (a * a))
    EB = RB * (mp.sqrt(1 - RB * RB / RH2) - mp.sqrt(chiB))
    S = pi * RA * RA
    TB = kappa_A(rA, a, M, k, H, Hd) / (2 * pi * mp.sqrt(chiB))
    return dict(rA=rA, RA=RA, rB=rB, chiB=chiB, EB=EB, S=S, TB=TB)


def residual(RB, a, M, k, H, Hd, h=None):
    """(delta E_B - T_B delta S)/delta M by central differences at fixed R_B, a, H, Hdot, k; also delta R_A/delta M."""
    h = h or M * mp.mpf('1e-8')
    W0 = wall_quantities(RB, a, M, k, H, Hd)
    Wp = wall_quantities(RB, a, M + h, k, H, Hd)
    Wm = wall_quantities(RB, a, M - h, k, H, Hd)
    dEB = (Wp['EB'] - Wm['EB']) / (2 * h)
    dS = (Wp['S'] - Wm['S']) / (2 * h)
    dRA = (Wp['RA'] - Wm['RA']) / (2 * h)
    return W0, dEB, dS, dRA, dEB - W0['TB'] * dS


print("Part 1: k = 0, Hdot != 0, Eq. (Hdot-residual)")
a, M = mp.mpf(1), mp.mpf(1)
for H, Hd_over_H2, RB in ((mp.mpf('0.01'), mp.mpf('-0.5'), mp.mpf(20)), (mp.mpf('0.05'), mp.mpf('-1.5'), mp.mpf(8)),
                          (mp.mpf('0.02'), mp.mpf('0.3'), mp.mpf(30))):
    Hd = Hd_over_H2 * H * H
    W, dEB, dS, dRA, res = residual(RB, a, M, mp.mpf(0), H, Hd)
    paper = Hd * W['RA'] * dRA / (2 * H * mp.sqrt(W['chiB']))
    print(f"  H={H} Hdot/H^2={Hd_over_H2} R_B={RB}: residual/dM = {mp.nstr(res, 12)}, Eq. (Hdot-residual) = {mp.nstr(paper, 12)}, "
          f"difference {mp.nstr(res - paper, 2)}")
    assert abs(res - paper) < mp.mpf('1e-9')

print("\nPart 2: k != 0 at constant H, Eq. (k-residual)")
print("  columns: eps = MH, l/eps, M/R_B, Omega_k, k r_B^2 | numerical residual*sqrt(chi_B)/dM, Eq. (k-residual), "
      "un-expanded form, ratio numerical/Eq. (k-residual)")
print("  un-expanded form = (1-5kr_B^2)/(1+kr_B^2)^{5/2} - 1 + l/[2(l+2eps)]")
cases = [  # (eps, l/eps, R_B/M)
    (mp.mpf('1e-2'), mp.mpf('0.05'), mp.mpf(5)),
    (mp.mpf('3e-3'), mp.mpf('0.05'), mp.mpf(8)),
    (mp.mpf('1e-3'), mp.mpf('0.05'), mp.mpf(10)),
    (mp.mpf('3e-4'), mp.mpf('0.05'), mp.mpf(20)),
    (mp.mpf('1e-4'), mp.mpf('0.05'), mp.mpf(30)),
    (mp.mpf('1e-3'), mp.mpf('0.2'), mp.mpf(10)),
    (mp.mpf('1e-3'), mp.mpf('0.01'), mp.mpf(10)),
    (mp.mpf('1e-2'), mp.mpf('-5e-4'), mp.mpf(5)),    # k < 0: Omega_k = -16 (l/eps)/eps = 0.8 <= 1, so |l/eps| <= eps/16
    (mp.mpf('1e-3'), mp.mpf('-5e-5'), mp.mpf(10)),   # k < 0 with Omega_k = 0.8
    (mp.mpf('1e-3'), mp.mpf('-0.05'), mp.mpf(10)),   # k < 0 with Omega_k > 1: negative background density, skipped
]
for eps, ratio_le, RBoverM in cases:
    H = eps / M
    l = ratio_le * eps
    k = 4 * a * a * l / (M * M)
    Omega_k = -4 * k / (a * a * H * H)
    RB = RBoverM * M
    if Omega_k > 1:
        print(f"  eps={mp.nstr(eps,3)} l/eps={mp.nstr(ratio_le,3)}: Omega_k = {mp.nstr(Omega_k,3)} > 1, negative background density, skipped")
        continue
    W, dEB, dS, dRA, res = residual(RB, a, M, k, H, mp.mpf(0))
    w = k * W['rB']**2
    sq = mp.sqrt(W['chiB'])
    paper = -(mp.mpf(15) / 2 * w - l / (4 * eps))
    unexp = (1 - 5 * w) / (1 + w)**mp.mpf('2.5') - 1 + l / (2 * (l + 2 * eps))
    num = res * sq
    wall_num, wall_paper = dEB * sq, (1 - 5 * w) / (1 + w)**mp.mpf('2.5')
    hor_num, hor_paper = W['TB'] * dS * sq, 1 - l / (2 * (l + 2 * eps))
    print(f"  {mp.nstr(eps,3):>6} {mp.nstr(ratio_le,3):>6} {mp.nstr(1/RBoverM,3):>6} {mp.nstr(Omega_k,3):>7} {mp.nstr(w,3):>8} | "
          f"{mp.nstr(num,6):>11} {mp.nstr(paper,6):>11} {mp.nstr(unexp,6):>11}  ratio {mp.nstr(num/paper,5)}")
    print(f"      wall term   delta E_B sqrt(chi_B)/dM = {mp.nstr(wall_num,7)}  vs (1-5kr_B^2)/(1+kr_B^2)^(5/2) = {mp.nstr(wall_paper,7)}")
    print(f"      horizon term T_B delta S sqrt(chi_B)/dM = {mp.nstr(hor_num,7)}  vs 1 - l/[2(l+2eps)] = {mp.nstr(hor_paper,7)}"
          f"  (1 - l/(4eps) = {mp.nstr(1 - l/(4*eps),7)})")

print("\nEq. (response): -(d chi/dM) at fixed areal R and M -> 0, against 2(1-5kr^2)/(a r (1+kr^2)^{3/2})")
for k, r in ((mp.mpf('0.01'), mp.mpf(3)), (mp.mpf('-0.002'), mp.mpf(5)), (mp.mpf('0.2'), mp.mpf('0.7'))):
    H = mp.mpf('0.03')
    Rfix = metric(r, a, mp.mpf(0), k)[3]
    def chi_at_fixed_R(Mv):
        rr = mp.findroot(lambda x: metric(x, a, Mv, k)[3] - Rfix, r)
        Y = metric(rr, a, Mv, k)[4]
        return Y * Y - H * H * Rfix * Rfix
    h = mp.mpf('1e-10')
    num = -(chi_at_fixed_R(h) - chi_at_fixed_R(-h)) / (2 * h)
    paper = 2 * (1 - 5 * k * r * r) / (a * r * (1 + k * r * r)**mp.mpf('1.5'))
    print(f"  k={k} r={r}: numerical {mp.nstr(num, 12)}, Eq. (response) {mp.nstr(paper, 12)}, difference {mp.nstr(num - paper, 2)}")
    assert abs(num - paper) < mp.mpf('1e-8')
print("\nDone.")
