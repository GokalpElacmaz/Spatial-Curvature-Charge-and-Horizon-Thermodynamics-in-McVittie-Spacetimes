"""Check that the counting argument of Sect. 2.3 carries over to the charged solution with |Q| <= M (Sect. 4.1).

The charged curve of double zeros, Eqs. (turn-charged) and (Mc-charged), is scanned in mu_A at fixed Q/M from the edge of
the chart (kr_A^2 -> -1) to the endpoint Y_A = 0 fixed by the cubic Eq. (merger-end-charged). For each Q/M it checks
  (a) Omega_k is largest at the triple zero (the maximum of M_c H), decreases monotonically on the merger part beyond it,
      and exceeds one on the rest of the curve, so each background with Omega_k < 1 has a single double zero as M varies,
  (b) r^2 d^2(Y-HR)/dr^2 < 0 on the merger part, so the double zero is a maximum and increasing M removes two zeros,
  (c) Eq. (dGdmu-charged), the closed form of d(Y-HR)/dmu at fixed Q/M, against numerical differentiation,
  (d) d(Y-HR)/dM < 0 at fixed Q/M on random points of the region N > 0 with H > 0.
Units a = M = 1, so mu = sqrt(1+k r^2)/(2 r), k r^2 = x and M_c H = H.
Run: python3 checks/verify_charged_counting.py"""
import random
import mpmath as mp

mp.mp.dps = 30
QM = [0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999]


def curve(mu, q):
    """Point of the curve of double zeros at compactness mu, Eqs. (turn-charged) and (Mc-charged)."""
    lam = q * mu
    Xi = (1 + mu)**2 - lam**2
    rhs = 6 * mu * (1 + mu)**2 - 2 * lam**2 * (3 * mu + 4)
    if rhs <= 0:
        return None
    s = mp.sqrt(rhs) / Xi                      # s = 1 + k r_A^2
    x = s - 1
    N = (1 - mu**2 + lam**2) / Xi
    Y = (N - x) / s
    McH = 2 * mu * mp.sqrt(s) / Xi * Y
    Om = -4 * x * Xi**2 / (s**2 * Y**2)
    return dict(x=x, Y=Y, McH=McH, Om=Om)


def G(r, a, M, q, k, H):
    """Y - HR and the lapse N of the charged metric, Eq. (charged-metric)."""
    mu = M * mp.sqrt(1 + k * r * r) / (2 * a * r)
    lam = q * mu
    Xi = (1 + mu)**2 - lam**2
    N = (1 - mu**2 + lam**2) / Xi
    R = a * r * Xi / (1 + k * r * r)
    Y = (N - k * r * r) / (1 + k * r * r)
    return Y - H * R, N


def derivatives_at_double_zero(mu, q):
    c = curve(mu, q)
    x = c['x']
    r = mp.sqrt(1 + x) / (2 * mu)
    k = x / r**2
    f = lambda rr: G(rr, 1, 1, q, k, c['McH'])[0]
    return mp.diff(f, r, 1), r * r * mp.diff(f, r, 2)


print("Curve of double zeros at fixed Q/M")
for q in QM:
    q = mp.mpf(q)
    c3 = 1 - q * q
    cubic = lambda m: 3 * c3 * m**3 + 4 * c3 * m**2 - m - 2      # Eq. (merger-end-charged)
    hi = mp.mpf(1)
    while cubic(hi) < 0:
        hi *= 2
    mu_end = mp.findroot(cubic, (mp.mpf(0), hi), solver='bisect')
    pts = [(m, curve(m, q)) for m in (mu_end * mp.mpf(i) / 2000 for i in range(1, 2000))]
    pts = [(m, c) for m, c in pts if c]
    i_max = max(range(len(pts)), key=lambda i: pts[i][1]['McH'])
    i_om = max(range(len(pts)), key=lambda i: pts[i][1]['Om'])
    mu_t, ct = pts[i_max]
    merger = [c for m, c in pts if m > mu_t]
    rest = [c for m, c in pts if m < mu_t]
    decreasing = all(merger[i + 1]['Om'] < merger[i]['Om'] for i in range(len(merger) - 1))
    min_rest = min(c['Om'] for c in rest)
    checks = [derivatives_at_double_zero(m, q) for m, c in pts[i_max + 5::60]]
    max_first = max(abs(d1) for d1, d2 in checks)
    max_second = max(d2 for d1, d2 in checks)
    x_one = [c['x'] for c in merger if c['Om'] <= 1][0]
    print(f"  Q/M={mp.nstr(q, 4):6}: triple zero at kr_A^2={mp.nstr(ct['x'], 4)} with M_cH={mp.nstr(ct['McH'], 4)}, "
          f"Omega_k={mp.nstr(ct['Om'], 4)}; Omega_k=1 at kr_A^2={mp.nstr(x_one, 4)}; endpoint mu_A={mp.nstr(mu_end, 5)}")
    print(f"           Omega_k decreasing on merger part: {decreasing}; min Omega_k on the rest: {mp.nstr(min_rest, 4)}; "
          f"max r^2 G'' on merger part: {mp.nstr(max_second, 3)}; max |G'|: {mp.nstr(max_first, 2)}")
    assert abs(i_om - i_max) <= 1, "Omega_k and M_c H should peak together at the triple zero"
    assert decreasing and min_rest > 1 and max_second < 0 and max_first < mp.mpf('1e-20')

random.seed(1)
print("\nEq. (dGdmu-charged): d(Y-HR)/dmu at fixed Q/M, closed form against numerical differentiation")
worst = 0
for _ in range(200):
    q = mp.mpf(random.uniform(0, 0.999)); mu = mp.mpf(random.uniform(0.01, 3))
    x = mp.mpf(random.uniform(-0.5, 0.5)); Har = mp.mpf(random.uniform(0.001, 0.5))
    c = 1 - q * q
    Xi = (1 + mu)**2 - (q * mu)**2
    HR = Har * Xi / (1 + x)
    Gmu = lambda m: ((1 - m**2 + (q * m)**2) / ((1 + m)**2 - (q * m)**2) - x) / (1 + x) \
        - Har * ((1 + m)**2 - (q * m)**2) / (1 + x)
    formula = -2 * (1 + 2 * c * mu + c * mu * mu) / (Xi**2 * (1 + x)) - 2 * HR * (1 + c * mu) / Xi
    worst = max(worst, abs(mp.diff(Gmu, mu) - formula))
print(f"  max deviation {mp.nstr(worst, 3)}")
assert worst < mp.mpf('1e-20')

print("\nd(Y-HR)/dM at fixed Q/M on random points with N > 0 and H > 0")
largest = -mp.inf
for _ in range(3000):
    q = mp.mpf(random.choice(QM)); k = mp.mpf(random.uniform(-0.01, 0.01)); H = mp.mpf(random.uniform(1e-3, 0.3))
    M = mp.mpf(random.uniform(0.01, 3)); r = mp.mpf(random.uniform(0.05, 30))
    if 1 + k * r * r <= 0 or G(r, 1, M, q, k, H)[1] <= 0:
        continue
    h = mp.mpf('1e-12')
    largest = max(largest, (G(r, 1, M + h, q, k, H)[0] - G(r, 1, M - h, q, k, H)[0]) / (2 * h))
print(f"  largest value {mp.nstr(largest, 4)} (negative everywhere)")
assert largest < 0

print("\nAll charged counting checks passed.")
