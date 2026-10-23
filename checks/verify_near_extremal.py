"""Check the near-extremal statements of Sect. 4.1 at l = 0, where the marginal condition is (R-M)^2 = M^2 g^2 + H^2 R^4.

1. Series of the outer root in eps = MH at fixed g = sqrt(1-Q^2/M^2): coefficients of eps^2, eps^4, eps^6
   (Eq. (bh-charged-eps4)).
2. The example g = 1e-4, eps = 3e-3: the eps^2 term of Eq. (bh-charged) is 0.045, yet the expansion is 4.2% off.
3. Accuracy of the near-extremal form R_s/M = 1 + sqrt(g^2+eps^2) + 2 eps^2, Eq. (bh-near-extremal)."""
import numpy as np
import sympy as sp
from scipy.optimize import brentq

g, e = sp.symbols('gamma epsilon', positive=True)
c2, c4, c6 = sp.symbols('c2 c4 c6')
rho = 1 + g + c2 * e**2 + c4 * e**4 + c6 * e**6
F = sp.expand((rho - 1)**2 - g**2 - e**2 * rho**4)
sol = {}
for c, n in [(c2, 2), (c4, 4), (c6, 6)]:
    sol[c] = sp.factor(sp.solve(sp.expand(F.subs(sol)).coeff(e, n), c)[0])
    print(f'coefficient of eps^{n}:', sol[c])


def exact(eps, gam):
    f = lambda r: (r - 1) - np.sqrt(gam**2 + eps**2 * r**4)
    rs = 1 + gam + np.linspace(0, 3, 300001)[1:]
    i = np.where(f(rs) > 0)[0][0]
    return brentq(f, 1 + gam if i == 0 else rs[i - 1], rs[i], xtol=1e-15)


def expansion(eps, gam):
    return (1 + gam) + eps**2 * (1 + gam)**4 / (2 * gam)


def near_extremal(eps, gam):
    return 1 + np.sqrt(gam**2 + eps**2) + 2 * eps**2


gam, eps = 1e-4, 3e-3
print(f'example: eps^2 term = {eps**2 * (1 + gam)**3 / (2 * gam):.4f}, Eq.(bh-charged) error = '
      f'{expansion(eps, gam) / exact(eps, gam) - 1:+.4f}, near-extremal error = {near_extremal(eps, gam) / exact(eps, gam) - 1:+.1e}')
for emax in [0.05, 0.1]:
    worst = max(abs(near_extremal(x, y) / exact(x, y) - 1)
                for y in np.linspace(0, 0.2, 41) for x in np.linspace(1e-4, emax, 40))
    print(f'near-extremal form, gamma <= 0.2, eps <= {emax}: max relative error {worst:.2e}')
