"""Check that the Vaidya-Shah metric, Eqs. (charged-metric), with the fluid density and pressure of
Eqs. (charged-density), (charged-pressure) and the Gauss charge q(r)=Q/(1+kr^2)^{3/2} of Eq. (enclosed-charge),
satisfies the Einstein equations G^a_b = 8 pi (T_fluid + T_EM)^a_b, including G^t_r = 0, at random points."""
import random
import sympy as sp

t, r, th, ph = sp.symbols('t r theta phi')
k, M, Q, H0, Hd = sp.symbols('k M Q H0 Hd')
a = sp.exp(H0*t + Hd*t**2/2)
u = 1 + k*r**2
mu = M*sp.sqrt(u)/(2*a*r); lam = Q*sp.sqrt(u)/(2*a*r)
Xi = (1 + mu)**2 - lam**2
N = (1 - mu**2 + lam**2)/Xi
B = a*Xi/u
x = [t, r, th, ph]
g = sp.diag(-N**2, B**2, B**2*r**2, B**2*r**2*sp.sin(th)**2)
ginv = g.inv()
# Christoffel symbols and Ricci tensor (diagonal metric)
Gam = [[[sum(ginv[i, l]*(sp.diff(g[l, j], x[kk]) + sp.diff(g[l, kk], x[j]) - sp.diff(g[j, kk], x[l]))
             for l in range(4))/2 for kk in range(4)] for j in range(4)] for i in range(4)]
def Ric(i, j):
    return (sum(sp.diff(Gam[l][i][j], x[l]) for l in range(4)) - sum(sp.diff(Gam[l][i][l], x[j]) for l in range(4))
            + sum(Gam[l][l][m]*Gam[m][i][j] for l in range(4) for m in range(4))
            - sum(Gam[l][j][m]*Gam[m][i][l] for l in range(4) for m in range(4)))
Rdd = sp.Matrix(4, 4, lambda i, j: Ric(i, j) if i == j or (i, j) in [(0, 1), (1, 0)] else 0)
Rs = sum(ginv[i, i]*Rdd[i, i] for i in range(4))
def G_mixed(i, j):  # G^i_j for the components we need
    return sum(ginv[i, l]*Rdd[l, j] for l in range(4)) - (Rs/2 if i == j else 0)
H = sp.diff(a, t)/a; Hdot = sp.diff(H, t)
rho = (3*H**2 + 12*k*(1 + mu)/(a**2*Xi**3))/(8*sp.pi)
p = (-2*Hdot/N - 3*H**2 - 4*k/(a**2*N*Xi**3))/(8*sp.pi)
R = B*r; q = Q/u**sp.Rational(3, 2); eem = q**2/(8*sp.pi*R**4)
targets = {(0, 0): 8*sp.pi*(-rho - eem), (1, 1): 8*sp.pi*(p - eem), (2, 2): 8*sp.pi*(p + eem), (0, 1): 0}
random.seed(3)
worst = 0
for trial in range(4):
    vals = {k: random.choice([-1, 1])*random.uniform(0.001, 0.004), M: 1.0, Q: random.uniform(0, 0.8),
            H0: random.uniform(0.05, 0.12), Hd: -random.uniform(0, 0.01), t: random.uniform(-0.5, 0.5),
            r: random.uniform(2.5, 6.0), th: 1.1}
    for (i, j), T in targets.items():
        lhs = complex(G_mixed(i, j).subs(vals).evalf(30)); rhs = complex(sp.sympify(T).subs(vals).evalf(30))
        scale = max(1e-12, abs(rhs), abs(lhs))
        worst = max(worst, abs(lhs - rhs)/scale if (i, j) != (0, 1) else abs(lhs))
        print(f'trial {trial} G^{i}_{j}: {lhs.real:+.12e}  8piT: {rhs.real:+.12e}')
print('worst relative mismatch (and |G^t_r|):', worst)
