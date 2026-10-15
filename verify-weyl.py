import sympy as sp, random
t,r,th,ph = sp.symbols('t r theta phi')
k,M,Q = sp.symbols('k M Q')
A0,A1,A2 = sp.symbols('A0 A1 A2')
a = sp.Function('a')(t)
mu = M*sp.sqrt(1+k*r**2)/(2*a*r); lam = Q*sp.sqrt(1+k*r**2)/(2*a*r)
Xi = (1+mu)**2 - lam**2
N = (1-mu**2+lam**2)/Xi;  B = a*Xi/(1+k*r**2)          # Vaidya-Shah, Eq. (43)
X = [t,r,th,ph]
g = sp.diag(-N**2, B**2, B**2*r**2, B**2*r**2*sp.sin(th)**2)
gi = [1/g[i,i] for i in range(4)]
G = [[[sum((gi[l] if s==l else 0)*(sp.diff(g[s,m],X[n])+sp.diff(g[s,n],X[m])-sp.diff(g[m,n],X[s])) for s in range(4))/2
       for n in range(4)] for m in range(4)] for l in range(4)]
def Riem(p,s,m,n):
    e = sp.diff(G[p][n][s],X[m]) - sp.diff(G[p][m][s],X[n])
    return e + sum(G[p][m][l]*G[l][n][s] - G[p][n][l]*G[l][m][s] for l in range(4))
Ric = [sum(Riem(p,s,p,s) for p in range(4)) for s in range(4)]           # diagonal Ricci
Rs = sum(gi[i]*Ric[i] for i in range(4))
C = gi[3]*Riem(2,3,2,3) - (gi[2]*Ric[2] + gi[3]*Ric[3])/2 + Rs/6         # C^{th ph}_{th ph}
R = B*r
chi = sp.diff(R,r)**2/B**2 - sp.diff(R,t)**2/N**2
EMS = R/2*(1-chi)
rho_tot = -(gi[0]*Ric[0] - Rs/2)/(8*sp.pi)                               # -G^t_t/(8 pi)
def num(e, vals):
    e = e.subs(sp.Derivative(a,(t,2)),A2).subs(sp.Derivative(a,t),A1).subs(a,A0)
    return sp.N(e.subs(vals), 30)
random.seed(1)
worst = {"Weyl": 0, "EMS": 0, "rho": 0}
for trial in range(8):
    kk = random.choice([-1,1])*random.uniform(0.01,0.3); QQ = random.choice([0, random.uniform(0.1,0.8)])
    vals = {r: random.uniform(0.6,1.4), th: 1.1, k: kk, M: 1.0, Q: QQ, A0: random.uniform(0.8,1.5),
            A1: random.uniform(0.05,0.5), A2: random.uniform(-0.4,0.4)}
    Rv = num(R,vals); q = QQ*(1+kk*vals[r]**2)**sp.Rational(-3,2)
    MW_formula = 1.0*(1+kk*vals[r]**2)**sp.Rational(-5,2) - q**2/Rv           # claimed charged Weyl mass
    MW_direct = Rv**3/2*num(C,vals)                                          # -R^3 Psi2 with Psi2 = -C/2
    H = vals[A1]/vals[A0]; m_ = num(mu,vals); X_ = num(Xi,vals)
    rho = (3*H**2 + 12*kk*(1+m_)/(vals[A0]**2*X_**3))/(8*sp.pi)               # Eq. (59)
    EMS_formula = 1.0*(1+kk*vals[r]**2)**sp.Rational(-5,2) + 4*sp.pi/3*rho*Rv**3 - q**2/(2*Rv)   # Eq. (56)
    worst["Weyl"] = max(worst["Weyl"], abs(MW_direct-MW_formula))
    worst["EMS"]  = max(worst["EMS"],  abs(num(EMS,vals)-EMS_formula))
    worst["rho"]  = max(worst["rho"],  abs(num(rho_tot,vals)-(rho+q**2/(8*sp.pi*Rv**4))))
    if trial < 3: print(f"k={kk:+.3f} Q={QQ:.2f}: -R^3 Psi2 = {float(MW_direct):.12f}  vs  M(1+kr^2)^(-5/2)-q^2/R = {float(MW_formula):.12f}"
                        f"  |  M(1+kr^2)^(-5/2) alone = {float((1+kk*vals[r]**2)**-2.5):.12f}")
print("max deviations over 8 random points (general a, adot, addot; k of both signs; Q=0 and Q!=0):", {k_: float(v) for k_,v in worst.items()})