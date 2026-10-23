"""On-shell Euclidean action of the RNdS black hole in a cavity, Eq. (euclid-action), evaluated term by term
(bulk with R=4Lambda and F^2=-2Q^2/R^4, Gibbons-Hawking term on the wall, minus empty de Sitter with the same wall data),
compared with Eq. (action-result): I = [R_B(sqrt(1-H^2R_B^2)-sqrt(chi_B)) - Phi_B Q]/T_B - pi R_A^2."""
import mpmath as mp
mp.mp.dps = 30
def check(RB, H, Q, RA):
    Lam = 3*H**2
    M = (RA + Q**2/RA - H**2*RA**3)/2            # f(R_A)=0
    f = lambda R: 1 - 2*M/R + Q**2/R**2 - H**2*R**2
    fp = lambda R: mp.diff(f, R)
    fB = f(RB); kap = fp(RA)/2
    TB = kap/(2*mp.pi*mp.sqrt(fB))
    PhiB = Q/mp.sqrt(fB)*(1/RA - 1/RB)
    c = 2*mp.pi*TB*mp.sqrt(fB)                   # n = sqrt(f)/c
    bulk = -(mp.mpf(1)/(16*mp.pi))*2*mp.pi*4*mp.pi/c*mp.quad(lambda R: (2*Lam + 2*Q**2/R**4)*R**2, [RA, RB])
    ghy = -(mp.mpf(1)/(8*mp.pi))*8*mp.pi**2*mp.sqrt(fB)*(fp(RB)/(2*c*mp.sqrt(fB))*RB**2 + 2*mp.sqrt(fB)/c*RB)
    f0 = lambda R: 1 - H**2*R**2; f0B = f0(RB); c0 = 2*mp.pi*TB*mp.sqrt(f0B)
    bulk0 = -(mp.mpf(1)/(16*mp.pi))*2*mp.pi*4*mp.pi/c0*mp.quad(lambda R: 2*Lam*R**2, [0, RB])
    ghy0 = -(mp.mpf(1)/(8*mp.pi))*8*mp.pi**2*mp.sqrt(f0B)*(mp.diff(f0, RB)/(2*c0*mp.sqrt(f0B))*RB**2 + 2*mp.sqrt(f0B)/c0*RB)
    I_direct = bulk + ghy - (bulk0 + ghy0)
    I_paper = (RB*(mp.sqrt(1 - H**2*RB**2) - mp.sqrt(fB)) - PhiB*Q)/TB - mp.pi*RA**2
    I_dS_paper = -RB*mp.sqrt(1 - H**2*RB**2)/TB
    print(f'R_B={RB} H={H} Q={Q} R_A={RA}: I_direct={mp.nstr(I_direct,15)}  I_paper={mp.nstr(I_paper,15)}  '
          f'diff={mp.nstr(I_direct-I_paper,3)} | I_dS direct={mp.nstr(bulk0+ghy0,12)} paper={mp.nstr(I_dS_paper,12)}')
for args in [(1, mp.mpf('0.3'), mp.mpf('0.2'), mp.mpf('0.5')), (1, mp.mpf('0.1'), mp.mpf('0.0'), mp.mpf('0.6')),
             (2, mp.mpf('0.2'), mp.mpf('0.5'), mp.mpf('1.1')), (1, mp.mpf('0'), mp.mpf('0.3'), mp.mpf('0.7'))]:
    check(*args)
