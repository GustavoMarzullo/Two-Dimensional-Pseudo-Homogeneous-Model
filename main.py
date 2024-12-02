import numpy as np
from scipy.optimize import root
from transporte import Re_leito, λer, Der, ra, init_estimativa

"""
Os vetores de concentração (C) e temperatura (T) são matrizes bidimensionais organizadas da seguinte forma:

- As linhas correspondem à direção axial (z) com N_z + 1 pontos:
  - z = 0 representa a entrada do reator.
  - z = L representa a saída do reator.

- As colunas correspondem à direção radial (r) com N_r + 1 pontos:
  - r = 0 representa o centro do reator.
  - r = R representa a parede do reator.

Cada elemento C[z, r] ou T[z, r] representa o valor na posição:
  - z na direção axial: Z = z * Δz.
  - r na direção radial: R = r * Δr.

Exemplo:
- C[0, 0]: Concentração na entrada (z = 0) e centro do reator (r = 0).
- T[N_z, N_r]: Temperatura na saída (z = L) e na parede (r = R).
"""

def pprint(X):
     print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in X]))

#vazão de entrada (acetona pura)
Fin = 500 #mol/s

#temperatura de entrada
Tin = 750 + 273.15 #K, entrada
Tr = 25 + 273.15 #K, ambiente
Tw = 65 + 273.15 #K, parede do trocador
Pin = 1 #atm

#características do leito
ε = 0.5
dp = 4 #mm
Ac = 7 #m²

#propriedades físico-químicas do fluido e da reação (fonte: aspen)
Cp = 2500 #J/(kg.K)
ΔHr = 91956 #J/mol
M = 58e-3 #kg/mol
ρG = 0.6 #kg/m³
λg = 0.09 #W/m.K
λs = 170 #W/m.K 
μg = 2.6e-05 #Pa.s
αw = 0.156e3 #J/(m².s.K)

Qin = Fin*M/ρG #m³/s
Cin = Fin/Qin
v = Qin/Ac #m³/s -> m/s
us = v*ε #velocidade superficial do gás

#discretização do reator na direção axial
L = 10 #comprimento do reator
N_z = 5 #número de espaços na direção axial
Δz = L/N_z #step size
Z_eval = np.linspace(0, L, N_z+1)

#discretização do reator na direção radial
R = np.sqrt(Ac/np.pi) #m
N_r = 5 #número de espaços na direção radial
Δr = R/N_r #step size
R_eval = np.linspace(0, R, N_r+1)

#criando a função objetivo
def fobj(vars):
    #array flat para matriz
    C = vars[:(N_z + 1) * (N_r + 1)].reshape((N_z + 1, N_r + 1)) #concentração
    T = vars[(N_z + 1) * (N_r + 1):].reshape((N_z + 1, N_r + 1))  #temperatura

    #calculando as concentrações
    C_cetena = Cin - C
    C_metano = Cin - C

    #arrays de resíduo, os nans são para verificar se todos os valores estão sendo preenchidos
    C_res = np.zeros((N_z + 1, N_r + 1)) * np.nan
    T_res = np.zeros((N_z + 1, N_r + 1)) * np.nan


    _Der = Der(us, dp, R)
    Re = Re_leito(ρG, us, ε, dp, μg)
    for z in range(1, N_z):
        for r in range(1, N_r):
            CT = np.array([C[z, r], C_cetena[z, r], C_metano[z, r]])
            _ra = ra(CT, T[z, r], Pin)
            _λder = λer(T[z, r], λg, λs, dp, ε, R, Re)

            C_res[z,r] = ε*_Der*(1/(2*Δr**2)*(C[z+1][r+1] - 2*C[z+1][r] + C[z+1][r-1] + C[z][r+1] - 2*C[z][r] + C[z][r-1]) + 1/R_eval[r] * 1/(4*Δr) * (C[z+1][r+1] - C[z+1][r-1] + C[z][r+1] - C[z][r-1])) - us*(C[z][r] - C[z-1][r])/Δz - _ra #concentração

            T_res[z,r] = _λder*(1/(2*Δr**2)*(T[z+1][r+1] - 2*T[z+1][r] + T[z+1][r-1] + T[z][r+1] - 2*T[z][r] + T[z][r-1]) + 1/R_eval[r] * 1/(4*Δr) * (T[z+1][r+1] - T[z+1][r-1] + T[z][r+1] - T[z][r-1])) - us*ρG*Cp*(T[z+1][r] - T[z-1][r])/(2*Δz) + _ra*(-ΔHr) #temperatura

    for r in range(1, N_r):
        z = N_z
        C_res[z, r] =  ε*_Der*(1/(Δr)*(C[z][r] - C[z][r-1]) + 1/R_eval[r] * 1/(Δr) * (C[z][r] - C[z][r-1])) - us*(C[z][r] - C[z-1][r])/Δz - _ra #concentração
        T_res[z, r] = _λder*(1/(Δr)*(T[z][r] - T[z][r-1]) + 1/R_eval[r] * 1/(Δr) * (T[z][r] - T[z][r-1])) - us*ρG*Cp*(T[z][r] - T[z-1][r])/Δz + _ra*(-ΔHr) #temperatura

    #condições de contorno
    for r in range(0, N_r+1):
        C_res[0,r] = C[0,r] - Cin #C.C. 1
        T_res[0,r] = T[0,r] - Tin #C.C. 2
    for z in range(1, N_z+1):
        C_res[z,0] = C[z,1] - C[z,0] #C.C. 3
        C_res[z, N_r] = C[z, N_r] - C[z, N_r-1] #C.C. 4
        T_res[z, 0] = T[z, 1] - T[z, 0] #C.C. 5
        T_med = (T[z, N_r] + T[z, N_r-1])/2
        T_res[z, N_r] = (T[z, N_r] - T[z, N_r-1])/Δr - αw/λer(T_med, λg, λs, dp, ε, R, Re)*(Tr-Tw)/100 #C.C. 6
    
    
    res = np.concatenate([C_res.flatten(), T_res.flatten()])

    return res

#criando as estimativas iniciais
X_est = 0.4
Tout_est = Tin-100
C_init, T_init = init_estimativa(N_z, N_r, Cin, Tin, Tout_est, R, X_est)
pprint(C_init)
pprint(T_init)
estimativa = np.concatenate([C_init.flatten(), T_init.flatten()])

#resolvendo o problema
print("\n")
print(f"C0 = {Cin:.1f} mol/m³")
resultado = root(fobj, estimativa, method='lm').x
residuo = np.linalg.norm(fobj(resultado))
print(f"\n Resíduo:{residuo:.3g}")

if residuo < 1e-3:
    C_res = resultado[:(N_z + 1) * (N_r + 1)].reshape((N_z + 1, N_r + 1)) #concentração
    T_res = resultado[(N_z + 1) * (N_r + 1):].reshape((N_z + 1, N_r + 1))-273.15  #temperatura
    print("Concentração (mol/m³):")
    pprint(C_res)
    print("Temperatura (ºC):")
    pprint(T_res)
    

   
    

             



