import numpy as np
from transporte import Re_leito, λer, Der, ra, init_estimativa

"""
Os vetores de concentração (C) e temperatura (T) são matrizes bidimensionais organizadas da seguinte forma:

- As linhas correspondem à direção axial (z) com N_z + 1 pontos:
  - z = 0 representa a entrada do reator.
  - z = L representa a saída do reator.

- As colunas correspondem à direção radial (r) com N_r + 1 pontos:
  - r = 0 representa o centro do reator.
  - r = R representa a parede do reator.

Cada elemento C[i, j] ou T[i, j] representa o valor na posição:
  - i na direção axial: z_i = i * Δz.
  - j na direção radial: r_j = j * Δr.

Exemplo:
- C[0, 0]: Concentração na entrada (z = 0) e centro do reator (r = 0).
- T[N_z, N_r]: Temperatura na saída (z = L) e na parede (r = R).
"""

#vazão de entrada (acetona pura)
Fin = 500 #mol/s

#temperatura de entrada
Tin = 750 + 273.15 #K, entrada
Tr = 25 + 273.15 #K, ambiente
Tw = 65 + 273.15 #K, parede do trocador
Pin = 1 #atm

#características do leito
ε = 0.4
dp = 2 #mm
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
L = 5 #comprimento do reator
N_z = 20 #número de espaços na direção axial
Δz = L/N_z #step size
Z_eval = np.linspace(0, L, N_z+1)

#discretização do reator na direção radial
R = np.sqrt(Ac/np.pi) #m
N_r = 5 #número de espaços na direção radial
Δr = R/N_r #step size
R_eval = np.linspace(0, R, N_r+1)

#criando as estimativas iniciais
X_est = 0.8
Tout_est = Tin-150
C_init, T_init = init_estimativa(N_z, N_r, Cin, Tin, Tout_est, R, X_est)
print(C_init, "\n", T_init)

