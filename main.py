import numpy as np
from transporte import Re_leito, λer, Der, ra

#vazão de entrada
FA = 500 #mol/s

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
Cp = 2000 #J/(kg.K)
ΔHr = 91956 #J/mol
M = 58e-3 #kg/mol
ρG = 1 #kg/m³
λg = 0.05 #W/m.K
λs = 170 #W/m.K
μg = 2e-05 #Pa.s

Q = FA*M/ρG #m³/s
CA = FA/Q
v = Q/Ac #m³/s -> m/s
us = v*ε #velocidade superficial do gás
print(us)

#discretização do reator na direção axial
L_z = 5 #comprimento do reator
N_z = 20 #número de espaços na direção axial
h_z = L_z/N_z #step size
L_eval = np.linspace(0, L_z, N_z+1)

#discretização do reator na direção radial
d = 2*np.sqrt(Ac/np.pi) #m
N_r = 5 #número de espaços na direção radial
h_r = d/N_r #step size
L_eval = np.linspace(0, d, N_r+1)