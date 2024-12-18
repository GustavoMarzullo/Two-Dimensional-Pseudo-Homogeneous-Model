import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from auxiliares import Re_leito, λer, Der, ra, init_estimativa
from sklearn.preprocessing import normalize


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
Fin = 1000 #mol/s

#temperatura de entrada
Tin = 750 + 273.15 #K, entrada
Tr = 25 + 273.15 #K, ambiente
Tw = 65 + 273.15 #K, parede do reator
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
λs = 3 #W/m.K 
μg = 2.6e-05 #Pa.s

αw = 50 #J/(m².s.K)

Qin = Fin*M/ρG #m³/s
Cin = Fin/Qin
v = Qin/Ac #m³/s -> m/s
us = v*ε #velocidade superficial do gás

#discretização do reator na direção axial
L = 10 #comprimento do reator, m
N_z = 31  # Número de pontos na direção axial (inclui entrada e saída)
Δz = L/(N_z - 1)  # Step size entre pontos
Z_eval = np.linspace(0, L, N_z)

#discretização do reator na direção radial
R = np.sqrt(Ac/np.pi) #raio do reator,m
N_r = 51  # Número de pontos na direção radial (inclui centro e parede)
Δr = R/(N_r - 1)  # Step size entre pontos
R_eval = np.linspace(0, R, N_r)

#criando a função objetivo
def fobj(vars):
    #array flat para matriz
    C = vars[:(N_z) * (N_r)].reshape((N_z , N_r)) #concentração
    T = vars[(N_z) * (N_r):].reshape((N_z, N_r))  #temperatura

    #calculando as concentrações
    C_cetena = Cin - C
    C_metano = Cin - C

    #arrays de resíduo, os nans são para verificar se todos os valores estão sendo preenchidos
    C_res = np.zeros((N_z, N_r)) * np.nan
    T_res = np.zeros((N_z, N_r)) * np.nan


    _Der = Der(us, dp, R)
    Re = Re_leito(ρG, us, ε, dp, μg)

    for z in range(1, N_z-1):
        for r in range(1, N_r-1):
          CT = np.array([C[z, r], C_cetena[z, r], C_metano[z, r]])
          _ra = ra(CT, T[z, r], Pin)
          _λder = λer(T[z, r], λg, λs, dp, ε, R, Re)

          C_res[z, r] = ε * _Der * (1 / (2 * Δr**2) * (C[z+1, r+1] - 2 * C[z+1, r] + C[z+1, r-1] + C[z, r+1] - 2 * C[z, r] + C[z, r-1]) + 1 / R_eval[r] * 1 / (4 * Δr) * (C[z+1, r+1] - C[z+1, r-1] + C[z, r+1] - C[z, r-1])) - us * (C[z, r] - C[z-1, r]) / Δz - _ra

          T_res[z, r] = _λder * (1 / (2 * Δr**2) * (T[z+1, r+1] - 2 * T[z+1, r] + T[z+1, r-1] + T[z, r+1] - 2 * T[z, r] + T[z, r-1]) + 1 / R_eval[r] * 1 / (4 * Δr) * (T[z+1, r+1] - T[z+1, r-1] + T[z, r+1] - T[z, r-1])) - us * ρG * Cp * (T[z, r] - T[z-1, r]) / Δz + _ra * (-ΔHr)

    for r in range(1, N_r-1):
        z = N_z-1
        CT = np.array([C[z, r], C_cetena[z, r], C_metano[z, r]])
        _ra = ra(CT, T[z, r], Pin)
        _λder = λer(T[z, r], λg, λs, dp, ε, R, Re)
        C_res[z, r] = ε * _Der * (1 / Δr**2 * (C[z, r] - 2*C[z, r-1] + C[z, r-2]) + 1 / R_eval[r] * 1 / Δr * (C[z, r] - C[z, r-1])) - us * (C[z, r] - C[z-1, r]) / Δz - _ra

        T_res[z, r] = _λder * (1 / Δr**2 * (T[z, r] - 2*T[z, r-1] + T[z, r-2]) + 1 / R_eval[r] * 1 / Δr * (T[z, r] - T[z, r-1])) - us * ρG * Cp * (T[z, r] - T[z-1, r]) / Δz + _ra * (-ΔHr)

    #condições de contorno
    for r in range(0, N_r):
        C_res[0,r] = C[0,r] - Cin #C.C. 1
        T_res[0,r] = T[0,r] - Tin #C.C. 2
    for z in range(1, N_z):
        C_res[z,0] = -3*C[z,0] + 4*C[z,1] - C[z,2] #C.C. 3
        C_res[z, -1] = 3*C[z, -1] - 4*C[z, -2] + C[z, -3]  #C.C. 4
        T_res[z, 0] = -3*T[z,0] + 4*T[z,1] - T[z,2] #C.C. 5
        T_res[z, -1] = (3*T[z, -1] - 4*T[z, -2] + T[z, -3])/(2*Δr) - αw/λer(T[z, -1], λg, λs, dp, ε, R, Re)*(Tr-Tw) #C.C. 6
    
    res = np.concatenate([C_res.flatten(), T_res.flatten()])

    return res

#criando as estimativas iniciais
X_est = 0.2
Tout_est = Tin-150
C_init, T_init = init_estimativa(N_z, N_r, Cin, Tin, Tout_est, R, X_est)
estimativa = np.concatenate([C_init.flatten(), T_init.flatten()])

#resolvendo o problema
print("\n")
print(f"C0 = {Cin:.1f} mol/m³")
resultado = root(fobj, estimativa, method='lm')
residuo = np.linalg.norm(fobj(resultado.x))
print(f"\nResíduo:{residuo:.3g}\n")

if residuo < 1e-3:
  C_res = resultado.x[:(N_z) * (N_r)].reshape((N_z, N_r)) #concentração
  T_res = resultado.x[(N_z) * (N_r):].reshape((N_z, N_r))-273.15  #temperatura
  X_final = (Cin-C_res[-1].mean())/Cin
  print(f"Conversão = {X_final:.1%}\n")
  """ print("Concentração (mol/m³):")
  pprint(C_res)
  print("\nTemperatura (ºC):")
  pprint(T_res) """

  ##PRINTANDO
  
  X, Y = np.meshgrid(R_eval, Z_eval)  # Usando np.meshgrid para criar as malhas

  #plt.pcolormesh(X, Y, T_res, cmap='seismic', shading='gouraud' )
  plt.contourf(X, Y, T_res, cmap='seismic', levels=20)
  plt.colorbar(label="Tempeatura (ºC)")
  plt.title("Distribuição de Temperatura no Reator")
  plt.ylabel("Comprimento (z) [m]")
  plt.xlabel("Raio (r) [m]")
  plt.savefig("Temperatura.pdf")
  plt.show()

  #plt.pcolormesh(X, Y, C_res, cmap='seismic', shading='gouraud')
  plt.contourf(X, Y, C_res, cmap='seismic', levels=20)
  plt.colorbar(label="Concentração (mol/m³)")
  plt.title(f"Distribuição de Concentração no Reator")
  plt.ylabel("Comprimento (z) [m]") 
  plt.xlabel("Raio (r) [m]")
  plt.savefig("Concentração.pdf")
  plt.show()

  ### CHAT GPT
  import pandas as pd
  # Create DataFrames for the results
  C_res_df = pd.DataFrame(C_res, columns=[f"R{i+1}" for i in range(N_r)])
  T_res_df = pd.DataFrame(T_res, columns=[f"R{i+1}" for i in range(N_r)])

  # Create a DataFrame for metadata
  metadata_df = pd.DataFrame({'Parameter': ['N_r', 'N_z'], 'Value': [N_r, N_z]})

  # Save everything into a single Excel file with separate sheets
  output_file = "results_with_metadata.xlsx"
  with pd.ExcelWriter(output_file) as writer:
      metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
      C_res_df.to_excel(writer, sheet_name='Concentration', index=False)
      T_res_df.to_excel(writer, sheet_name='Temperature', index=False)

  print(f"Results and metadata saved to {output_file}")

   
    

             



