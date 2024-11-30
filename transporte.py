import numpy as np 

def Re_leito(ρ:float, us:float, ε:float, dp:float, μ:float):
    """Calcula o número de Reynolds em um leito empacotado
    Argumentos:   
        ρ (float): densidade do fluido [kg/m3]  
        us (float): velocidade do fluido [m/s]
        ε (float): porosidade do leito [adimensional]
        dp (float): diâmetro médio das partículas do leito [mm]
        μ (float): viscosidade do fluido [Pa s]
    Retorna:
        float: número de Reynolds [adimensional]"""
    
    dp = dp*1e-3 #mm -> m
    return (ρ*us*ε*dp)/μ

def λer(T:float, λg:float, λs:float ,dp:float, ε:float, dt:float, Re:float):
    """Calcula a condutividade térmica efetiva em um leito empacotado
    Argumentos:   
        T (float): temperatura do fluido [K]  
        λg (float): condutividade térmica do fluido [W/mK]
        λs (float): condutividade térmica do leito [W/mK]
        dp (float): diâmetro médio das partículas do leito [mm]
        ε (float): porosidade do leito [adimensional]
        dt (float): diâmetro do reator [m]
        Re (float): número de Reynolds do leito [adimensional]
    Retorna:
        float: condutividade térmica efetiva [W/mK]"""
    
    dp = dp*1e-3 #mm -> m
    p = 0.95 #emissividade do sólido
    β = 0.95 #"a coefficient that depends on the particle geometry and the packing density, comprised between 0.9 and 1.0" (Froment)
    γ = 2/3
    dp = dp*1e-3 #mm -> m
    αrs = 0.1952 * p/(2-p) * T**3/100
    αrv = 0.1952/(1+ε/(2*(1-ε))*(1-p)/p) * T**3/100
    λer0 = λg*(ε*(1+β*(dp*αrv)/λg) + (β*(1-ε))/(1/(1/γ + (αrs*dp)/λg)+γ*λg/λs))

    return λer0 + 0.0025/(1+46*(dp/dt)**2)*Re


def Der(us:float, dp:float, dt:float):
    """Calcula a difusividade do fluido em um leito empacotado
    Argumentos:   
        us (float): velocidade do fluido [m/s]  
        dp (float): diâmetro médio das partículas do leito [mm]
        dt (float): diâmetro do reator [m]
    Retorna:
        float: difusividade efetiva do fluido [m2/s]"""
    
    dp = dp*1e-3 #mm -> m
    Ψ = 0.14/(1+46*(dp/dt)**2)
    return Ψ*us*dp

def ra(CT:np.array, T:float , P:float):
    """
    Função para calcular a taxa de reação.
    Fonte: Bischoff (1979)

    Parameters
    ----------
    CT
        Concentração molar  no reator (acetona, cetena, metano) [mol/s].
    T
        Temperatura [K].
    P
        Pressão [atm].
    Returns
    -------
    Taxa de reação. [mol/s]
    """
    Y = CT/sum(CT)
    Y_acetona = Y[0]
    p_acetona = P*Y_acetona #atm
    r_a = np.exp(22.780 - 26660/T)*p_acetona**1.5 #kmol/(m³ s)

    return r_a*1000 #kmol/(m³ s) -> mol/(m³ s)

if __name__ == "__main__":
    print(f"Der = {Der(2.5, 2, 1.5):.3e}")
    Re = Re_leito(0.3, 2.5, 0.4, 1.5, 40e-6)
    print(f"Re = {Re:0f}")
    print(f"λer = {λer(750+273, 70e-3, 170, 2, 0.4, 1.5, Re):.3e}") 