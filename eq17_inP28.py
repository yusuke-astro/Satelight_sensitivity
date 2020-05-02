import  numpy as np
from scipy import integrate
from my_module import my_sens_func as st
import matplotlib.pyplot as plt

GHz = 1e9
p = 1e-12
Tera=1e12
W = 0.08883*p
u = 1e-6
m = 1e-3
e_0 = 8.85418782*10**(-12)
h = 6.62607004*10**(-34)
k_B = 1.38064852*10**(-23)
c = 299792458

def equation(f):
    return (1.0/k_B)*((h*f)/(T_cmb*(np.exp((h*f)/(k_B*T_cmb))-1.0)))**2 * np.exp((h*f)/(k_B*T_cmb))



#f = [40,50,60,68,78,89,100,119,140]
freq = np.array([40,50,60,68,78,89,100,119,140,166,195,235,280,337,402])*GHz
T_cmb = 2.725
INTEGRATE = []
for i in range(len(freq)):
    #print(freq[i])
    WL = c/freq[i]
    F = freq[i]
    BW = st.width(F)
    #print(BW)

    INTEGRATE.append(integrate.quad(equation, F-BW, F+BW)[0])
INTEGRATE = np.array(INTEGRATE)
INTEGRATE

print(freq[8], INTEGRATE[8])

ratio = INTEGRATE[8]/INTEGRATE
print(ratio)
plt.figure()
plt.grid()
plt.title("dP(f)/dT_CMB")
plt.xlabel("Frequency[GHz]")
plt.ylabel("P/T")
plt.plot(freq*1e-9, INTEGRATE, "o")

plt.figure()
plt.grid()
plt.title("[dP(140GHz)/dT_CMB]/[dP(f)/dT_CMB]")
plt.xlabel("Frequency[GHz]")
plt.ylabel("Ratio")
plt.plot(freq*1e-9, ratio, "o")
