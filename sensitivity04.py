"""
このプログラムはsensitivity03.pyでP_optを求めた計算から引き続きSensitivityなどを求めるようにしたコードである．
switchによるDの制御をfor loop内のiの制御に置き換えた．
"""

import  numpy as np
from scipy import integrate
from my_module import my_sens_func as st
import matplotlib.pyplot as plt
import pandas as pd

GHz = 1e9
p = 1e-12
a =1e-18
Tera=1e12
W = 0.08883*p
u = 1e-6
m = 1e-3
e_0 = 8.85418782*10**(-12)
h = 6.62607004*10**(-34)
k_B = 1.38064852*10**(-23)
c = 299792458

def P_CMB(f):
    T = 2.725
    return st.ita_HWP_LFT(F)*st.ita_Apt_LFT(f,D)*st.ita_Mir(f)*st.ita_Mir(f)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T) * 1.000

def P_HWP(f):
    T = 20.00
    T_r = 5.000
    HWP_e = st.ita_Apt_LFT(f,D)*st.ita_Mir(f)*st.ita_Mir(f)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T) * st.epsi_HWP_LFT(F)
    HWP_r = st.ita_Apt_LFT(f,D)*st.ita_Mir(f)*st.ita_Mir(f)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T_r) * st.r_HWP_LFT(F)
    return HWP_e + HWP_r

def P_Ape(f):
    T = 5.000
    return st.ita_5K(f,D)*st.ita_Mir(f)*st.ita_Mir(f)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T)

def P_m1(f):
    T = 5.000
    Pri = st.ita_Mir(f)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T)
    Pri_e = Pri * st.epsi_Mir(f)
    Pri_r = Pri * st.r_Mir(f)
    return Pri_e + Pri_r

def P_m2(f):
    T = 5.000
    Sec = st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T)
    Sec_e = Sec * st.epsi_Mir(f)
    Sec_r = Sec * st.r_Mir(f)
    return Sec_e + Sec_r

def P_20K(f):
    T = 20.0
    return st.ita_20K(F,D)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T) * 1.000

def P_2KF(f):
    T = 2.000
    T_r = 0.100
    Flt2K_e = st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T) * st.epsi_2KF(f)
    Flt2K_r = st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T_r) * st.r_2KF()
    return Flt2K_e + Flt2K_r

def P_Lens(f):
    T = 0.100
    T_r = 5.000
    Lens_e = st.ita_det_LFT()*st.Bose(f,T) * st.epsi_Lenslet(f)
    Lens_r = st.ita_det_LFT()*st.Bose(f,T_r) * st.r_Lenslet()
    return Lens_e + Lens_r

def P_Det(f):
    T = 0.100
    Det_e = st.Bose(f,T) * st.r_det_LFT()
    Det_r = st.Bose(f,T) * st.ita_det_LFT()
    return Det_e + Det_r

def P_Opt(f):
    return P_CMB(f)+P_HWP(f)+P_Ape(f)+P_m1(f)+P_m2(f)+P_20K(f)+P_2KF(f)+P_Lens(f)+P_Det(f)

def in_nep_ph(f):
    return 2.0*h*f*P_Opt(f) + 2.0*(P_Opt(f))**2

def nep_ph(f):
    return np.sqrt((integrate.quad(in_nep_ph, F-BW, F+BW)[0]))

def nep_g(P_opt):
    T_b = 0.100
    T_c = 0.171
    n = 3.0
    ratio = T_c/T_b
    A = (n+1.)**2/((2.*n)+3.)
    B = ((ratio**(2.*n+3.))-1.)/(((ratio**(n+1.))-1.)**2.)
    value = np.sqrt(4.0*k_B*2.5*P_opt*T_b*A*B)
    return value

def nep_read(a,b):
    value = np.sqrt(0.21)*np.sqrt(a**2+b**2)
    return value

def nep_int(a,b,c):
    return np.sqrt(a**2+b**2+c**2)


def dPdT_CMB(f):
    def eta(f):
        return st.ita_HWP_LFT(F)*st.ita_Apt_LFT(f,D)*st.ita_Mir(f)*st.ita_Mir(f)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()

    T_cmb = 2.725
    kT = k_B*T_cmb
    return (eta(f)/k_B) * (( (h*f) / (T_cmb*(np.exp((h*f)/kT)-1.0)))**2) * np.exp((h*f)/kT)

freq = np.array([40,60,78,50,68,89,68,89,119,78,100,140])*GHz

print("CSV-file generated...")
INTEGRATE = []#積分されたP_Optが入る
NEP_ph = []
NEP_g = []
NEP_read = []
NEP_int = []
NEP_ext = []
NEP_det = []
dPdT = []
NET_det = []
NET_arr = []
sigma_s = []
t_obs = 94672800.*0.95*0.95
with open('sensitivity04.csv', 'w') as FILE:
    print("Freq[GHz],Band_width[GHz],D(Pixel_Pich),P_CMB,P_HWP,P_APT,P_20K,P_m1,P_m2,P_2KF,P_Lens,P_Det,P_opt,\
    INTEG_P_CMB,INTEG_P_opt[p],NEP_ph[a],NEP_g[a],NEP_read[a],NEP_int[a],NEP_ext[a],NEP_det[a],NET_det[u],NET_arr[u],dPdT_CMB,sigma_s[u]",file=FILE)
    for i in range(len(freq)):
        F = freq[i]
        WL = c/freq[i]
        BW = st.width(F)

        if i <= 5:
            D = 24.0
            N_dat = 64.
        else:
            D = 16.0
            N_dat = 144.

        INTEGRATE.append(integrate.quad(P_Opt, F-BW, F+BW)[0])
        NEP_ph.append(nep_ph(freq[i]))
        NEP_g.append(nep_g(INTEGRATE[i]))
        NEP_read.append(nep_read(NEP_ph[i],NEP_g[i]))
        NEP_int.append(nep_int(NEP_ph[i],NEP_g[i],NEP_read[i]))
        NEP_ext.append(np.sqrt(0.32*NEP_int[i]**2))
        NEP_det.append(np.sqrt(NEP_int[i]**2 + NEP_ext[i]**2))
        dPdT.append(integrate.quad(dPdT_CMB, F-BW, F+BW)[0])
        NET_det.append(NEP_det[i]/(np.sqrt(2.)*dPdT[i]))
        NET_arr.append(NET_det[i]/np.sqrt(N_dat*0.8))
        sigma_s.append(np.sqrt((4.*np.pi*1.0 * 2. * NET_arr[i]**2)/t_obs)*(10800./np.pi))

        print("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(F/GHz,BW/GHz,D,P_CMB(freq[i]),P_HWP(freq[i]),P_Ape(freq[i]),\
        P_20K(freq[i]),P_m1(freq[i]),P_m2(freq[i]),P_2KF(freq[i]),P_Lens(freq[i]),P_Det(freq[i]),P_Opt(freq[i]),\
        integrate.quad(P_CMB, F-BW, F+BW)[0],INTEGRATE[i]/p,NEP_ph[i]/a,NEP_g[i]/a,NEP_read[i]/a,NEP_int[i]/a,NEP_ext[i]/a,\
        NEP_det[i]/a,NET_det[i]/u,NET_arr[i]/u,dPdT[i],sigma_s[i]/u) ,file=FILE)

df = pd.read_csv("sensitivity04.csv")
df
NET_overlaped_78GHz = np.sqrt(1./(1./(df.loc[2][15:23]**2 + df.loc[7][15:23]**2)).sum())
NET_overlaped_78GHz

NET_overlaped_68GHz = np.sqrt(1./(1./(df.loc[4][15:23]**2 + df.loc[6][15:23]**2)).sum())
NET_overlaped_89GHz = np.sqrt(1./(1./(df.loc[5][15:23]**2 + df.loc[7][15:23]**2)).sum())


hasebe_GHz = [40,
50,
60,
68,
78,
89,
100,
119,
140]

hasebe_sensi = [39.76,
25.76,
20.69,
12.72,
10.39,
8.95,
6.43,
4.3,
4.43]
plt.figure()
plt.title("Sensitivity")
plt.xlabel("Frequency[GHz]")
plt.ylabel("\sigma_s")
plt.plot(df["Freq[GHz]"],df["sigma_s[u]"],"o",color="red",label="takase", alpha=0.5)
plt.plot(hasebe_GHz,hasebe_sensi,"o",color="blue",label="hasebe-san", alpha=0.5)
plt.legend()
plt.grid()
