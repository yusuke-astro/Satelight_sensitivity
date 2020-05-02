"""
このプログラムはsensitivity02.pyでP_optを求めた計算から引き続きNEPなどを求めるようにしたコードである．
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

print("Define Pixel_pitch(D)...")
print("If you use D=16 -->> input 0, D=24 -->> input 1")
switch = 1 #= int(input(">>>"))

def define_D(f):
    #周波数に対してDが2つ存在するところで，使用するDの値を24にするか16にするかを決める仮想パラメータ
    #switch=1なら78,68,89GHzにおいてD=24.0でswitch=0ならD=16.0が選択される
    global switch
    if f == 40*GHz:
        D = 24.0
    if f == 60*GHz:
        D = 24.0
    if f == 78*GHz:
        if switch == 1:
            D = 24.0
        if switch == 0:
            D = 16.0
    if f == 50*GHz:
        D = 24.0
    if f == 68*GHz:
        if switch == 1:
            D = 24.0
        if switch == 0:
            D = 16.0
    if f == 89*GHz:
        if switch == 1:
            D = 24.0
        if switch == 0:
            D = 16.0
    if f == 100*GHz:
        D = 16.0
    if f == 119*GHz:
        D = 16.0
    if f == 140*GHz:
        D = 16.0
    return D

def define_Num_dat(f):
    global switch
    if f == 40*GHz:
        Num_det = 64.
    if f == 60*GHz:
        Num_det = 64.
    if f == 78*GHz:
        if switch == 1:
            Num_det = 64.
        if switch == 0:
            Num_det = 144.
    if f == 50*GHz:
        Num_det = 64.
    if f == 68*GHz:
        if switch == 1:
            Num_det = 64.
        if switch == 0:
            Num_det = 144.
    if f == 89*GHz:
        if switch == 1:
            Num_det = 64.
        if switch == 0:
            Num_det = 144.
    if f == 100*GHz:
        Num_det = 144.
    if f == 119*GHz:
        Num_det = 144.
    if f == 140*GHz:
        Num_det = 144.
    return Num_det

def P_CMB(f):
    global F
    #D = define_D(F)
    T = 2.725
    return st.ita_HWP_LFT(F)*st.ita_Apt_LFT(f,D)*st.ita_Mir(f)*st.ita_Mir(f)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T) * 1.000

def P_HWP(f):
    global F
    #D = define_D(F)
    T = 20.00
    T_r = 5.000
    HWP_e = st.ita_Apt_LFT(f,D)*st.ita_Mir(f)*st.ita_Mir(f)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T) * st.epsi_HWP_LFT(F)
    HWP_r = st.ita_Apt_LFT(f,D)*st.ita_Mir(f)*st.ita_Mir(f)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T_r) * st.r_HWP_LFT(F)
    return HWP_e + HWP_r

def P_Ape(f):
    global F
    #D = define_D(F)
    T = 5.000
    return st.ita_5K(f,D)*st.ita_Mir(f)*st.ita_Mir(f)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T)

def P_m1(f):
    global F
    T = 5.000
    Pri = st.ita_Mir(f)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T)
    Pri_e = Pri * st.epsi_Mir(f)
    Pri_r = Pri * st.r_Mir(f)
    return Pri_e + Pri_r

def P_m2(f):
    global F
    T = 5.000
    Sec = st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T)
    Sec_e = Sec * st.epsi_Mir(f)
    Sec_r = Sec * st.r_Mir(f)
    return Sec_e + Sec_r

def P_20K(f):
    global F
    #D = define_D(F)
    T = 20.0
    return st.ita_20K(F,D)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T) * 1.000

def P_2KF(f):
    global F
    T = 2.000
    T_r = 0.100
    Flt2K_e = st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T) * st.epsi_2KF(f)
    Flt2K_r = st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T_r) * st.r_2KF()
    return Flt2K_e + Flt2K_r

def P_Lens(f):
    global F
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
    global F
    global BW
    #print("F=",F,BW)
    return np.sqrt((integrate.quad(in_nep_ph, F-BW, F+BW)[0]))

def nep_g(P_opt):
    global F
    global BW
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
    T_cmb = 2.725
    kT = k_B*T_cmb

    return (1.0/k_B) * (( (h*f) / (T_cmb*(np.exp((h*f)/kT)-1.0)))**2) * np.exp((h*f)/kT)

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
t_obs = 94672800.*0.72
with open('sensitivity03.csv', 'w') as FILE:
    print("Freq[GHz],Band_width[GHz],D(Pixel_Pich),P_CMB,P_HWP,P_APT,P_20K,P_m1,P_m2,P_2KF,P_Lens,P_Det,P_opt,\
    INTEG_P_CMB,INTEG_P_opt,NEP_ph,NEP_g,NEP_read,NEP_int,NEP_ext,NEP_det,dPdT_CMB,NET_det,NET_arr,sigma_s",file=FILE)
    for i in range(len(freq)):
        F = freq[i]
        #D = define_D(F)
        if i<=5:
            D=24.0
        else:
            D=16.0


        WL = c/freq[i]
        BW = st.width(F)
        INTEGRATE.append(integrate.quad(P_Opt, F-BW, F+BW)[0])
        NEP_ph.append(nep_ph(freq[i]))
        NEP_g.append(nep_g(INTEGRATE[i]))
        NEP_read.append(nep_read(NEP_ph[i],NEP_g[i]))
        NEP_int.append(nep_int(NEP_ph[i],NEP_g[i],NEP_read[i]))
        NEP_ext.append(np.sqrt(0.32*NEP_int[i]**2))
        NEP_det.append(np.sqrt(NEP_int[i]**2 + NEP_ext[i]**2))
        dPdT.append(integrate.quad(dPdT_CMB, F-BW, F+BW)[0])
        NET_det.append(NEP_det[i]/(np.sqrt(2.)*dPdT[i]))
        NET_arr.append(NET_det[i]/np.sqrt(define_Num_dat(freq[i])*0.8))
        sigma_s.append(np.sqrt((4.*np.pi*1.0*2.*NET_arr[i]**2)/t_obs)*(10800./np.pi))

        print("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(F/GHz,BW/GHz,D,P_CMB(freq[i]),P_HWP(freq[i]),P_Ape(freq[i]),\
        P_20K(freq[i]),P_m1(freq[i]),P_m2(freq[i]),P_2KF(freq[i]),P_Lens(freq[i]),P_Det(freq[i]),P_Opt(freq[i]),\
        integrate.quad(P_CMB, F-BW, F+BW)[0],INTEGRATE[i],NEP_ph[i]/a,NEP_g[i]/a,NEP_read[i]/a,NEP_int[i]/a,NEP_ext[i]/a,\
        NEP_det[i]/a,dPdT[i],NET_det[i]/u,NET_arr[i]/u,sigma_s[i]/u) ,file=FILE)
INTEGEATE = np.array(INTEGRATE)
NEP_ph = np.array(NEP_ph)
NEP_g = np.array(NEP_g)
NET_arr
sigma_s


df = pd.read_csv("sensitivity03.csv")
print(df)
