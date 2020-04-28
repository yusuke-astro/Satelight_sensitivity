"""
このプログラムはsensitivity01.pyで標準出力させていた計算結果をcsv出力できるようにしたものである．
D=16で計算したい場合は0を，D=24で計算したいときは1を標準入力する．
"""

import  numpy as np
from scipy import integrate
from my_module import my_sens_func as st
import matplotlib.pyplot as plt
import pandas as pd

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
############################################
print("Define Pixel_pitch(D)...")
print("If you use D=16 -->> input 0, D=24 -->> input 1")
switch = int(input(">>>"))
#freqに対してDが2つあるところがある
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

def P_CMB(f):
    global F
    D = define_D(F)
    T = 2.725
    return st.ita_HWP_LFT(F)*st.ita_Apt_LFT(f,D)*st.ita_Mir(f)*st.ita_Mir(f)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T) * 1.000

def P_HWP(f):
    global F
    D = define_D(F)
    T = 20.00
    T_r = 5.000
    HWP_e = st.ita_Apt_LFT(f,D)*st.ita_Mir(f)*st.ita_Mir(f)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T) * st.epsi_HWP_LFT(F)
    HWP_r = st.ita_Apt_LFT(f,D)*st.ita_Mir(f)*st.ita_Mir(f)*st.ita_2K(f)*st.ita_Lenslet(f)*st.ita_det_LFT()*st.Bose(f,T_r) * st.r_HWP_LFT(F)
    return HWP_e + HWP_r

def P_Ape(f):
    global F
    D = define_D(F)
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
    D = define_D(F)
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

#freq_note = np.array([40,60,78,50,68,89,119,100,140])
#hasebe = np.array([0.36,0.29,0.28,0.38,0.28,0.28,0.36,0.34,0.40,0.36,0.33,0.38])
if switch == 1:
    hasebe = np.array([0.36,0.29,0.28,0.38,0.28,0.28,0.40,0.33,0.38])
if switch == 0:
    hasebe = np.array([0.36,0.29,0.36,0.38,0.36,0.34,0.40,0.33,0.38])



freq = np.array([40,60,78,50,68,89,119,100,140])*GHz

print("CSV-file generated...")
INTEGRATE = []
with open('sensitivity.csv', 'w') as FILE:
    print("Freq[GHz],Band_width[GHz],D(Pixel_Pich),P_CMB,P_HWP,P_APT,P_20K,P_m1,P_m2,P_2KF,P_Lens,P_Det,P_opt,INTEG_P_CMB,INTEG_P_opt",file=FILE)
    for i in range(len(freq)):
        F = freq[i]
        D = define_D(F)
        WL = c/freq[i]
        BW = st.width(F)
        INTEGRATE.append(integrate.quad(P_Opt, F-BW, F+BW)[0])
        print("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(F/GHz,BW/GHz,D,P_CMB(freq[i]),P_HWP(freq[i]),P_Ape(freq[i]),\
        P_20K(freq[i]),P_m1(freq[i]),P_m2(freq[i]),P_2KF(freq[i]),P_Lens(freq[i]),P_Det(freq[i]),P_Opt(freq[i]),\
        integrate.quad(P_CMB, F-BW, F+BW)[0],INTEGRATE[i] ),file=FILE)
INTEGEATE = np.array(INTEGRATE)

df = pd.read_csv("sensitivity.csv")
df
#hasebe_D1 = np.array([0.36,0.29,0.28,0.38,0.28,])
delta = hasebe*p - INTEGEATE
delta
plt.figure()
plt.title("Integrated P_opt")
plt.xlabel("Freqency[GHz]")
plt.ylabel("INTEG_P_opt[pW]")
plt.plot(freq/GHz,hasebe,"o",label="Hasebe-san")
plt.plot(freq/GHz,INTEGEATE/p,"o",label="Takase")
plt.grid()
plt.legend()

plt.figure()
plt.title("Hasebe_san - Takase")
plt.xlabel("Freqency[GHz]")
plt.ylabel("Delta")
plt.plot(freq/GHz, delta,"o", label="Hase-Taka")
plt.grid()
plt.legend()

plt.show()
